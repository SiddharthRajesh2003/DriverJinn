import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional, Dict
from utils.logging_manager import get_logger
import numpy as np
import networkx as nx
import pandas as pd

logger = get_logger(__name__)

from model.support_models import CurvatureAwareGNN, ProjectionHead, BinaryClassifier
from model.multi_layer_attention import MultiLayerAttention
try:
    from graph_builder.curvature_calculator import EdgeCurvature
    EDGE_CURVATURE_AVAILABLE = True
except ImportError:
    EDGE_CURVATURE_AVAILABLE = False
    logger.warning("EdgeCurvature not available, using approximation methods only")





class ContrastiveDriverGenePredictor(nn.Module):
    """
    Model for cancer driver prediction with potential driver identification:
    - Binary classification: driver (1) vs non-driver (0)
    - Post-hoc identification of potential drivers from false positives
      based on curvature features, confidence, and node properties
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        projection_dim: int = 128,
        num_gnn_layers: int = 3,
        curvature_types: List[str] = ['positive', 'negative', 'both'],
        num_attention_heads: int = 4,
        temperature: float = 0.4,
        dropout: float = 0.2
    ):
        super().__init__()
        self.temperature = temperature
        self.curvature_types = curvature_types
        self.hidden_channels = hidden_channels
        
        # Encoder: CurvatureAwareGNN
        self.encoder = CurvatureAwareGNN(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=num_gnn_layers,
            curvature_types=curvature_types,
            use_attention=True,
            dropout=dropout
        )
        
        # Multi-Layer Attention for each curvature type
        self.layer_attentions = nn.ModuleDict({
            curv_type: MultiLayerAttention(
                hidden_dim=hidden_channels,
                num_heads=num_attention_heads,
                dropout=dropout
            ) for curv_type in self.curvature_types
        })
        
        # Cross-curvature attention to aggregate different curvature views
        self.cross_curvature_attention = MultiLayerAttention(
            hidden_dim=hidden_channels,
            num_heads=num_attention_heads,
            dropout=dropout
        )
        
        # Projection head for contrastive learning
        self.projection = ProjectionHead(
            input_dim=in_channels,
            hidden_dim=hidden_channels,
            out_dim=projection_dim
        )
        
        # Binary Classifier
        self.classifier = BinaryClassifier(
            input_dim=hidden_channels,
            hidden_dim=hidden_channels,
            dropout=dropout
        )
        
    def encode(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_curvature: torch.Tensor,
        return_attention: bool = True
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Encode graph into representation vector
        """
        # Get layer outputs for each curvature type
        curvature_outputs = self.encoder(x, edge_index, edge_curvature)
        
        attention_weights = {} if return_attention else None
        
        curvature_representations = []
        for curv_type in self.curvature_types:
            layer_outputs = curvature_outputs[curv_type]
            aggregated, attn = self.layer_attentions[curv_type](
                layer_outputs,
                return_attention = return_attention
            )
            curvature_representations.append(aggregated)
            
            if return_attention:
                attention_weights[f'{curv_type}_layer_attention'] = attn
        
        # Aggregate across curvature types            
        final_repr, cross_attn = self.cross_curvature_attention(
            curvature_representations,
            return_attention
        )
        
        if return_attention:
            attention_weights['cross_curvature_attention'] = cross_attn
            # Store individual curvature representations for analysis
            attention_weights['curvature_representations'] = {
                curv_type: rep for curv_type, rep in 
                zip(self.curvature_types, curvature_representations)
            }
    
        return final_repr, attention_weights
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_curvature: torch.Tensor,
        return_embeddings: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for binary classification
        
        Returns:
            logits: [num_nodes] binary logits
            embeddings: [num_nodes, hidden_channels] if return_embeddings=True
        """
        h, _ = self.encode(x, edge_index, edge_curvature)
        logits = self.classifier(h)
        
        if return_embeddings:
            return logits, h
        
        return logits, None
    
    def get_contrastive_projection(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_curvature: torch.Tensor
    ) -> torch.Tensor:
        """
        Get normalized projection for contrastive loss
        """
        h, _ = self.encode(x, edge_index, edge_curvature)
        z = self.projection(h)
        return F.normalize(z, dim=-1)
    
    def compute_contrastive_loss(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        node_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute NT-Xent (InfoNCE) contrastive loss between two views
        """
        if node_mask is not None:
            z1 = z1[node_mask]
            z2 = z2[node_mask]
        
        batch_size = z1.shape[0]
        z = torch.cat([z1, z2], dim = 0)
        sim_matrix = torch.mm(z, z.T) / self.temperature
        
        mask = torch.eye(batch_size, device=z.device)
        mask = mask.repeat(2, 2)
        mask = 1 - mask
        
        labels = torch.cat([
            torch.arange(batch_size, 2*batch_size, device=z.device),
            torch.arange(batch_size, device=z.device)
        ])
        
        sim_matrix = sim_matrix * mask
        exp_sim = torch.exp(sim_matrix)
        log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True))
        pos_sim = log_prob[torch.arange(2*batch_size, device=z.device)]
        loss = -pos_sim.mean()
        
        return loss
    
    def compute_classification_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        pos_weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute weighted binary cross-entropy loss
        
        Args:
            logits: [num_nodes] binary logits
            labels: [num_nodes] with values {0, 1}
            mask: [num_nodes] boolean mask for labeled nodes
            pos_weight: Weight for positive class (driver genes)
        """
        
        if mask is not None:
            logits = logits[mask]
            labels = labels[mask].float()
            
        if pos_weight is not None:
            loss = F.binary_cross_entropy_with_logits(
                logits, labels, pos_weight
            )
        else:
            loss = F.binary_cross_entropy_with_logits(
                logits, labels
            )
            
        return loss
    
    def compute_augmented_curvature(
        self,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor],
        x: torch.Tensor,
        original_curvature: torch.Tensor,
        original_edge_index: torch.Tensor,
        method: str = 'hybrid',
        node_names: Optional[List] = None,
        curvature_type: str = 'ollivier'
    ) -> torch.Tensor:
        """
        Compute curvature for augmented graph edges
        
        Methods:
        - 'transfer': Transfer curvature from original edges, 0 for new edges (fastest)
        - 'hybrid': Transfer original + approximate new edges based on neighborhoods (recommended)
        - 'recompute': Recompute curvature using EdgeCurvature class (most accurate but slow)
        
        Args:
            edge_index: [2, num_edges] augmented edge index
            edge_weight: [num_edges] optional edge weights
            x: [num_nodes, num_features] node features
            original_curvature: [num_original_edges] original curvature values
            original_edge_index: [2, num_original_edges] original edge index
            method: 'transfer', 'hybrid', or 'recompute'
            node_names: List of node names (required for 'recompute')
            curvature_type: 'ollivier' or 'forman' (for 'recompute' method)
        """
        device = edge_index.device
        num_edges = edge_index.shape[1]
        edge_curvature = torch.zeros(num_edges, device=device)
        
        if method == 'recompute' and EDGE_CURVATURE_AVAILABLE:
            logger.info("Recomputing curvature using EdgeCurvature class...")
            return self.recompute_curvature_exact(
                edge_index, x, node_names, curvature_type
            )
        elif method == 'recompute' and not EDGE_CURVATURE_AVAILABLE:
            logger.warning("EdgeCurvature not available, falling back to hybrid method")
            method = 'hybrid'
            
        # Create edge lookup dictionary from original graph
        original_edges = {}
        for i in range(original_edge_index.shape[1]):
            src, dst = original_edge_index[0, i].item(), original_edge_index[1, i].item()
            # Store both directions
            original_edges[(src, dst)] = original_curvature[i].item()
            original_edges[(dst, src)] = original_curvature[i].item()
            
        if method == 'transfer':
            # Simple transfer: original edges keep curvature, new edges get 0
            for i in range(num_edges):
                src, dst = edge_index[0, i].item(), edge_index[1, i].item()
                edge_curvature[i] = original_edges.get((src, dst), 1.0)
        
        elif method == 'hybrid':
            # Build adjacency for computing neighborhood-based approximations
            num_nodes = x.shape[0]
            adjacency = {i: set() for i in range(num_nodes)}
            node_curvatures = {i: [] for i in range(num_nodes)}
            
            # Build original adjacency and collect curvatures per node
            for i in range(original_edge_index.shape[1]):
                src, dst = original_edge_index[0, i].item(), original_edge_index[1, i].item()
                adjacency[src].add(dst)
                adjacency[dst].add(src)
                curv = original_curvature[i].item()
                node_curvatures[src].append(curv)
                node_curvatures[dst].append(curv)
                
            avg_node_curvatures = {}
            for node, curvs in node_curvatures.items():
                avg_node_curvatures[node] = np.mean(curvs) if curvs else 0
            
            for i in range(num_edges):
                src, dst = edge_index[0, i].item(), edge_index[1, i].item()
                
                if (src, dst) in original_edges:
                    # Original edge: use original curvature
                    edge_curvature[i] = original_edges[(src, dst)]
                else:
                    # New edge from Schur complement: approximate curvature
                    # Method 1: Average of endpoint node curvatures
                    approx_curv = (avg_node_curvatures[src] + avg_node_curvatures[dst]) / 2.0
                    
                    # Method 2: Check common neighbors (triangle closure)
                    common_neighbors = adjacency[src] & adjacency[dst]
                    if common_neighbors:
                        # Edges connecting nodes with many common neighbors tend to have positive curvature
                        neighbour_curvs = [] 
                        for neighbour in common_neighbors:
                            if (src, neighbour) in original_edges:
                                neighbour_curvs.append(original_edges[(src, neighbour)])
                            if (dst, neighbour) in common_neighbors:
                                neighbour_curvs.append(original_edges[(dst, neighbour)])
                        
                        if neighbour_curvs:
                            # Weight towards positive for high clustering
                            approx_curv = np.mean(neighbour_curvs) * 0.8  # Slightly dampen
                    
                    edge_curvature[i] = approx_curv
            
        else:
            raise ValueError(f"Unknown curvature computation method: {method}")

        return edge_curvature
    
    def recompute_curvature_exact(
        self,
        edge_index: torch.Tensor,
        x: torch.Tensor,
        node_names: Optional[List],
        curvature_type: str = 'ollivier'
    ) -> torch.Tensor:
        """
        Recompute exact curvature for augmented graph using EdgeCurvature class
        
        This is the most accurate but also the slowest method.
        Use this when you need exact curvature values for augmented graphs.
        """
        if node_names is None:
            node_names = [f"node_{i}" for i in range(x.shape[0])]
            
        G = nx.Graph()
        G.add_nodes_from(range(x.shape[0]))
        
        edge_list = edge_index.cpu().detach().numpy()
        G.add_edges_from([(int(src), int(dst)) for src, dst in edge_list])
        
        # Create feature dataframe
        feature_df = pd.DataFrame(
            x.cpu().numpy(),
            index=node_names
        )
        
        edge_curv_calculator = EdgeCurvature(G, feature_df) if EDGE_CURVATURE_AVAILABLE else None
        edge_curv_calculator.calculate_edge_curvature(method = curvature_type)
        
        # Extract curvature values
        curvature_dict = edge_curv_calculator.edge_curvature.get(
            'OllivierRicci' if curvature_type == 'ollivier' else 'FormanRicci',
            {}
        )
        
        # Map back to tensor
        edge_curvature = torch.zeros(edge_index.shape[1], device=edge_index.device)
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            # Try both edge directions
            curv = curvature_dict.get((src, dst), curvature_dict.get((dst, src), 0.0))
            edge_curvature[i] = curv
        
        logger.info(f"Recomputed exact {curvature_type} curvature for {edge_index.shape[1]} edges")
        
        return edge_curvature
                    
    
    def train_step(
        self,
        view1: Dict,
        view2: Dict,
        original_data: Dict,
        labels: torch.Tensor,
        train_mask: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        contrastive_weight: float = 0.3,
        pos_weight: Optional[torch.Tensor] = None,
        curvature_type: str = 'ollivier',
        curvature_method: str = 'hybrid',
        node_names: Optional[List] = None
    ) -> Dict[str, float]:
        """
        Combined training step with contrastive and classification objectives
        
        Args:
            view1: First augmented view {'edge_index', 'edge_weight', 'x', 'metadata'}
            view2: Second augmented view
            original_data: Original graph with curvature {'feature', 'edge_index', 'ollivier_curvature'}
            labels: Node labels [num_nodes] with values {0, 1}
            train_mask: Training mask
            optimizer: Optimizer
            contrastive_weight: Weight for contrastive loss (0-1)
            pos_weight: Positive class weight for imbalanced data
            curvature_type: 'ollivier' or 'forman'
            curvature_method: 'transfer', 'hybrid', or 'recompute'
            node_names: Node names (required for 'recompute' method)
        """
        self.train()
        optimizer.zero_grad()
        
        # Get original curvature
        original_edge_index = original_data['edge_index']
        original_curvature = original_data[f'{curvature_type}_curvature']
        
        curv1 = self.compute_augmented_curvature(
            view1['edge_index'],
            view1['edge_weight'],
            view1['x'],
            original_curvature=original_curvature,
            original_edge_index=original_edge_index,
            method='recompute',
            node_names=node_names,
            curvature_type=curvature_type
        )
        
        curv2 = self.compute_augmented_curvature(
            view2['edge_index'],
            view2['edge_weight'],
            view2['x'],
            original_curvature=original_curvature,
            original_edge_index=original_edge_index,
            method='recompute',
            node_names=node_names,
            curvature_type=curvature_type
        )
        
        # Contrastive learning: Learn invariant representations across augmented views
        z1 = self.get_contrastive_projection(
            view1['x'],
            view1['edge_index'],
            curv1
        )
        
        z2 = self.get_contrastive_projection(
            view2['x'],
            view2['edge_index'],
            curv2
        )
        
        contrastive_loss = self.compute_contrastive_loss(z1, z2)
        
        # Classification on both views (average predictions)
        logits1, _ = self.forward(
            x = view1['x'],
            edge_index = view1['edge_index'],
            edge_curvature = curv1
        )
        
        logits2, _ = self.forward(
            x = view2['x'],
            edge_index = view2['edge_index'],
            edge_curvature=curv2
        )
        
        # Average logits from both views for more robust predictions
        logits = (logits1 + logits2) / 2.0
        
        classification_loss = self.compute_classification_loss(
            logits=logits, labels=labels, mask=train_mask, pos_weight=pos_weight
        )
        
        # Combined Loss
        total_loss = (contrastive_weight * contrastive_loss + 
                     (1 - contrastive_weight) * classification_loss)
        
        total_loss.backward()
        optimizer.step()
        
        # Compute training metrics
        with torch.no_grad():
            probs = torch.sigmoid(logits[train_mask])
            pred = (probs > 0.5).long()
            train_acc = (pred == labels[train_mask]).float().mean()
        
        return {
            'total_loss': total_loss.item(),
            'contrastive_loss': contrastive_loss.item(),
            'classification_loss': classification_loss.item(),
            'train_accuracy': train_acc.item()
        }
    
    @torch.no_grad()
    def evaluate(
        self,
        data: Dict,
        labels: torch.Tensor,
        mask: torch.Tensor
    ) -> Dict[str, float]:
        """
        Evaluate model on validation/test set
        """
        self.eval()