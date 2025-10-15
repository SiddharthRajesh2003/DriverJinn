import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional, Dict
import numpy as np
import networkx as nx
import pandas as pd
import time
from torch.utils.checkpoint import checkpoint as gradient_checkpoint

# Assuming these are available from your codebase
from utils.logging_manager import get_logger
from model.support_models import ProjectionHead, BinaryClassifier
from model.curvature_aware_gnn import CurvatureAwareGNN
from model.multi_layer_attention import MultiLayerAttention

logger = get_logger(__name__)

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
        dropout: float = 0.2,
        device: torch.device = None
    ):
        super().__init__()
        self.temperature = temperature
        self.curvature_types = curvature_types
        self.hidden_channels = hidden_channels
        self.device = device
        self.training_step_counter = 0  # For gradient accumulation
        
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
            input_dim=hidden_channels,
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
        curvature_outputs = gradient_checkpoint(self.encoder, x, edge_index, edge_curvature,
                                                use_reentrant=False)
        
        attention_weights = {} if return_attention else None
        
        curvature_representations = []
        for curv_type in self.curvature_types:
            layer_outputs = curvature_outputs[curv_type]
            aggregated, attn = self.layer_attentions[curv_type](
                layer_outputs,
                return_attention=return_attention
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
        
        FIXED: Correct positive pair selection
        """
        if node_mask is not None:
            z1 = z1[node_mask]
            z2 = z2[node_mask]
        
        batch_size = z1.shape[0]
        z = torch.cat([z1, z2], dim=0)  # [2*batch_size, dim]
        
        # Compute similarity matrix
        sim_matrix = torch.mm(z, z.T) / self.temperature  # [2*batch_size, 2*batch_size]
        
        # Create mask to remove self-similarity
        mask = torch.eye(2 * batch_size, device=z.device, dtype=torch.bool)
        mask_value = torch.finfo(sim_matrix.dtype).min
        sim_matrix = sim_matrix.masked_fill(mask, mask_value)
        
        # Labels: for each sample i, its positive pair is at i + batch_size (or i - batch_size)
        labels = torch.cat([
            torch.arange(batch_size, 2 * batch_size, device=z.device),
            torch.arange(batch_size, device=z.device)
        ])
        
        # Compute log softmax
        log_prob = F.log_softmax(sim_matrix, dim=1)
        
        # Extract positive similarities using labels
        pos_sim = log_prob[torch.arange(2 * batch_size, device=z.device), labels]
        
        # NT-Xent loss
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
                logits, labels, pos_weight=pos_weight
            )
        else:
            loss = F.binary_cross_entropy_with_logits(
                logits, labels
            )
            
        return loss
    
    def compute_focal_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        alpha: float = 0.25,
        gamma: float = 2.0
    ) -> torch.Tensor:
        """
        Compute focal loss to handle class imbalance better than BCE
        
        Args:
            logits: Model predictions
            labels: Ground truth labels
            mask: Optional node mask
            alpha: Balancing factor
            gamma: Focusing parameter
        """
        if mask is not None:
            logits = logits[mask]
            labels = labels[mask].float()
        
        # Compute probabilities
        probs = torch.sigmoid(logits)
        
        # Compute focal loss
        ce_loss = F.binary_cross_entropy_with_logits(
            logits, labels, reduction='none'
        )
        
        p_t = probs * labels + (1 - probs) * (1 - labels)
        focal_weight = (1 - p_t) ** gamma
        
        if alpha >= 0:
            alpha_t = alpha * labels + (1 - alpha) * (1 - labels)
            focal_weight = alpha_t * focal_weight
        
        loss = (focal_weight * ce_loss).mean()
        
        return loss
    
    def match_curvature_to_edges(
        self,
        edge_index: torch.Tensor,
        edge_curvature: torch.Tensor,
        view_name: str = "view"
    ) -> torch.Tensor:
        """
        Match curvature values to edges when they were computed on undirected graph
        but edge_index contains both directions.
        
        This creates a proper mapping using edge pairs.
        """
        num_edges = edge_index.shape[1]
        num_curvatures = edge_curvature.shape[0]
        device = edge_curvature.device
        
        logger.info(f"{view_name}: Matching {num_curvatures} curvatures to {num_edges} edges")
        
        # Build mapping from canonical edges to curvature indices
        curvature_map = {}
        
        # First pass: identify unique undirected edges from edge_index
        edges_set = set()
        edge_list = edge_index.t().cpu().numpy()
        
        for i, (src, dst) in enumerate(edge_list):
            src, dst = int(src), int(dst)
            # Canonical form: smaller node first
            canonical = (min(src, dst), max(src, dst))
            edges_set.add(canonical)
        
        # Sort for consistent ordering
        unique_edges = sorted(list(edges_set))
        
        if len(unique_edges) != num_curvatures:
            logger.warning(
                f"{view_name}: Number of unique edges ({len(unique_edges)}) != "
                f"number of curvatures ({num_curvatures})"
            )
            # Use the minimum to avoid index errors
            n_to_map = min(len(unique_edges), num_curvatures)
        else:
            n_to_map = num_curvatures
        
        # Map curvatures to canonical edges
        for i in range(n_to_map):
            curvature_map[unique_edges[i]] = edge_curvature[i].item()
        
        # Second pass: assign curvatures to all directed edges
        matched_curvature = torch.zeros(num_edges, device=device, dtype=edge_curvature.dtype)
        
        for i, (src, dst) in enumerate(edge_list):
            src, dst = int(src), int(dst)
            canonical = (min(src, dst), max(src, dst))
            
            if canonical in curvature_map:
                matched_curvature[i] = curvature_map[canonical]
            else:
                # Use mean of available curvatures as fallback
                matched_curvature[i] = edge_curvature.mean().item()
        
        logger.info(f"{view_name}: Successfully matched curvatures to edges")
        return matched_curvature
    
    def validate_and_fix_curvature_dimensions(
        self,
        edge_index: torch.Tensor,
        edge_curvature: torch.Tensor,
        view_name: str = "view"
    ) -> torch.Tensor:
        """
        Ensure edge_curvature matches edge_index dimensions
        
        Handles the case where curvature is computed for undirected edges
        but edge_index contains directed edges (both directions)
        """
        num_edges = edge_index.shape[1]
        num_curvatures = edge_curvature.shape[0]
        
        if num_curvatures == num_edges:
            return edge_curvature
        
        elif num_curvatures * 2 == num_edges:
            # Perfect 2:1 ratio - curvature for undirected, edges are directed
            logger.info(f"{view_name}: Mapping undirected curvatures to directed edges")
            return self.match_curvature_to_edges(edge_index, edge_curvature, view_name)
        
        elif num_edges * 2 == num_curvatures:
            # Edge index is undirected but curvature is for both directions
            logger.info(f"{view_name}: Averaging curvature for undirected edges")
            return edge_curvature.reshape(-1, 2).mean(dim=1)
        
        else:
            # Non-standard ratio - use smart matching
            logger.warning(
                f"{view_name}: Non-standard curvature dimension: "
                f"{num_curvatures} curvatures for {num_edges} edges"
            )
            return self.match_curvature_to_edges(edge_index, edge_curvature, view_name)
    
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
                edge_curvature[i] = original_edges.get((src, dst), 0.0)
        
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
                            if (dst, neighbour) in original_edges:
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
        curvature_type = curvature_type.lower()
        if node_names is None:
            node_names = [f"node_{i}" for i in range(x.shape[0])]
            
        G = nx.Graph()
        G.add_nodes_from(range(x.shape[0]))
        
        edge_list = edge_index.cpu().detach().numpy().T
        G.add_edges_from([(int(src), int(dst)) for src, dst in edge_list])
        
        # Create feature dataframe
        feature_df = pd.DataFrame(
            x.cpu().numpy(),
            index=node_names
        )
        
        edge_curv_calculator = EdgeCurvature(G, feature_df)
        edge_curv_calculator.calculate_edge_curvature(method=curvature_type)
        
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
    
    def map_original_mask_to_augmented(
        self,
        original_mask: torch.Tensor,
        eliminated_node_ids: List[int],
        num_original_nodes: int
    ) -> torch.Tensor:
        """
        Map original node mask to augmented graph (after Schur complement elimination)
        
        Args:
            original_mask: [num_original_nodes] boolean mask from original graph
            eliminated_node_ids: List of node IDs that were eliminated in augmentation
            num_original_nodes: Total number of nodes in original graph
        
        Returns:
            augmented_mask: [num_augmented_nodes] boolean mask for augmented graph
        """
        # Create mapping from original to augmented node indices
        eliminated_set = set(eliminated_node_ids)
        
        # Build mapping: original_idx -> augmented_idx
        original_to_augmented = {}
        augmented_idx = 0
        
        for orig_idx in range(num_original_nodes):
            if orig_idx not in eliminated_set:
                original_to_augmented[orig_idx] = augmented_idx
                augmented_idx += 1
        
        # Map the mask
        num_augmented_nodes = num_original_nodes - len(eliminated_node_ids)
        augmented_mask = torch.zeros(num_augmented_nodes, dtype=torch.bool)
        
        for orig_idx in range(num_original_nodes):
            if orig_idx in original_to_augmented:
                aug_idx = original_to_augmented[orig_idx]
                augmented_mask[aug_idx] = bool(original_mask[orig_idx])
        
        return augmented_mask
    
    def map_augmented_predictions_to_original(
        self,
        augmented_predictions: torch.Tensor,
        eliminated_node_ids: List[int],
        num_original_nodes: int,
        fill_value: float = 0.0
    ) -> torch.Tensor:
        """
        Map predictions from augmented graph back to original node indices
        
        Args:
            augmented_predictions: [num_augmented_nodes] predictions
            eliminated_node_ids: List of eliminated node IDs
            num_original_nodes: Total number of original nodes
            fill_value: Value to assign to eliminated nodes
        
        Returns:
            original_predictions: [num_original_nodes] predictions in original indexing
        """
        eliminated_set = set(eliminated_node_ids)
        
        original_predictions = torch.full(
            (num_original_nodes,),
            fill_value,
            dtype=augmented_predictions.dtype,
            device=augmented_predictions.device
        )
        
        augmented_idx = 0
        for orig_idx in range(num_original_nodes):
            if orig_idx not in eliminated_set:
                original_predictions[orig_idx] = augmented_predictions[augmented_idx]
                augmented_idx += 1
        
        return original_predictions
    
    @torch.cuda.amp.autocast()
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
        node_names: Optional[List] = None,
        device: torch.device = None,
        batch_size: int = 2048,
        use_focal_loss: bool = False,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0
    ) -> Dict[str, float]:
        """
        Combined training step with contrastive and classification objectives
        
        FIXED: All device issues, dimension mismatches, and edge validation
        """
        self.train()
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # FIX 1: Ensure consistent device usage
        device = device if device else self.device
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Move model to device if not already
        self.to(device)
        
        curvature_type = curvature_type.lower()
        curv_key = f'{curvature_type}_curvature'
        
        # FIX 2: Move all data to device BEFORE any operations
        view1['x'] = view1['x'].to(device)
        view1['edge_index'] = view1['edge_index'].to(device)
        view2['x'] = view2['x'].to(device)
        view2['edge_index'] = view2['edge_index'].to(device)
        
        # Move labels and masks to device - FIX 4
        labels = labels.to(device)
        train_mask = train_mask
        if pos_weight is not None:
            pos_weight = pos_weight.to(device)
        
        # FIX 3: Validate edge indices before processing
        def validate_edge_index(edge_index, num_nodes, view_name):
            if edge_index.numel() == 0:
                logger.warning(f"{view_name}: Empty edge index")
                return edge_index, None
                
            # Filter out invalid edges
            valid_edges = (edge_index[0] < num_nodes) & (edge_index[1] < num_nodes)
            if not torch.all(valid_edges):
                invalid_count = (~valid_edges).sum().item()
                logger.warning(f"{view_name}: Filtered {invalid_count} edges with invalid node indices")
                edge_index = edge_index[:, valid_edges]
                
            return edge_index, valid_edges
        
        view1['edge_index'], valid_edges1 = validate_edge_index(
            view1['edge_index'], view1['x'].shape[0], "View1"
        )
        view2['edge_index'], valid_edges2 = validate_edge_index(
            view2['edge_index'], view2['x'].shape[0], "View2"
        )
        
        # Get or compute curvatures
        if curv_key in view1 and curv_key in view2:
            curv1 = view1[curv_key].to(device)
            curv2 = view2[curv_key].to(device)
            
            # Validate and fix curvature dimensions
            curv1 = self.validate_and_fix_curvature_dimensions(
                view1['edge_index'], curv1, "View1"
            )
            curv2 = self.validate_and_fix_curvature_dimensions(
                view2['edge_index'], curv2, "View2"
            )
        else:
            # Use hybrid method for speed
            original_edge_index = original_data['edge_index'].to(device)
            original_curvature = original_data[curv_key].to(device)
            
            curv1 = self.compute_augmented_curvature(
                view1['edge_index'],
                view1.get('edge_weight'),
                view1['x'],
                original_curvature=original_curvature,
                original_edge_index=original_edge_index,
                method='hybrid',
                node_names=node_names,
                curvature_type=curvature_type
            )
            
            curv2 = self.compute_augmented_curvature(
                view2['edge_index'],
                view2.get('edge_weight'),
                view2['x'],
                original_curvature=original_curvature,
                original_edge_index=original_edge_index,
                method='hybrid',
                node_names=node_names,
                curvature_type=curvature_type
            )
        
        # Map masks and labels to augmented views
        num_original_nodes = labels.shape[0]
        eliminated_ids_1 = view1['metadata']['eliminated_node_ids']
        eliminated_ids_2 = view2['metadata']['eliminated_node_ids']
        
        aug_train_mask1 = self.map_original_mask_to_augmented(
            train_mask, eliminated_ids_1, num_original_nodes
        ).to(device)
        
        aug_train_mask2 = self.map_original_mask_to_augmented(
            train_mask, eliminated_ids_2, num_original_nodes
        ).to(device)
        
        # FIX 6: Properly convert labels
        aug_labels1 = self.map_original_mask_to_augmented(
            labels.bool(), eliminated_ids_1, num_original_nodes
        ).long().to(device)
        
        aug_labels2 = self.map_original_mask_to_augmented(
            labels.bool(), eliminated_ids_2, num_original_nodes
        ).long().to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Contrastive learning
        z1 = self.get_contrastive_projection(view1['x'], view1['edge_index'], curv1)
        z2 = self.get_contrastive_projection(view2['x'], view2['edge_index'], curv2)
        
        contrastive_loss = self.compute_contrastive_loss(z1, z2)
        
        # Classification on both views
        logits1, _ = self.forward(
            x=view1['x'],
            edge_index=view1['edge_index'],
            edge_curvature=curv1
        )
        
        logits2, _ = self.forward(
            x=view2['x'],
            edge_index=view2['edge_index'],
            edge_curvature=curv2
        )
        
        # FIX 7: Ensure logits are 1D
        if logits1.dim() > 1:
            logits1 = logits1.squeeze(-1)
        if logits2.dim() > 1:
            logits2 = logits2.squeeze(-1)
        
        # Choose loss function
        if use_focal_loss:
            classification_loss1 = self.compute_focal_loss(
                logits1, aug_labels1, aug_train_mask1, focal_alpha, focal_gamma
            )
            classification_loss2 = self.compute_focal_loss(
                logits2, aug_labels2, aug_train_mask2, focal_alpha, focal_gamma
            )
        else:
            classification_loss1 = self.compute_classification_loss(
                logits1, aug_labels1, aug_train_mask1, pos_weight
            )
            classification_loss2 = self.compute_classification_loss(
                logits2, aug_labels2, aug_train_mask2, pos_weight
            )
        
        classification_loss = (classification_loss1 + classification_loss2) / 2.0
        
        # Combined loss
        total_loss = (contrastive_weight * contrastive_loss + 
                    (1 - contrastive_weight) * classification_loss)
        
        total_loss.backward()
        
        # FIX 8: Gradient clipping to prevent instability
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        
        optimizer.step()
        torch.cuda.empty_cache()
        
        # Training metrics
        with torch.no_grad():
            if aug_train_mask1.sum() > 0:
                probs1 = torch.sigmoid(logits1[aug_train_mask1])
                pred1 = (probs1 > 0.5).long()
                train_acc = (pred1 == aug_labels1[aug_train_mask1]).float().mean()
            else:
                train_acc = torch.tensor(0.0)
        
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
        mask: torch.Tensor,
        curvature_type: str = 'ollivier',
        device: torch.device = None
    ) -> Dict[str, float]:
        """
        Evaluate model on validation/test set
        
        Args:
            data: Graph data (original or augmented view with curvature)
            labels: Node labels
            mask: Evaluation mask
            curvature_type: 'ollivier' or 'forman'
        """
        self.eval()
        
        device = device if device else self.device
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        curv_key = f'{curvature_type.lower()}_curvature'
        if curv_key not in data:
            raise ValueError(f"'{curv_key}' not found in data. Available keys: {data.keys()}")
        
        edge_curvature = data[curv_key].to(device)
        
        features = data.get('feature', data.get('x'))
        if features is None:
            raise ValueError("Neither 'feature' nor 'x' found in data")
        
        features = features.to(device)
        edge_index = data['edge_index'].to(device)
        labels = labels.to(device)
        
        
        # Validate curvature dimensions
        edge_curvature = self.validate_and_fix_curvature_dimensions(
            edge_index, edge_curvature, "Evaluation"
        )
        
        logits, _ = self.forward(features, edge_index, edge_curvature)
        
        # Ensure logits are 1D
        if logits.dim() > 1:
            logits = logits.squeeze(-1)
        
        logits = logits[mask]
        labels = labels[mask]
        
        probs = torch.sigmoid(logits)
        pred = (probs > 0.5).long()
        
        accuracy = (pred == labels).float().mean().item()
        
        tp = ((pred == 1) & (labels == 1)).sum().item()
        fp = ((pred == 1) & (labels == 0)).sum().item()
        tn = ((pred == 0) & (labels == 0)).sum().item()
        fn = ((pred == 0) & (labels == 1)).sum().item()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        loss = F.binary_cross_entropy_with_logits(logits, labels.float())
        
        return {
            'loss': loss.item(),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'true_positives': tp,
            'false_positives': fp,
            'true_negatives': tn,
            'false_negatives': fn
        }
    
    @torch.no_grad()
    def predict_probability(
        self,
        data: Dict,
        mask: Optional[torch.Tensor] = None,
        curvature_type: str = 'ollivier',
        device: torch.device = None
    ) -> torch.Tensor:
        """
        Get prediction probabilities for nodes
        
        FIXED: Added missing method
        
        Args:
            data: Graph data dictionary
            mask: Optional mask for specific nodes
            curvature_type: 'ollivier' or 'forman'
            device: Device to use
        
        Returns:
            probs: [num_nodes] or [num_masked_nodes] probability of being driver
        """
        self.eval()
        
        device = device if device else self.device
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        curv_key = f'{curvature_type.lower()}_curvature'
        features = data.get('feature', data.get('x')).to(device)
        edge_index = data['edge_index'].to(device)
        edge_curvature = data[curv_key].to(device)
        
        # Validate curvature dimensions
        edge_curvature = self.validate_and_fix_curvature_dimensions(
            edge_index, edge_curvature, "PredictProba"
        )
        
        logits, _ = self.forward(features, edge_index, edge_curvature)
        
        # Ensure logits are 1D
        if logits.dim() > 1:
            logits = logits.squeeze(-1)
        
        probs = torch.sigmoid(logits)
        
        if mask is not None:
            probs = probs[mask]
        
        return probs
        
    @torch.no_grad()
    def identify_potential_drivers(
        self,
        data: Dict,
        labels: torch.Tensor,
        mask: torch.Tensor,
        confidence_threshold: float = 0.6,
        curvature_threshold: float = 0.0,
        feature_criteria: Optional[Dict[str, Tuple[int, float]]] = None,
        curvature_type: str = 'ollivier',
        device: torch.device = None
    ) -> Dict[str, any]:
        """
        Identify potential driver genes from false positives
        
        Args:
            data: Graph data dictionary
            labels: True labels (0: non-driver, 1: driver)
            mask: Mask for nodes to analyze
            confidence_threshold: Minimum prediction confidence for potential drivers
            curvature_threshold: Minimum mean curvature for potential drivers
            feature_criteria: Dict mapping feature name to (feature_idx, min_value)
            curvature_type: 'ollivier' or 'forman'
        """
        self.eval()
        
        device = device if device else self.device
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Get curvature
        curv_key = f'{curvature_type}_curvature'
        edge_curvature = data[curv_key].to(device)

        # Handle both 'feature' and 'x' keys
        features = data.get('feature', data.get('x')).to(device)
        edge_index = data['edge_index'].to(device)
        feature_names = data.get('feature_name', [])
        
        labels = labels.to(device)
        
        # Validate curvature dimensions
        edge_curvature = self.validate_and_fix_curvature_dimensions(
            edge_index, edge_curvature, "IdentifyDrivers"
        )
        
        logits, embeddings = self.forward(
            features, 
            edge_index,
            edge_curvature,
            return_embeddings=True
        )
        
        # Ensure logits are 1D
        if logits.dim() > 1:
            logits = logits.squeeze(-1)
        
        # Get attention weights and curvature-specific representations
        _, attention_info = self.encode(
            features,
            edge_index,
            edge_curvature,
            return_attention=True
        )
        
        probs = torch.sigmoid(logits)
        pred = (probs > 0.5).long()
        
        # Identify false positives (predicted as driver but labeled as non-driver)
        mask_tensor = torch.tensor(mask, dtype=torch.bool).to(device) if isinstance(mask, np.ndarray) else mask
        pred_tensor = torch.tensor(pred).to(device) if isinstance(pred, np.ndarray) else pred
        labels_tensor = torch.tensor(labels).to(device) if isinstance(labels, np.ndarray) else labels

        fp_mask = mask_tensor & (pred_tensor == 1) & (labels_tensor == 0)       
        
        fp_indices = torch.where(fp_mask)[0]
        
        # Filter based on high confidence
        high_conf_mask = probs > confidence_threshold
        candidate_mask = fp_mask & high_conf_mask
        
        # Compute per-node curvature statistics
        node_curvature_stats = self.compute_node_curvature_features(
            edge_index, edge_curvature, features.shape[0]
        )
        
        # Apply curvature threshold
        curv_mask = node_curvature_stats['mean_curvature'] > curvature_threshold
        candidate_mask = candidate_mask & curv_mask
        
        # Apply additional feature criteria if provided
        if feature_criteria is not None and feature_names:
            for feat_name, (feat_idx, min_val) in feature_criteria.items():
                feat_mask = features[:, feat_idx] > min_val
                candidate_mask = candidate_mask & feat_mask
                
        # Get final potential driver candidates
        potential_indices = torch.where(candidate_mask)[0]
        
        # Compute scores and reasons for each candidate
        scores = []
        reasons = []
        detailed_features = []
        
        for idx in potential_indices:
            idx_item = idx.item()
            
            conf = probs[idx].item()
            
            # Curvature features
            curv_features = {
                'mean_curvature': node_curvature_stats['mean_curvature'][idx].item(),
                'positive_ratio': node_curvature_stats['positive_ratio'][idx].item(),
                'negative_ratio': node_curvature_stats['negative_ratio'][idx].item()
            }
            
            # Extract important node features
            node_features = {}
            important_features = [
                'ppin_hub', 'ppin_degree', 'ppin_betweenness',
                'essentiality_percentage', 'complexes', 'mirna'
            ]
            
            for feat_name in important_features:
                if feat_name in feature_names:
                    feat_idx = feature_names.index(feat_name)
                    node_features[feat_name] = features[idx_item, feat_idx].item()
            
            # Attention weights (which curvature types are important)
            cross_attn = attention_info['cross_curvature_attention'][idx]
            curv_importance = {
                curv_type: cross_attn[i].item()
                for i, curv_type in enumerate(self.curvature_types)
            }
            
            reason_parts = []
            reason_parts.append(f"High confidence: {conf:.3f}")
            
            if curv_features['mean_curvature'] > 0:
                reason_parts.append(f"Positive curvature: {curv_features['mean_curvature']:.3f}")
            
            if 'ppin_hub' in node_features and node_features['ppin_hub'] > 0.5:
                reason_parts.append("Hub protein")
            
            if 'essentiality_percentage' in node_features and node_features['essentiality_percentage'] > 0.3:
                reason_parts.append(f"Essential: {node_features['essentiality_percentage']:.3f}")
            
            # Dominant curvature pathway
            dominant_curv = max(curv_importance.items(), key=lambda x: x[1])
            reason_parts.append(f"Dominant: {dominant_curv[0]} curvature")
            
            scores.append(conf)
            reasons.append("; ".join(reason_parts))
            detailed_features.append({
                'confidence': conf,
                'curvature': curv_features,
                'node_features': node_features,
                'curvature_importance': curv_importance
            })
        
        # Get node names
        node_names = data.get('node_name', None)
        if node_names is not None:
            potential_names = [node_names[idx] for idx in potential_indices]
        else:
            potential_names = [f"Node_{idx.item()}" for idx in potential_indices]
        
        return {
            'potential_driver_mask': candidate_mask,
            'potential_driver_indices': potential_indices,
            'scores': torch.tensor(scores),
            'reasons': reasons,
            'detailed_features': detailed_features,
            'total_false_positives': fp_indices.shape[0],
            'num_potential_drivers': potential_indices.shape[0],
            'node_names': potential_names
        }
    
    def compute_node_curvature_features(
        self,
        edge_index: torch.Tensor,
        edge_curvature: torch.Tensor,
        num_nodes: int
    ) -> Dict[str, torch.Tensor]:
        """
        Compute curvature-based features for each node
        """
        # Initialize
        mean_curvature = torch.zeros(num_nodes, device=edge_index.device)
        positive_ratio = torch.zeros(num_nodes, device=edge_index.device)
        negative_ratio = torch.zeros(num_nodes, device=edge_index.device)
        degree = torch.zeros(num_nodes, device=edge_index.device)
        
        # Aggregate curvature for each node
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i], edge_index[1, i]
            curv = edge_curvature[i]
            
            mean_curvature[src] += curv
            mean_curvature[dst] += curv
            
            degree[src] += 1
            degree[dst] += 1
            
            if curv > 0:
                positive_ratio[src] += 1
                positive_ratio[dst] += 1
            elif curv < 0:
                negative_ratio[src] += 1
                negative_ratio[dst] += 1
        
        # Normalize
        valid_mask = degree > 0
        mean_curvature[valid_mask] /= degree[valid_mask]
        positive_ratio[valid_mask] /= degree[valid_mask]
        negative_ratio[valid_mask] /= degree[valid_mask]
        
        return {
            'mean_curvature': mean_curvature,
            'positive_ratio': positive_ratio,
            'negative_ratio': negative_ratio,
            'degree': degree
        }
    
    def visualize_attention_weights(
        self,
        data: Dict,
        node_idx: int,
        curvature_type: str = 'ollivier',
        save_path: Optional[str] = None,
        device: torch.device = None
    ):
        """
        Visualize attention weights for a specific node
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            logger.error("matplotlib and seaborn required for visualization")
            return
        
        self.eval()
        
        device = device if device else self.device
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        features = data.get('feature', data.get('x')).to(device)
        edge_index = data['edge_index'].to(device)
        edge_curvature = data[f'{curvature_type}_curvature'].to(device)
        
        # Validate curvature dimensions
        edge_curvature = self.validate_and_fix_curvature_dimensions(
            edge_index, edge_curvature, "Visualization"
        )
        
        _, attention_info = self.encode(
            features,
            edge_index,
            edge_curvature,
            return_attention=True
        )
        
        # Plot cross-curvature attention
        cross_attn = attention_info['cross_curvature_attention'][node_idx]
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Cross-curvature attention
        axes[0].bar(self.curvature_types, cross_attn.cpu().numpy())
        axes[0].set_title(f'Cross-Curvature Attention (Node {node_idx})')
        axes[0].set_ylabel('Attention Weight')
        axes[0].set_ylim([0, 1])
        
        # Layer attention for each curvature type
        layer_attns = []
        for curv_type in self.curvature_types:
            key = f'{curv_type}_layer_attention'
            if key in attention_info:
                layer_attns.append(attention_info[key][node_idx].cpu().numpy())
        
        if layer_attns:
            layer_attns = np.array(layer_attns)
            sns.heatmap(
                layer_attns,
                xticklabels=[f'L{i}' for i in range(layer_attns.shape[1])],
                yticklabels=self.curvature_types,
                ax=axes[1],
                cmap='viridis',
                annot=True,
                fmt='.3f',
                cbar_kws={'label': 'Attention Weight'}
            )
            axes[1].set_title(f'Layer Attention Weights (Node {node_idx})')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved attention visualization to {save_path}")
        
        plt.show()
    
    def save_checkpoint(
        self,
        path: str,
        epoch: int,
        optimizer: torch.optim.Optimizer,
        metrics: Dict,
        metadata: Optional[Dict] = None
    ):
        """Save model checkpoint with versioning and metadata"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'model_config': {
                'hidden_channels': self.hidden_channels,
                'curvature_types': self.curvature_types,
                'temperature': self.temperature
            },
            'metadata': metadata or {},
            'timestamp': time.time()
        }
        
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path} (epoch {epoch}, F1: {metrics.get('f1', 'N/A')})")

    def load_checkpoint(
        self,
        path: str,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: torch.device = None
    ) -> Dict:
        """Load model checkpoint"""
        device = device if device else self.device
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        checkpoint = torch.load(path, map_location=device)
        
        self.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        if 'metrics' in checkpoint:
            logger.info(f"  Metrics: {checkpoint['metrics']}")
        
        return checkpoint