import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional, Dict
import numpy as np
import networkx as nx
import pandas as pd
import time
from torch.utils.checkpoint import checkpoint as gradient_checkpoint
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
from statsmodels.stats.multitest import multipletests


# Assuming these are available from your codebase
from utils.logging_manager import get_logger
from model.support_models import ProjectionHead, BinaryClassifier, RankingLoss
from model.curvature_aware_gnn import CurvatureAwareGNN
from model.multi_layer_attention import HybridAggregator

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
        hop_types: List[str] = ['one_hop', 'two_hop'],
        num_attention_heads: int = 4,
        use_attention:bool = True,
        attention_mode:str = 'hybrid',
        pathway_aggregator: str = 'hierarchical',
        concat:bool = True,
        min_edge_ratio: float = 0.15,
        negative_slope: float = 0.2,
        temperature: float = 0.4,
        dropout: float = 0.2,
        device: torch.device = None
    ):
        super().__init__()
        self.temperature = temperature
        self.curvature_types = curvature_types
        self.hidden_channels = hidden_channels
        self.device = device
        self.hop_types = hop_types
        self.training_step_counter = 0  # For gradient accumulation
        
        # Encoder: CurvatureAwareGNN
        self.encoder = CurvatureAwareGNN(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=num_gnn_layers,
            curvature_types=curvature_types,
            hop_types=hop_types,
            use_attention=use_attention,
            dropout=dropout,
            min_edge_ratio=min_edge_ratio,
            attention_mode=attention_mode,
            concat=concat,
            negative_slope=negative_slope,
            heads = num_attention_heads
        )
        
        self.aggregator = HybridAggregator(
            hidden_channels=hidden_channels,
            num_curvature_types=len(curvature_types),
            num_hop_types=len(hop_types),
            num_heads=num_attention_heads,
            dropout=dropout,
            pathway_aggregation=pathway_aggregator,
            concat_heads=concat
        )
        
        # Projection head for contrastive learning
        self.projection = ProjectionHead(
            input_dim=hidden_channels,
            hidden_dim=hidden_channels,
            out_dim=projection_dim
        )
        
        self.ranking_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1) # Single score per node
        )
        
        # Deprecated 
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
        # Get representations from all curvature + hop pathways
        # Returns: {curvature: {hop: [layer_outputs]}}
        # Get pathway outputs with checkpointing
        pathway_outputs, gnn_attention = gradient_checkpoint(
                self._encoder_wrapper, x, edge_index, edge_curvature, 
                return_all_layers=True,
                return_attention=return_attention,
                use_reentrant=False
            )
        
        # Aggregate pathways
        final_repr, pathway_attention = self.aggregator(
            pathway_outputs,
            return_attention=return_attention
        )

        attention_info = None
        if return_attention:
            attention_info = {
                'pathway_attention': pathway_attention,
                'pathway_outputs': pathway_outputs
            }
        
        return final_repr, attention_info

    def _encoder_wrapper(self, x, edge_index, edge_curvature, return_all_layers, return_attention):
        """Wrapper for gradient checkpointing to handle tuple returns"""
        return self.encoder(x, edge_index, edge_curvature, return_all_layers, return_attention)
    
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
        
        # Clamp to prevent overflow in exp()
        sim_matrix = torch.clamp(sim_matrix, min=-50, max=50)
        
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
    
    @torch.no_grad()
    def predict_probability(
        self,
        data: Dict,
        mask: Optional[torch.Tensor] = None,
        curvature_type: str = 'ollivier',
        device: torch.device = None
    ) -> torch.Tensor:
        """        
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
        
        try:
            curv_key = f'{curvature_type.lower()}_curvature'
            
            # Get features
            features = data.get('feature', data.get('x'))
            if features is None:
                raise ValueError("Neither 'feature' nor 'x' found in data")
            
            features = features.to(device)
            edge_index = data['edge_index'].to(device)
            edge_curvature = data[curv_key].to(device)
            
            # CHECKPOINT 1: Check input data for NaN
            if torch.isnan(features).any():
                logger.error(f"NaN detected in features! {torch.isnan(features).sum().item()} values")
                nan_features = torch.isnan(features).any(dim=1)
                logger.error(f"Affected nodes: {torch.where(nan_features)[0].tolist()[:10]}...")
                # Replace NaN features with 0
                features = torch.nan_to_num(features, nan=0.0)
                logger.warning("Replaced NaN features with 0")
            
            if torch.isnan(edge_curvature).any():
                logger.error(f"NaN detected in edge_curvature! {torch.isnan(edge_curvature).sum().item()} values")
                # Replace NaN curvature with 0
                edge_curvature = torch.nan_to_num(edge_curvature, nan=0.0)
                logger.warning("Replaced NaN curvature with 0")
            
            # Validate curvature dimensions
            edge_curvature = self.validate_and_fix_curvature_dimensions(
                edge_index, edge_curvature, "PredictProba"
            )
            
            # Clamp curvature to reasonable range
            edge_curvature = torch.clamp(edge_curvature, min=-10.0, max=10.0)
            
            # CHECKPOINT 2: Forward pass with error catching
            try:
                logits, embeddings = self.forward(
                    features, edge_index, edge_curvature, return_embeddings=True
                )
            except RuntimeError as e:
                logger.error(f"Runtime error in forward pass: {e}")
                raise
            
            # CHECKPOINT 3: Check logits for NaN
            if logits.dim() > 1:
                logits = logits.squeeze(-1)
            
            if torch.isnan(logits).any():
                logger.error(f"NaN detected in logits! {torch.isnan(logits).sum().item()} out of {logits.numel()} values")
                nan_nodes = torch.where(torch.isnan(logits))[0]
                logger.error(f"Nodes with NaN logits: {nan_nodes.tolist()[:10]}...")
                
                # Check embeddings
                if embeddings is not None and torch.isnan(embeddings).any():
                    logger.error(f"NaN also in embeddings: {torch.isnan(embeddings).sum().item()} values")
                
                # Replace NaN logits with 0 (neutral)
                logits = torch.nan_to_num(logits, nan=0.0)
                logger.warning("Replaced NaN logits with 0")
            
            # CHECKPOINT 4: Compute probabilities
            probs = torch.sigmoid(logits)
            
            # Final NaN check
            if torch.isnan(probs).any():
                logger.error(f"NaN detected in probabilities after sigmoid!")
                probs = torch.nan_to_num(probs, nan=0.5)  # Use 0.5 as neutral probability
                logger.warning("Replaced NaN probabilities with 0.5")
            
            # Apply mask if provided
            if mask is not None:
                mask = mask.to(device)
                probs = probs[mask]
            
            return probs
            
        except Exception as e:
            logger.error(f"Error in predict_probability: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            import traceback
            logger.error(traceback.format_exc())
            
            # Return safe default probabilities
            num_nodes = data.get('feature', data.get('x')).shape[0]
            if mask is not None:
                num_nodes = mask.sum().item()
            logger.warning(f"Returning default probabilities (0.5) for {num_nodes} nodes")
            return torch.full((num_nodes,), 0.5, device=device)
    
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
        
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        
        self.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        if 'metrics' in checkpoint:
            logger.info(f"  Metrics: {checkpoint['metrics']}")
        
        return checkpoint
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_curvature: torch.Tensor,
        return_embeddings:bool = False
    ) -> torch.Tensor:
        """
        Compute driver likelihood scores for ranking (NOT binary classification).
        
        Returns:
            scores: [num_nodes] unbounded scores (higher = more likely driver)
        """
        h, _ = self.encode(x, edge_index, edge_curvature, return_attention=False)
        scores = self.ranking_head(h).squeeze(-1)
        
        if return_embeddings:
            return scores, h
        return scores, None
    
    def train_step(
        self,
        view1: Dict,
        view2: Dict,
        original_data: Dict,
        labels: torch.Tensor,
        train_mask: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        contrastive_weight: float = 0.3,
        ranking_loss_type: str = 'pairwise',
        margin: float = 1.0,
        curvature_type: str = 'ollivier',
        node_names: Optional[List] = None,
        device: torch.device = None,
        backward: bool = True
    ) -> Dict[str, float]:
        """
        Training step with ranking loss instead of classification loss.
        """
        self.train()
            
        device = device if device else self.device
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.to(device)
        curv_key = f'{curvature_type.lower()}_curvature'
        
        # NEW: Check if data is already on device
        if view1['x'].device != device:
            view1['x'] = view1['x'].to(device)
            view1['edge_index'] = view1['edge_index'].to(device)
        if view2['x'].device != device:
            view2['x'] = view2['x'].to(device)
            view2['edge_index'] = view2['edge_index'].to(device)
        # Ensure labels and mask are on device
        if labels.device != device:
            labels = labels.to(device)
        if train_mask.device != device:
            train_mask = train_mask.to(device)

        # Validate edge indices before processing
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
        
        view1['edge_index'], _ = validate_edge_index(
            view1['edge_index'], view1['x'].shape[0], "View1"
        )
        view2['edge_index'], _ = validate_edge_index(
            view2['edge_index'], view2['x'].shape[0], "View2"
        )
        
        # Get or compute curvatures
        if curv_key in view1 and curv_key in view2:
            curv1 = view1[curv_key].to(device) if view1[curv_key].device != device else view1[curv_key]
            curv2 = view2[curv_key].to(device) if view2[curv_key].device != device else view2[curv_key]
            
            curv1 = self.validate_and_fix_curvature_dimensions(view1['edge_index'], curv1, "View1")
            curv2 = self.validate_and_fix_curvature_dimensions(view2['edge_index'], curv2, "View2")
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
        
        # ROBUST FIX: Handle label and mask mapping for Schur complement augmentation
        num_original_nodes = labels.shape[0]
        aug_size_1 = view1['x'].shape[0]
        aug_size_2 = view2['x'].shape[0]
        
        eliminated_ids_1 = set(view1['metadata']['eliminated_node_ids'])
        eliminated_ids_2 = set(view2['metadata']['eliminated_node_ids'])
        
        # Create mapping from original indices to augmented indices
        def create_index_mapping(num_orig, eliminated_set):
            """Map original node indices to augmented node indices"""
            orig_to_aug = {}
            aug_idx = 0
            for orig_idx in range(num_orig):
                if orig_idx not in eliminated_set:
                    orig_to_aug[orig_idx] = aug_idx
                    aug_idx += 1
            return orig_to_aug, aug_idx
        
        orig_to_aug1, mapped_count1 = create_index_mapping(num_original_nodes, eliminated_ids_1)
        orig_to_aug2, mapped_count2 = create_index_mapping(num_original_nodes, eliminated_ids_2)
        
        # Initialize augmented labels and masks
        # Use -100 (ignore index) for unmapped nodes
        aug_labels1 = torch.full((aug_size_1,), -100, dtype=torch.long, device=device)
        aug_labels2 = torch.full((aug_size_2,), -100, dtype=torch.long, device=device)
        aug_train_mask1 = torch.zeros(aug_size_1, dtype=torch.bool, device=device)
        aug_train_mask2 = torch.zeros(aug_size_2, dtype=torch.bool, device=device)
        
        # Map labels and masks using the mapping
        for orig_idx in range(num_original_nodes):
            if orig_idx in orig_to_aug1:
                aug_idx = orig_to_aug1[orig_idx]
                if aug_idx < aug_size_1:  # Safety check
                    aug_labels1[aug_idx] = labels[orig_idx]
                    aug_train_mask1[aug_idx] = train_mask[orig_idx]
            
            if orig_idx in orig_to_aug2:
                aug_idx = orig_to_aug2[orig_idx]
                if aug_idx < aug_size_2:  # Safety check
                    aug_labels2[aug_idx] = labels[orig_idx]
                    aug_train_mask2[aug_idx] = train_mask[orig_idx]
        
        # Handle case where augmented graph is larger than expected
        # (Schur complement may create "virtual" nodes)
        if mapped_count1 < aug_size_1:
            logger.warning(f"View1: Mapped {mapped_count1} nodes but graph has {aug_size_1} nodes. "
                        f"Extra {aug_size_1 - mapped_count1} nodes will be ignored in training.")
            # Set extra nodes' labels to 0 (non-driver) and exclude from training
            aug_labels1[mapped_count1:] = 0
            aug_train_mask1[mapped_count1:] = False
        
        if mapped_count2 < aug_size_2:
            logger.warning(f"View2: Mapped {mapped_count2} nodes but graph has {aug_size_2} nodes. "
                        f"Extra {aug_size_2 - mapped_count2} nodes will be ignored in training.")
            aug_labels2[mapped_count2:] = 0
            aug_train_mask2[mapped_count2:] = False
        
        # Verify we have training samples
        if aug_train_mask1.sum() == 0 or aug_train_mask2.sum() == 0:
            raise ValueError("No training samples after mapping! Check eliminated_node_ids.")
        
        # Zero gradients
        optimizer.zero_grad()
        
        # NEW: Use no_grad for contrastive learning part to save memory
        with torch.amp.autocast('cuda') if torch.cuda.is_available() else torch.no_grad():
            z1 = self.get_contrastive_projection(view1['x'], view1['edge_index'], curv1)
            z2 = self.get_contrastive_projection(view2['x'], view2['edge_index'], curv2)
        contrastive_loss = self.compute_contrastive_loss(z1, z2)
        
        # Detach contrastive loss to save memory
        contrastive_loss_value = contrastive_loss.item()
        del z1, z2
        
        # Ranking loss
        ranking_criterion = RankingLoss(margin=margin, loss_type=ranking_loss_type)
        
        scores1, _ = self.forward(view1['x'], view1['edge_index'], curv1)
        scores2, _ = self.forward(view2['x'], view2['edge_index'], curv2)
        
        ranking_loss1 = ranking_criterion(scores1, aug_labels1, aug_train_mask1)
        ranking_loss2 = ranking_criterion(scores2, aug_labels2, aug_train_mask2)
        ranking_loss = (ranking_loss1 + ranking_loss2) / 2.0
        
        # NEW: Clean up intermediate tensors
        del scores1, scores2, ranking_loss1, ranking_loss2
        
        # Combined loss
        total_loss = (contrastive_weight * contrastive_loss
                        + (1-contrastive_weight) * ranking_loss)
        
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            logger.error(f"Invalid total loss: {total_loss.item()}")
            logger.error(f"  Contrastive loss: {contrastive_loss.item()}")
            logger.error(f"  Classification loss: {ranking_loss}")
            return {
                'total_loss': float('nan'),
                'contrastive_loss': contrastive_loss.item() if not torch.isnan(contrastive_loss) else float('nan'),
                'ranking_loss': ranking_loss if not torch.isnan(ranking_loss) else float('nan'),
                'train_ndcg': 0.0
            }
        
        # NEW: Only do backward if requested (for gradient accumulation)
        if backward:
            total_loss.backward()
            
            # Check for NaN gradients
            has_nan_grad = False
            for param in self.parameters():
                if param.grad is not None and torch.isnan(param.grad).any():
                    has_nan_grad = True
                    break
            
            if has_nan_grad:
                logger.error("NaN gradient detected! Skipping update.")
                if hasattr(optimizer, 'zero_grad'):
                    optimizer.zero_grad()
                return {
                    'total_loss': float('nan'),
                    'contrastive_loss': contrastive_loss_value,
                    'ranking_loss': ranking_loss.item(),
                    'train_ndcg': 0.0
                }
        
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            # Optimizer step
            optimizer.step()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        with torch.no_grad():
            train_ndcg = compute_ndcg(
                scores1[aug_train_mask1].detach().cpu().numpy(),
                aug_labels1[aug_train_mask1].detach().cpu().numpy()
            )
            
        return {
        'total_loss': total_loss.item(),
        'contrastive_loss': contrastive_loss.item(),
        'ranking_loss': ranking_loss,
        'train_ndcg': train_ndcg
        }
        
    def evaluate(
        self,
        data: Dict,
        labels: torch.Tensor,
        mask: torch.Tensor,
        curvature_type: str = 'ollivier',
        device: torch.device = None,
        k_values: List[int] = [10, 20, 50, 100]
    ) -> Dict[str, float]:
        """
        Evaluate model using ranking metrics.
        """

        self.eval()
        device = device if device else self.device
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        curv_key = f'{curvature_type.lower()}_curvature'
        edge_curvature = data[curv_key].to(device)
        features = data.get('feature', data.get('x')).to(device)
        edge_index = data['edge_index'].to(device)
        labels = labels.to(device)
        
        edge_curvature = self.validate_and_fix_curvature_dimensions(
            edge_index, edge_curvature, "Evaluation"
        )
        
        scores, _ = self.forward(features, edge_index, edge_curvature)
        scores = scores[mask]
        labels = labels[mask]
        
        scores_np = scores.detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()
        
        metrics = {}
        
        # AUROC and AUPRC
        metrics['auroc'] = roc_auc_score(labels_np, scores_np)
        metrics['auprc'] = average_precision_score(labels_np, scores_np)
        
        # F1 Score (using optimal threshold from precision-recall curve)
        precision, recall, thresholds = precision_recall_curve(labels_np, scores_np)
        # Calculate F1 for each threshold
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        best_f1_idx = np.argmax(f1_scores)
        metrics['f1'] = f1_scores[best_f1_idx]
        metrics['best_threshold'] = thresholds[best_f1_idx] if best_f1_idx < len(thresholds) else 0.5
        
        # Also add F1 at default threshold of 0.5
        predictions_05 = (scores_np >= 0.5).astype(int)
        tp = ((predictions_05 == 1) & (labels_np == 1)).sum()
        fp = ((predictions_05 == 1) & (labels_np == 0)).sum()
        fn = ((predictions_05 == 0) & (labels_np == 1)).sum()
        
        precision_05 = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall_05 = tp / (tp + fn) if (tp + fn) > 0 else 0
        metrics['f1@0.5'] = 2 * (precision_05 * recall_05) / (precision_05 + recall_05 + 1e-10)
        
        # Precision@K and Recall@K
        n_positives = labels_np.sum()
        sorted_indices = np.argsort(-scores_np)
        
        for k in k_values:
            if k > len(scores_np):
                continue
            
            top_k_indices = sorted_indices[:k]
            top_k_labels = labels_np[top_k_indices]
            
            precision_at_k = top_k_labels.sum() / k
            recall_at_k = top_k_labels.sum() / n_positives if n_positives > 0 else 0
            
            metrics[f'precision@{k}'] = precision_at_k
            metrics[f'recall@{k}'] = recall_at_k
        
        # NDCG@K
        for k in k_values:
            if k > len(scores_np):
                continue
            ndcg_k = compute_ndcg(scores_np, labels_np, k=k)
            metrics[f'ndcg@{k}'] = ndcg_k
        
        # Mean Reciprocal Rank
        mrr = compute_mrr(scores_np, labels_np)
        metrics['mrr'] = mrr
        
        # Median rank of known drivers
        driver_indices = np.where(labels_np == 1)[0]
        if len(driver_indices) > 0:
            driver_ranks = [np.where(sorted_indices == idx)[0][0] + 1 
                        for idx in driver_indices]
            metrics['median_driver_rank'] = np.median(driver_ranks)
            metrics['mean_driver_rank'] = np.mean(driver_ranks)
        
        return metrics
    
    @torch.no_grad()
    def score_all_genes(
        self,
        data: Dict,
        curvature_type: str = 'ollivier',
        device: torch.device = None,
        save_path: Optional[str] = None,
        save_prefix: str = ''
    ) -> pd.DataFrame:
        """
        Score ALL genes and return ranked DataFrame.
        
        Args:
            data: Graph data dictionary
            curvature_type: Type of curvature to use
            device: Device to use for computation
            save_path: Optional directory path to save CSV (if None, no file saved)
            save_prefix: Optional prefix for output filename
        
        Returns:
            DataFrame with all genes scored and ranked
        
        Example:
            >>> df = model.score_all_genes(
            ...     data,
            ...     save_path='results/',
            ...     save_prefix='experiment1'
            ... )
            # Saves to: results/experiment1_gene_rankings.csv
        """
        self.eval()
        
        device = device if device else self.device
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        curv_key = f'{curvature_type.lower()}_curvature'
        edge_curvature = data[curv_key].to(device)
        features = data.get('feature', data.get('x')).to(device)
        edge_index = data['edge_index'].to(device)
        
        edge_curvature = self.validate_and_fix_curvature_dimensions(
            edge_index, edge_curvature, "ScoreAllGenes"
        )
        
        scores, _ = self.forward(features, edge_index, edge_curvature)
        scores_normalized = torch.sigmoid(scores)
        
        node_names = data.get('node_name', [f"Gene_{i}" for i in range(len(scores))])
        true_labels = data.get('label', torch.zeros(len(scores)))
        
        df = pd.DataFrame({
            'gene_name': node_names,
            'driver_score': scores.cpu().numpy(),
            'driver_probability': scores_normalized.cpu().numpy(),
            'true_label': true_labels.cpu().numpy(),
            'rank': 0
        })
        
        df = df.sort_values('driver_score', ascending=False).reset_index(drop=True)
        df['rank'] = df.index + 1
        df['percentile'] = df['rank'] / len(df) * 100
        
        # Save to CSV if path provided
        if save_path is not None:
            from pathlib import Path
            save_dir = Path(save_path)
            save_dir.mkdir(exist_ok=True, parents=True)
            
            prefix = f"{save_prefix}_" if save_prefix else ""
            output_file = save_dir / f"{prefix}gene_rankings.csv"
            df.to_csv(output_file, index=False)
            
            print(f"✓ Saved gene rankings to: {output_file}")
        
        return df

    
    @torch.no_grad()
    def score_genes_with_statistics(
        self,
        data: Dict,
        labels: torch.Tensor,
        curvature_type: str = 'ollivier',
        node_names: List = None,
        num_permutations: int = 1000,
        device: torch.device = None,
        save_path: Optional[str] = None,
        save_prefix: str = ''
    ) -> pd.DataFrame:
        """
        Score all genes and compute statistical significance via permutation testing.
        
        Args:
            data: Graph data dictionary
            labels: True labels
            curvature_type: Type of curvature to use
            node_names: List of gene names corresponding to nodes
            num_permutations: Number of permutations for p-value calculation
            device: Device to use for computation
            save_path: Directory path to save CSV files (if None, no files saved)
            save_prefix: Prefix for output filenames (e.g., 'fold1', 'experiment_A')
        
        Returns:
            DataFrame with all genes scored and ranked with statistical significance
        
        Example:
            >>> df = model.score_genes_with_statistics(
            ...     data, labels,
            ...     node_names=gene_list,
            ...     save_path='results/',
            ...     save_prefix='fold_1'
            ... )
            # Saves to: results/fold_1_all_genes_scored.csv
            #           results/fold_1_significant_genes.csv
            #           results/fold_1_known_drivers.csv
        """
        print("\n" + "="*80)
        print("SCORING ALL GENES WITH STATISTICAL SIGNIFICANCE")
        print("="*80)
        
        print(f"Scoring {labels.shape[0]} genes with {num_permutations} permutations...")
    
        # Get node names (stored as a list)
        num_genes = labels.shape[0]
        if node_names is None:
            logger.warning("No node names provided. Using gene indices.")
            node_names = np.array([f"Gene_{i}" for i in range(num_genes)])
        elif isinstance(node_names, list):
            # Convert list to numpy array
            node_names = np.array(node_names)
        elif isinstance(node_names, torch.Tensor):
            node_names = node_names.cpu().numpy()
        elif not isinstance(node_names, np.ndarray):
            # Handle any other iterable
            node_names = np.array(list(node_names))
        
        # Validate length
        if len(node_names) != num_genes:
            logger.error(f"Node names length ({len(node_names)}) doesn't match labels ({num_genes})")
            node_names = np.array([f"Gene_{i}" for i in range(num_genes)])
        
        df_scores = self.score_all_genes(data, curvature_type, device)
        
        # Add gene IDs and gene names to the dataframe
        df_scores['gene_id'] = np.arange(len(df_scores))
        df_scores['gene_name'] = node_names
        
        device = device if device else self.device
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        labels = labels.to(device)
        
        known_drivers = labels == 1
        known_driver_indices = torch.where(known_drivers)[0].cpu().numpy()
        unknown_indices = torch.where(~known_drivers)[0].cpu().numpy()
        
        print(f"\nKnown drivers: {len(known_driver_indices)}")
        print(f"Unknown genes: {len(unknown_indices)}")
        
        driver_scores = df_scores.loc[known_driver_indices, 'driver_score'].values
        driver_mean = driver_scores.mean()
        driver_std = driver_scores.std()
        
        print(f"\nKnown Driver Score Statistics:")
        print(f"  Mean: {driver_mean:.4f}")
        print(f"  Std: {driver_std:.4f}")
        print(f"  Median: {np.median(driver_scores):.4f}")
        
        # Compute p-values via empirical comparison
        print(f"\nComputing empirical p-values...")
        
        # Initialize p-values array
        pvalues = np.ones(len(df_scores))
        
        # For unknown genes, compute empirical p-value
        # P-value = proportion of known drivers with score >= this gene's score
        for idx in unknown_indices:
            if idx % 500 == 0:
                print(f"  Processing gene {idx}/{len(unknown_indices)}...", end='\r')
            
            observed_score = df_scores.loc[idx, 'driver_score']
            
            # Empirical p-value: what fraction of known drivers score as high or higher?
            count_higher = (driver_scores >= observed_score).sum()
            pvalue = (count_higher + 1) / (len(driver_scores) + 1)  # Add 1 to avoid p=0
            pvalues[idx] = pvalue
        
        print(f"  Processing gene {len(unknown_indices)}/{len(unknown_indices)}... Done!")
        
        # Known drivers get p-value of 0 (they are the reference)
        pvalues[known_driver_indices] = 0.0
        
        # Print p-value distribution
        unknown_pvalues_for_stats = pvalues[unknown_indices]
        print(f"\nP-value distribution for unknown genes:")
        print(f"  Min: {unknown_pvalues_for_stats.min():.6f}")
        print(f"  Max: {unknown_pvalues_for_stats.max():.6f}")
        print(f"  Mean: {unknown_pvalues_for_stats.mean():.6f}")
        print(f"  Median: {np.median(unknown_pvalues_for_stats):.6f}")
        print(f"  P < 0.01: {(unknown_pvalues_for_stats < 0.01).sum()}")
        print(f"  P < 0.05: {(unknown_pvalues_for_stats < 0.05).sum()}")
        
        # Multiple testing correction
        unknown_pvalues = pvalues[unknown_indices]
        _, adjusted_pvalues_unknown, _, _ = multipletests(unknown_pvalues, method='fdr_bh')
        
        adjusted_pvalues = np.ones(len(df_scores))
        adjusted_pvalues[known_driver_indices] = 0.0
        adjusted_pvalues[unknown_indices] = adjusted_pvalues_unknown
        
        df_scores['pvalue'] = pvalues
        df_scores['adjusted_pvalue'] = adjusted_pvalues
        df_scores['is_known_driver'] = df_scores['true_label'] == 1
        df_scores['is_significant'] = (adjusted_pvalues < 0.05) & (~df_scores['is_known_driver'])
        df_scores['z_score'] = (df_scores['driver_score'] - driver_mean) / driver_std
        
        print(f"\n" + "="*80)
        print("RESULTS")
        print("="*80)
        
        significant_unknowns = df_scores[df_scores['is_significant']]
        print(f"\nSignificant unknown genes (FDR < 0.05): {len(significant_unknowns)}")
        
        if len(significant_unknowns) > 0:
            print(f"\nTop 20 Significant Unknown Genes:")
            print("-" * 80)
            display_cols = ['rank', 'gene_name', 'driver_score', 'z_score', 
                        'adjusted_pvalue', 'percentile']
            print(significant_unknowns[display_cols].head(20).to_string(index=False))
        
        # Save to CSV if path provided
        if save_path is not None:
            from pathlib import Path
            save_dir = Path(save_path)
            save_dir.mkdir(exist_ok=True, parents=True)
            
            # Add prefix if provided
            prefix = f"{save_prefix}_" if save_prefix else ""
            
            # Save all genes
            all_genes_file = save_dir / f"{prefix}all_genes_scored.csv"
            df_scores.to_csv(all_genes_file, index=False)
            print(f"\n✓ Saved all genes to: {all_genes_file}")
            
            # Save significant genes only
            if len(significant_unknowns) > 0:
                sig_file = save_dir / f"{prefix}significant_genes.csv"
                significant_unknowns.to_csv(sig_file, index=False)
                print(f"✓ Saved {len(significant_unknowns)} significant genes to: {sig_file}")
            
            # Save known drivers for reference
            known_drivers_df = df_scores[df_scores['is_known_driver']]
            known_file = save_dir / f"{prefix}known_drivers_scores.csv"
            known_drivers_df.to_csv(known_file, index=False)
            print(f"✓ Saved {len(known_drivers_df)} known drivers to: {known_file}")
            
            # Save summary statistics
            summary_file = save_dir / f"{prefix}scoring_summary.txt"
            with open(summary_file, 'w') as f:
                f.write("="*80 + "\n")
                f.write("GENE SCORING SUMMARY\n")
                f.write("="*80 + "\n\n")
                
                f.write(f"Total genes analyzed: {len(df_scores)}\n")
                f.write(f"Known driver genes: {len(known_driver_indices)}\n")
                f.write(f"Unknown genes: {len(unknown_indices)}\n")
                f.write(f"Significant unknown genes (FDR < 0.05): {len(significant_unknowns)}\n\n")
                
                f.write("Known Driver Score Statistics:\n")
                f.write(f"  Mean: {driver_mean:.4f}\n")
                f.write(f"  Std: {driver_std:.4f}\n")
                f.write(f"  Median: {np.median(driver_scores):.4f}\n")
                f.write(f"  Min: {driver_scores.min():.4f}\n")
                f.write(f"  Max: {driver_scores.max():.4f}\n\n")
                
                if len(significant_unknowns) > 0:
                    f.write("Significant Gene Score Statistics:\n")
                    f.write(f"  Mean: {significant_unknowns['driver_score'].mean():.4f}\n")
                    f.write(f"  Median: {significant_unknowns['driver_score'].median():.4f}\n")
                    f.write(f"  Min rank: {significant_unknowns['rank'].min()}\n")
                    f.write(f"  Max rank: {significant_unknowns['rank'].max()}\n\n")
                    
                    f.write("Top 10 Significant Genes:\n")
                    f.write("-"*80 + "\n")
                    top_10 = significant_unknowns.head(10)[
                        ['rank', 'gene_name', 'driver_score', 'adjusted_pvalue']
                    ]
                    f.write(top_10.to_string(index=False))
            
            print(f"✓ Saved summary to: {summary_file}")
            print(f"\n{'='*80}\n")
        
        return df_scores


def compute_ndcg(
    scores: np.ndarray, 
    labels: np.ndarray,
    k: Optional[int] = None
) -> float:
    """Compute Normalized Discounted Cumulative Gain."""
    if k is None:
        k = len(scores)

    if isinstance(scores, torch.Tensor):
        scores = scores.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    
    sorted_indices = np.argsort(-scores)[:k]
    sorted_labels = labels[sorted_indices]

    gains = sorted_labels
    discounts = 1.0 / np.log2(np.arange(2, len(gains) + 2))
    dcg = np.sum(gains * discounts)

    ideal_sorted = np.sort(labels)[::-1][:k]
    idcg = np.sum(ideal_sorted * discounts[:len(ideal_sorted)])

    if idcg == 0:
        return 0.0

    return dcg / idcg

def compute_mrr(
    scores: np.ndarray, 
    labels: np.ndarray
) -> float:
    
    """Compute Mean Reciprocal Rank."""
    sorted_indices = np.argsort(-scores)
    driver_indices = np.where(labels == 1)[0]
    
    if len(driver_indices) == 0:
        return 0.0
    
    reciprocal_ranks = []
    for driver_idx in driver_indices:
        rank = np.where(sorted_indices == driver_idx)[0][0] + 1
        reciprocal_ranks.append(1.0 / rank)
    
    return np.mean(reciprocal_ranks)
'''
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

# Deprecated train step as it uses classification which did not really give meaningful
# results due to the class imbalance and did not identify any significant potential
# drivers

@torch.amp.autocast('cuda')
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
    
    FIXED: Robust mask and label mapping handling Schur complement augmentation
    """
    self.train()
    
    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Ensure consistent device usage
    device = device if device else self.device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Move model to device if not already
    self.to(device)
    
    curvature_type = curvature_type.lower()
    curv_key = f'{curvature_type}_curvature'
    
    # Move all data to device BEFORE any operations
    view1['x'] = view1['x'].to(device)
    view1['edge_index'] = view1['edge_index'].to(device)
    view2['x'] = view2['x'].to(device)
    view2['edge_index'] = view2['edge_index'].to(device)
    
    # Move labels and masks to device
    labels = labels.to(device)
    train_mask = train_mask.to(device)
    if pos_weight is not None:
        pos_weight = pos_weight.to(device)
    
    # Validate edge indices before processing
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
    
    view1['edge_index'], _ = validate_edge_index(
        view1['edge_index'], view1['x'].shape[0], "View1"
    )
    view2['edge_index'], _ = validate_edge_index(
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
    
    # ROBUST FIX: Handle label and mask mapping for Schur complement augmentation
    num_original_nodes = labels.shape[0]
    aug_size_1 = view1['x'].shape[0]
    aug_size_2 = view2['x'].shape[0]
    
    eliminated_ids_1 = set(view1['metadata']['eliminated_node_ids'])
    eliminated_ids_2 = set(view2['metadata']['eliminated_node_ids'])
    
    # Create mapping from original indices to augmented indices
    def create_index_mapping(num_orig, eliminated_set):
        """Map original node indices to augmented node indices"""
        orig_to_aug = {}
        aug_idx = 0
        for orig_idx in range(num_orig):
            if orig_idx not in eliminated_set:
                orig_to_aug[orig_idx] = aug_idx
                aug_idx += 1
        return orig_to_aug, aug_idx
    
    orig_to_aug1, mapped_count1 = create_index_mapping(num_original_nodes, eliminated_ids_1)
    orig_to_aug2, mapped_count2 = create_index_mapping(num_original_nodes, eliminated_ids_2)
    
    # Initialize augmented labels and masks
    # Use -100 (ignore index) for unmapped nodes
    aug_labels1 = torch.full((aug_size_1,), -100, dtype=torch.long, device=device)
    aug_labels2 = torch.full((aug_size_2,), -100, dtype=torch.long, device=device)
    aug_train_mask1 = torch.zeros(aug_size_1, dtype=torch.bool, device=device)
    aug_train_mask2 = torch.zeros(aug_size_2, dtype=torch.bool, device=device)
    
    # Map labels and masks using the mapping
    for orig_idx in range(num_original_nodes):
        if orig_idx in orig_to_aug1:
            aug_idx = orig_to_aug1[orig_idx]
            if aug_idx < aug_size_1:  # Safety check
                aug_labels1[aug_idx] = labels[orig_idx]
                aug_train_mask1[aug_idx] = train_mask[orig_idx]
        
        if orig_idx in orig_to_aug2:
            aug_idx = orig_to_aug2[orig_idx]
            if aug_idx < aug_size_2:  # Safety check
                aug_labels2[aug_idx] = labels[orig_idx]
                aug_train_mask2[aug_idx] = train_mask[orig_idx]
    
    # Handle case where augmented graph is larger than expected
    # (Schur complement may create "virtual" nodes)
    if mapped_count1 < aug_size_1:
        logger.warning(f"View1: Mapped {mapped_count1} nodes but graph has {aug_size_1} nodes. "
                    f"Extra {aug_size_1 - mapped_count1} nodes will be ignored in training.")
        # Set extra nodes' labels to 0 (non-driver) and exclude from training
        aug_labels1[mapped_count1:] = 0
        aug_train_mask1[mapped_count1:] = False
    
    if mapped_count2 < aug_size_2:
        logger.warning(f"View2: Mapped {mapped_count2} nodes but graph has {aug_size_2} nodes. "
                    f"Extra {aug_size_2 - mapped_count2} nodes will be ignored in training.")
        aug_labels2[mapped_count2:] = 0
        aug_train_mask2[mapped_count2:] = False
    
    # Verify we have training samples
    if aug_train_mask1.sum() == 0 or aug_train_mask2.sum() == 0:
        raise ValueError("No training samples after mapping! Check eliminated_node_ids.")
    
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
    
    # Ensure logits are 1D
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
    
    # ===== ROBUST ERROR HANDLING =====
    # Check for NaN/Inf in loss BEFORE backward
    if torch.isnan(total_loss) or torch.isinf(total_loss):
        logger.error(f"Invalid total loss: {total_loss.item()}")
        logger.error(f"  Contrastive loss: {contrastive_loss.item()}")
        logger.error(f"  Classification loss: {classification_loss.item()}")
        return {
            'total_loss': float('nan'),
            'contrastive_loss': contrastive_loss.item() if not torch.isnan(contrastive_loss) else float('nan'),
            'classification_loss': classification_loss.item() if not torch.isnan(classification_loss) else float('nan'),
            'train_accuracy': 0.0
        }
    
    # Backward pass
    total_loss.backward()
    
    # Check for NaN gradients
    has_nan_grad = False
    max_grad = 0.0
    for param in self.parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any():
                has_nan_grad = True
                break
            max_grad = max(max_grad, param.grad.abs().max().item())
    
    if has_nan_grad:
        logger.error("NaN gradient detected! Skipping update.")
        logger.error(f"  Max grad before NaN: {max_grad}")
        optimizer.zero_grad()
        return {
            'total_loss': float('nan'),
            'contrastive_loss': contrastive_loss.item(),
            'classification_loss': classification_loss.item(),
            'train_accuracy': 0.0
        }
    
    # Log if gradients are unusually large
    if max_grad > 100:
        logger.warning(f"Large gradient detected: {max_grad:.2f}")
    
    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
    
    # Optimizer step
    optimizer.step()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # ===== COMPUTE METRICS =====
    with torch.no_grad():
        # Compute training accuracy safely
        try:
            if aug_train_mask1.sum() > 0:
                probs1 = torch.sigmoid(logits1[aug_train_mask1])
                pred1 = (probs1 > 0.5).long()
                train_acc = (pred1 == aug_labels1[aug_train_mask1]).float().mean()
            else:
                train_acc = torch.tensor(0.0, device=device)
            
            # Ensure train_acc is a valid number
            if torch.isnan(train_acc) or torch.isinf(train_acc):
                train_acc = torch.tensor(0.0, device=device)
                
        except Exception as e:
            logger.error(f"Error computing training accuracy: {e}")
            train_acc = torch.tensor(0.0, device=device)
    
    return {
        'total_loss': total_loss.item(),
        'contrastive_loss': contrastive_loss.item(),
        'classification_loss': classification_loss.item(),
        'train_accuracy': train_acc.item()
    }

# Deprecated Evaluation method that tests the model classification

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

# Deprecated method as it did not identify any significant potential 
# drivers from the false positives

@torch.no_grad()
def identify_potential_drivers(
    self,
    data: Dict,
    labels: torch.Tensor,
    mask: torch.Tensor,
    confidence_threshold: float = 0.2,
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

# Deprecated Helper function for the previous classification method

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

# Deprecated loss functions for classification

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
'''