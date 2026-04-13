#!/usr/bin/env python

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


from utils.logging_manager import get_logger
from model.support_models import ProjectionHead
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
    - Ranking genes based on their features and interactions to be potential drivers
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
        aggregation: str = 'add',
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
        self.temperature = temperature                  # NT-Xent temparature tau; lower = sharper contrastive distribution
        self.curvature_types = curvature_types          # ['positive', 'negative', 'both']
        self.hidden_channels = hidden_channels          # base hidden dim; saved for checkpoint metadata
        self.device = device                            # Target device; used as fallback in evaluate/score methods
        self.hop_types = hop_types                      
        self.training_step_counter = 0  # Tracks gradient accumulation across forward passes
        
        # Encoder: CurvatureAwareGNN
        self.encoder = CurvatureAwareGNN(               # Multi-pathway GNN; produces nested dict of per-pathway layer embeddings
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=num_gnn_layers,
            curvature_types=curvature_types,
            hop_types=hop_types,
            use_attention=use_attention,
            aggregation=aggregation,
            dropout=dropout,
            min_edge_ratio=min_edge_ratio,
            attention_mode=attention_mode,
            concat=concat,
            negative_slope=negative_slope,
            heads = num_attention_heads
        )
        
        self.aggregator = HybridAggregator(             # two-stage aggregator; first collapses layers, then collapses pathways -> [num_nodes, hidden_channels]
            hidden_channels=hidden_channels,
            num_curvature_types=len(curvature_types),
            num_hop_types=len(hop_types),
            num_heads=num_attention_heads,
            dropout=dropout,
            pathway_aggregation=pathway_aggregator,
            concat_heads=concat
        )
        
        # Projection head for contrastive learning
        self.projection = ProjectionHead(               # MLP that maps aggregated embeddings into the contrastive projection space
            input_dim=hidden_channels,
            hidden_dim=hidden_channels,
            out_dim=projection_dim
        )
        
        self.ranking_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),# First linear layer, allows non-linear scoring
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1), # Single score per gene; higher = more likely driver
            nn.Sigmoid()                    # Squash to (0, 1) so scores are interpretable as probabilities
        )
        
        # Deprecated 
        # Binary Classifier
        # self.classifier = BinaryClassifier(
        #    input_dim=hidden_channels,
        #    hidden_dim=hidden_channels,
        #    dropout=dropout
        # )
        
    def encode(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_curvature: torch.Tensor,
        return_attention: bool = True,
        return_all_layers: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Encode graph into representation vector

        Args:
            return_all_layers: If False, only return final layer (saves memory during training)
        """
        # Get representations from all curvature + hop pathways
        # Returns: {curvature: {hop: [layer_outputs]}}
        # Get pathway outputs with checkpointing
        pathway_outputs, gnn_attention = gradient_checkpoint(
                self._encoder_wrapper, x, edge_index, edge_curvature,
                return_all_layers=return_all_layers,
                return_attention=return_attention,
                use_reentrant=False                     # Recomputes activations on backward instead of storing them; trades compute for memory
            )

        # Aggregate pathways
        final_repr, pathway_attention = self.aggregator(        # Collapses all (curvature, hop, layer) pathways into a single [num_nodes, hidden_channels] embedding
            pathway_outputs,
            return_attention=return_attention
        )

        attention_info = None
        if return_attention:
            attention_info = {
                'pathway_attention': pathway_attention,     # Which (curvature,hop) pathways were most important
                'pathway_outputs': pathway_outputs,         # Raw per-pathway embeddings before aggregation
                'gnn_attention': gnn_attention              # Per-edge attention weights from the GNN layers
            }

        return final_repr, attention_info                   # Final node embeddings + optional attention breakdown

    def _encoder_wrapper(self, x, edge_index, edge_curvature, return_all_layers, return_attention):
        """Wrapper for gradient checkpointing to handle tuple returns"""
        return self.encoder(x, edge_index, edge_curvature, return_all_layers, return_attention)     # Thin wrapper needed because gradient_checkpoint cannot handle functions that return multiple tensors directly; wrapping normalizes the interface
    
    def get_contrastive_projection(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_curvature: torch.Tensor
    ) -> torch.Tensor:
        """
        Get normalized projection for contrastive loss
        """
        h, _ = self.encode(x, edge_index, edge_curvature)           # Run full encoder + aggregator to get node embeddings
        z = self.projection(h)                                      # Project embeddings into the contrastive space
        return F.normalize(z, dim=-1)                               # L2-normalize so cosine similarity == dot product; required for NT-Xent
    
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
        num_edges = edge_index.shape[1]                 # Total directed edges (both directions includedd)
        num_curvatures = edge_curvature.shape[0]        # Number of curvature values (one per undirected edge)
        device = edge_curvature.device                  # Keep tensor output on same device as input
        
        logger.info(f"{view_name}: Matching {num_curvatures} curvatures to {num_edges} edges")
        
        # Build mapping from canonical edges to curvature indices
        curvature_map = {}      # Maps (min_node, max_node) -> curvature scalar
        
        # First pass: identify unique undirected edges from edge_index
        edges_set = set()
        edge_list = edge_index.t().cpu().numpy()    # Transpose to [num_edges, 2] for row-wise iteration
        
        for i, (src, dst) in enumerate(edge_list):
            src, dst = int(src), int(dst)
            # Canonical form: smaller node first
            canonical = (min(src, dst), max(src, dst))      # Canonical form: Smaller index first; ensures both directions map to the same key
            edges_set.add(canonical)
        
        # Sort for consistent ordering
        unique_edges = sorted(list(edges_set))              # Deterministic ordering so curvature[i] always maps to the same edge
        
        if len(unique_edges) != num_curvatures:
            logger.warning(
                f"{view_name}: Number of unique edges ({len(unique_edges)}) != "
                f"number of curvatures ({num_curvatures})"
            )
            # Use the minimum to avoid index errors
            n_to_map = min(len(unique_edges), num_curvatures)           # Map only as many as we safely can to avoid index errors
        else:
            n_to_map = num_curvatures                           # Counts match; safe to map all
        
        # Map curvatures to canonical edges
        for i in range(n_to_map):
            curvature_map[unique_edges[i]] = edge_curvature[i].item()       # Associate each sorted edge with its curvature value
        
        # Second pass: assign curvatures to all directed edges
        matched_curvature = torch.zeros(num_edges, device=device, dtype=edge_curvature.dtype)       # Output tensor; will be filled per directed edge
        
        for i, (src, dst) in enumerate(edge_list):
            src, dst = int(src), int(dst)
            canonical = (min(src, dst), max(src, dst))              # Look up canonical regardless of direction
            
            if canonical in curvature_map:
                matched_curvature[i] = curvature_map[canonical]     # Assign the curvature of the undirected edge to this directed edge
            else:
                # Use mean of available curvatures as fallback
                matched_curvature[i] = edge_curvature.mean().item() # Fallback; use global mean for any edge not found in the map
        
        logger.info(f"{view_name}: Successfully matched curvatures to edges")
        return matched_curvature        # [num_edges] with curvature values aligned to directed edge_index
    
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
        num_edges = edge_index.shape[1]             # Number of directed edges in the graph
        num_curvatures = edge_curvature.shape[0]    # Number of curvature values provided
        
        if num_curvatures == num_edges:
            return edge_curvature                   # Dimensions already match; nothing to do
        
        elif num_curvatures * 2 == num_edges:
            # Perfect 2:1 ratio - curvature for undirected, edges are directed
            logger.info(f"{view_name}: Mapping undirected curvatures to directed edges")
            return self.match_curvature_to_edges(edge_index, edge_curvature, view_name)     # Expand each undirected curvature to both directed edges
        
        elif num_edges * 2 == num_curvatures:
            # Edge index is undirected but curvature is for both directions
            logger.info(f"{view_name}: Averaging curvature for undirected edges")
            return edge_curvature.reshape(-1, 2).mean(dim=1)                                # Collapse both directions into one value per undirected edge
        
        else:
            # Non-standard ratio - use smart matching
            logger.warning(
                f"{view_name}: Non-standard curvature dimension: "
                f"{num_curvatures} curvatures for {num_edges} edges"
            )
            return self.match_curvature_to_edges(edge_index, edge_curvature, view_name)     # Best-effort alignment via canonical edge matching
    
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
        device = edge_index.device                                      # Keep all tensors on the same device
        num_edges = edge_index.shape[1]                                 # Total edges in the augmented graph
        edge_curvature = torch.zeros(num_edges, device=device)          # Initialize output; new edges default to 0
        
        if method == 'recompute' and EDGE_CURVATURE_AVAILABLE:
            logger.info("Recomputing curvature using EdgeCurvature class...")
            return self.recompute_curvature_exact(
                edge_index, x, node_names, curvature_type               # Exact recomputation; accurate but slow
            )
        elif method == 'recompute' and not EDGE_CURVATURE_AVAILABLE:
            logger.warning("EdgeCurvature not available, falling back to hybrid method")
            method = 'hybrid'                                           # Graceful downgrade when dependency is missing
            
        # Create edge lookup dictionary from original graph
        original_edges = {}
        for i in range(original_edge_index.shape[1]):
            src, dst = original_edge_index[0, i].item(), original_edge_index[1, i].item()
            # Store both directions
            original_edges[(src, dst)] = original_curvature[i].item()           # Forward direction
            original_edges[(dst, src)] = original_curvature[i].item()           # Reverse direction; curvature is undirected
            
        if method == 'transfer':
            # Simple transfer: original edges keep curvature, new edges get 0
            for i in range(num_edges):
                src, dst = edge_index[0, i].item(), edge_index[1, i].item()
                edge_curvature[i] = original_edges.get((src, dst), 0.0)         # Use original curvature if edge existed; 0 for fill-in edges
        
        elif method == 'hybrid':
            # Build adjacency for computing neighborhood-based approximations
            num_nodes = x.shape[0]
            adjacency = {i: set() for i in range(num_nodes)}                    # Adjacency list for the original graph
            node_curvatures = {i: [] for i in range(num_nodes)}                 # Accumulates curvatures of all edges incident to each node
            
            # Build original adjacency and collect curvatures per node
            for i in range(original_edge_index.shape[1]):
                src, dst = original_edge_index[0, i].item(), original_edge_index[1, i].item()
                adjacency[src].add(dst)                                         # Add neighbor in both directions
                adjacency[dst].add(src)
                curv = original_curvature[i].item()
                node_curvatures[src].append(curv)                               # Collect curvature for source node
                node_curvatures[dst].append(curv)                               # Collect curvature for destination node
                
            avg_node_curvatures = {}
            for node, curvs in node_curvatures.items():
                avg_node_curvatures[node] = np.mean(curvs) if curvs else 0      # Mean curvature of all edges touching this node; used as fallback for new edges
            
            for i in range(num_edges):
                src, dst = edge_index[0, i].item(), edge_index[1, i].item()
                
                if (src, dst) in original_edges:
                    # Original edge: use original curvature
                    edge_curvature[i] = original_edges[(src, dst)]              # Exact value; no approximation needed
                else:
                    # New edge from Schur complement: approximate curvature
                    # Method 1: Average of endpoint node curvatures
                    approx_curv = (avg_node_curvatures[src] + avg_node_curvatures[dst]) / 2.0       # Baseline: average curvature of src and dst neighbourhoods
                    
                    # Method 2: Check common neighbors (triangle closure)
                    common_neighbors = adjacency[src] & adjacency[dst]          # Nodes that both src and dst are connected to in the original graph
                    if common_neighbors:
                        # Edges connecting nodes with many common neighbors tend to have positive curvature
                        neighbour_curvs = [] 
                        for neighbour in common_neighbors:
                            if (src, neighbour) in original_edges:
                                neighbour_curvs.append(original_edges[(src, neighbour)])    # Curvature of the src-neighbour edge
                            if (dst, neighbour) in original_edges:
                                neighbour_curvs.append(original_edges[(dst, neighbour)])    # Curvature of the dst-neighbour edge
                        
                        if neighbour_curvs:
                            # Weight towards positive for high clustering
                            approx_curv = np.mean(neighbour_curvs) * 0.8  # use mean of triangle edges; dampen by 0.8 since fill-in edges are structurally weaker
                    
                    edge_curvature[i] = approx_curv
            
        else:
            raise ValueError(f"Unknown curvature computation method: {method}")     # Guard against invalid method strings

        return edge_curvature                   # [num_edges] curvature values aligned to augmented edge_index
    
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
        curvature_type = curvature_type.lower()                                 # Normalize to lowercase to match API keys (ollivier, forman)
        if node_names is None:
            node_names = [f"node_{i}" for i in range(x.shape[0])]               # Generate placeholder names if not provided
            
        G = nx.Graph()
        G.add_nodes_from(range(x.shape[0]))                                     # Add all nodes (0-indexed integers)
        
        edge_list = edge_index.cpu().detach().numpy().T                         # Convert to [num_edges, 2] numpy array for networkx
        G.add_edges_from([(int(src), int(dst)) for src, dst in edge_list])      # Populate graph with all edges
        
        # Create feature dataframe
        feature_df = pd.DataFrame(
            x.cpu().numpy(),
            index=node_names                                                    # Index by gene name so EdgeCurvature can look up features by name
        )
        
        edge_curv_calculator = EdgeCurvature(G, feature_df)                     # Instantiate curvature calculator with graph + features
        edge_curv_calculator.calculate_edge_curvature(method=curvature_type)    # Run the full curvature computation
        
        # Extract curvature values
        curvature_dict = edge_curv_calculator.edge_curvature.get(
            'OllivierRicci' if curvature_type == 'ollivier' else 'FormanRicci', # Select the right key for the requested curvature type
            {}
        )
        
        # Map back to tensor
        edge_curvature = torch.zeros(edge_index.shape[1], device=edge_index.device) # Output tensor aligned to edge_index
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            # Try both edge directions
            curv = curvature_dict.get((src, dst), curvature_dict.get((dst, src), 0.0))  # Try forward direction first, then reverse, then default to 0
            edge_curvature[i] = curv
        
        logger.info(f"Recomputed exact {curvature_type} curvature for {edge_index.shape[1]} edges")
        
        return edge_curvature                                                   # [num_edges] exact curvature values for the augmented graph
    
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
        eliminated_set = set(eliminated_node_ids)                   # Convert to set for O(1) membership checks
        
        # Build mapping: original_idx -> augmented_idx
        original_to_augmented = {}                                  # Maps each surviving node's original index to its compacted index
        augmented_idx = 0                                           # Counter that increments only for non-eliminated nodes
        
        for orig_idx in range(num_original_nodes):
            if orig_idx not in eliminated_set:
                original_to_augmented[orig_idx] = augmented_idx     # Assign next available augmented index
                augmented_idx += 1                                  # Advance counter only for surviving nodes
        
        # Map the mask
        num_augmented_nodes = num_original_nodes - len(eliminated_node_ids) # Size of the augmented graph
        augmented_mask = torch.zeros(num_augmented_nodes, dtype=torch.bool) # Initialize all-False mask for the augmented graph
        
        for orig_idx in range(num_original_nodes):
            if orig_idx in original_to_augmented:
                aug_idx = original_to_augmented[orig_idx]
                augmented_mask[aug_idx] = bool(original_mask[orig_idx])     # Transfer the original mask value to the new index position
        
        return augmented_mask                                           # [num_augmented_nodes] boolean mask aligned to the augmented graph's node ordering
    
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
        eliminated_set = set(eliminated_node_ids)                       # Set for O(1) membership checks
        
        original_predictions = torch.full(
            (num_original_nodes,),
            fill_value,                                                 # Pre-fill all positions with fill_value (default 0.0) so eliminated scores get a defined score
            dtype=augmented_predictions.dtype,
            device=augmented_predictions.device
        )
        
        augmented_idx = 0                                               # Tracks position in the augmented (compacted) prediction array
        for orig_idx in range(num_original_nodes):
            if orig_idx not in eliminated_set:
                original_predictions[orig_idx] = augmented_predictions[augmented_idx]       # Copy prediction back to its original node slot
                augmented_idx += 1                                                          # Advance augmented only for surviving nodes (same logic as mapping construction)
        
        return original_predictions                                     # [num_original_nodes] predictions with eliminated nodes filled with fill_value
    
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
            z1 = z1[node_mask]                              # Restrict view 1 to training nodes only; test nodes must not contribute to contrastive loss
            z2 = z2[node_mask]                              # Same mask applied to view 2
        
        batch_size = z1.shape[0]
        z = torch.cat([z1, z2], dim=0)  # concatenate both views -> [2*batch_size, dim]; first half = view1, second half=view2
        
        # Compute similarity matrix
        sim_matrix = torch.mm(z, z.T) / self.temperature  # All-pairs cosine similarity (L2-normalized inputs) scaled by 1/tau -> [2*batch_size, 2*batch_size]
        
        # Clamp to prevent overflow in exp()
        sim_matrix = torch.clamp(sim_matrix, min=-50, max=50)       # Avoid numerical overflow when exponentiating large similarity values
        
        # Create mask to remove self-similarity
        mask = torch.eye(2 * batch_size, device=z.device, dtype=torch.bool)     # Diagonal mask; each sample shouldn't attend to itself
        mask_value = torch.finfo(sim_matrix.dtype).min                          # Use the most negative representable float
        sim_matrix = sim_matrix.masked_fill(mask, mask_value)                   # Set diagonal to -inf so softmax assigns ~0 probability to self-pairs
        
        # Labels: for each sample i, its positive pair is at i + batch_size (or i - batch_size)
        labels = torch.cat([
            torch.arange(batch_size, 2 * batch_size, device=z.device),          # For each view1 node, positive is at index i+batch_size (view2)
            torch.arange(batch_size, device=z.device)                           # For each view2 node i+batch_size, positive is at index i (view1)
        ])
        
        # Compute log softmax
        log_prob = F.log_softmax(sim_matrix, dim=1)                             # Log-probability distribution over all other samples for each query
        
        # Extract positive similarities using labels
        pos_sim = log_prob[torch.arange(2 * batch_size, device=z.device), labels]       # Select the log-prob assigned to the correct positive pair for each sample
        
        # NT-Xent loss
        loss = -pos_sim.mean()                                                  # Maximize log-prob of positive pairs = minimize NT-Xent loss; averaged over both views
        
        return loss
    
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
        logger.info(f"Saved checkpoint to {path} (epoch {epoch}, NDCG@50: {metrics.get('ndcg@50', 'N/A')})")

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
        return_embeddings:bool = False,
        return_all_layers: bool = False
    ) -> torch.Tensor:
        """
        Compute driver likelihood scores for ranking (NOT binary classification).

        Args:
            return_all_layers: If False, only use final layer (saves memory during training)

        Returns:
            scores: [num_nodes] unbounded scores (higher = more likely driver)
        """
        h, _ = self.encode(x, edge_index, edge_curvature,
                          return_attention=False,                   # Skip attention computation during training to save memory
                          return_all_layers=return_all_layers)
        scores = self.ranking_head(h).squeeze(-1)                   # Pass embeddings through MLP + sigmoid -> [num_nodes]; squeeze removes trailing dim-1

        if return_embeddings:
            return scores, h                                        # Also return raw embeddings when caller needs them (eg for contrastive loss)
        return scores, None                                         # Return none for embeddings by default to avoid keeping large tensors in memory
    
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

        self.eval()                                                     # Disable dropout and batch norm training mode
        device = device if device else self.device
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # Fallback device resolution
        
        curv_key = f'{curvature_type.lower()}_curvature'                # Build dict key eg 'ollivier_curvature'
        edge_curvature = data[curv_key].to(device)
        features = data.get('feature', data.get('x')).to(device)        # Support both 'feature' and 'x' key names
        edge_index = data['edge_index'].to(device)
        labels = labels.to(device)
        
        edge_curvature = self.validate_and_fix_curvature_dimensions(    # Align curvature tensor to directed edge_index
            edge_index, edge_curvature, "Evaluation"
        )

        scores, _ = self.forward(features, edge_index, edge_curvature)  # Run full forward pass -> [num_nodes] scores
        scores = scores[mask]                                           # Restrict to eval nodes only
        labels = labels[mask]

        scores_np = scores.detach().cpu().numpy()                       # Move to CPU for sklearn metric functions
        labels_np = labels.detach().cpu().numpy()

        # CRITICAL: Explicitly delete GPU tensors to free memory immediately
        del scores, features, edge_index, edge_curvature                # Free GPU memory before metric computation; avoids OOM error on large graphs
        torch.cuda.empty_cache()

        metrics = {}
        
        # AUROC and AUPRC
        metrics['auroc'] = roc_auc_score(labels_np, scores_np)          # Area under ROC curve; threshold-independent discrimination
        metrics['auprc'] = average_precision_score(labels_np, scores_np)# Area under precision-recall curve; better for imbalanced labels
        
        # F1 Score (using optimal threshold from precision-recall curve)
        precision, recall, thresholds = precision_recall_curve(labels_np, scores_np)
        # Calculate F1 for each threshold
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)                 # F1 at each threshold; 1e-10 prevents division by 0
        best_f1_idx = np.argmax(f1_scores)                                                  # Index of the threshold that maximizes F1
        metrics['f1'] = f1_scores[best_f1_idx]                                              # Best-achievable F1 scores across all thresholds
        metrics['best_threshold'] = thresholds[best_f1_idx] if best_f1_idx < len(thresholds) else 0.5       # Threshold that achieved best F1; fallback to 0.5 if out of bounds
        
        # Also add F1 at default threshold of 0.5
        predictions_05 = (scores_np >= 0.5).astype(int)                     # Hard predictions at fixed 0.5 threshold
        tp = ((predictions_05 == 1) & (labels_np == 1)).sum()               # True positives
        fp = ((predictions_05 == 1) & (labels_np == 0)).sum()               # False positives
        fn = ((predictions_05 == 0) & (labels_np == 1)).sum()               # False negatives
        
        precision_05 = tp / (tp + fp) if (tp + fp) > 0 else 0               # Precision at 0.5 threshold
        recall_05 = tp / (tp + fn) if (tp + fn) > 0 else 0                  # Recall at 0.5 threshold
        metrics['f1@0.5'] = 2 * (precision_05 * recall_05) / (precision_05 + recall_05 + 1e-10) # F1 at the standard 0.5 threshold
        
        # Precision@K and Recall@K
        n_positives = labels_np.sum()                                           # Total number of known driver genes in the eval set
        sorted_indices = np.argsort(-scores_np)                                 # Indices sorted by descending score
        
        for k in k_values:
            if k > len(scores_np):                                              # Skip k values larger than the eval set
                continue
            
            top_k_indices = sorted_indices[:k]                                  # Indices of the top-k scoring genes
            top_k_labels = labels_np[top_k_indices]                             # True labels of top-k genes
            
            precision_at_k = top_k_labels.sum() / k                             # Fraction of top-k that are true drivers
            recall_at_k = top_k_labels.sum() / n_positives if n_positives > 0 else 0    # Fraction of all drivers recovered in top-k
            
            metrics[f'precision@{k}'] = precision_at_k
            metrics[f'recall@{k}'] = recall_at_k
        
        # NDCG@K
        for k in k_values:
            if k > len(scores_np):
                continue
            ndcg_k = compute_ndcg(scores_np, labels_np, k=k)                    # Normalized DCG at k
            metrics[f'ndcg@{k}'] = ndcg_k
        
        # Mean Reciprocal Rank
        mrr = compute_mrr(scores_np, labels_np)                                 # Average of 1/rank for each known driver
        metrics['mrr'] = mrr
        
        # Median rank of known drivers
        driver_indices = np.where(labels_np == 1)[0]                            # Positions of known drivers in the eval set
        if len(driver_indices) > 0:
            driver_ranks = [np.where(sorted_indices == idx)[0][0] + 1 
                        for idx in driver_indices]                              # 1-indexed rank of each known driver in the full ranking
            metrics['median_driver_rank'] = np.median(driver_ranks)             # Median rank across all known drivers; lower = better
            metrics['mean_driver_rank'] = np.mean(driver_ranks)                 # Mean rank; more sensitive to outliers than median
        
        return metrics
    
    @torch.no_grad()
    def score_all_genes(
        self,
        data: Dict,
        labels: torch.Tensor,
        node_names: Optional[np.ndarray] = None,  # <-- ADD THIS
        curvature_type: str = 'ollivier',
        eliminated_node_ids: Optional[set] = None,
        device: torch.device = None,
        save_path: Optional[str] = None,
        save_prefix: str = ''
    ) -> pd.DataFrame:
        """Score ALL genes and return ranked DataFrame."""
        self.eval()                                                                         # Disable dropout; ensure deterministic scoring
        
        device = device if device else self.device
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        curv_key = f'{curvature_type.lower()}_curvature'                                    # eg 'ollivier_curvature'
        edge_curvature = data[curv_key].to(device)
        features = data.get('feature', data.get('x')).to(device)                            # Support both key naming conventions
        edge_index = data['edge_index'].to(device)
        labels = labels.to(device)
        
        edge_curvature = self.validate_and_fix_curvature_dimensions(                        # Align curvature dimensions to directed edge_index
            edge_index, edge_curvature, "ScoreAllGenes"
        )
        
        scores, _ = self.forward(features, edge_index, edge_curvature)                      # Forward pass -> [num_nodes] raw scores
        scores_normalized = torch.sigmoid(scores)                                           # Map raw scores to (0, 1); used as driver probability
        
        # Use provided node_names or generate defaults
        if node_names is None:
            node_names = [f"Gene_{i}" for i in range(len(scores))]                          # Fallback gene labels if no names provided
        
        if eliminated_node_ids is not None:
            kept_mask = np.array([i not in eliminated_node_ids
                                    for i in range(len(scores))])                           # Boolean array marking which nodes survived Schur complement elimination
        
        # IMPORTANT: Store everything with original_index BEFORE sorting
        df = pd.DataFrame({
            'original_index': np.arange(len(scores)),                                       # Preserve original node ordering for alignment checks
            'gene_id': np.arange(len(scores)),  # Original gene ID                          # Same as original_index; kept for compatibility
            'gene_name': node_names,  # <-- Use provided names in original order            # Gene symbol aligned to original node order
            'driver_score': scores.cpu().numpy(),                                           # Raw scores from ranking head
            'driver_probability': scores_normalized.cpu().numpy(),                          # Sigmoid-normalized scores
            'true_label': labels.cpu().numpy(),                                             # Ground truth: 1=known_driver, 0=non-driver
        })
        
        # NOW sort by score
        df = df.sort_values('driver_score', ascending=False).reset_index(drop=True)         # Sort genes by predicted driver likelihood
        df['rank'] = df.index + 1                                                           # 1-indexed rank; rank 1 = highest scoring gene    
        df['percentile'] = df['rank'] / len(df) * 100                                       # Percentile rank; rank 1 -> ~0%, last rank -> 100%
        
        # Save to CSV if path provided
        if save_path is not None:
            from pathlib import Path
            save_dir = Path(save_path)
            save_dir.mkdir(exist_ok=True, parents=True)                                     # Create output directory if it doesn't exist
            
            prefix = f"{save_prefix}_" if save_prefix else ""
            output_file = save_dir / f"{prefix}gene_rankings.csv"
            df.to_csv(output_file, index=False)
            
            print(f"✓ Saved gene rankings to: {output_file}")
        
        return df                                                                           # Dataframe sorted by driver_score with rank, percentile, and true label columns

    
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
            node_names = np.array([f"Gene_{i}" for i in range(num_genes)])                      # Fallback: Integer indexed gene names
        elif isinstance(node_names, list):
            # Convert list to numpy array
            node_names = np.array(node_names)                                                   # Convert list to numpy array for consistent indexing
        elif isinstance(node_names, torch.Tensor):
            node_names = node_names.cpu().numpy()                                               # Move tensor to CPU numpy
        elif not isinstance(node_names, np.ndarray):
            # Handle any other iterable
            node_names = np.array(list(node_names))                                             # Handle any other iterable type
        
        # Validate length
        if len(node_names) != num_genes:
            logger.error(f"Node names length ({len(node_names)}) doesn't match labels ({num_genes})")
            node_names = np.array([f"Gene_{i}" for i in range(num_genes)])                      # Fallback to index-based names on mismatch
        
        df_scores = self.score_all_genes(
            data = data, labels = labels, node_names = node_names,                              # Get raw scores and ranks for all genes
            curvature_type=curvature_type,
            device=device
        )

        # Add this verification:
        logger.info("\n=== VERIFYING DATAFRAME ALIGNMENT ===")
        tp53_rows = df_scores[df_scores['gene_name'] == 'TP53']                                 # Spot-check alignment using TP53 (a well-known driver)
        if len(tp53_rows) > 0:
            tp53_row = tp53_rows.iloc[0]
            logger.info(f"TP53 in DataFrame:")
            logger.info(f"  original_index: {tp53_row['original_index']}")
            logger.info(f"  gene_name: {tp53_row['gene_name']}")
            logger.info(f"  true_label: {tp53_row['true_label']}")
            logger.info(f"  driver_score: {tp53_row['driver_score']:.4f}")
            
            # Verify against labels tensor
            orig_idx = int(tp53_row['original_index'])
            actual_label = labels[orig_idx].item()
            logger.info(f"  labels[{orig_idx}]: {actual_label}")                                # Look up TP53's label in the original tensor using stored original_index
            
            if tp53_row['true_label'] == actual_label == 1:
                logger.info("  ✓ TP53 alignment is CORRECT!")                                  # Gene name, score, and label are consistent
            else:
                logger.error("  ❌ TP53 alignment is BROKEN!")                                 # Mismatch indicates a node ordering bug
        logger.info("=" * 40 + "\n")
        
        device = device if device else self.device
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        labels = labels.to(device)
        
        known_drivers = labels == 1                                                             
        known_driver_indices = torch.where(known_drivers)[0].cpu().numpy()                      # Integer indices of all known driver genes
        unknown_indices = torch.where(~known_drivers)[0].cpu().numpy()                          # Integer indices of all non-driver genes
        
        print(f"\nKnown drivers: {len(known_driver_indices)}")
        print(f"Unknown genes: {len(unknown_indices)}")
        
        driver_scores = df_scores.loc[known_driver_indices, 'driver_score'].values              # Raw scores of known drivers; used as reference distribution
        driver_mean = driver_scores.mean()                                                      # Mean driver score; used to compute z-scores
        driver_std = driver_scores.std()                                                        # Std of driver scores; used to compute z-scores
        
        print(f"\nKnown Driver Score Statistics:")
        print(f"  Mean: {driver_mean:.4f}")
        print(f"  Std: {driver_std:.4f}")
        print(f"  Median: {np.median(driver_scores):.4f}")
        
        # Compute p-values via empirical comparison
        print(f"\nComputing empirical p-values...")
        
        # Initialize p-values array
        pvalues = np.ones(len(df_scores))                                                       # Initialize all p-values to 1.0 (non-significant)
        
        # For unknown genes, compute empirical p-value
        # P-value = proportion of known drivers with score >= this gene's score
        for idx in unknown_indices:
            if idx % 500 == 0:
                print(f"  Processing gene {idx}/{len(unknown_indices)}...", end='\r')
            
            observed_score = df_scores.loc[idx, 'driver_score']
            
            # Empirical p-value: what fraction of known drivers score as high or higher?
            count_higher = (driver_scores >= observed_score).sum()                              # How many known drivers score atleast as high as this gene
            pvalue = (count_higher + 1) / (len(driver_scores) + 1)                              # Laplace-smoothed empirical p-value; +1 numerator/denominator avoids p=0
            pvalues[idx] = pvalue
        
        print(f"  Processing gene {len(unknown_indices)}/{len(unknown_indices)}... Done!")
        
        # Known drivers get p-value of 0 (they are the reference)
        pvalues[known_driver_indices] = 0.0                                                     # Known drivers are the reference class; assign p=0 (not being tested)
        
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
        _, adjusted_pvalues_unknown, _, _ = multipletests(unknown_pvalues, method='fdr_bh')    # Benjamini-Hochberg FDR correction; controls false discovery rate across all tested genes
        
        adjusted_pvalues = np.ones(len(df_scores))                                             # Initialize all p-adjusted values to 1.0
        adjusted_pvalues[known_driver_indices] = 0.0                                           # Known drivers get adjusted to p=0
        adjusted_pvalues[unknown_indices] = adjusted_pvalues_unknown                           # Fill in BH-corrected values for unknown genes
        
        df_scores['pvalue'] = pvalues                                                          # Raw empirical p-values
        df_scores['adjusted_pvalue'] = adjusted_pvalues                                        # FDR-corrected p-values
        df_scores['is_known_driver'] = df_scores['true_label'] == 1                            # Boolean flag for known drivers
        df_scores['is_significant'] = (adjusted_pvalues < 0.05) & (~df_scores['is_known_driver'])   # Significant novel candidates: FDR < 0.05 AND not already a known driver
        df_scores['z_score'] = (df_scores['driver_score'] - driver_mean) / driver_std          # How many std deviations above/below the mean driver scores
        
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
            all_genes_file = save_dir / f"{prefix}all_genes_scored.csv"                     # Completed scored gene table
            df_scores.to_csv(all_genes_file, index=False)
            print(f"\n✓ Saved all genes to: {all_genes_file}")
            
            # Save significant genes only
            if len(significant_unknowns) > 0:
                sig_file = save_dir / f"{prefix}significant_genes.csv"                      # Subset: Only FDR-significant novel candidates
                significant_unknowns.to_csv(sig_file, index=False)
                print(f"✓ Saved {len(significant_unknowns)} significant genes to: {sig_file}")
            
            # Save known drivers for reference
            known_drivers_df = df_scores[df_scores['is_known_driver']]
            known_file = save_dir / f"{prefix}known_drivers_scores.csv"                     # Scores of all known drivers; useful as performance reference
            known_drivers_df.to_csv(known_file, index=False)
            print(f"✓ Saved {len(known_drivers_df)} known drivers to: {known_file}")
            
            # Save summary statistics
            summary_file = save_dir / f"{prefix}scoring_summary.txt"                        # Human-readable text summary of counts and statistics
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
        
        return df_scores                                                                   # Full dataframe with scores, ranks, p-values, z-scores, and significance flags for every gene


def compute_ndcg(
    scores: np.ndarray, 
    labels: np.ndarray,
    k: Optional[int] = None
) -> float:
    """Compute Normalized Discounted Cumulative Gain."""
    if k is None:
        k = len(scores)                                                                    # Default to full list NDCG when no cutoff is specified

    if isinstance(scores, torch.Tensor):
        scores = scores.detach().cpu().numpy()                                             # Detach from autograd graph before converting to numpy
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    
    sorted_indices = np.argsort(-scores)[:k]                                               # Indices of top-k genes sorted by descending score
    sorted_labels = labels[sorted_indices]                                                 # True labels of the top-k genes in rank order

    gains = sorted_labels                                                                  # Relevance gains: 1 for known drivers, 0 otherwise
    discounts = 1.0 / np.log2(np.arange(2, len(gains) + 2))                                # Position discounts: 1/log2(rank+1); rank 1 gets discount 1.0, rank 2 gets 0.63, etc
    dcg = np.sum(gains * discounts)                                                        # DCG: Sum of discounted gains for the predicted ranking

    ideal_sorted = np.sort(labels)[::-1][:k]                                               # Ideal ranking: Put all drivers at the top
    idcg = np.sum(ideal_sorted * discounts[:len(ideal_sorted)])                            # Ideal DCG: maximum achievable DCG at cutoff k

    if idcg == 0:
        return 0.0                                                                         # No Positive labels in the eval set; NDCG is undefined, return 0

    return dcg / idcg                                                                      # NDCG in [0, 1]; 1.0 means all drivers were ranked at the top

def compute_mrr(
    scores: np.ndarray, 
    labels: np.ndarray
) -> float:
    
    """Compute Mean Reciprocal Rank."""
    sorted_indices = np.argsort(-scores)                                                   # All gene indices sorted by descending score
    driver_indices = np.where(labels == 1)[0]                                              # Position of known driver genes in the original (unsorted) array
    
    if len(driver_indices) == 0:
        return 0.0                                                                         # No drivers in eval set; MRR is undefined
    
    reciprocal_ranks = []
    for driver_idx in driver_indices:
        rank = np.where(sorted_indices == driver_idx)[0][0] + 1                            # 1-indexed position of this driver in the sorted ranking
        reciprocal_ranks.append(1.0 / rank)                                                # Reciprocal rank: rank 1 -> 1.0, rank 10 -> 0.1 etc
    
    return np.mean(reciprocal_ranks)                                                       # MRR: Mean of 1/rank across all known drivers; higher = drivers appear earlier in the ranking