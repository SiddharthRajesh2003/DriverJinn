import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from typing import Optional, Tuple
from utils.logging_manager import get_logger

logger = get_logger(__name__)

class CurvatureConstrainedMessagePassing(MessagePassing):
    """
    Curvature-Constrained Message Passing Layer
    
    Implements the algorithm from the paper that selectively propagates
    information based on edge curvature values.
    
    Args:
        in_channels: Input feature dimensionality
        out_channels: Output feature dimensionality
        curvature_type: 'positive', 'negative', or 'both'
        hop_type: 'one_hop' or 'two_hop'
        aggregation: Aggregation method ('add', 'mean', 'max')
        use_attention: Whether to use attention mechanism
    """
    
    def __init__(
        self, 
        in_channels: int,
        out_channels: int,
        curvature_type:str = 'positive',
        hop_type:str = 'one_hop',
        aggregation: str = 'add',
        use_attention: bool = False,
        dropout: float = 0.0
    ):
        
        super().__init__(aggr=aggregation)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.curvature_type = curvature_type.lower()
        self.hop_type = hop_type.lower()
        self.use_attention = use_attention
        self.dropout = dropout
        
        # Transformation Weights
        self.lin = nn.Linear(self.in_channels, self.out_channels)
        
        if use_attention:
            # IMPORTANT: These should expect out_channels, not in_channels
            self.att_src = nn.Linear(out_channels, 1)
            self.att_dst = nn.Linear(out_channels, 1)
            # Optional: attention weight parameter
            self.att_weight = nn.Parameter(torch.Tensor(1, out_channels))
            nn.init.xavier_uniform_(self.att_weight)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        self.lin.reset_parameters()
        if self.use_attention:
            self.att_src.reset_parameters()
            self.att_dst.reset_parameters()
    
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_curvature: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with curvature-constrained message passing
        
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Graph connectivity [2, num_edges]
            edge_curvature: Curvature values for each edge [num_edges]
            edge_attr: Optional edge attributes [num_edges, edge_dim]
        
        Returns:
            Updated node features [num_nodes, out_channels]
        """
        
        # Validate inputs
        num_nodes = x.shape[0]
        max_node_idx = edge_index.max().item() if edge_index.numel() > 0 else -1
        
        if max_node_idx >= num_nodes:
            raise ValueError(f"Edge index contains node {max_node_idx} but only {num_nodes} nodes in features")
        
        # Transform node features
        x = self.lin(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Filter edges based on curvature
        filtered_edge_index, filtered_edge_attr = self.filter_edges_by_curvature(
            edge_index, edge_curvature, edge_attr
        )
        
        # Check if we have any edges left after filtering
        if filtered_edge_index.shape[1] == 0:
            logger.warning(f"No edges remain after filtering for curvature_type='{self.curvature_type}'")
            return x  # Return transformed features without message passing
        
        if self.hop_type.lower() == 'two_hop':
            out = self.two_hop_propagation(x, filtered_edge_index, filtered_edge_attr)
        else:
            # Standard message passing with size argument for proper indexing
            out = self.propagate(
                filtered_edge_index, 
                x=x, 
                edge_attr=filtered_edge_attr,
                size=(num_nodes, num_nodes)  # Explicitly set size
            )

        return out
    
    def filter_edges_by_curvature(
        self,
        edge_index: torch.Tensor,
        edge_curvature: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Filter edges based on curvature type (positive/negative)
        
        Returns:
            Filtered edge_index and edge_attr
        """
        # Sanity: ensure edge_index is [2, num_edges]
        num_edges_index = edge_index.shape[1]
        num_curv = edge_curvature.shape[0]

        # Use a small epsilon for numerical stability
        eps = 1e-8
        
        if self.curvature_type == 'positive':
            threshold = edge_curvature.mean() if edge_curvature.numel() > 0 else 0.0
            base_mask = edge_curvature > (threshold + eps)
        elif self.curvature_type == 'negative':
            threshold = edge_curvature.mean() if edge_curvature.numel() > 0 else 0.0
            base_mask = edge_curvature <= (threshold - eps)
        elif self.curvature_type == 'both':
            base_mask = torch.ones_like(edge_curvature, dtype=torch.bool)
        else:
            raise ValueError(f'Invalid Curvature type: {self.curvature_type}')

        # Handle different edge_curvature and edge_index dimensions
        if num_curv == num_edges_index:
            mask = base_mask
        elif num_curv * 2 == num_edges_index:
            # Undirected graph: curvature per undirected edge, but edge_index has both directions
            mask = torch.repeat_interleave(base_mask, 2)
        else:
            raise IndexError(
                f"Edge curvature length ({num_curv}) does not match edge_index edges ({num_edges_index}). "
                "Expected either same length or half (for undirected graphs)."
            )

        filtered_edge_index = edge_index[:, mask]
        filtered_edge_attr = edge_attr[mask] if edge_attr is not None else None

        logger.debug(f"Filtered edges: {mask.sum().item()} / {len(mask)} "
                    f"({100 * mask.sum().item() / max(len(mask), 1):.1f}%) "
                    f"for curvature_type='{self.curvature_type}'")

        return filtered_edge_index, filtered_edge_attr
    
    def two_hop_propagation(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Implement two-hop message passing along curvature-constrained paths
        
        This creates paths of length 2 following only edges with specific curvature
        """
        num_nodes = x.shape[0]
        
        # First hop
        out_1hop = self.propagate(
            edge_index, 
            x=x, 
            edge_attr=edge_attr,
            size=(num_nodes, num_nodes)
        )
        
        # Second hop: ALSO along curvature-constrained edges
        out_2hop = self.propagate(
            edge_index, 
            x=out_1hop, 
            edge_attr=edge_attr,
            size=(num_nodes, num_nodes)
        )
        
        # Combine with residual connection
        return x + 0.5 * out_1hop + 0.5 * out_2hop
    
    def message(
        self,
        x_j: torch.Tensor,
        x_i: Optional[torch.Tensor] = None,
        edge_index_i: Optional[torch.Tensor] = None,
        size_i: Optional[int] = None
    ) -> torch.Tensor:
        """
        Construct messages from neighbors
        
        Args:
            x_j: Features of neighbor nodes [num_edges, out_channels]
            x_i: Features of center nodes [num_edges, out_channels]
            edge_index_i: Edge indices for center nodes
            size_i: Number of center nodes
        
        Returns:
            Messages to be aggregated [num_edges, out_channels]
        """
        
        if self.use_attention and x_i is not None:
            # Ensure dimensions match
            if x_i.shape[1] != self.out_channels or x_j.shape[1] != self.out_channels:
                raise ValueError(
                    f"Dimension mismatch in attention: x_i shape {x_i.shape}, "
                    f"x_j shape {x_j.shape}, expected dim {self.out_channels}"
                )
            
            alpha = self.compute_attention(x_i, x_j, edge_index_i, size_i)
            alpha = F.dropout(alpha, p=self.dropout, training=self.training)
            return x_j * alpha.view(-1, 1)
        else:
            return x_j
        
    def compute_attention(
        self,
        x_i: torch.Tensor,
        x_j: torch.Tensor,
        edge_index_i: Optional[torch.Tensor] = None,
        size_i: Optional[int] = None
    ) -> torch.Tensor:
        """
        Compute attention coefficients between nodes
        
        Similar to GAT attention mechanism
        """
        # Remove debug prints in production
        # print("DEBUG: x_i.shape, x_j.shape:", x_i.shape, x_j.shape)
        # print("DEBUG: x_i.dtype, device:", x_i.dtype, x_i.device)
        
        # Compute attention scores
        att_src = self.att_src(x_i)  # [num_edges, 1]
        att_dst = self.att_dst(x_j)  # [num_edges, 1]
        
        # Combine attention scores
        alpha = att_src + att_dst
        alpha = F.leaky_relu(alpha, negative_slope=0.2)
        
        
        alpha = F.softmax(alpha, dim=0)
        
        return alpha