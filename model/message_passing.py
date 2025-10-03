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
            self.att_src = nn.Linear(out_channels, 1)
            self.att_dst = nn.Linear(out_channels, 1)
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
        
        # Transform node features
        x = self.lin(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        filtered_edge_index, filtered_edge_attr = self.filter_edges_by_curvature(
            edge_index, edge_curvature, edge_attr
        )
        
        if self.hop_type.lower() == 'two_hop':
            out = self.two_hop_propagation(x, filtered_edge_index, filtered_edge_attr)
        else:
            out = self.propagate(filtered_edge_index, x = x, edge_attr = filtered_edge_attr)

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
        
        if self.curvature_type == 'positive':
            mask = edge_curvature > 0
        elif self.curvature_type == 'negative':
            mask =  edge_curvature < 0
        elif self.curvature_type == 'both':
            mask = torch.ones_like(edge_curvature, dtype = torch.bool)
        else:
            raise ValueError(f'Invalid Curvature type: {self.curvature_type}')
        
        filtered_edge_index = edge_index[:, mask]
        filtered_edge_attr = edge_attr[mask] if edge_attr is not None else None
        
        logger.debug(f"Filtered edges: {mask.sum().item()} / {len(mask)} "
                    f"({100 * mask.sum().item() / len(mask):.1f}%) "
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
        
        out_1hop = self.propagate(edge_index, x=x, edge_attr = edge_attr)
        
        # Second hop: propagate messages from first hop
        out_2hop = self.propagate(edge_index, x = out_1hop, edge_attr = edge_attr)
        
        # Combine original features with 2-hop aggregation
        return out_1hop + out_2hop
    
    def message(
        self,
        x_j: torch.Tensor,
        x_i: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Construct messages from neighbors
        
        Args:
            x_j: Features of neighbor nodes [num_edges, out_channels]
            x_i: Features of center nodes [num_edges, out_channels]
            edge_attr: Edge attributes
        
        Returns:
            Messages to be aggregated [num_edges, out_channels]
        """
        
        if self.use_attention and x_i is not None:
            alpha = self.compute_attention(x_i, x_j)
            alpha = F.dropout(alpha, p = self.dropout, training = self.training)
            return x_j * alpha.view(-1, 1)
        else:
            return x_j
        
    def compute_attention(
        self,
        x_i: torch.Tensor,
        x_j: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute attention coefficients between nodes
        
        Similar to GAT attention mechanism
        """
        # Attention mechanism: a(x_i, x_j)
        att_src = self.att_src(x_i)
        att_dst = self.att_dst(x_j)
        alpha = att_src + att_dst
        alpha = F.leaky_relu(alpha, negative_slope=0.2)
        alpha = F.softmax(alpha, dim = 0)
        
        return alpha

