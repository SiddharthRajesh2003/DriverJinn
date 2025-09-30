import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_self_loops, degree
import numpy as np
from typing import Optional, Dict, Tuple
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
        self.curvature_type = curvature_type
        self.hop_type = hop_type
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