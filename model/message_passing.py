import torch
import torch.nn as nn
import torch.functional as F
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
        
        self.lin = nn.Linear(self.in_channels, self.out_channels)
        