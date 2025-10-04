import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List
from utils.logging_manager import get_logger
from model.message_passing import CurvatureConstrainedMessagePassing


logger = get_logger(__name__)

class CurvatureConstrainedGNN(nn.Module):
    """
    Multi-layer GNN with curvature-constrained message passing
    
    Implements flexible framework allowing:
    - Different curvature types per layer
    - One-hop and two-hop propagation
    - Attention mechanisms
    
    Args:
        in_channels: Input feature dimensionality
        hidden_channels: Hidden layer dimensionality
        out_channels: Output dimensionality (num classes)
        num_layers: Number of GNN layers
        layer_configs: List of configs for each layer, each dict contains:
            - curvature_type: 'positive', 'negative', or 'both'
            - hop_type: 'one_hop' or 'two_hop'
            - use_attention: bool
        dropout: Dropout rate
        use_batch_norm: Whether to use batch normalization
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
        layer_configs: Optional[List] = None,
        dropout: float = 0.5,
        use_batch_norm: bool = True,
        aggregation: str = 'add'
    ):
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm
        
        if layer_configs is None:
            # Strategy from paper: positive curvature first, then negative
            layer_configs = [
                {'curvature_type': 'positive', 'hop_type': 'one_hop', 'use_attention': False},
                {'curvature_type': 'negative', 'hop_type': 'two_hop', 'use_attention': True},
                {'curvature_type': 'both', 'hop_type': 'two_hop', 'use_attention': True}
            ]
        
        while len(layer_configs) < num_layers:
            layer_configs.append(layer_configs[-1])
        
        self.layer_configs = layer_configs[:num_layers]
        
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        for i in range(num_layers):
            in_dim = in_channels if i == 0 else hidden_channels
            out_dim = out_channels if i == num_layers - 1 else hidden_channels
            
            config = self.layer_configs[i]
            conv = CurvatureConstrainedMessagePassing(
                in_channels = in_dim,
                out_channels = out_dim,
                curvature_type = config.get('curvature_type', 'positive'),
                hop_type = config.get('hop_type', 'one_hop'),
                aggregation = aggregation,
                use_attention = config.get('use_attention', False),
                dropout = dropout
            )
            self.convs.append(conv)
            
            if self.use_batch_norm and i < num_layers - 1:
                self.batch_norms.append(nn.BatchNorm1d(out_dim))
        
        logger.info(f"Created CurvatureConstrainedGNN with {num_layers} layers")
        for i, config in enumerate(self.layer_configs):
            logger.info(f"  Layer {i}: {config}")
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_curvature: torch.Tensor,
        edge_attr: Optional[torch.Tensor]
    )   -> torch.Tensor:
        """
        Forward pass through the GNN
        
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Graph connectivity [2, num_edges]
            edge_curvature: Curvature values [num_edges]
            edge_attr: Optional edge attributes
        
        Returns:
            Node embeddings/predictions [num_nodes, out_channels]
        """
        
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_curvature, edge_attr)
        
            if i < self.num_layers - 1:
                if self.use_batch_norm:
                    x = self.batch_norms[i](x)
                
                x = F.relu(x)
                x = F.dropout(x, p = self.dropout, training = self.training)
                
        return x

class AdaptiveCurvatureGNN(nn.Module):
    """
    Adaptive GNN that learns to weight positive vs negative curvature paths
    
    This extends the basic approach by learning which curvature type
    is more important for each layer/task.
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
        dropout: float = 0.5,
        use_batch_norm: bool = True,
        device: str = 'cpu'
    ):
        super().__init__()
        
        self.dropout = dropout
        self.num_layers = num_layers
        self.device = device
        
        self.pos_convs = nn.ModuleList()
        self.neg_convs = nn.ModuleList()
        self.curvature_weights = nn.ParameterList()
        self.batch_norms = nn.ModuleList()
        
        for i in range(self.num_layers):
            in_dim = in_channels if i == 0 else hidden_channels
            out_dim = out_channels if i == num_layers - 1 else hidden_channels
            
            self.pos_convs.append(
                CurvatureConstrainedMessagePassing(
                    in_dim, out_dim, curvature_type = 'positive',
                    hop_type = 'one_hop', dropout = self.dropout
                )
            )
            
            self.neg_convs.append(
                CurvatureConstrainedMessagePassing(
                    in_dim, out_dim, curvature_type = 'negative',
                    hop_type = 'one_hop', dropout = self.dropout
                )
            )
            
            self.curvature_weights.append(nn.Parameter(torch.Tensor([0.5, 0.5])))
            if use_batch_norm and i < num_layers - 1:
                self.batch_norms.append(nn.BatchNorm1d(out_dim))
                
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_curvature: torch.Tensor,
        edge_attr: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Forward pass with adaptive curvature weighting"""
        for i in range(self.num_layers):
            x_pos = self.pos_convs[i](x, edge_index, edge_curvature, edge_attr)
            x_neg = self.neg_convs[i](x, edge_index, edge_curvature, edge_attr)
            
            weights = F.softmax(self.curvature_weights[i], dim = 0)
            x = weights[0] * x_pos + weights[1] * x_neg
            
            if i < self.num_layers - 1:
                if hasattr(self, 'batch_norms') and len(self.batch_norms) > i:
                    x = self.batch_norms[i](x)
                
                x = F.relu(x)
                x = F.dropout(x, p = self.dropout, training = self.training)
                
        return x

def main():
    import argparse
    import pickle
    
    parser = argparse.ArgumentParser(description='Run the curvature constrained GNN model')
    parser.add_argument('--dataset_file', default=None)
    parser.add_argument('--outdir', default=None)
    
    args = parser.parse_args()
    
    dataset_file = args.dataset_file
    
    with open(dataset_file, 'rb') as f:
        data_dict = pickle.load(f)
        f.close()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    node_names = data_dict['node_name']
    edge_index = data_dict['edge_index'].to(device)
    features = data_dict['feature'].to(device)
    curvature = data_dict['ollivier_curvature'].to(device) 
    
    
    
    configs = [
        {'curvature_type': 'positive', 'hop_type': 'one_hop', 'use_attention': False},
        {'curvature_type': 'negative', 'hop_type': 'two_hop', 'use_attention': True}
    ]
    
    model = AdaptiveCurvatureGNN(
        in_channels=73,
        hidden_channels = 128,
        out_channels=2,
        num_layers = 3,
        dropout=0.5
    ).to(device)
    
    out = model(features, edge_index, curvature, edge_attr = data_dict['edge_features'].to(device))
    print(f'Output Head: {out.cpu().detach().numpy()[:, :5]}')
    