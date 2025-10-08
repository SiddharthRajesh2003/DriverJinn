import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List
from utils.logging_manager import get_logger
from model.message_passing import CurvatureConstrainedMessagePassing

logger = get_logger(__name__)

class CurvatureAwareGNN(nn.Module):
    """
    Multi-layer GNN with curvature-constrained message passing
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int = 3,
        curvature_types:List[str] = ['positive', 'negative', 'both'],
        hop_type: str = 'one_hop',
        use_attention: bool = True,
        dropout: float = 0.2
    ):
        super().__init__()
        self.num_layers = num_layers
        self.curvature_types = curvature_types
        
        # Input projection
        self.input_proj = nn.Linear(in_channels, hidden_channels)
        
        # Create multiple message passing layers for each curvature type
        self.conv_layers = nn.ModuleDict()
        for curv_type in self.curvature_types:
            layers = nn.ModuleList()
            for i in range(num_layers):
                layers.append(
                    CurvatureConstrainedMessagePassing(
                        in_channels=hidden_channels,
                        out_channels=hidden_channels,
                        curvature_type=curv_type,
                        hop_type=hop_type,
                        aggregation='add',
                        use_attention=use_attention,
                        dropout=dropout
                    )
                )
        
            self.conv_layers[curv_type] = layers
    
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_channels) for _ in range(num_layers)
        ])
        
        self.dropout = dropout
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_curvature: torch.Tensor,
        return_all_layers: bool = True
    ) -> Dict[str, List[torch.Tensor]]:
        """
        Forward pass through all curvature-specific pathways
        
        Returns:
            Dictionary mapping curvature_type -> list of layer outputs
        """
        
        x = self.input_proj(x)
        x = F.relu(x)
        
        outputs = {curv_type: [] for curv_type in self.curvature_types}
        
        for curv_type in self.curvature_types:
            h = x
            layer_outputs = []
            
            for i, conv in enumerate(self.conv_layers[curv_type]):
                h = conv(h, edge_index, edge_curvature)
                h = self.batch_norms[i](h)
                h = F.relu(h)
                h = F.dropout(h, p = self.dropout, training = self.training)
                
                if return_all_layers:
                    layer_outputs.append(h)
                    
            outputs[curv_type] = layer_outputs if return_all_layers else [h]
        
        return outputs
    
class ProjectionHead(nn.Module):
    """
    Projection head for contrastive learning
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int = 2
    ):
        super().__init__()
        
        layers = []
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2: # No activation on last layer
                layers.append(nn.BatchNorm1d(dims[i+1]))
                layers.append(nn.ReLU())
                
        self.projection = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x)
    
class BinaryClassifier(nn.Module):
    """
    Binary classifier for known driver vs non-driver genes
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1) # Binary output
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x).squeeze(-1)