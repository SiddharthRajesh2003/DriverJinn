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
    
    Parameters:
        in_channels: Input feature dimensionality
        out_channels: Output feature dimensionality
        curvature_type: 'positive', 'negative', or 'both'
        hop_types: List of hop types ['one_hop', 'two_hop']
        use_attention: Whether to use attention mechanism
        dropout: Dropout rate for regularization
        min_edge_ratio: Minimum fraction of edges to keep after filtering
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int = 3,
        curvature_types:List[str] = ['positive', 'negative', 'both'],
        hop_types: List[str] = ['one_hop', 'two_hop'],
        use_attention: bool = True,
        dropout: float = 0.2,
        min_edge_ratio: float = 0.15
    ):
        super().__init__()
        self.num_layers = num_layers
        self.curvature_types = curvature_types
        self.hop_types = hop_types
        
        # Log configuration
        logger.info(f"Initializing CurvatureAwareGNN with:")
        logger.info(f"  Curvature types: {curvature_types}")
        logger.info(f"  Hop types: {hop_types}")
        logger.info(f"  Total pathways: {len(curvature_types)} × {len(hop_types)} = {len(curvature_types) * len(hop_types)}")
        
        # Input projection (shared across all pathways)
        self.input_proj = nn.Linear(in_channels, hidden_channels)
        
        # Create message passing layers for EACH (curvature_type, hop_type) combination
        self.conv_layers = nn.ModuleDict()
        
        for curv_type in self.curvature_types:
            for hop_type in self.hop_types:
                # Create unique key for each pathway
                pathway_key = f"{curv_type}_{hop_type}"
                
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
                            dropout=dropout,
                            min_edge_ratio=min_edge_ratio
                        )
                    )
                
                self.conv_layers[pathway_key] = layers
                logger.debug(f"  Created pathway: {pathway_key} with {num_layers} layers")
        
        # Batch normalization layers (shared across pathways within each layer)
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_channels) for _ in range(num_layers)
        ])
        
        # Optional: Pathway-specific batch norms for better separation
        # self.pathway_batch_norms = nn.ModuleDict({
        #     pathway_key: nn.ModuleList([
        #         nn.BatchNorm1d(hidden_channels) for _ in range(num_layers)
        #     ]) for pathway_key in self.conv_layers.keys()
        # })
        
        self.dropout = dropout
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_curvature: torch.Tensor,
        return_all_layers: bool = True
    ) -> Dict[str, List[torch.Tensor]]:
        """
        Forward pass through all curvature-specific AND hop-specific pathways.
        
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Graph connectivity [2, num_edges]
            edge_curvature: Edge curvature values [num_edges]
            return_all_layers: If True, return outputs from all layers
        
        Returns:
            Dictionary with two levels:
            - Top level: curvature_type ('positive', 'negative', 'both')
            - Each curvature_type contains a dict with hop_type keys
            
            Example structure:
            {
                'positive': {
                    'one_hop': [layer1_output, layer2_output, layer3_output],
                    'two_hop': [layer1_output, layer2_output, layer3_output]
                },
                'negative': {
                    'one_hop': [...],
                    'two_hop': [...]
                },
                'both': {
                    'one_hop': [...],
                    'two_hop': [...]
                }
            }
        """
        
        x = self.input_proj(x)
        x = F.relu(x)
        
        outputs = {curv_type: {hop_type: [] for hop_type in self.hop_types} 
                    for curv_type in self.curvature_types}
        
        for curv_type in self.curvature_types:
            for hop_type in self.hop_types:
                pathway_key = f"{curv_type}_{hop_type}"
                h = x.clone()  # Start fresh for each pathway
                
                layer_outputs = []
                
                for i, conv in enumerate(self.conv_layers[pathway_key]):
                    # Message passing
                    h = conv(h, edge_index, edge_curvature)
                    
                    # Normalization
                    h = self.batch_norms[i](h)
                    
                    # Activation
                    h = F.relu(h)
                    
                    # Dropout
                    h = F.dropout(h, p=self.dropout, training=self.training)
                    
                    if return_all_layers:
                        layer_outputs.append(h)
                
                outputs[curv_type][hop_type] = layer_outputs if return_all_layers else [h]
        
        return outputs
    
    def get_pathway_names(self) -> List[str]:
        """
        Get list of all pathway names for easy iteration.
        
        Returns:
            List of pathway keys like ['positive_one_hop', 'positive_two_hop', ...]
        """
        return list(self.conv_layers.keys())
    
    def count_parameters(self) -> Dict[str, int]:
        """
        Count parameters for each pathway and total.
        
        Returns:
            Dictionary with parameter counts
        """
        counts = {}
        total = 0
        
        # Shared parameters
        shared_params = sum(p.numel() for p in self.input_proj.parameters())
        shared_params += sum(p.numel() for p in self.batch_norms.parameters())
        counts['shared'] = shared_params
        total += shared_params
        
        # Pathway-specific parameters
        for pathway_key, layers in self.conv_layers.items():
            pathway_params = sum(
                p.numel() for layer in layers for p in layer.parameters()
            )
            counts[pathway_key] = pathway_params
            total += pathway_params
        
        counts['total'] = total
        
        return counts