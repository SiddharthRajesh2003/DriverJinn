#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from torch.utils.checkpoint import checkpoint as gradient_checkpoint
from utils.logging_manager import get_logger
from model.message_passing import CurvatureConstrainedMessagePassing

logger = get_logger(__name__)

class CurvatureAwareGNN(nn.Module):
    """
    Multi-layer GNN with curvature-constrained message passing
    
    This architecture creates multiple pathways based on:
    - Curvature types: How edges are filtered/weighted ('positive', 'negative', 'both')
    - Hop types: Message passing distance ('one_hop', 'two_hop')
    - Attention modes: How curvature influences attention ('standard', 'edge_feature', 'bias', 'gated', 'hybrid')
    
    Parameters:
        in_channels: Input feature dimensionality
        hidden_channels: Hidden layer dimensionality
        num_layers: Number of message passing layers per pathway
        curvature_types: List of curvature types ['positive', 'negative', 'both']
        hop_types: List of hop types ['one_hop', 'two_hop']
        use_attention: Whether to use attention mechanism
        attention_mode: How to incorporate curvature in attention
            - 'standard': No curvature in attention (uses filtering only)
            - 'edge_feature': Concatenate curvature with node features
            - 'bias': Add curvature as bias to attention scores
            - 'gated': Learn to mix node and curvature attention
            - 'hybrid': Combine edge_feature + bias (recommended)
        heads: Number of attention heads
        concat: Whether to concatenate or average multi-head outputs
        dropout: Dropout rate for regularization
        min_edge_ratio: Minimum fraction of edges to keep after filtering
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int = 3,
        curvature_types: List[str] = ['positive', 'negative', 'both'],
        hop_types: List[str] = ['one_hop', 'two_hop'],
        use_attention: bool = True,
        attention_mode: str = 'hybrid',
        heads: int = 4,
        concat: bool = True,
        dropout: float = 0.2,
        min_edge_ratio: float = 0.15,
        negative_slope: float = 0.2
    ):
        super().__init__()
        self.num_layers = num_layers
        self.curvature_types = curvature_types
        self.hop_types = hop_types
        self.attention_mode = attention_mode
        self.heads = heads
        self.concat = concat
        
        # Calculate output dimension after multi-head attention
        if use_attention and concat:
            layer_out_channels = hidden_channels * heads
        else:
            layer_out_channels = hidden_channels
        
        # Log configuration
        logger.info(f"Initializing CurvatureAwareGNN with:")
        logger.info(f"  Curvature types: {curvature_types}")
        logger.info(f"  Hop types: {hop_types}")
        logger.info(f"  Attention mode: {attention_mode}")
        logger.info(f"  Multi-head attention: {heads} heads, concat={concat}")
        logger.info(f"  Total pathways: {len(curvature_types)} × {len(hop_types)} = {len(curvature_types) * len(hop_types)}")
        
        # Input projection (shared across all pathways)
        self.input_proj = nn.Linear(in_channels, hidden_channels)
        
        # Create message passing layers for EACH (curvature_type, hop_type) combination
        self.conv_layers = nn.ModuleDict()
        
        for curv_type in curvature_types:
            for hop_type in hop_types:
                # Create unique key for each pathway
                pathway_key = f"{curv_type}_{hop_type}"
                
                layers = nn.ModuleList()
                for i in range(num_layers):
                    # First layer takes hidden_channels, outputs layer_out_channels
                    # Subsequent layers need to match dimensions
                    in_ch = hidden_channels if i == 0 else layer_out_channels
                    
                    layers.append(
                        CurvatureConstrainedMessagePassing(
                            in_channels=in_ch,
                            out_channels=hidden_channels,  # Per-head dimension
                            curvature_type=curv_type,
                            hop_type=hop_type,
                            aggregation='add',
                            use_attention=use_attention,
                            attention_mode=attention_mode,
                            heads=heads,
                            concat=concat,
                            dropout=dropout,
                            min_edge_ratio=min_edge_ratio,
                            negative_slope=negative_slope
                        )
                    )
                
                self.conv_layers[pathway_key] = layers
                logger.debug(f"  Created pathway: {pathway_key} with {num_layers} layers")
        
        # Batch normalization layers (shared across pathways within each layer)
        # All layers output layer_out_channels (hidden_channels * heads if concat=True)
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(layer_out_channels) for _ in range(num_layers)
        ])
        
        # Optional: Pathway-specific batch norms for better separation
        self.use_pathway_norms = False  # Set to True to enable
        if self.use_pathway_norms:
            self.pathway_batch_norms = nn.ModuleDict({
                pathway_key: nn.ModuleList([
                    nn.BatchNorm1d(layer_out_channels) for _ in range(num_layers)
                ]) for pathway_key in self.conv_layers.keys()
            })
        
        self.dropout = dropout
        self.use_pathway_checkpointing = True  # Enable per-pathway checkpointing

    def _process_single_pathway(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_curvature: torch.Tensor,
        pathway_key: str,
        curv_type: str,
        return_all_layers: bool,
        return_attention: bool
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Process a single pathway through all layers.
        This function is checkpointed to save memory.
        """
        h = x.clone()
        layer_outputs = []
        layer_attentions = []

        for i, conv in enumerate(self.conv_layers[pathway_key]):
            # Message passing
            h, alpha = conv(h, edge_index, edge_curvature, return_attention)

            # Normalization
            if self.use_pathway_norms:
                h = self.pathway_batch_norms[pathway_key][i](h)
            else:
                h = self.batch_norms[i](h)

            # Activation
            h = F.relu(h)

            # Dropout
            h = F.dropout(h, p=self.dropout, training=self.training)

            if return_all_layers:
                layer_outputs.append(h)
                if return_attention and alpha is not None:
                    layer_attentions.append(alpha)

        # If not return_all_layers, only return final output
        if not return_all_layers:
            layer_outputs = [h]
            if return_attention and alpha is not None:
                layer_attentions = [alpha]

        return layer_outputs, layer_attentions

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_curvature: torch.Tensor,
        return_all_layers: bool = True,
        return_attention: bool = False
    ) -> Tuple[Dict[str, Dict[str, List[torch.Tensor]]], Optional[Dict[str, Dict[str, List[torch.Tensor]]]]]:
        """
        Forward pass through all curvature-specific AND hop-specific pathways.
        
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Graph connectivity [2, num_edges]
            edge_curvature: Edge curvature values [num_edges]
            return_all_layers: If True, return outputs from all layers
            return_attention: If True, return attention weights from all layers
        
        Returns:
            Tuple of (outputs, attention_weights):
            
            outputs: Dictionary with two levels:
            {
                'positive': {
                    'one_hop': [layer1_output, layer2_output, layer3_output],
                    'two_hop': [layer1_output, layer2_output, layer3_output]
                },
                'negative': {...},
                'both': {...}
            }
            
            attention_weights (if return_attention=True): Same structure as outputs
            but contains attention weight tensors instead of node features
        """
        
        # Project input features
        x = self.input_proj(x)
        x = F.relu(x)
        x = F.dropout(x, p = self.dropout, training=self.training)
        
        # Initialize output structures
        outputs = {curv_type: {hop_type: [] for hop_type in self.hop_types} 
                    for curv_type in self.curvature_types}
        
        attention_weights = None
        if return_attention:
            attention_weights = {
                curv_type: {hop_type: [] for hop_type in self.hop_types}
                for curv_type in self.curvature_types
            }
        
        for curv_type in self.curvature_types:
            for hop_type in self.hop_types:
                pathway_key = f"{curv_type}_{hop_type}"

                # CRITICAL: Use checkpointing during training for memory efficiency
                if self.training and self.use_pathway_checkpointing:
                    # Checkpoint each pathway to save memory
                    layer_outputs, layer_attentions = gradient_checkpoint(
                        self._process_single_pathway,
                        x, edge_index, edge_curvature,
                        pathway_key, curv_type,
                        return_all_layers, return_attention,
                        use_reentrant=False
                    )
                else:
                    # During eval: process pathway and immediately detach to save memory
                    layer_outputs, layer_attentions = self._process_single_pathway(
                        x, edge_index, edge_curvature,
                        pathway_key, curv_type,
                        return_all_layers, return_attention
                    )

                    # CRITICAL: Detach outputs during eval to free computational graph
                    if not self.training:
                        layer_outputs = [out.detach() for out in layer_outputs]
                        if layer_attentions is not None:
                            layer_attentions = [att.detach() if att is not None else None
                                                for att in layer_attentions]

                # Store pathway outputs
                outputs[curv_type][hop_type] = layer_outputs

                if return_attention:
                    attention_weights[curv_type][hop_type] = layer_attentions
        
        return outputs, attention_weights
    
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
    
    def get_attention_statistics(
        self,
        attention_weights: Dict[str, Dict[str, List[torch.Tensor]]]
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Compute statistics about attention weights across pathways.
        
        Args:
            attention_weights: Attention weights from forward pass
        
        Returns:
            Dictionary with statistics for each pathway and layer:
            {
                'positive_one_hop': {
                    'layer_0': {'mean': 0.25, 'std': 0.1, 'max': 0.8, 'min': 0.0},
                    'layer_1': {...},
                    ...
                }
            }
        """
        stats = {}
        
        for curv_type in self.curvature_types:
            for hop_type in self.hop_types:
                pathway_key = f"{curv_type}_{hop_type}"
                stats[pathway_key] = {}
                
                attentions = attention_weights[curv_type][hop_type]
                
                for layer_idx, alpha in enumerate(attentions):
                    if alpha is not None:
                        stats[pathway_key][f'layer_{layer_idx}'] = {
                            'mean': alpha.mean().item(),
                            'std': alpha.std().item(),
                            'max': alpha.max().item(),
                            'min': alpha.min().item()
                        }
        
        return stats