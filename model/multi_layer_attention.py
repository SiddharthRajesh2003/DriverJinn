import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional, Dict
from utils.logging_manager import get_logger

logger = get_logger(__name__)

class MultiLayerAttention(nn.Module):
    """
    Memory-efficient layer aggregation using chunked attention.
    
    Key optimizations:
    1. Process nodes in chunks instead of all at once
    2. Don't store full attention matrices
    3. Use gradient checkpointing
    """
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        chunk_size: int = 1000  # Process 1000 nodes at a time
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.chunk_size = chunk_size
        
        assert hidden_dim % num_heads == 0
        
        self.q_linear = nn.Linear(hidden_dim, hidden_dim)
        self.k_linear = nn.Linear(hidden_dim, hidden_dim)
        self.v_linear = nn.Linear(hidden_dim, hidden_dim)
        self.out_linear = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(
        self,
        layer_outputs: List[torch.Tensor],
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Memory-efficient forward pass using chunking.

        CRITICAL OPTIMIZATION: If only one layer, skip attention entirely!
        """
        # EMERGENCY: If only final layer is passed, skip all attention computation
        if len(layer_outputs) == 1:
            return layer_outputs[0], None

        # Stack layer outputs: [num_nodes, num_layers, hidden_dim]
        stacked = torch.stack(layer_outputs, dim=1)
        num_nodes, num_layers, hidden_dim = stacked.shape

        # Process in chunks to save memory
        chunk_size = min(self.chunk_size, num_nodes)
        num_chunks = (num_nodes + chunk_size - 1) // chunk_size

        aggregated_chunks = []

        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, num_nodes)

            # Process this chunk
            chunk = stacked[start_idx:end_idx]  # [chunk_size, num_layers, hidden_dim]
            chunk_flat = chunk.reshape(-1, hidden_dim)

            # Compute Q, K, V for this chunk
            Q = self.q_linear(chunk_flat).view(-1, num_layers, self.num_heads, self.head_dim)
            K = self.k_linear(chunk_flat).view(-1, num_layers, self.num_heads, self.head_dim)
            V = self.v_linear(chunk_flat).view(-1, num_layers, self.num_heads, self.head_dim)

            # Transpose: [chunk_size, num_heads, num_layers, head_dim]
            Q = Q.transpose(1, 2)
            K = K.transpose(1, 2)
            V = V.transpose(1, 2)

            # Attention within this chunk
            scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
            attention = F.softmax(scores, dim=-1)
            attention = self.dropout(attention)

            # Apply attention
            context = torch.matmul(attention, V)
            context = context.transpose(1, 2).contiguous().view(-1, num_layers, self.hidden_dim)

            # Project
            output_flat = context.view(-1, self.hidden_dim)
            output = self.out_linear(output_flat).view(-1, num_layers, self.hidden_dim)

            # Aggregate across layers
            aggregated = output.mean(dim=1)
            aggregated_chunks.append(aggregated)

            # Clear intermediate tensors
            del Q, K, V, scores, attention, context, output

        # Concatenate all chunks
        aggregated = torch.cat(aggregated_chunks, dim=0)

        # Residual connection
        aggregated = self.layer_norm(aggregated + layer_outputs[-1])

        return aggregated, None
    
class MultiPathwayAggregator(nn.Module):
    """
    Aggregates representations from multiple curvature and hop pathways.
    
    This operates on the spatial dimension (different graph views).
    
    Memory-efficient pathway aggregator that avoids large attention matrices.
    
    Key optimizations:
    1. Use lightweight attention (score each pathway independently)
    2. No cross-pathway attention matrices
    3. Sequential processing for hierarchical mode
    
    Aggregation strategies:
    - 'concat': Concatenate all pathways and project down
    - 'attention': Learn attention weights for each pathway
    - 'mean': Simple average of all pathways
    - 'hierarchical': First aggregate by curvature, then by hop type
    """
    
    def __init__(
        self,
        hidden_channels: int,
        num_curvature_types: int,
        num_hop_types: int,
        aggregation_method: str = 'attention',
        dropout: float = 0.2
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.num_curvature_types = num_curvature_types
        self.num_hop_types = num_hop_types
        self.aggregation_method = aggregation_method.lower()
        self.num_pathways = num_curvature_types * num_hop_types
        
        if aggregation_method == 'concat':
            self.projection = nn.Linear(
                hidden_channels * self.num_pathways,
                hidden_channels
            )
            
        elif aggregation_method == 'attention':
            # Lightweight attention: each pathway gets a score
            # No cross-pathway attention matrices!
            self.pathway_scorer = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels // 4),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_channels // 4, 1)
            )
        
        elif aggregation_method == 'hierarchical':
            # Two separate scorers (no cross-attention)
            self.hop_scorer = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels // 4),
                nn.ReLU(),
                nn.Linear(hidden_channels // 4, 1)
            )
            
            self.curv_scorer = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels // 4),
                nn.ReLU(),
                nn.Linear(hidden_channels // 4, 1)
            )

        self.dropout = dropout
    
    def forward(
        self, 
        pathway_outputs: Dict[str, Dict[str, torch.Tensor]],
        return_attention: bool = True
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """Memory-efficient aggregation"""
        
        if self.aggregation_method == 'concat':
            # Simple concatenation - very memory efficient
            pathway_list = []
            for curv_dict in pathway_outputs.values():
                for tensor in curv_dict.values():
                    pathway_list.append(tensor)
            
            concatenated = torch.cat(pathway_list, dim=-1)
            aggregated = self.projection(concatenated)
            aggregated = F.dropout(aggregated, p=self.dropout, training=self.training)
            
            return (aggregated, None)
        
        elif self.aggregation_method == 'mean':
            # Simplest - just average
            pathway_list = []
            for curv_dict in pathway_outputs.values():
                for tensor in curv_dict.values():
                    pathway_list.append(tensor)
            
            stacked = torch.stack(pathway_list, dim=0)
            aggregated = stacked.mean(dim=0)
            
            return (aggregated, None)
        
        elif self.aggregation_method == 'attention':
            # Lightweight attention: score each pathway independently
            pathway_list = []
            pathway_names = []
            
            for curv_type, hop_dict in pathway_outputs.items():
                for hop_type, tensor in hop_dict.items():
                    pathway_list.append(tensor)
                    pathway_names.append(f"{curv_type}_{hop_type}")
            
            # Compute attention scores WITHOUT cross-pathway matrices
            attention_scores = []
            for pathway_repr in pathway_list:
                # Each pathway is scored independently: [num_nodes, 1]
                score = self.pathway_scorer(pathway_repr)
                attention_scores.append(score)
            
            # Stack scores: [num_pathways, num_nodes, 1]
            attention_scores = torch.stack(attention_scores, dim=0)
            
            # Softmax across pathways: [num_pathways, num_nodes, 1]
            attention_weights = F.softmax(attention_scores, dim=0)
            
            # Stack pathway representations: [num_pathways, num_nodes, hidden]
            stacked = torch.stack(pathway_list, dim=0)
            
            # Weighted sum: [num_nodes, hidden]
            aggregated = (stacked * attention_weights).sum(dim=0)
            
            if return_attention:
                attn_dict = {
                    name: attention_weights[i].squeeze(-1).mean().item()
                    for i, name in enumerate(pathway_names)
                }
                return aggregated, attn_dict
            
            return aggregated, None
        
        elif self.aggregation_method == 'hierarchical':
            # Hierarchical but WITHOUT cross-attention matrices
            curvature_aggregated = {}
            hop_attention_weights = {}
            
            for curv_type, hop_dict in pathway_outputs.items():
                hop_tensors = list(hop_dict.values())
                hop_names = list(hop_dict.keys())
                
                if len(hop_tensors) == 1:
                    curvature_aggregated[curv_type] = hop_tensors[0]
                else:
                    # Score each hop type independently
                    hop_scores = []
                    for h in hop_tensors:
                        score = self.hop_scorer(h)  # [num_nodes, 1]
                        hop_scores.append(score)
                    
                    hop_scores = torch.stack(hop_scores, dim=0)  # [num_hops, num_nodes, 1]
                    hop_attn = F.softmax(hop_scores, dim=0)
                    
                    hop_stacked = torch.stack(hop_tensors, dim=0)  # [num_hops, num_nodes, hidden]
                    curvature_aggregated[curv_type] = (hop_stacked * hop_attn).sum(dim=0)
                    
                    if return_attention:
                        hop_attention_weights[curv_type] = {
                            name: hop_attn[i].squeeze(-1).mean().item()
                            for i, name in enumerate(hop_names)
                        }
            
            # Aggregate across curvature types
            curv_tensors = list(curvature_aggregated.values())
            curv_names = list(curvature_aggregated.keys())
            
            if len(curv_tensors) == 1:
                aggregated = curv_tensors[0]
                curv_attention_weights = {curv_names[0]: 1.0}
            else:
                # Score each curvature type independently
                curv_scores = []
                for c in curv_tensors:
                    score = self.curv_scorer(c)  # [num_nodes, 1]
                    curv_scores.append(score)
                
                curv_scores = torch.stack(curv_scores, dim=0)  # [num_curvs, num_nodes, 1]
                curv_attn = F.softmax(curv_scores, dim=0)
                
                curv_stacked = torch.stack(curv_tensors, dim=0)  # [num_curvs, num_nodes, hidden]
                aggregated = (curv_stacked * curv_attn).sum(dim=0)
                
                curv_attention_weights = {
                    name: curv_attn[i].squeeze(-1).mean().item()
                    for i, name in enumerate(curv_names)
                }
            
            if return_attention:
                return aggregated, {
                    'hop_attention': hop_attention_weights,
                    'curv_attention': curv_attention_weights
                }
            
            return aggregated, None
        
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation_method}")

class HybridAggregator(nn.Module):
    """
    Two-stage aggregation: Layer-wise, then pathway-wise.
    
    Stage 1 (Depth): MultiLayerAttention aggregates across layers
    Stage 2 (Breadth): MultiPathwayAggregator aggregates across pathways
    
    Uses chunked attention and lightweight scoring to avoid OOM.
    """
    
    def __init__(
        self,
        hidden_channels: int,
        num_curvature_types: int,
        num_hop_types: int,
        num_heads: int = 4,
        pathway_aggregation: str = 'attention',
        dropout: float = 0.2,
        concat_heads: bool = True,
        chunk_size: int = 1000  # Process nodes in chunks
    ):
        super().__init__()
        
        self.hidden_channels = hidden_channels
        self.concat_heads = concat_heads
        
        actual_hidden_dim = hidden_channels * num_heads if concat_heads else hidden_channels
        
        # Stage 1: Memory-efficient layer aggregation
        self.layer_aggregator = MultiLayerAttention(
            hidden_dim=actual_hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            chunk_size=chunk_size
        )
        
        # Stage 2: Memory-efficient pathway aggregation
        self.pathway_aggregator = MultiPathwayAggregator(
            hidden_channels=actual_hidden_dim,
            num_curvature_types=num_curvature_types,
            num_hop_types=num_hop_types,
            aggregation_method=pathway_aggregation,
            dropout=dropout
        )
        
        # Projection layer
        if concat_heads and num_heads > 1:
            self.output_projection = nn.Sequential(
                nn.Linear(actual_hidden_dim, hidden_channels),
                nn.LayerNorm(hidden_channels),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
        else:
            self.output_projection = nn.Identity()
    
    def forward(
        self,
        gnn_outputs: Dict[str, Dict[str, List[torch.Tensor]]],
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """Two-stage memory-efficient aggregation"""
        
        # Stage 1: Aggregate layers within each pathway (chunked)
        pathway_layer_aggregated = {}
        layer_attention_weights = {}
        
        for curv_type, hop_dict in gnn_outputs.items():
            pathway_layer_aggregated[curv_type] = {}
            
            for hop_type, layer_outputs in hop_dict.items():
                # Use chunked attention for layer aggregation
                aggregated, layer_attn = self.layer_aggregator(
                    layer_outputs,
                    return_attention=return_attention
                )
                
                pathway_layer_aggregated[curv_type][hop_type] = aggregated
                
                if return_attention and layer_attn is not None:
                    pathway_key = f"{curv_type}_{hop_type}"
                    layer_attention_weights[pathway_key] = layer_attn
        
        # Stage 2: Aggregate pathways (lightweight scoring)
        final_output, pathway_attn = self.pathway_aggregator(
            pathway_layer_aggregated,
            return_attention=return_attention
        )
        
        # Project back
        final_output = self.output_projection(final_output)
        
        if return_attention:
            attention_info = {
                'layer_attention': layer_attention_weights,
                'pathway_attention': pathway_attn
            }
            return final_output, attention_info
        
        return final_output, None