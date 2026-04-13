#!/usr/bin/env python

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
        self.hidden_dim = hidden_dim    # Store total embedding size used throughout
        self.num_heads = num_heads      # Number of attention heads to split hidden_dim across
        self.head_dim = hidden_dim // num_heads # Dimension per head after splitting
        self.chunk_size = chunk_size    # Max nodes processed at once to cap peak memory
        
        assert hidden_dim % num_heads == 0  # Ensure hidden_dim splits evenly across heads, otherwise head_dim would be fractional
        
        self.q_linear = nn.Linear(hidden_dim, hidden_dim)   # Projects node-layer features to query vectors
        self.k_linear = nn.Linear(hidden_dim, hidden_dim)   # Projects node-layer features to key vectors
        self.v_linear = nn.Linear(hidden_dim, hidden_dim)   # Projects node-layer feature to value vectors
        self.out_linear = nn.Linear(hidden_dim, hidden_dim) # Mixes multi-head context back into hidden_dim
        
        self.dropout = nn.Dropout(dropout)                  # Dropout applied to attention weights to prevent over-reliance on specific layers
        self.layer_norm = nn.LayerNorm(hidden_dim)          # Normalizes aggregated output; used in residual connection at the end
        
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
            return layer_outputs[0], None   # Nothing to aggregate across; return the single-layer as is

        # Stack layer outputs: [num_nodes, num_layers, hidden_dim]
        stacked = torch.stack(layer_outputs, dim=1)     # Stack list of [num_nodes, hidden_dim] tensors along a new layer dimension
        num_nodes, num_layers, hidden_dim = stacked.shape   # Unpack dimensions for use in reshaping below

        # Process in chunks to save memory
        chunk_size = min(self.chunk_size, num_nodes)    # Don't use a chunk larger than the total number of nodes
        num_chunks = (num_nodes + chunk_size - 1) // chunk_size # Ceiling division: number of chunks needed to cover all nodes

        aggregated_chunks = []      # Collects the aggregated output for each chunk

        for i in range(num_chunks):
            start_idx = i * chunk_size      # First node index for this chunk
            end_idx = min((i + 1) * chunk_size, num_nodes)  # Last node index (Clamped to avoid out-of-bounds)

            # Process this chunk
            chunk = stacked[start_idx:end_idx]  # slice [chunk_size, num_layers, hidden_dim] from stacked
            chunk_flat = chunk.reshape(-1, hidden_dim)  # Flatten nodes x layers into a single batch dim -> [chunk_size*num_layers, hidden_dim]

            # Compute Q, K, V for this chunk
            Q = self.q_linear(chunk_flat).view(-1, num_layers, self.num_heads, self.head_dim)   # Project then reshape to [chunk_size, num_layers, heads, head_dim]
            K = self.k_linear(chunk_flat).view(-1, num_layers, self.num_heads, self.head_dim)   # Same for keys
            V = self.v_linear(chunk_flat).view(-1, num_layers, self.num_heads, self.head_dim)   # Same for values

            # Transpose: [chunk_size, num_heads, num_layers, head_dim]
            Q = Q.transpose(1, 2)   # Move heads before layers so matmul operates over the layer dim
            K = K.transpose(1, 2)   # Same for keys
            V = V.transpose(1, 2)   # Same for values

            # Attention within this chunk
            scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # Scaled dot-product: how much each layer attends to every other layer -> [chunk_size, heads, num_layers, num_layers]
            attention = F.softmax(scores, dim=-1)   # Normalize scores across the key (layer) attention -> attention distribution per query layer
            attention = self.dropout(attention)     # Randomly zero some attention entries to regularize layer importance

            # Apply attention
            context = torch.matmul(attention, V)    # Weighted sum of value vectors -> [chunk_size, heads, num_layers, head_dim]
            context = context.transpose(1, 2).contiguous().view(-1, num_layers, self.hidden_dim)    # Move heads back, merge head_dim -> [chunk_size, num_layers, hidden_dim]

            # Project
            output_flat = context.view(-1, self.hidden_dim) # Flatten to [chunk_size * num_layers, hidden_dim] for linear layer
            output = self.out_linear(output_flat).view(-1, num_layers, self.hidden_dim) # Project and restore shape -> [chunk_size, num_layers, hidden_dim]

            # Aggregate across layers
            aggregated = output.mean(dim=1) # Average over the layer dim -> [chunk_size, hidden_dim]; simple mean is cheaper
            aggregated_chunks.append(aggregated)    # Store this chunk's result

            # Clear intermediate tensors
            del Q, K, V, scores, attention, context, output # Free GPU memory before the next chunk

        # Concatenate all chunks
        aggregated = torch.cat(aggregated_chunks, dim=0)    # Reassemble full [num_nodes, hidden_dim] tensor from chunks

        # Residual connection
        aggregated = self.layer_norm(aggregated + layer_outputs[-1])    # Add last layer's output as residual to preserve fine-grained info, then normalize

        return aggregated, None # Return aggregated node embeddings; attention weights are not returned to save memory
    
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
        self.hidden_channels = hidden_channels  # Embedding size shared across all pathways
        self.num_curvature_types = num_curvature_types  # 3 for ['positive', 'negative', 'both']
        self.num_hop_types = num_hop_types              # 2 for ['one_hop', 'two_hop']
        self.aggregation_method = aggregation_method.lower()    # Normalize to lowercase to avoid mismatch
        self.num_pathways = num_curvature_types * num_hop_types # Total number of (curvature, hop) pathway combinations
        
        if aggregation_method == 'concat':
            self.projection = nn.Linear(
                hidden_channels * self.num_pathways,        # Input is all pathways concatenated along feature dim
                hidden_channels                             # Project back down to hidden_channels
            )
            
        elif aggregation_method == 'attention':
            # Lightweight attention: each pathway gets a score
            # No cross-pathway attention matrices!
            self.pathway_scorer = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels // 4),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_channels // 4, 1)      # Output a single scalar importance score per node per pathway
            )
        
        elif aggregation_method == 'hierarchical':
            # Two separate scorers (no cross-attention)
            self.hop_scorer = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels // 4),       # scores each hop_type within a curvature group
                nn.ReLU(),
                nn.Linear(hidden_channels // 4, 1)                      # Scalar score per node per hop type
            )
            
            self.curv_scorer = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels // 4),       # Scores each curvature after hop aggregation
                nn.ReLU(),
                nn.Linear(hidden_channels // 4, 1)                      # Scalar score per node per curvature type
            )

        self.dropout = dropout      # Stored as float for use with F.dropout calls in forward
    
    def forward(
        self, 
        pathway_outputs: Dict[str, Dict[str, torch.Tensor]],
        return_attention: bool = True
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """Memory-efficient aggregation"""
        
        if self.aggregation_method == 'concat':
            # Simple concatenation - very memory efficient
            pathway_list = []
            for curv_dict in pathway_outputs.values():      # Iterate over curvature types
                for tensor in curv_dict.values():           # Iterate over hop_types within each curvature type
                    pathway_list.append(tensor)             # Collect all pathway tensors [num_nodes, hidden_channels]
            
            concatenated = torch.cat(pathway_list, dim=-1)  # Concatenate along feature dim -> [num_nodes, hidden_channels * num_pathways] 
            aggregated = self.projection(concatenated)      # Linear projection back to hidden_channels -> [num_nodes, hidden_channels*num_pathways]
            aggregated = F.dropout(aggregated, p=self.dropout, training=self.training)  # Regularize after projection
            
            return (aggregated, None)       # No attention weights to return in concat mode
        
        elif self.aggregation_method == 'mean':
            # Simplest - just average
            pathway_list = []
            for curv_dict in pathway_outputs.values():  # Iterate over curvature types
                for tensor in curv_dict.values():       # Iterate over hop_types
                    pathway_list.append(tensor)         # Collect all pathway tensors
            
            stacked = torch.stack(pathway_list, dim=0)  # Stack into [num_pathways, num_nodes, hidden_channels]
            aggregated = stacked.mean(dim=0)            # Average across pathways -> [num_nodes, hidden_channels]
            
            return (aggregated, None)                   # No attention weights in mean mode
        
        elif self.aggregation_method == 'attention':
            # Lightweight attention: score each pathway independently
            pathway_list = []
            pathway_names = []
            
            for curv_type, hop_dict in pathway_outputs.items():
                for hop_type, tensor in hop_dict.items():
                    pathway_list.append(tensor)                         # Collect pathway tensors
                    pathway_names.append(f"{curv_type}_{hop_type}")     # Record pathway name for attention dict keys
            
            # Compute attention scores WITHOUT cross-pathway matrices
            attention_scores = []
            for pathway_repr in pathway_list:
                # Each pathway is scored independently: [num_nodes, 1]
                score = self.pathway_scorer(pathway_repr)               # MLP produces per-node importance scalar for this pathway
                attention_scores.append(score)
            
            # Stack scores: [num_pathways, num_nodes, 1]
            attention_scores = torch.stack(attention_scores, dim=0)     # Stack scalar scores into [num_pathways, num_nodes, 1]
            
            # Softmax across pathways: [num_pathways, num_nodes, 1]
            attention_weights = F.softmax(attention_scores, dim=0)      # Normalize so all pathway scores sum upto 1 per node
            
            # Stack pathway representations: [num_pathways, num_nodes, hidden]
            stacked = torch.stack(pathway_list, dim=0)                  # Stack all pathway embeddings for weighted sum
            
            # Weighted sum: [num_nodes, hidden]
            aggregated = (stacked * attention_weights).sum(dim=0)       # Each node's embedding is a convex combination of pathway embeddings
            
            if return_attention:
                attn_dict = {
                    name: attention_weights[i].squeeze(-1).mean().item()# Mean attention weight across nodes for this pathway; useful for diagnostics
                    for i, name in enumerate(pathway_names)
                }
                return aggregated, attn_dict                            # Return aggregated embeddings and per-pathway mean attention
            
            return aggregated, None                                     # Return aggregated embeddings without attention info
        
        elif self.aggregation_method == 'hierarchical':
            # Hierarchical but WITHOUT cross-attention matrices
            curvature_aggregated = {}                                   # Stores hop-aggregated embedding per curvature type
            hop_attention_weights = {}                                  # Stores per-hop attention weights per curvature type (for diagnostics)
            
            for curv_type, hop_dict in pathway_outputs.items():
                hop_tensors = list(hop_dict.values())                   # List [num_nodes, hidden] tensors, one per hop_type
                hop_names = list(hop_dict.keys())                       # Corresponding hop type names
                
                if len(hop_tensors) == 1:
                    curvature_aggregated[curv_type] = hop_tensors[0]    # Only one hop type; no aggregation needed
                else:
                    # Score each hop type independently
                    hop_scores = []
                    for h in hop_tensors:
                        score = self.hop_scorer(h)                      # [num_nodes, 1]; MLP scores this hop type's importance per node
                        hop_scores.append(score)
                    
                    hop_scores = torch.stack(hop_scores, dim=0)         # [num_hops, num_nodes, 1]
                    hop_attn = F.softmax(hop_scores, dim=0)             # normalize across hop types per node
                    
                    hop_stacked = torch.stack(hop_tensors, dim=0)       # [num_hops, num_nodes, hidden]
                    curvature_aggregated[curv_type] = (hop_stacked * hop_attn).sum(dim=0) # Weighted sum over hop types -> [num_nodes, hidden]
                    
                    if return_attention:
                        hop_attention_weights[curv_type] = {
                            name: hop_attn[i].squeeze(-1).mean().item() # Mean hop attention weight across nodes
                            for i, name in enumerate(hop_names)
                        }
            
            # Aggregate across curvature types
            curv_tensors = list(curvature_aggregated.values())          # List of hop-aggregated tensors, one per curvature type
            curv_names = list(curvature_aggregated.keys())              # Corresponding curvature type names
            
            if len(curv_tensors) == 1:
                aggregated = curv_tensors[0]                            # Only 1 curvature type; no aggregation needed
                curv_attention_weights = {curv_names[0]: 1.0}           # trivial weight of 1.0
            else:
                # Score each curvature type independently
                curv_scores = []
                for c in curv_tensors:
                    score = self.curv_scorer(c)  # [num_nodes, 1]' MLP scores this curvature type's importance per node
                    curv_scores.append(score)
                
                curv_scores = torch.stack(curv_scores, dim=0)  # [num_curvs, num_nodes, 1]
                curv_attn = F.softmax(curv_scores, dim=0)      # Normalize across curvature types per node
                
                curv_stacked = torch.stack(curv_tensors, dim=0)  # [num_curvs, num_nodes, hidden]
                aggregated = (curv_stacked * curv_attn).sum(dim=0)  # Weighted sum over curvature type -> [num_nodes, hidden]
                
                curv_attention_weights = {
                    name: curv_attn[i].squeeze(-1).mean().item()    # Mean curvature attention weight across nodes
                    for i, name in enumerate(curv_names)
                }
            
            if return_attention:
                return aggregated, {
                    'hop_attention': hop_attention_weights,         # Per-curvature type hop weights
                    'curv_attention': curv_attention_weights        # Per-curvature type weights
                }
            
            return aggregated, None                                 # Return final aggregated embedding without attention info
        
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation_method}")  # Guard against typos or unsupported methods

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
        
        self.hidden_channels = hidden_channels      # base embedding dimension before head expansion
        self.concat_heads = concat_heads            # If true, multi-head outputs are concatenated (expands dim by num_heads); if False, averaged
        
        actual_hidden_dim = hidden_channels * num_heads if concat_heads else hidden_channels    # Actual feature dim flowing through the aggregator depending on how the GNN heads were combined
        
        # Stage 1: Memory-efficient layer aggregation
        self.layer_aggregator = MultiLayerAttention(
            hidden_dim=actual_hidden_dim,   # Must match the dim of tensors coming out of CurvatureAwareGNN
            num_heads=num_heads,
            dropout=dropout,
            chunk_size=chunk_size           # Controls peak memory per forward pass
        )
        
        # Stage 2: Memory-efficient pathway aggregation
        self.pathway_aggregator = MultiPathwayAggregator(
            hidden_channels=actual_hidden_dim,      # Input dim matches layer aggregator output
            num_curvature_types=num_curvature_types,
            num_hop_types=num_hop_types,
            aggregation_method=pathway_aggregation,     # one of 'concat', 'attention', 'mean', 'hierarchical'
            dropout=dropout
        )
        
        # Projection layer
        if concat_heads and num_heads > 1:
            self.output_projection = nn.Sequential(
                nn.Linear(actual_hidden_dim, hidden_channels),      # Project concatenated multi-head dim back down to hidden channels
                nn.LayerNorm(hidden_channels),                      # Normalize before activation
                nn.ReLU(),
                nn.Dropout(dropout)
            )
        else:
            self.output_projection = nn.Identity()                  # no projection needed if heads were averaged or there's only 1 head
    
    def forward(
        self,
        gnn_outputs: Dict[str, Dict[str, List[torch.Tensor]]],
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """Two-stage memory-efficient aggregation"""
        
        # Stage 1: Aggregate layers within each pathway (chunked)
        pathway_layer_aggregated = {}       # Stores layer-aggregated embedding per (curvature_type, hop_type)
        layer_attention_weights = {}        # Store layer attention per pathway (Only populated if return attention = True)
        
        for curv_type, hop_dict in gnn_outputs.items():
            pathway_layer_aggregated[curv_type] = {}            # Initialize inner dict for this curvature type
            
            for hop_type, layer_outputs in hop_dict.items():
                # Use chunked attention for layer aggregation
                aggregated, layer_attn = self.layer_aggregator(
                    layer_outputs,                              # List of per-layer [num_nodes, hidden] tensors
                    return_attention=return_attention           # Pass through caller's preference
                )
                
                pathway_layer_aggregated[curv_type][hop_type] = aggregated      # Store the single aggregated embedding for this pathway
                
                if return_attention and layer_attn is not None:
                    pathway_key = f"{curv_type}_{hop_type}"
                    layer_attention_weights[pathway_key] = layer_attn           # Record which layers were most important for this pathway
        
        # Stage 2: Aggregate pathways (lightweight scoring)
        final_output, pathway_attn = self.pathway_aggregator(
            pathway_layer_aggregated,                   # Nested dict of layer-aggregated embeddings per pathway
            return_attention=return_attention           # Pass through caller's preference
        )
        
        # Project back
        final_output = self.output_projection(final_output)     # Compress concatenated multi-head dim back to hidden_channels (identity if heads were averaged)
        
        if return_attention:
            attention_info = {
                'layer_attention': layer_attention_weights,     # Which GNN layers were most informative per pathway
                'pathway_attention': pathway_attn               # Which (curvature, hop) pathways were most informative
            }
            return final_output, attention_info                 # Return final node embeddings and full attention breakdown
        
        return final_output, None                               # Return final node embeddings without attention info