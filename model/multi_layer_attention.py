import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional, Dict
from utils.logging_manager import get_logger

logger = get_logger(__name__)

class MultiLayerAttention(nn.Module):
    """
    Aggregates representations across GNN layers (depth dimension).
    
    Uses multi-head attention to learn which layers are most informative.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
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
        Args:
            layer_outputs: List of [num_nodes, hidden_dim] tensors from different layers
            return_attention: Whether to return attention weights
        
        Returns:
            aggregated_output: [num_nodes, hidden_dim]
            attention_weights: [num_nodes, num_layers] if return_attention=True
        """
        
        # Stack layer outputs: [num_nodes, num_layers, hidden_dim]
        stacked = torch.stack(layer_outputs, dim=1)
        batch_size, num_layers, _ = stacked.shape
        
        # Compute queries, keys, values
        Q = self.q_linear(stacked).view(batch_size, num_layers, self.num_heads, self.head_dim)
        K = self.k_linear(stacked).view(batch_size, num_layers, self.num_heads, self.head_dim)
        V = self.v_linear(stacked).view(batch_size, num_layers, self.num_heads, self.head_dim)
        
        # Transpose for attention: [batch_size, num_heads, num_layers, head_dim]
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        
        # Apply attention to values
        context = torch.matmul(attention, V)
        
        # Reshape and project: [batch_size, num_layers, hidden_dim]
        context = context.transpose(1, 2).contiguous().view(batch_size, num_layers, self.hidden_dim)
        output = self.out_linear(context)
        
        # Aggregate across layers (mean pooling)
        aggregated = output.mean(dim=1)
        aggregated = self.layer_norm(aggregated + layer_outputs[-1])  # Residual from last layer
        
        if return_attention:
            # Average attention weights across heads: [batch_size, num_layers]
            avg_attention = attention.mean(dim=1).mean(dim=-1)
            return aggregated, avg_attention
        
        return aggregated, None
    
class MultiPathwayAggregator(nn.Module):
    """
    Aggregates representations from multiple curvature and hop pathways.
    
    This operates on the spatial dimension (different graph views).
    
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
            # Project concatenated pathways back to hidden_channels
            self.projection = nn.Linear(
                hidden_channels * self.num_pathways,
                hidden_channels
            )
            
        elif aggregation_method == 'attention':
            # Learn attention weights for each pathway
            self.pathway_attention = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_channels // 2, 1)
            )
        
        elif aggregation_method == 'hierarchical':
            # First aggregate by hop_type and then by curvature type
            
            self.hop_attn = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels//2),
                nn.ReLU(),
                nn.Linear(hidden_channels//2, 1)
            )
            
            self.curv_attn = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels // 2),
                nn.ReLU(),
                nn.Linear(hidden_channels // 2, 1)
            )
            

        self.dropout = dropout
    
    def forward(
        self, 
        pathway_outputs: Dict[str, Dict[str, torch.Tensor]],
        return_attention: bool = True
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Aggregate outputs from multiple pathways.
        
        Args:
            pathway_outputs: Nested dict from layer aggregation
                {curvature_type: {hop_type: tensor}}
                Each tensor is [num_nodes, hidden_channels]
            return_attention: Whether to return attention weights
        
        Returns:
            aggregated: [num_nodes, hidden_channels] aggregated representation
            attention_weights: Optional dict of attention weights
        """
        
        # Flatten pathway outputs into a list
        pathway_list = []
        pathway_names = []
        
        for curv_type, hop_dict in pathway_outputs.items():
            for hop_type, tensor in hop_dict.items():
                pathway_list.append(tensor)
                pathway_names.append(f"{curv_type}_{hop_type}")
                

        if self.aggregation_method == 'concat':
            # Concatenate all pathways
            concatenated = torch.cat(pathway_list, dim = -1)
            aggregated = self.projection(concatenated)
            aggregated = F.dropout(aggregated, p = self.dropout, training= self.training)
            
            return (aggregated, None) if not return_attention else (aggregated, {})
        
        elif self.aggregation_method == 'mean':
            # Simple average
            stacked = torch.stack(pathway_list, dim=0)  # [num_pathways, num_nodes, hidden]
            aggregated = stacked.mean(dim=0)
            
            return (aggregated, None) if not return_attention else (aggregated, {})
        
        elif self.aggregation_method == 'attention':
            # Attention-weighted aggregation
            stacked = torch.stack(pathway_list, dim = 0)  # [num_pathways, num_nodes, hidden]
            
            # Compute attention scores for each pathway
            attention_scores = []
            for pathway_repr in pathway_list:
                score = self.pathway_attention(pathway_repr) # [num_nodes, 1]
                attention_scores.append(score)
                
            attention_scores = torch.stack(attention_scores, dim = 0)  # [num_pathways, num_nodes, 1]
            attention_weights = F.softmax(attention_scores, dim=0)  # Normalize across pathways
            
            # Weighted sum
            aggregated = (stacked * attention_weights).sum(dim=0)
            
            if return_attention:
                # Create attention weight dictionary
                attn_dict = {
                    name: attention_weights[i].squeeze(-1).mean().item()
                    for i, name in enumerate(pathway_names)
                }
                return aggregated, attn_dict
            else:
                return aggregated, None
        
        elif self.aggregation_method == 'hierarchical':
            # First aggregate by hop type within each curvature type
            curvature_aggregated = {}
            hop_attention_weights = {}
            
            for curv_type, hop_type in pathway_outputs.items():
                hop_tensors = list(hop_type.values())
                hop_names = list(hop_type.keys())
                
                if len(hop_tensors) == 1:
                    curvature_aggregated[curv_type] = hop_tensors[0]
                else:
                    # Attention over hop types
                    hop_stacked = torch.stack(hop_tensors, dim=0)
                    hop_scores = torch.stack([
                        self.hop_attn(h) for h in hop_tensors
                    ], dim = 0)
                    
                    hop_attn = F.softmax(hop_scores, dim=0)
                    
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
                curv_stacked = torch.stack(curv_tensors, dim=0)
                curv_scores = torch.stack([
                    self.curv_attn(c) for c in curv_tensors
                ], dim=0)
                curv_attn = F.softmax(curv_scores, dim=0)
                
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
            else:
                return aggregated, None
            
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation_method}")
    

class HybridAggregator(nn.Module):
    """
    Two-stage aggregation: Layer-wise, then pathway-wise.
    
    Stage 1 (Depth): MultiLayerAttention aggregates across layers
    Stage 2 (Breadth): MultiPathwayAggregator aggregates across pathways
    
    This is the RECOMMENDED approach for your architecture!
    """
    
    def __init__(
        self,
        hidden_channels: int,
        num_curvature_types: int,
        num_hop_types: int,
        num_heads: int = 4,
        pathway_aggregation: str = 'attention',
        dropout: float = 0.2
    ):
        super().__init__()
        
        # Stage 1: Aggregate layers within each pathway
        self.layer_aggregator = MultiLayerAttention(
            hidden_dim=hidden_channels,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Stage 2: Aggregate across pathways
        self.pathway_aggregator = MultiPathwayAggregator(
            hidden_channels=hidden_channels,
            num_curvature_types=num_curvature_types,
            num_hop_types=num_hop_types,
            aggregation_method=pathway_aggregation,
            dropout=dropout
        )
    
    def forward(
        self,
        gnn_outputs: Dict[str, Dict[str, List[torch.Tensor]]],
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Two-stage aggregation of GNN outputs.
        
        Args:
            gnn_outputs: From CurvatureAwareGNN.forward()
                {
                    'positive': {
                        'one_hop': [layer0, layer1, layer2],
                        'two_hop': [layer0, layer1, layer2]
                    },
                    'negative': {...},
                    'both': {...}
                }
            return_attention: Whether to return attention weights
        
        Returns:
            final_representation: [num_nodes, hidden_channels]
            attention_info: Dict with layer and pathway attention weights
        """
        
        # Stage 1: Aggregate layers within each pathway
        pathway_layer_aggregated = {}
        layer_attention_weights = {}
        
        for curv_type, hop_dict in gnn_outputs.items():
            pathway_layer_aggregated[curv_type] = {}
            
            for hop_type, layer_outputs in hop_dict.items():
                # Aggregate across layers
                aggregated, layer_attn = self.layer_aggregator(
                    layer_outputs,
                    return_attention=return_attention
                )
                
                pathway_layer_aggregated[curv_type][hop_type] = aggregated
                
                if return_attention and layer_attn is not None:
                    pathway_key = f"{curv_type}_{hop_type}"
                    layer_attention_weights[pathway_key] = layer_attn
        
        # Stage 2: Aggregate across pathways
        final_output, pathway_attn = self.pathway_aggregator(
            pathway_layer_aggregated,
            return_attention=return_attention
        )
        
        if return_attention:
            attention_info = {
                'layer_attention': layer_attention_weights,
                'pathway_attention': pathway_attn
            }
            return final_output, attention_info
        
        return final_output, None
