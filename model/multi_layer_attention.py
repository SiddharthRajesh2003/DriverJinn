import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional
from utils.logging_manager import get_logger

logger = get_logger(__name__)

class MultiLayerAttention(nn.Module):
    """
    Multi-layer attention mechanism for aggregating representations
    across different GNN layers and curvature-based views.
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
        self.head_dim = hidden_dim //num_heads
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        self.q_linear = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.k_linear = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.v_linear = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.out_linear = nn.Linear(self.hidden_dim, self.hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(self.hidden_dim)
        
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
        scores = torch.matmul(Q, K.transpose(-2, -1)) / ((self.head_dim)**0.5)
        attention = F.softmax(scores, dim=1)
        attention = self.dropout(attention)
        
        # Apply attention to values
        context = torch.matmul(attention, V)
        
        # Reshape and project: [batch_size, num_layers, hidden_dim]
        context = context.transpose(1, 2).contiguous().view(batch_size, num_layers, self.hidden_dim)
        output = self.out_linear(context)
        
        # Aggregate across layers (mean pooling)
        aggregated = output.mean(dim=1)
        aggregated = self.layer_norm(aggregated + layer_outputs[-1]) # Residual connection
        
        if return_attention:
            # Average attention weights across heads: [batch_size, num_layers]
            avg_attention = attention.mean(dim=1).mean(dim=-1)
            return aggregated, avg_attention
        
        return aggregated, None