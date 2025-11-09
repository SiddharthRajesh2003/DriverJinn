import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from typing import Optional, Tuple
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
        attention_mode: 'standard', 'edge_feature', 'bias', 'gated', 'hybrid'
        dropout: float, Control the proportion of neurons being dropped to avoid overfitting
        min_edge_ratio: Minimum fraction of edges to keep when filtering
        heads: Number of attention heads
        concat: Concatenate or average heads
        negative_slope: For LeakyReLU in attention
    """
    
    def __init__(
        self, 
        in_channels: int,
        out_channels: int,
        curvature_type: str = 'positive',
        hop_type: str = 'one_hop',
        aggregation: str = 'add',
        use_attention: bool = True,
        attention_mode: str = 'bias',
        dropout: float = 0.0,
        min_edge_ratio: float = 0.1,
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2
    ):
        super().__init__(aggr=aggregation)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.curvature_type = curvature_type.lower()
        self.hop_type = hop_type.lower()
        self.use_attention = use_attention
        self.attention_mode = attention_mode.lower()
        self.dropout = dropout
        self.min_edge_ratio = min_edge_ratio
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        
        # Output dimension after multi-head attention
        if use_attention and concat:
            self.output_channels = out_channels * heads
        else:
            self.output_channels = out_channels
        
        # Transformation Weights
        if use_attention:
            # Transform to multi-head space
            self.lin = nn.Linear(in_channels, out_channels * heads, bias=False)
        else:
            self.lin = nn.Linear(in_channels, out_channels)
        
        # Attention mechanism - different for each mode
        if use_attention:
            self._setup_attention_layers()
        
        self.reset_parameters()

    def _setup_attention_layers(self):
        """Setup attention layers based on attention_mode."""
        if self.attention_mode == 'standard':
            # Original GAT: [h_i || h_j]
            self.att = nn.Parameter(torch.Tensor(1, self.heads, 2 * self.out_channels))
        
        elif self.attention_mode == 'edge_feature':
            # Curvature concatenated: [h_i || h_j || c_ij]
            self.att = nn.Parameter(torch.Tensor(1, self.heads, 2 * self.out_channels + 1))
            self.curvature_scale = nn.Parameter(torch.ones(1))
        
        elif self.attention_mode == 'bias':
            # Curvature as additive bias
            self.att = nn.Parameter(torch.Tensor(1, self.heads, 2 * self.out_channels))
            self.curvature_bias_weight = nn.Parameter(torch.ones(1, self.heads) * 0.1)
        
        elif self.attention_mode == 'gated':
            # Gated mixing of node and curvature attention
            self.att_node = nn.Parameter(torch.Tensor(1, self.heads, 2 * self.out_channels))
            self.att_curv = nn.Parameter(torch.Tensor(1, self.heads, 1))
            
            # Gate network
            self.gate_net = nn.Sequential(
                nn.Linear(2 * self.out_channels + 1, self.heads),
                nn.Sigmoid()
            )
        
        elif self.attention_mode == 'hybrid':
            # Hybrid: Combines edge_feature + bias approaches
            self.att_feature = nn.Parameter(torch.Tensor(1, self.heads, 2 * self.out_channels + 1))
            self.curvature_scale = nn.Parameter(torch.ones(1))
            
            self.att_bias = nn.Parameter(torch.Tensor(1, self.heads, 2 * self.out_channels))
            self.curvature_bias_weight = nn.Parameter(torch.ones(1, self.heads) * 0.1)
            
            # Learnable mixing weights
            self.hybrid_weight = nn.Parameter(torch.tensor([0.5, 0.5]))
        
        else:
            raise ValueError(
                f"Unknown attention_mode: {self.attention_mode}. "
                f"Choose from: 'standard', 'edge_feature', 'bias', 'gated', 'hybrid'"
            )

    def reset_parameters(self):
        self.lin.reset_parameters()
        
        if self.use_attention:
            if hasattr(self, 'att'):
                nn.init.xavier_uniform_(self.att)
            if hasattr(self, 'att_node'):
                nn.init.xavier_uniform_(self.att_node)
                nn.init.xavier_uniform_(self.att_curv)
            if hasattr(self, 'att_feature'):
                nn.init.xavier_uniform_(self.att_feature)
                nn.init.xavier_uniform_(self.att_bias)
            if hasattr(self, 'gate_net'):
                for layer in self.gate_net:
                    if hasattr(layer, 'weight'):
                        nn.init.xavier_uniform_(layer.weight)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_curvature: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with curvature-constrained message passing
        
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Graph connectivity [2, num_edges]
            edge_curvature: Curvature values for each edge [num_edges]
            edge_attr: Optional edge attributes [num_edges, edge_dim]
            return_attention: Whether to return attention weights
        
        Returns:
            out: Updated node features [num_nodes, output_channels]
            attention_weights: Optional attention weights if return_attention=True
        """
        
        # Validate inputs
        num_nodes = x.shape[0]
        max_node_idx = edge_index.max().item() if edge_index.numel() > 0 else -1
        
        if max_node_idx >= num_nodes:
            raise ValueError(f"Edge index contains node {max_node_idx} but only {num_nodes} nodes in features")
        
        # Store for message passing
        self.edge_curvature = edge_curvature
        self.return_attention = return_attention
        
        # Transform node features - KEEP IN 2D for propagate()
        h = self.lin(x)  # [num_nodes, heads * out_channels]
        h = F.dropout(h, p=self.dropout, training=self.training)
        
        # Filter edges if using standard attention mode
        if self.attention_mode == 'standard' and self.curvature_type != 'both':
            logger.info(
                f"Using legacy filtering mode for '{self.attention_mode}' attention. "
                f"Consider using curvature-aware attention modes for better performance."
            )
            filtered_edge_index, filtered_edge_curvature = self.filter_edges_by_curvature(
                edge_index, edge_curvature, edge_attr
            )
            
            if filtered_edge_index.shape[1] == 0:
                logger.warning(
                    f"No edges remain after filtering for curvature_type='{self.curvature_type}'"
                )
                if self.use_attention and self.concat:
                    return h, None
                else:
                    return h.view(num_nodes, self.heads, self.out_channels).mean(dim=1), None
        else:
            filtered_edge_index = edge_index
            filtered_edge_curvature = edge_curvature
        
        # Update stored curvature for message passing
        self.edge_curvature = filtered_edge_curvature
        
        if self.hop_type == 'two_hop':
            out = self.two_hop_propagation(h, filtered_edge_index, num_nodes)
        else:
            # Standard message passing - pass 2D tensor
            out = self.propagate(
                filtered_edge_index, 
                x=h,  # Pass as 2D: [num_nodes, heads * out_channels]
                size=(num_nodes, num_nodes)
            )
        
        # Reshape output from multi-head
        if self.use_attention:
            if self.concat:
                # Already in correct shape: [num_nodes, heads * out_channels]
                pass
            else:
                # Average over heads
                out = out.view(num_nodes, self.heads, self.out_channels).mean(dim=1)
        
        # Return attention weights if requested
        attention_weights = None
        if return_attention and hasattr(self, '_alpha'):
            attention_weights = self._alpha
            delattr(self, '_alpha')
        
        return out, attention_weights
    
    def filter_edges_by_curvature(
        self,
        edge_index: torch.Tensor,
        edge_curvature: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        DEPRECATED: Legacy edge filtering by curvature.
        
        Filter edges based on curvature type (positive/negative)
        """
        logger.warning(
            "filter_edges_by_curvature is DEPRECATED. "
            "Use curvature-aware attention modes instead."
        )
        
        num_edges_index = edge_index.shape[1]
        num_curv = edge_curvature.shape[0]
        
        if num_curv == 0 or num_edges_index == 0:
            return edge_index, edge_curvature
        
        # Initialize base_mask
        if self.curvature_type == 'both':
            base_mask = torch.ones_like(edge_curvature, dtype=torch.bool)
        else:
            if self.curvature_type == 'positive':
                threshold = torch.median(edge_curvature)
                base_mask = edge_curvature >= threshold
            elif self.curvature_type == 'negative':
                threshold = torch.median(edge_curvature)
                base_mask = edge_curvature <= threshold
            else:
                raise ValueError(f'Invalid Curvature type: {self.curvature_type}')
            
            # Ensure minimum edges
            if base_mask.sum() < int(num_curv * self.min_edge_ratio):
                k = max(1, int(num_curv * self.min_edge_ratio))
                if self.curvature_type == 'positive':
                    _, indices = torch.topk(edge_curvature, k)
                else:
                    _, indices = torch.topk(edge_curvature, k, largest=False)
                base_mask = torch.zeros_like(edge_curvature, dtype=torch.bool)
                base_mask[indices] = True

        # Handle undirected graphs
        if num_curv == num_edges_index:
            mask = base_mask
        elif num_curv * 2 == num_edges_index:
            mask = torch.repeat_interleave(base_mask, 2)
        else:
            raise IndexError(
                f"Edge curvature length ({num_curv}) does not match edge_index ({num_edges_index})"
            )

        filtered_edge_index = edge_index[:, mask]
        filtered_edge_curvature = edge_curvature[mask] if num_curv == num_edges_index else edge_curvature

        return filtered_edge_index, filtered_edge_curvature
    
    def two_hop_propagation(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        num_nodes: int
    ) -> torch.Tensor:
        """Implement two-hop message passing"""
        
        # First hop
        out_1hop = self.propagate(
            edge_index, 
            x=x,
            size=(num_nodes, num_nodes)
        )
        
        # Second hop
        out_2hop = self.propagate(
            edge_index, 
            x=out_1hop,
            size=(num_nodes, num_nodes)
        )
        
        # Combine with residual
        return x + 0.5 * out_1hop + 0.5 * out_2hop
    
    def message(
        self,
        x_j: torch.Tensor,
        x_i: Optional[torch.Tensor] = None,
        index: Optional[torch.Tensor] = None,
        size_i: Optional[int] = None
    ) -> torch.Tensor:
        """
        Construct messages with curvature-aware attention.
        
        Args:
            x_j: Neighbor features [num_edges, heads * out_channels]
            x_i: Center features [num_edges, heads * out_channels]
            index: Edge indices for center nodes
            size_i: Number of center nodes
        """
        
        if not self.use_attention:
            return x_j
        
        if x_i is None:
            x_i = x_j
        
        # Reshape to multi-head format
        # [num_edges, heads * out_channels] -> [num_edges, heads, out_channels]
        num_edges = x_j.shape[0]
        x_i = x_i.view(num_edges, self.heads, self.out_channels)
        x_j = x_j.view(num_edges, self.heads, self.out_channels)
        
        # Get edge curvature
        edge_curvature = self.edge_curvature
        
        # Compute attention based on mode
        if self.attention_mode == 'standard':
            alpha = self.compute_standard_attention(x_i, x_j, index, size_i)
        elif self.attention_mode == 'edge_feature':
            alpha = self.compute_edge_feature_attention(x_i, x_j, edge_curvature, index, size_i)
        elif self.attention_mode == 'bias':
            alpha = self.compute_bias_attention(x_i, x_j, edge_curvature, index, size_i)
        elif self.attention_mode == 'gated':
            alpha = self.compute_gated_attention(x_i, x_j, edge_curvature, index, size_i)
        elif self.attention_mode == 'hybrid':
            alpha = self.compute_hybrid_attention(x_i, x_j, edge_curvature, index, size_i)
        
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        
        if self.return_attention:
            self._alpha = alpha
        
        # Apply attention: [num_edges, heads, out_channels] * [num_edges, heads, 1]
        out = x_j * alpha.unsqueeze(-1)
        
        # Flatten back to 2D for aggregation
        out = out.view(num_edges, self.heads * self.out_channels)
        
        return out
    
    def compute_standard_attention(
        self,
        x_i: torch.Tensor,
        x_j: torch.Tensor,
        index: Optional[torch.Tensor] = None,
        size_i: Optional[int] = None
    ) -> torch.Tensor:
        """Standard GAT attention (node features only)."""
        # [h_i || h_j]
        alpha_input = torch.cat([x_i, x_j], dim=-1)
        alpha = (alpha_input * self.att).sum(dim=-1)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = self.softmax_per_node(alpha, index, size_i)
        return alpha
    
    def softmax_per_node(
        self,
        alpha: torch.Tensor,
        index: torch.Tensor,
        size_i: Optional[int]
    ) -> torch.Tensor:
        """Apply softmax normalization per destination node."""
        if size_i is None:
            size_i = int(index.max()) + 1
        
        alpha_list = []
        
        for head in range(self.heads):
            alpha_head = alpha[:, head]
            
            # Numerical stability
            alpha_max = torch.zeros(size_i, device=alpha.device)
            alpha_max.scatter_reduce_(0, index, alpha_head, reduce='amax', include_self=False)
            alpha_head = alpha_head - alpha_max[index]
            
            # Exp and normalize
            alpha_head = alpha_head.exp()
            alpha_sum = torch.zeros(size_i, device=alpha.device)
            alpha_sum.scatter_add_(0, index, alpha_head)
            alpha_head = alpha_head / (alpha_sum[index] + 1e-16)
            
            alpha_list.append(alpha_head)
        
        return torch.stack(alpha_list, dim=1)

    def compute_edge_feature_attention(
        self,
        x_i: torch.Tensor,
        x_j: torch.Tensor,
        edge_curvature: torch.Tensor,
        index: torch.Tensor,
        size_i: Optional[int]
    ) -> torch.Tensor:
        """Attention with curvature as edge feature."""
        curv_expanded = (edge_curvature * self.curvature_scale).unsqueeze(-1).unsqueeze(-1)
        curv_expanded = curv_expanded.expand(-1, self.heads, -1)
        
        alpha_input = torch.cat([x_i, x_j, curv_expanded], dim=-1)
        alpha = (alpha_input * self.att).sum(dim=-1)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = self.softmax_per_node(alpha, index, size_i)
        return alpha
    
    def compute_bias_attention(
        self,
        x_i: torch.Tensor,
        x_j: torch.Tensor,
        edge_curvature: torch.Tensor,
        index: torch.Tensor,
        size_i: int 
    ) -> torch.Tensor:
        """Attention with curvature as additive bias."""
        alpha_input = torch.cat([x_i, x_j], dim=-1)
        alpha = (alpha_input * self.att).sum(dim=-1)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        
        # Add curvature bias
        curvature_bias = edge_curvature.unsqueeze(-1) * self.curvature_bias_weight
        alpha = alpha + curvature_bias
        
        alpha = self.softmax_per_node(alpha, index, size_i)
        return alpha
    
    def compute_gated_attention(
        self,
        x_i: torch.Tensor,
        x_j: torch.Tensor,
        edge_curvature: torch.Tensor,
        index: torch.Tensor,
        size_i: int
    ) -> torch.Tensor:
        """Gated attention: learned mixing."""
        # Node attention
        alpha_node_input = torch.cat([x_i, x_j], dim=-1)
        alpha_node = (alpha_node_input * self.att_node).sum(dim=-1)
        alpha_node = F.leaky_relu(alpha_node, self.negative_slope)
        
        # Curvature attention
        curv_expanded = edge_curvature.unsqueeze(-1).unsqueeze(-1).expand(-1, self.heads, -1)
        alpha_curv = (curv_expanded * self.att_curv).squeeze(-1)
        alpha_curv = F.leaky_relu(alpha_curv, self.negative_slope)
        
        # Gate
        gate_input = torch.cat([
            x_i.mean(dim=1),
            x_j.mean(dim=1),
            edge_curvature.unsqueeze(-1)
        ], dim=-1)
        gate = self.gate_net(gate_input)
        
        # Mix
        alpha = gate * alpha_node + (1 - gate) * alpha_curv
        alpha = self.softmax_per_node(alpha, index, size_i)
        return alpha
    
    def compute_hybrid_attention(
        self,
        x_i: torch.Tensor,
        x_j: torch.Tensor,
        edge_curvature: torch.Tensor, 
        index: torch.Tensor,
        size_i: int
    ) -> torch.Tensor:
        """Hybrid: Combines edge_feature and bias."""
        # Edge feature component
        curv_expanded = (edge_curvature * self.curvature_scale).unsqueeze(-1).unsqueeze(-1)
        curv_expanded = curv_expanded.expand(-1, self.heads, -1)
        
        alpha_feature_input = torch.cat([x_i, x_j, curv_expanded], dim=-1)
        alpha_feature = (alpha_feature_input * self.att_feature).sum(dim=-1)
        alpha_feature = F.leaky_relu(alpha_feature, self.negative_slope)
        
        # Bias component
        alpha_bias_input = torch.cat([x_i, x_j], dim=-1)
        alpha_bias = (alpha_bias_input * self.att_bias).sum(dim=-1)
        alpha_bias = F.leaky_relu(alpha_bias, self.negative_slope)
        
        curvature_bias = edge_curvature.unsqueeze(-1) * self.curvature_bias_weight
        alpha_bias = alpha_bias + curvature_bias
        
        # Combine
        weights = F.softmax(self.hybrid_weight, dim=0)
        alpha = weights[0] * alpha_feature + weights[1] * alpha_bias
        
        alpha = self.softmax_per_node(alpha, index, size_i)
        return alpha
