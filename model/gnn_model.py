from torch_geometric.utils import add_self_loops, degree
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
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
        aggregation: str = 'add',
        device: str = 'cpu'
    ):
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm
        self.device = device
        
        if layer_configs is None:
            # Strategy from paper: positive curvature first, then negative
            layer_configs = [
                {'curvature_type': 'positive', 'hop_type': 'one_hop', 'use_attention': False},
                {'curvature_type': 'negative', 'hop_type': 'two_hop', 'use_attention': True}
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


# ============================================================================
# ANALYSIS AND TESTING
# ============================================================================

def analyze_curvature_filtering(edge_curvature):
    """Analyze how edges are distributed by curvature"""
    positive = (edge_curvature > 0).sum().item()
    negative = (edge_curvature < 0).sum().item()
    zero = (edge_curvature == 0).sum().item()
    total = len(edge_curvature)
    
    print("\n" + "="*60)
    print("CURVATURE DISTRIBUTION ANALYSIS")
    print("="*60)
    print(f"Total edges: {total}")
    print(f"Positive curvature: {positive} ({100*positive/total:.1f}%)")
    print(f"Negative curvature: {negative} ({100*negative/total:.1f}%)")
    print(f"Zero curvature: {zero} ({100*zero/total:.1f}%)")
    print(f"Curvature range: [{edge_curvature.min():.3f}, {edge_curvature.max():.3f}]")
    print(f"Mean curvature: {edge_curvature.mean():.3f}")
    print(f"Std curvature: {edge_curvature.std():.3f}")


def test_model_configurations(device):
    """Test different model configurations and compare results"""
    print("\n" + "="*60)
    print("TESTING DIFFERENT MODEL CONFIGURATIONS")
    print("="*60)
    
    # Setup
    num_nodes = 100
    num_edges = 500
    in_channels = 32
    hidden_channels = 64
    out_channels = 10
    
    # Create data
    torch.manual_seed(123)
    x = torch.randn(num_nodes, in_channels).to(device)
    edge_index = torch.randint(0, num_nodes, (2, num_edges)).to(device)
    edge_curvature = torch.randn(num_edges).to(device)
    
    analyze_curvature_filtering(edge_curvature)
    
    configs_to_test = [
        {
            'name': 'Positive Only (1-hop)',
            'configs': [
                {'curvature_type': 'positive', 'hop_type': 'one_hop', 'use_attention': False},
                {'curvature_type': 'positive', 'hop_type': 'one_hop', 'use_attention': False}
            ]
        },
        {
            'name': 'Negative Only (1-hop)',
            'configs': [
                {'curvature_type': 'negative', 'hop_type': 'one_hop', 'use_attention': False},
                {'curvature_type': 'negative', 'hop_type': 'one_hop', 'use_attention': False}
            ]
        },
        {
            'name': 'Mixed (Pos→Neg)',
            'configs': [
                {'curvature_type': 'positive', 'hop_type': 'one_hop', 'use_attention': False},
                {'curvature_type': 'negative', 'hop_type': 'two_hop', 'use_attention': True}
            ]
        },
        {
            'name': 'Both Curvatures',
            'configs': [
                {'curvature_type': 'both', 'hop_type': 'one_hop', 'use_attention': False},
                {'curvature_type': 'both', 'hop_type': 'one_hop', 'use_attention': False}
            ]
        }
    ]
    
    results = []
    
    for config_dict in configs_to_test:
        print(f"\n{'-'*60}")
        print(f"Testing: {config_dict['name']}")
        print(f"{'-'*60}")
        
        model = CurvatureConstrainedGNN(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=2,
            layer_configs=config_dict['configs'],
            dropout=0.5,
            device=device
        ).to(device)
        
        # Forward pass
        model.eval()
        with torch.no_grad():
            out = model(x, edge_index, edge_curvature, edge_attr=None)
        
        # Analyze output
        print(f"Output shape: {out.shape}")
        print(f"Output range: [{out.min():.3f}, {out.max():.3f}]")
        print(f"Output mean: {out.mean():.3f}")
        print(f"Output std: {out.std():.3f}")
        print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        results.append({
            'name': config_dict['name'],
            'output': out.cpu(),
            'mean': out.mean().item(),
            'std': out.std().item(),
            'params': sum(p.numel() for p in model.parameters())
        })
    
    return results


def compare_hop_types(device):
    """Compare 1-hop vs 2-hop propagation"""
    print("\n" + "="*60)
    print("COMPARING 1-HOP vs 2-HOP PROPAGATION")
    print("="*60)
    
    num_nodes = 50
    num_edges = 200
    in_channels = 16
    hidden_channels = 32
    out_channels = 8
    
    x = torch.randn(num_nodes, in_channels).to(device)
    edge_index = torch.randint(0, num_nodes, (2, num_edges)).to(device)
    edge_curvature = torch.randn(num_edges).to(device)
    
    for hop_type in ['one_hop', 'two_hop']:
        print(f"\n{hop_type.upper()}:")
        
        configs = [
            {'curvature_type': 'positive', 'hop_type': hop_type, 'use_attention': False},
            {'curvature_type': 'negative', 'hop_type': hop_type, 'use_attention': False}
        ]
        
        model = CurvatureConstrainedGNN(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=2,
            layer_configs=configs,
            dropout=0.0,
            device=device
        ).to(device)
        
        model.eval()
        with torch.no_grad():
            out = model(x, edge_index, edge_curvature, edge_attr = None)
        
        print(f"  Output mean: {out.mean():.4f}")
        print(f"  Output std: {out.std():.4f}")
        print(f"  Output norm: {out.norm():.4f}")
    
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("="*60)
    print(f"CURVATURE-CONSTRAINED GNN ANALYSIS")
    print("="*60)
    print(f"Device: {device}")
    print(f"PyTorch version: {torch.__version__}")
    
    # Run tests
    results = test_model_configurations(device)
    compare_hop_types(device)
    
    print("\n" + "="*60)
    print("SUMMARY OF RESULTS")
    print("="*60)
    for r in results:
        print(f"{r['name']:25} | Mean: {r['mean']:7.3f} | Std: {r['std']:6.3f} | Params: {r['params']:,}")
    
    print("\n✓ All tests completed successfully!")
    print("="*60)