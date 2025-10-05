import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional, Dict
from utils.logging_manager import get_logger
import numpy as np

from model.support_models import CurvatureAwareGNN, ProjectionHead, BinaryClassifier
from model.multi_layer_attention import MultiLayerAttention

class ContrastiveDriverGenePredictor(nn.Module):
    """
    Model for cancer driver prediction with potential driver identification:
    - Binary classification: driver (1) vs non-driver (0)
    - Post-hoc identification of potential drivers from false positives
      based on curvature features, confidence, and node properties
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        projection_dim: int = 128,
        num_gnn_layers: int = 3,
        curvature_types: List[str] = ['positive', 'negative', 'both'],
        num_attention_heads: int = 4,
        temperature: float = 0.4,
        dropout: float = 0.2
    ):
        super().__init__()
        self.temperature = temperature
        self.curvature_types = curvature_types
        self.hidden_channels = hidden_channels
        
        # Encoder: CurvatureAwareGNN
        self.encoder = CurvatureAwareGNN(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=num_gnn_layers,
            curvature_types=curvature_types,
            use_attention=True,
            dropout=dropout
        )
        
        # Multi-Layer Attention for each curvature type
        self.layer_attentions = nn.ModuleDict({
            curv_type: MultiLayerAttention(
                hidden_dim=hidden_channels,
                num_heads=num_attention_heads,
                dropout=dropout
            ) for curv_type in self.curvature_types
        })
        
        # Cross-curvature attention to aggregate different curvature views
        self.cross_curvature_attention = MultiLayerAttention(
            hidden_dim=hidden_channels,
            num_heads=num_attention_heads,
            dropout=dropout
        )
        
        # Projection head for contrastive learning
        self.projection = ProjectionHead(
            input_dim=in_channels,
            hidden_dim=hidden_channels,
            out_dim=projection_dim
        )
        
        # Binary Classifier
        self.classifier = BinaryClassifier(
            input_dim=hidden_channels,
            hidden_dim=hidden_channels,
            dropout=dropout
        )
        
    def encode(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_curvature: torch.Tensor,
        return_attention: bool = True
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Encode graph into representation vector
        """
        # Get layer outputs for each curvature type
        curvature_outputs = self.encoder(x, edge_index, edge_curvature)
        
        attention_weights = {} if return_attention else None
        
        curvature_representations = []
        for curv_type in self.curvature_types:
            layer_outputs = curvature_outputs[curv_type]
            aggregated, attn = self.layer_attentions[curv_type](
                layer_outputs,
                return_attention = return_attention
            )
            curvature_representations.append(aggregated)
            
            if return_attention:
                attention_weights[f'{curv_type}_layer_attention'] = attn
        
        # Aggregate across curvature types            
        final_repr, cross_attn = self.cross_curvature_attention(
            curvature_representations,
            return_attention
        )
        
        if return_attention:
            attention_weights['cross_curvature_attention'] = cross_attn
            # Store individual curvature representations for analysis
            attention_weights['curvature_representations'] = {
                curv_type: rep for curv_type, rep in 
                zip(self.curvature_types, curvature_representations)
            }
    
        return final_repr, attention_weights
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_curvature: torch.Tensor,
        return_embeddings: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for binary classification
        
        Returns:
            logits: [num_nodes] binary logits
            embeddings: [num_nodes, hidden_channels] if return_embeddings=True
        """
        h, _ = self.encode(x, edge_index, edge_curvature)
        logits = self.classifier(h)
        
        if return_embeddings:
            return logits, h
        
        return logits, None
    
    def get_contrastive_projection(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_curvature: torch.Tensor
    ) -> torch.Tensor:
        """
        Get normalized projection for contrastive loss
        """
        h, _ = self.encode(x, edge_index, edge_curvature)
        z = self.projection(h)
        return F.normalize(z, dim=-1)
    
    def compute_contrastive_loss(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        node_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute NT-Xent (InfoNCE) contrastive loss between two views
        """
        if node_mask is not None:
            z1 = z1[node_mask]
            z2 = z2[node_mask]
        
        batch_size = z1.shape[0]
        z = torch.cat([z1, z2], dim = 0)
        sim_matrix = torch.mm(z, z.T) / self.temperature
        
        mask = torch.eye(batch_size, device=z.device)
        mask = mask.repeat(2, 2)
        mask = 1 - mask
        
        labels = torch.cat([
            torch.arange(batch_size, 2*batch_size, device=z.device),
            torch.arange(batch_size, device=z.device)
        ])
        
        sim_matrix = sim_matrix * mask
        exp_sim = torch.exp(sim_matrix)
        log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True))
        pos_sim = log_prob[torch.arange(2*batch_size, device=z.device)]
        loss = -pos_sim.mean()
        
        return loss
    
    def compute_classification_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        pos_weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute weighted binary cross-entropy loss
        
        Args:
            logits: [num_nodes] binary logits
            labels: [num_nodes] with values {0, 1}
            mask: [num_nodes] boolean mask for labeled nodes
            pos_weight: Weight for positive class (driver genes)
        """
        
        if mask is not None:
            logits = logits[mask]
            labels = labels[mask].float()
            
        if pos_weight is not None:
            loss = F.binary_cross_entropy_with_logits(
                logits, labels, pos_weight
            )
        else:
            loss = F.binary_cross_entropy_with_logits(
                logits, labels
            )
            
        return loss
    
    def train_step(
        self,
        original_data: Dict,
        augmented_data: Dict,
        labels: torch.Tensor,
        train_mask: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        contrastive_weight: float = 0.3,
        pos_weight: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        Combined training step with contrastive and classification objectives
        """
        
        