import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.logging_manager import get_logger

logger = get_logger(__name__)

class ProjectionHead(nn.Module):
    """
    A projection head for contrastive learning that transforms input representations
    into a space where contrastive learning is performed.
    
    The projection head consists of multiple layers of linear transformations,
    each followed by batch normalization and ReLU activation (except the last layer).
    This architecture helps in learning better representations for contrastive learning.
    
    Args:
        input_dim (int): Dimension of input features
        hidden_dim (int): Dimension of hidden layers
        out_dim (int): Dimension of output projection
        num_layers (int, optional): Number of layers in projection head. Defaults to 2
        
    Architecture:
        - Multiple Linear layers with dimensions: input_dim -> hidden_dim -> ... -> out_dim
        - BatchNorm1d after each layer (except the last)
        - ReLU activation after each layer (except the last)
        
    Example:
        >>> proj = ProjectionHead(
        ...     input_dim=256,
        ...     hidden_dim=128,
        ...     out_dim=64,
        ...     num_layers=2
        ... )
        >>> x = torch.randn(32, 256)  # batch_size=32, features=256
        >>> out = proj(x)  # Shape: (32, 64)
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
    A binary classifier for predicting driver genes in cancer genomics.
    This classifier takes graph-based gene representations and predicts
    whether a gene is a cancer driver or non-driver.

    The model uses a multi-layer architecture with batch normalization
    and dropout for robust prediction of driver genes. It processes
    gene features learned from multiple biological networks (GGNet,
    PathNet, and PPNet) to make final predictions.

    Args:
        input_dim (int): Dimension of input features from the GNN
        hidden_dim (int, optional): Dimension of hidden layers. Defaults to 256
        dropout (float, optional): Dropout rate for regularization. Defaults to 0.3

    Architecture:
        Layer 1:
            - Linear: input_dim → hidden_dim
            - BatchNorm1d
            - ReLU
            - Dropout(0.3)
        Layer 2:
            - Linear: hidden_dim → hidden_dim//2
            - BatchNorm1d
            - ReLU
            - Dropout(0.3)
        Output:
            - Linear: hidden_dim//2 → 1 (binary classification)

    Returns:
        torch.Tensor: Logits for binary classification
            - Positive values suggest driver genes
            - Negative values suggest non-driver genes

    Example:
        >>> classifier = BinaryClassifier(
        ...     input_dim=256,  # GNN output dimension
        ...     hidden_dim=128,
        ...     dropout=0.3
        ... )
        >>> gene_features = torch.randn(100, 256)  # 100 genes
        >>> predictions = classifier(gene_features)  # Shape: (100,)
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

class EarlyStopping:
    """
    Early stopping mechanism to prevent overfitting in the driver gene prediction model.
    Monitors a metric (e.g., validation performance) and stops training when no improvement
    is seen for a specified number of epochs.

    This is particularly important for GNN training where overfitting can lead to
    poor generalization on unseen genes and cancer types.

    Args:
        patience (int, optional): Number of epochs to wait for improvement 
            before stopping. Defaults to 50.
        min_delta (float, optional): Minimum change in monitored quantity to 
            qualify as an improvement. Defaults to 0.0001.
        mode (str, optional): One of {'min', 'max'}. In 'min' mode, training 
            stops when the quantity monitored stops decreasing; in 'max' mode it 
            will stop when the quantity monitored stops increasing. Defaults to 'max'.

    Attributes:
        counter (int): Number of epochs without improvement
        best_score (float): Best score observed
        early_stop (bool): True if early stopping criteria is met

    Example:
        >>> early_stopping = EarlyStopping(patience=20, mode='max')
        >>> for epoch in range(num_epochs):
        ...     val_auc = validate_model()
        ...     if early_stopping(val_auc):
        ...         print("Early stopping triggered")
        ...         break
    """
    def __init__(self, patience=50, min_delta=0.0001, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'max':
            if score > self.best_score + self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
        else:
            if score < self.best_score - self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
        
        if self.counter >= self.patience:
            self.early_stop = True
            return True
        
        return False


class WarmupScheduler:
    """
    Learning rate warmup scheduler for the driver gene prediction model.
    Gradually increases the learning rate from an initial value to a target value
    over a specified number of epochs.

    Warmup is particularly beneficial for GNN training as it:
    1. Helps stabilize early training with complex graph structures
    2. Allows the model to learn better node representations before fine-tuning
    3. Reduces the risk of early training instability with curvature-aware mechanisms

    Args:
        optimizer (torch.optim.Optimizer): The optimizer whose learning rate will be scheduled
        warmup_epochs (int): Number of epochs over which to increase learning rate
        initial_lr (float): Starting learning rate
        target_lr (float): Final learning rate after warmup

    Attributes:
        current_epoch (int): Current epoch number
        optimizer (torch.optim.Optimizer): The optimizer being scheduled

    Example:
        >>> optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        >>> scheduler = WarmupScheduler(
        ...     optimizer,
        ...     warmup_epochs=10,
        ...     initial_lr=1e-6,
        ...     target_lr=1e-3
        ... )
        >>> for epoch in range(num_epochs):
        ...     train_epoch()
        ...     scheduler.step()
        ...     current_lr = scheduler.get_lr()
    """
    def __init__(self, optimizer, warmup_epochs, initial_lr, target_lr):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.initial_lr = initial_lr
        self.target_lr = target_lr
        self.current_epoch = 0
    
    def step(self):
        if self.current_epoch < self.warmup_epochs:
            lr = self.initial_lr + (self.target_lr - self.initial_lr) * \
                (self.current_epoch / self.warmup_epochs)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        self.current_epoch += 1
    
    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']