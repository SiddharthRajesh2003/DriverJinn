import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.logging_manager import get_logger

logger = get_logger(__name__)

class ProjectionHead(nn.Module):
    """
    Projection head for contrastive learning
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
    Binary classifier for known driver vs non-driver genes
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
    """Early stopping to prevent overfitting"""
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
    """Learning rate warmup scheduler"""
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