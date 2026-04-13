#!/usr/bin/env python

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
        
        layers = []                                                             # List to accumulate Linear/BN/ReLU modules before wrapping in Sequential
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]        # Build dimension schedule: [input_dim, hidden_dim, ..., out_dim]
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))                        # Add Linear layer mapping from current dim to next
            if i < len(dims) - 2: # No activation on last layer                 # Skip BN and ReLU after the final linear layer
                layers.append(nn.BatchNorm1d(dims[i+1]))                        # Normalize activations to stabilize contrastive learning
                layers.append(nn.ReLU())                                        # Non-linearity between layers
                
        self.projection = nn.Sequential(*layers)                                # Pack into a single callable module
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x)                                               # Pass node embeddings through the MLP stack; output lives in the contrastive projection space
    
class BinaryClassifier(nn.Module):
    """
    DEPRECATED: Converted the DriverGenePredictor to use ranking instead of classification.
    
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
        return self.classifier(x).squeeze(-1)           # Run through the MLP and remove the trailing size-1 dim -> [num_nodes] logits

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
        self.best_epoch = 0
        
    def __call__(self, score, epoch):
        if self.best_score is None:     # First call: no baseline yet
            self.best_score = score     # Set the initial best score
            self.best_epoch = epoch     # Record which epoch produced it
            return False                # Never stop on the very first call
        
        if self.mode == 'max':
            if score > self.best_score + self.min_delta:        # Improvement is large enough to count(must beat best by atleast min_delta)
                self.best_score = score # Update best score
                self.best_epoch = epoch # Update epoch of best score
                self.counter = 0        # Reset patience counter since we improved
            else:
                self.counter += 1       # NO meaningful improvement; increment stale epoch count
        else:
            if score < self.best_score - self.min_delta:    # in 'min' mode: improvement means score dropped by atleast min_delta
                self.best_score = score
                self.best_epoch = epoch
                self.counter = 0
            else:
                self.counter += 1       # No meaningful improvement in 'min' mode either
        
        if self.counter >= self.patience:   # Stale epochs have exhausted the patience budget
            self.early_stop = True
            return True                 # Signal Caller to stop training
        
        return False                    # Still within patience; continue training


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
        if self.current_epoch < self.warmup_epochs:     # Only adjust LR during the warmup window
            lr = self.initial_lr + (self.target_lr - self.initial_lr) * \
                (self.current_epoch / self.warmup_epochs)       # linear interpolation from initial_lr to target_lr over warmup_epochs
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr                          # Write the computed LR into every parameter group of the optimizer
        self.current_epoch += 1                                 # Advance epoch counter regardless of whether we're still in warmup
    
    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']             # Read current LR from the first parameter group (representative for the whole optimizer)


class RankingLoss(nn.Module):
    """
    Ranking loss that directly optimizes for driver genes to score higher.

    Improvements for stable training:
    - Hard negative mining to focus on informative pairs
    - Sampled pairwise to reduce memory and variance
    - Focal weighting to focus on hard examples
    """
    def __init__(
        self,
        margin: float = 1.0,
        loss_type: str = 'pairwise',
        num_samples: int = 256,
        focal_gamma: float = 2.0,
        use_focal: bool = True
    ):
        super().__init__()
        self.margin = margin
        self.loss_type = loss_type
        self.num_samples = num_samples
        self.focal_gamma = focal_gamma
        self.use_focal = use_focal

    def forward(
        self,
        scores: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor,
        progress: float = 0.0
    ) -> torch.Tensor:
        """
        Args:
            scores: [num_nodes] driver likelihood scores
            labels: [num_nodes] binary labels (1=driver, 0=non-driver)
            mask: [num_nodes] training mask
            progress: training progress in [0, 1] for curriculum scheduling
        """

        scores = scores[mask]       # Restrict to training nodes only; validation/test nodes must not influence the loss
        labels = labels[mask]       # Correspondingly mask labels

        driver_mask = (labels == 1)     # boolean mask selecting known driver genes in the training set
        non_driver_mask = (labels == 0) # Boolean mask selecting non-driver genes

        n_drivers = driver_mask.sum().item()                # Count of driver genes available for pairing
        n_non_drivers = non_driver_mask.sum().item()        # Count of non-driver genes available for pairing

        if n_drivers == 0 or n_non_drivers == 0:
            return torch.tensor(0.0, device=scores.device, requires_grad=True)  # Degenerate fold: Can't form any pairs, return a zero loss that still carries a gradient

        driver_scores = scores[driver_mask]             # Scores for driver genes only -> [n_driver]
        non_driver_scores = scores[non_driver_mask]     # Scores for non-driver genes only -> [n_non_drivers]

        if self.loss_type == 'pairwise':
            return self._pairwise_loss(driver_scores, non_driver_scores)

        elif self.loss_type == 'sampled_pairwise':
            return self._sampled_pairwise_loss(driver_scores, non_driver_scores)

        elif self.loss_type == 'listwise':
            return self._listwise_loss(scores, labels)

        elif self.loss_type == 'approxndcg':
            return self._approxndcg_loss(scores, labels)

        elif self.loss_type == 'bpr':
            # Bayesian Personalized Ranking - more stable than standard pairwise
            return self._bpr_loss(driver_scores, non_driver_scores, progress=progress)

        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")    # Guard against invalid loss_type strings

    def _pairwise_loss(
        self,
        driver_scores: torch.Tensor,
        non_driver_scores: torch.Tensor
    ) -> torch.Tensor:
        """Standard pairwise margin ranking loss with optional focal weighting."""
        driver_expanded = driver_scores.unsqueeze(1)  # reshape to [n_drivers, 1] for broadcasting over all non-drivers
        non_driver_expanded = non_driver_scores.unsqueeze(0)  # [1, n_non_drivers]  for broadcasting over all drivers

        # margin_loss[i,j] = max(0, margin - driver[i] + non_driver[j])
        margin_violations = self.margin - driver_expanded + non_driver_expanded # Positive when driver[i] fails to beat non-driver[j] by atleast margin -> [n_drivers, n_non_drivers]
        loss_matrix = F.relu(margin_violations)     # HingeL Only penalize violated pairs, zero out correctly ranked pairs

        if self.use_focal:
            # Focal weighting: focus on hard examples (where violation is large)
            # Probability of correct ranking
            prob_correct = torch.sigmoid(driver_expanded - non_driver_expanded)     # Probability that driver outranks non-driver; close to 0.5 means nearly tied(hard pair)
            focal_weight = (1 - prob_correct) ** self.focal_gamma                   # Hard pairs (low prob_correct) get weight near 1; easy pairs get weight near 0
            loss_matrix = focal_weight * loss_matrix                                # Re-weight loss matrix to focus gradient on hard pairs

        return loss_matrix.mean()                                                   # Average over all n_drivers x n_non_drivers pairs

    def _sampled_pairwise_loss(
        self,
        driver_scores: torch.Tensor,
        non_driver_scores: torch.Tensor
    ) -> torch.Tensor:
        """Sampled pairwise loss for memory efficiency and reduced variance."""
        n_drivers = len(driver_scores)
        n_non_drivers = len(non_driver_scores)

        # Sample pairs instead of computing all n_d * n_nd pairs
        n_pairs = min(self.num_samples, n_drivers * n_non_drivers)      # Cap at num_samples to avoid O(n^2) memory cost

        # Sample driver indices (with replacement if needed)
        driver_idx = torch.randint(0, n_drivers, (n_pairs,), device=driver_scores.device)       # Randomly sample driver indices
        non_driver_idx = torch.randint(0, n_non_drivers, (n_pairs,), device=driver_scores.device)   # Randomly sample non-driver indices

        sampled_driver = driver_scores[driver_idx]                  # Get sampled driver scores -> [n_pairs]
        sampled_non_driver = non_driver_scores[non_driver_idx]      # Gather sampled non-driver scores -> [n_pairs]

        # BPR-style loss: -log(sigmoid(driver - non_driver))
        diff = sampled_driver - sampled_non_driver                  # Positive diff means driver outranks non-driver (correct ordering)
        loss = F.softplus(-diff)  # log(1 + exp(-diff)) = -log(sigmoid(diff)); differentiable BPR loss

        if self.use_focal:
            prob_correct = torch.sigmoid(diff)                      # Probability of correct ranking for each sampled pair
            focal_weight = (1 - prob_correct) ** self.focal_gamma   # Down-weight easy-pairs, up-weight hard pairs
            loss = focal_weight * loss

        return loss.mean()      # Average over all sampled pairs

    def _bpr_loss(
        self,
        driver_scores: torch.Tensor,
        non_driver_scores: torch.Tensor,
        progress: float = 0.0
    ) -> torch.Tensor:
        """Bayesian Personalized Ranking loss with curriculum hard negative mining and focal weighting.
        hard_frac ramps from 0.25 → 0.75 over training so early epochs use
        easier negatives for stability and later epochs use harder ones for refinement.
        """
        n_drivers = len(driver_scores)
        n_non_drivers = len(non_driver_scores)

        n_pairs = min(self.num_samples, n_drivers * n_non_drivers)      # Total pairs to form this step

        # Curriculum: linearly ramp hard fraction from 25% → 75% over training
        hard_frac = 0.25 + 0.5 * progress           # At progress=0 (start): 25% hard; at progress=1 (end): 75% hard
        n_hard = int(n_pairs * hard_frac)           # Number of hard negative pairs this step
        n_random = n_pairs - n_hard                 # Remaining pairs are sampled randomly for stability

        # Hard negatives: globally top-scoring non-drivers (most informative gradient)
        if n_non_drivers > n_hard:
            _, top_indices = torch.topk(non_driver_scores, k=n_hard)        # Pick the n_hard non-drivers with highest scores (hardest to distinguish from drivers)
            hard_neg_idx = top_indices
        else:
            hard_neg_idx = torch.randint(0, n_non_drivers, (n_hard,), device=non_driver_scores.device)      # Fallback if fewer non-driver indices

        # Random negatives for stability
        random_neg_idx = torch.randint(0, n_non_drivers, (n_random,), device=non_driver_scores.device)      # Uniform random non-driver indices

        non_driver_idx = torch.cat([hard_neg_idx, random_neg_idx])                                          # Combine hard and random negatives into 1 index tensor
        driver_idx = torch.randint(0, n_drivers, (n_pairs,), device=driver_scores.device)                   # Randomly pair each negative with a driver

        diff = driver_scores[driver_idx] - non_driver_scores[non_driver_idx]                                # Score margin for each pair; positive means driver correctly outranks non-driver

        # BPR loss: -log(sigmoid(x_ui - x_uj))
        loss = -F.logsigmoid(diff)              # Log-sigmoid version of BPR; numerically stable, equals log(1 + exp(-diff))

        # Focal weighting: down-weight easy pairs, focus on hard ones
        if self.use_focal:
            prob_correct = torch.sigmoid(diff)          # Prob that driver outranks non-driver
            focal_weight = (1 - prob_correct) ** self.focal_gamma   # Near-zero weight for easy pairs, near-one-weight for hard pairs
            loss = focal_weight * loss

        return loss.mean()                          # Average BPR loss over all n_pairs

    def _listwise_loss(
        self,
        scores: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """ListNet-style listwise loss."""
        ideal_probs = labels.float()                                # Cast binary labels to float; drivers=1.0, non-drivers=0.0
        ideal_probs = ideal_probs / (ideal_probs.sum() + 1e-10)     # Normalize to a probability distribution over nodes; 1e-10 prevents by zero
        pred_probs = F.softmax(scores, dim=0)                       # Convert raw scores to predicted probability distribution over all nodes
        loss = -torch.sum(ideal_probs * torch.log(pred_probs + 1e-10))  # Cross-entropy between ideal and predicted distributions; minimizing this pushes driver scores to the stop
        return loss

    def _approxndcg_loss(
        self,
        scores: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """Approximate NDCG loss using differentiable soft ranking.

        Soft rank: rank[i] ≈ 1 + Σ_j sigmoid((s_j - s_i) / τ)
        As s_i increases, soft_rank[i] decreases (moves toward rank 1).
        Gradient flows through scores via the sigmoid pairwise differences.

        Stratified sampling: always include ALL driver genes + sample num_samples
        non-drivers. This ensures NDCG is optimized over the full driver set,
        matching how validation NDCG is computed globally.
        """
        driver_mask = (labels == 1)             # Boolean mask for driver genes
        nondriver_mask = ~driver_mask           # Boolean mask for non-driver genes

        driver_idx = driver_mask.nonzero(as_tuple=True)[0]          # Integer indices of all driver genes
        nondriver_pool = nondriver_mask.nonzero(as_tuple=True)[0]   # Integer indices of all non-driver genes

        # Sample non-drivers; always keep all drivers
        n_sample = min(self.num_samples, len(nondriver_pool))       # Cap non-driver sample to num_samples
        perm = torch.randperm(len(nondriver_pool), device=scores.device)[:n_sample] # Random permutation to sample without replacement
        nondriver_idx = nondriver_pool[perm]                        # Selecte non-driver indices

        idx = torch.cat([driver_idx, nondriver_idx])            # Combine all drivers with sampled non-drivers into one index set
        scores = scores[idx]                                    # Restrict scores to this subset
        labels = labels[idx]                                    # Restrict labels to this subset

        tau = 0.1  # sharper soft ranks → stronger gradient signal
        # [N, N]: s_j - s_i; sigmoid shrinks as s_i grows → lower soft rank
        pair_diff = scores.unsqueeze(0) - scores.unsqueeze(1)           # Pairwise score differences [N, N]; entry [j, i] = s_j - s_i
        soft_rank = torch.sigmoid(pair_diff / tau).sum(dim=0) + 1.0  # For node i: sum over j of sigmoid((s_j - s_i)/tau) = number of nodes ranked above i; +1 makes it 1-indexed

        discount = 1.0 / torch.log2(soft_rank + 1.0)        # DCG position discount; nodes ranked higher (smaller soft_rank) get larger discount
        gains = labels.float()                              # Relevance gains: 1.0 for drivers, 0.0 for non-drivers
        dcg = (gains * discount).sum()                      # Differentiable DCG: sum of discounted gains for driver genes

        # idcg is a constant normalizer — no gradient needed
        with torch.no_grad():
            ideal_ranks = torch.arange(1, len(gains) + 1, device=scores.device).float()     # Ideal ranks 1, 2, ..., N
            ideal_gains = torch.sort(gains, descending=True)[0]                             # Place all drivers at the top (ideal ordering)
            idcg = (ideal_gains / torch.log2(ideal_ranks + 1.0)).sum()                      # Ideal DCG; used to normalize DCG into [0, 1]

        loss = 1.0 - dcg / (idcg + 1e-10)               # 1-NDCG; minimizing this maximizes the differentiable NDCG approximation
        return loss

class EMA:
    """
    Exponential Moving Average for model weights.
    Maintains a shadow copy of model parameters that's updated with exponential decay.
    This smooths out training dynamics and often improves generalization.
    """
    def __init__(self, model: nn.Module, decay = 0.999, device: torch.device = None):
        self.model = model      # Reference to the live model whose parameters will be tracked
        self.decay = decay      # EMA smoothing factor; 0.999 means shadow moves very slowly towards current weights
        self.device = device if device is not None else torch.device('cpu')     # Store shadow params on CPU by default to save GPU memory
        
        # Create shadow parameters (stored on CPU to save GPU memory)
        self.shadow_params = {}         # Dict mapping parameter name -> shadow tensor
        self.register_params(model)     # Populate shadow_params with initial copies of all trainable parameters
        
    def register_params(
        self,
        model: nn.Module
    ):
        """Store initial model paramters"""
        for name, param in model.named_parameters():
            if param.requires_grad:                                             # Skip frozen parameters; only track trainable ones
                self.shadow_params[name] = param.data.clone().to(self.device)   # Store a CPU copy of the initial parameter values
    
    def update(
        self,
        model: nn.Module
    ):
        """Update shadow parameters with exponential moving average"""
        with torch.no_grad():                                                   # Shadow updates are not part of the computational graph; no gradients needed
            for name, param in model.named_parameters():
                if param.requires_grad and name in self.shadow_params:
                    # EMA update: shadow = decay * shadow + (1-decay) * current
                    self.shadow_params[name].mul_(self.decay).add_(             # in-place: shadow *= decay, then shadow += (1-decay) * current
                        param.data.to(self.device), alpha = 1.0 - self.decay    # Move current param to shadow device before blending
                    )
    
    def apply_shadow(
        self,
        model: nn.Module
    ):
        """Replace model parameters with EMA shadow parameters (for evaluation)"""
        
        self.backup_params = {}                                     # Temporary store for live weights so they can be restored after evaluation
        for name, params in model.named_parameters():
            if params.requires_grad and name in self.shadow_params:
                self.backup_params[name] = params.data.clone()              # Save a copy of the current live weights
                params.data.copy_(self.shadow_params[name].to(params.device))   # Overwrite live weights with the smoothed shadow weights for evaluation
                
    
    def restore(
        self,
        model: nn.Module
    ):
        """Restore original model parameters (after evaluation)"""
        for name, param in model.named_parameters():
            if name in self.backup_params:
                param.data.copy_(self.backup_params[name])      # Write the backed-up live weights back into the model
        
        self.backup_params = {}                                 # Clear the backup to free memory now that restoration is complete
    
    def state_dict(self):
        """Get EMA state for checkpointing"""
        return {
            'decay': self.decay,                                # Save the smoothing factor so it can be restored exactly
            'shadow_params': self.shadow_params                 # Save all shadow parameter tensors for checkpointing
        }
    
    def load_state_dict(
        self, state_dict
    ):
        """Load EMA state from checkpoint"""
        self.decay = state_dict['decay']                        # Restore the smoothing factor from the checkpoint
        self.shadow_params = state_dict['shadow_params']        # Restore all shadow parameter tensors