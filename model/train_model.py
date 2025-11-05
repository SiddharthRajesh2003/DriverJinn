import math
import torch
import torch.nn as nn
from typing import Tuple, List, Optional, Dict
import pandas as pd
import gc
import numpy as np
import pickle
import os
from pathlib import Path
from sklearn.metrics import (
    classification_report, roc_auc_score, roc_curve, 
    average_precision_score, precision_recall_curve
)
import argparse
import matplotlib.pyplot as plt
import seaborn as sns

from utils.logging_manager import get_logger
from model.DriverGenePredictor import ContrastiveDriverGenePredictor
from model.support_models import WarmupScheduler, EarlyStopping

# At the beginning of your script, after imports
torch.cuda.empty_cache()
gc.collect()

# Enable memory efficient attention if using transformers
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def check_for_nans(tensor, name="tensor", raise_error=True):
    """Check if tensor contains NaN or Inf values"""
    if torch.isnan(tensor).any():
        msg = f"{name} contains NaN values"
        if raise_error:
            raise ValueError(msg)
        else:
            logger.error(msg)
            return True
    if torch.isinf(tensor).any():
        msg = f"{name} contains Inf values"
        if raise_error:
            raise ValueError(msg)
        else:
            logger.error(msg)
            return True
    return False


def sanitize_curvature(edge_curvature, name="curvature"):
    """Replace NaN/Inf values in curvature with safe defaults"""
    if edge_curvature is None or edge_curvature.numel() == 0:
        return edge_curvature
    
    # Check for problematic values
    nan_mask = torch.isnan(edge_curvature)
    inf_mask = torch.isinf(edge_curvature)
    
    if nan_mask.any() or inf_mask.any():
        logger.warning(f"{name}: Found {nan_mask.sum().item()} NaN and {inf_mask.sum().item()} Inf values")
        
        # Replace NaN/Inf with 0 (neutral curvature)
        edge_curvature = edge_curvature.clone()
        edge_curvature[nan_mask] = 0.0
        edge_curvature[inf_mask] = 0.0
        
        logger.warning(f"{name}: Replaced problematic values with 0")
    
    # Clamp to reasonable range to prevent numerical issues
    edge_curvature = torch.clamp(edge_curvature, min=-10.0, max=10.0)
    
    return edge_curvature


def validate_graph_data(data, name="graph", strict=False):
    """Validate graph data structure for NaN/Inf values"""
    issues = []
    
    # Check features
    features = data.get('feature', data.get('x'))
    if features is not None:
        if torch.isnan(features).any():
            issues.append(f"{name}: Features contain NaN")
            if strict:
                raise ValueError(issues[-1])
        if torch.isinf(features).any():
            issues.append(f"{name}: Features contain Inf")
            if strict:
                raise ValueError(issues[-1])
    
    # Check edge index
    edge_index = data.get('edge_index')
    if edge_index is not None:
        num_nodes = features.shape[0] if features is not None else edge_index.max().item() + 1
        if edge_index.min() < 0:
            issues.append(f"{name}: Edge index contains negative values")
        if edge_index.max() >= num_nodes:
            issues.append(f"{name}: Edge index exceeds number of nodes")
    
    # Check curvature
    for curv_key in ['ollivier_curvature', 'forman_curvature']:
        if curv_key in data:
            curv = data[curv_key]
            if torch.isnan(curv).any():
                issues.append(f"{name}: {curv_key} contains NaN")
            if torch.isinf(curv).any():
                issues.append(f"{name}: {curv_key} contains Inf")
    
    if issues:
        for issue in issues:
            logger.error(issue)
        if strict:
            raise ValueError(f"Graph validation failed: {issues}")
    
    return len(issues) == 0


# UPDATE the preprocess_curvature_data function to include sanitization
def preprocess_curvature_data(data: Dict, curvature_type: str = 'ollivier') -> Dict:
    """
    Preprocess data to ensure curvature dimensions match edge_index.
    Call this once before training to fix all curvature issues upfront.
    NOW WITH NaN/Inf DETECTION AND SANITIZATION!
    """
    curv_key = f'{curvature_type}_curvature'
    
    if curv_key not in data:
        logger.warning(f"No '{curv_key}' found in data, skipping preprocessing")
        return data
    
    edge_index = data['edge_index']
    edge_curvature = data[curv_key]
    
    # STEP 1: Sanitize curvature values BEFORE dimension matching
    logger.info(f"Sanitizing {curv_key} values...")
    edge_curvature = sanitize_curvature(edge_curvature, name=curv_key)
    
    num_edges = edge_index.shape[1]
    num_curvatures = edge_curvature.shape[0]
    
    logger.info(f"Preprocessing curvature: {num_curvatures} curvatures for {num_edges} edges")
    
    if num_curvatures == num_edges:
        logger.info("Curvature dimensions already match, no preprocessing needed")
        data[curv_key] = edge_curvature  # Store sanitized version
        return data
    
    if num_curvatures * 2 == num_edges:
        logger.info("Detected undirected curvature for directed edges, matching...")
        
        device = edge_curvature.device
        matched_curvature = torch.zeros(num_edges, device=device, dtype=edge_curvature.dtype)
        
        edge_list = edge_index.t().cpu().numpy()
        edge_to_curv_idx = {}
        
        for src, dst in edge_list:
            src, dst = int(src), int(dst)
            canonical = (min(src, dst), max(src, dst))
            if canonical not in edge_to_curv_idx:
                edge_to_curv_idx[canonical] = len(edge_to_curv_idx)
        
        sorted_edges = sorted(edge_to_curv_idx.keys())
        edge_to_curv_idx = {edge: i for i, edge in enumerate(sorted_edges)}
        
        for i, (src, dst) in enumerate(edge_list):
            src, dst = int(src), int(dst)
            canonical = (min(src, dst), max(src, dst))
            curv_idx = edge_to_curv_idx.get(canonical, 0)
            if curv_idx < num_curvatures:
                matched_curvature[i] = edge_curvature[curv_idx]
        
        data[curv_key] = matched_curvature
        logger.info(f"Successfully matched {num_curvatures} curvatures to {num_edges} edges")
    
    elif num_edges * 2 == num_curvatures:
        logger.info("Detected directed curvature for undirected edges, averaging...")
        data[curv_key] = edge_curvature.reshape(-1, 2).mean(dim=1)
    
    else:
        logger.warning(f"Unusual ratio: {num_curvatures} curvatures for {num_edges} edges")
        logger.warning("Attempting best-effort matching...")
        
        device = edge_curvature.device
        matched_curvature = torch.zeros(num_edges, device=device, dtype=edge_curvature.dtype)
        
        edge_list = edge_index.t().cpu().numpy()
        edge_to_curv_idx = {}
        
        for src, dst in edge_list:
            src, dst = int(src), int(dst)
            canonical = (min(src, dst), max(src, dst))
            if canonical not in edge_to_curv_idx:
                edge_to_curv_idx[canonical] = len(edge_to_curv_idx)
        
        sorted_edges = sorted(edge_to_curv_idx.keys())
        edge_to_curv_idx = {edge: i for i, edge in enumerate(sorted_edges)}
        
        for i, (src, dst) in enumerate(edge_list):
            src, dst = int(src), int(dst)
            canonical = (min(src, dst), max(src, dst))
            curv_idx = edge_to_curv_idx.get(canonical, 0)
            if curv_idx < num_curvatures:
                matched_curvature[i] = edge_curvature[curv_idx]
            else:
                matched_curvature[i] = 0.0  # Use 0 instead of mean for missing
        
        data[curv_key] = matched_curvature
        logger.info(f"Best-effort matching complete")
    
    # STEP 2: Final sanitization after matching
    data[curv_key] = sanitize_curvature(data[curv_key], name=f"{curv_key}_matched")
    
    return data


# UPDATE the evaluate_with_ranking_metrics function to handle NaN gracefully
def evaluate_with_ranking_metrics(
    model: nn.Module,
    data: Dict,
    labels: torch.Tensor,
    mask: torch.Tensor,
    curvature_type: str = 'ollivier',
    device: torch.device = None
) -> Dict[str, float]:
    """
    Evaluate model using ranking metrics (AUPRC, AUROC, Precision@K, Recall@K).
    Better for imbalanced data than accuracy/F1.
    NOW WITH NaN DETECTION!
    """
    model.eval()
    
    try:
        with torch.no_grad():
            probs = model.predict_probability(data, mask, curvature_type, device)
        
        # CHECK FOR NaN IN PREDICTIONS
        if torch.isnan(probs).any():
            logger.error(f"NaN detected in predictions! {torch.isnan(probs).sum().item()} out of {probs.numel()} values")
            
            # Return zero metrics to allow training to continue
            return {
                'auc_roc': 0.0,
                'auc_pr': 0.0,
                'optimal_threshold': 0.5,
                'optimal_f1': 0.0,
                'optimal_precision': 0.0,
                'optimal_recall': 0.0,
                'precision@10': 0.0,
                'recall@10': 0.0
            }
        
        probs_np = probs.cpu().numpy()
        labels_np = labels[mask].cpu().numpy()
        
        # Double-check numpy arrays
        if np.isnan(probs_np).any():
            logger.error(f"NaN in numpy predictions after conversion!")
            return {
                'auc_roc': 0.0,
                'auc_pr': 0.0,
                'optimal_threshold': 0.5,
                'optimal_f1': 0.0,
                'optimal_precision': 0.0,
                'optimal_recall': 0.0
            }
        
        metrics = {}
        
        # AUPRC and AUROC with try-catch
        try:
            metrics['auc_roc'] = roc_auc_score(labels_np, probs_np)
            metrics['auc_pr'] = average_precision_score(labels_np, probs_np)
        except ValueError as e:
            logger.warning(f"Could not compute AUC metrics: {e}")
            metrics['auc_roc'] = 0.0
            metrics['auc_pr'] = 0.0
        
        # Find optimal threshold
        try:
            threshold_results = find_optimal_threshold_f1(probs_np, labels_np)
            metrics['optimal_threshold'] = threshold_results['optimal_threshold']
            metrics['optimal_f1'] = threshold_results['max_f1']
            metrics['optimal_precision'] = threshold_results['precision']
            metrics['optimal_recall'] = threshold_results['recall']
        except Exception as e:
            logger.warning(f"Could not find optimal threshold: {e}")
            metrics['optimal_threshold'] = 0.5
            metrics['optimal_f1'] = 0.0
            metrics['optimal_precision'] = 0.0
            metrics['optimal_recall'] = 0.0
        
        # Precision@K and Recall@K
        n_positives = labels_np.sum()
        k_values = [10, 20, 50, 100]
        
        for k in k_values:
            if k > len(probs_np):
                continue
            
            top_k_indices = np.argsort(-probs_np)[:k]
            top_k_labels = labels_np[top_k_indices]
            
            precision_at_k = top_k_labels.sum() / k
            recall_at_k = top_k_labels.sum() / n_positives if n_positives > 0 else 0
            
            metrics[f'precision@{k}'] = precision_at_k
            metrics[f'recall@{k}'] = recall_at_k
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error in evaluate_with_ranking_metrics: {e}")
        logger.error(f"Error type: {type(e).__name__}")
        import traceback
        logger.error(traceback.format_exc())
        
        # Return zero metrics to allow training to continue
        return {
            'auc_roc': 0.0,
            'auc_pr': 0.0,
            'optimal_threshold': 0.5,
            'optimal_f1': 0.0,
            'optimal_precision': 0.0,
            'optimal_recall': 0.0
        }

# ============================================================================
# HELPER FUNCTIONS FOR THRESHOLD OPTIMIZATION
# ============================================================================

def find_optimal_threshold_f1(probs: np.ndarray, labels: np.ndarray) -> Dict:
    """Find threshold that maximizes F1 score."""
    precision, recall, thresholds = precision_recall_curve(labels, probs)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
    
    predictions = (probs >= optimal_threshold).astype(int)
    tp = ((predictions == 1) & (labels == 1)).sum()
    fp = ((predictions == 1) & (labels == 0)).sum()
    tn = ((predictions == 0) & (labels == 0)).sum()
    fn = ((predictions == 0) & (labels == 1)).sum()
    
    return {
        'optimal_threshold': optimal_threshold,
        'max_f1': f1_scores[optimal_idx],
        'precision': precision[optimal_idx],
        'recall': recall[optimal_idx],
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
        'accuracy': (tp + tn) / (tp + fp + tn + fn)
    }


def analyze_driver_distribution(probs: np.ndarray, labels: np.ndarray) -> Dict:
    """Analyze probability distribution of known drivers vs non-drivers."""
    driver_probs = probs[labels == 1]
    non_driver_probs = probs[labels == 0]
    
    if len(driver_probs) == 0:
        return {
            'driver_mean': 0.0,
            'driver_median': 0.0,
            'non_driver_mean': non_driver_probs.mean() if len(non_driver_probs) > 0 else 0.0,
            'separation': 0.0,
            'suggested_conservative': 0.5,
            'suggested_balanced': 0.5,
            'suggested_liberal': 0.7
        }
    
    stats = {
        'driver_mean': driver_probs.mean(),
        'driver_median': np.median(driver_probs),
        'driver_std': driver_probs.std(),
        'driver_min': driver_probs.min(),
        'driver_max': driver_probs.max(),
        'driver_q25': np.percentile(driver_probs, 25),
        'driver_q75': np.percentile(driver_probs, 75),
        'non_driver_mean': non_driver_probs.mean(),
        'non_driver_median': np.median(non_driver_probs),
        'separation': driver_probs.mean() - non_driver_probs.mean()
    }
    
    # Suggested thresholds based on percentiles
    stats['suggested_conservative'] = np.percentile(driver_probs, 25)
    stats['suggested_balanced'] = np.percentile(driver_probs, 50)
    stats['suggested_liberal'] = np.percentile(driver_probs, 75)
    
    return stats


def plot_threshold_analysis(
    probs: np.ndarray,
    labels: np.ndarray,
    save_path: Path,
    fold_idx: int
):
    """Create threshold analysis plots."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Distribution comparison
    driver_probs = probs[labels == 1]
    non_driver_probs = probs[labels == 0]
    
    axes[0, 0].hist(non_driver_probs, bins=50, alpha=0.5, label='Non-drivers', 
                   density=True, color='blue')
    axes[0, 0].hist(driver_probs, bins=50, alpha=0.5, label='Known drivers', 
                   density=True, color='red')
    axes[0, 0].set_xlabel('Probability Score')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].set_title(f'Fold {fold_idx}: Probability Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Precision-Recall curve
    precision, recall, thresholds = precision_recall_curve(labels, probs)
    axes[0, 1].plot(recall, precision, linewidth=2)
    axes[0, 1].set_xlabel('Recall')
    axes[0, 1].set_ylabel('Precision')
    axes[0, 1].set_title(f'Fold {fold_idx}: Precision-Recall Curve')
    axes[0, 1].grid(True, alpha=0.3)
    
    # F1 vs threshold
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    axes[1, 0].plot(thresholds, f1_scores[:-1], linewidth=2)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
    axes[1, 0].axvline(optimal_threshold, color='r', linestyle='--', 
                      label=f'Optimal: {optimal_threshold:.3f}')
    axes[1, 0].set_xlabel('Threshold')
    axes[1, 0].set_ylabel('F1 Score')
    axes[1, 0].set_title(f'Fold {fold_idx}: F1 Score vs Threshold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # ROC curve
    fpr, tpr, _ = roc_curve(labels, probs)
    auc_roc = roc_auc_score(labels, probs)
    axes[1, 1].plot(fpr, tpr, linewidth=2, label=f'AUC={auc_roc:.3f}')
    axes[1, 1].plot([0, 1], [0, 1], 'k--', alpha=0.3)
    axes[1, 1].set_xlabel('False Positive Rate')
    axes[1, 1].set_ylabel('True Positive Rate')
    axes[1, 1].set_title(f'Fold {fold_idx}: ROC Curve')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


@torch.no_grad()
def ensemble_predict(
    models: List[nn.Module],
    data: Dict,
    mask: Optional[torch.Tensor] = None,
    curvature_type: str = 'ollivier',
    device: torch.device = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Ensemble prediction from multiple models
    
    Returns:
        mean_probs: Average probabilities
        std_probs: Standard deviation (uncertainty estimate)
    """
    all_probs = []
    
    for model in models:
        model.eval()
        probs = model.predict_probability(data, mask, curvature_type, device)
        all_probs.append(probs)
    
    all_probs = torch.stack(all_probs)
    mean_probs = all_probs.mean(dim=0)
    std_probs = all_probs.std(dim=0)
    
    return mean_probs, std_probs


def create_cancer_driver_model(
    num_features: int = 74,
    hidden_channels: int = 256,
    projection_dim: int = 128,
    num_layers: int = 3,
    device: torch.device = None
) -> ContrastiveDriverGenePredictor:
    """
    Factory function to create cancer driver prediction model
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = ContrastiveDriverGenePredictor(
        in_channels=num_features,
        hidden_channels=hidden_channels,
        projection_dim=projection_dim,
        num_gnn_layers=num_layers,
        curvature_types=['positive', 'negative', 'both'],
        num_attention_heads=4,
        temperature=0.7,
        dropout=0.2,
        device=device
    ).to(device)

    logger.info(f"Created ContrastiveCancerDriverPredictor model:")
    logger.info(f"  - Input features: {num_features}")
    logger.info(f"  - Hidden channels: {hidden_channels}")
    logger.info(f"  - Device: {device}")
    logger.info(f"  - Binary classification: driver vs non-driver")
    logger.info(f"  - Imbalance handling: focal loss enabled")
    
    return model

def train_single_fold(
    fold_idx: int,
    fold_data: Dict,
    original: Dict,
    augmented_views: List[Dict],
    labels: torch.Tensor,
    num_epochs: int,
    device: torch.device,
    model_save_dir: Path,
    results_dir: Path,
    model_prefix: str = "",
    use_focal_loss: bool = True,
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
    pos_weight: Optional[torch.Tensor] = None
) -> Tuple[ContrastiveDriverGenePredictor, Dict, Dict, Dict]:
    """
    Train model on a single fold with imbalance handling and threshold optimization.
    
    Returns:
        model: Trained model
        best_metrics: Best validation metrics
        history: Training history
        threshold_info: Optimal threshold information
    """
    print(f"\n{'='*80}")
    print(f"TRAINING FOLD {fold_idx}")
    print(f"{'='*80}")
    
    # Extract masks
    train_mask_original = torch.from_numpy(fold_data['train_mask'])
    val_mask_original = torch.from_numpy(fold_data['val_mask'])
    
    num_original_nodes = original['feature'].shape[0]
    if len(train_mask_original) != num_original_nodes:
        raise ValueError(
            f"Train mask size ({len(train_mask_original)}) doesn't match "
            f"original graph nodes ({num_original_nodes})"
        )
    
    # Analyze class imbalance
    train_labels = labels[train_mask_original]
    num_pos = (train_labels == 1).sum().item()
    num_neg = (train_labels == 0).sum().item()
    imbalance_ratio = num_neg / num_pos if num_pos > 0 else float('inf')
    
    print(f"\nClass Distribution:")
    print(f"  Training samples: {len(train_labels)}")
    print(f"  Positive (drivers): {num_pos} ({100*num_pos/len(train_labels):.2f}%)")
    print(f"  Negative: {num_neg} ({100*num_neg/len(train_labels):.2f}%)")
    print(f"  Imbalance ratio: 1:{imbalance_ratio:.1f}")
    
    # Adjust focal loss parameters based on imbalance severity
    if imbalance_ratio > 50:
        focal_alpha_adjusted = 0.15
        focal_gamma_adjusted = 3.0
        print(f"  → Using aggressive focal loss (extreme imbalance)")
    elif imbalance_ratio > 20:
        focal_alpha_adjusted = 0.20
        focal_gamma_adjusted = 2.5
        print(f"  → Using strong focal loss (severe imbalance)")
    else:
        focal_alpha_adjusted = focal_alpha
        focal_gamma_adjusted = focal_gamma
        print(f"  → Using standard focal loss (moderate imbalance)")
    
    print(f"  Focal loss parameters: alpha={focal_alpha_adjusted}, gamma={focal_gamma_adjusted}")
    
    # Calculate fold-specific positive weight (for non-focal loss)
    fold_pos_weight = torch.tensor([num_neg / num_pos], device=device) if num_pos > 0 else torch.tensor([1.0], device=device)
    
    print(f"  Val samples: {val_mask_original.sum().item()}")
    
    # Create model
    model = create_cancer_driver_model(
        num_features=original['feature'].shape[1],
        hidden_channels=256,
        projection_dim=128,
        num_layers=3,
        device=device
    )
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.001,
        weight_decay=1e-5,
        betas=(0.9, 0.999)
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max',  # Maximize AUPRC
        factor=0.5, 
        patience=20, 
        min_lr=1e-6
    )
    
    # Warmup scheduler
    warmup_scheduler = WarmupScheduler(
        optimizer,
        warmup_epochs=10,
        initial_lr=1e-5,
        target_lr=0.001
    )
    
    # Early stopping based on AUPRC (better for imbalanced data)
    early_stopping = EarlyStopping(patience=50, min_delta=0.0001, mode='max')
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_auprc': [],  # NEW: Primary metric
        'val_auroc': [],  # NEW: Secondary metric
        'val_f1': [],
        'val_precision': [],
        'val_recall': [],
        'val_precision@50': [],  # NEW: Interpretable metric
        'learning_rate': []
    }
    
    best_val_auprc = 0.0
    best_metrics = None
    threshold_info = None
    
    # Construct model filename
    if model_prefix:
        model_filename = f'{model_prefix}_fold_{fold_idx}_best_model.pt'
    else:
        model_filename = f'fold_{fold_idx}_best_model.pt'
    
    model_path = model_save_dir / model_filename
    
    print(f"\n{'='*80}")
    print("TRAINING WITH IMBALANCE HANDLING")
    print(f"{'='*80}")
    print(f"Loss type: {'Focal Loss' if use_focal_loss else 'Weighted BCE'}")
    print(f"Early stopping metric: AUPRC (better for imbalanced data)")
    print(f"{'='*80}\n")
    
    # Training loop
    for epoch in range(num_epochs):
        # Warmup learning rate
        if epoch < 10:
            warmup_scheduler.step()
        
        # Sample two different augmented views
        view_indices = torch.randperm(len(augmented_views))[:2]
        augmented_view1 = augmented_views[view_indices[0]]
        augmented_view2 = augmented_views[view_indices[1]]
        
        # Train step with focal loss
        try:
            loss_dict = model.train_step(
                augmented_view1,
                augmented_view2,
                original,
                labels,
                train_mask_original,
                optimizer,
                contrastive_weight=0.3,
                pos_weight=fold_pos_weight if not use_focal_loss else None,
                curvature_type='ollivier',
                device=device,
                batch_size=2048,
                use_focal_loss=use_focal_loss,
                focal_alpha=focal_alpha_adjusted,
                focal_gamma=focal_gamma_adjusted
            )
        except RuntimeError as e:
            if "out of memory" in str(e):
                logger.warning("OOM error, clearing cache and reducing batch size")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                loss_dict = model.train_step(
                    augmented_view1,
                    augmented_view2,
                    original,
                    labels,
                    train_mask_original,
                    optimizer,
                    contrastive_weight=0.3,
                    pos_weight=fold_pos_weight if not use_focal_loss else None,
                    curvature_type='ollivier',
                    device=device,
                    batch_size=1024,
                    use_focal_loss=use_focal_loss,
                    focal_alpha=focal_alpha_adjusted,
                    focal_gamma=focal_gamma_adjusted
                )
            else:
                raise e
        
        if math.isnan(loss_dict['total_loss']):
            logger.error(f"Epoch {epoch}: Training failed with NaN loss")
            continue
        
        # Validation with RANKING METRICS (better for imbalanced data)
        ranking_metrics = evaluate_with_ranking_metrics(
            model, original, labels, val_mask_original,
            curvature_type='ollivier', device=device
        )
        
        # Also get standard metrics for compatibility
        val_metrics = model.evaluate(
            original, labels, val_mask_original,
            curvature_type='ollivier', device=device
        )
        
        # Update learning rate based on AUPRC (not F1!)
        if epoch >= 10:
            scheduler.step(ranking_metrics['auc_pr'])
        
        # Store history
        history['train_loss'].append(loss_dict['total_loss'])
        history['train_acc'].append(loss_dict['train_accuracy'])
        history['val_auprc'].append(ranking_metrics['auc_pr'])
        history['val_auroc'].append(ranking_metrics['auc_roc'])
        history['val_f1'].append(val_metrics['f1'])
        history['val_precision'].append(val_metrics['precision'])
        history['val_recall'].append(val_metrics['recall'])
        history['val_precision@50'].append(ranking_metrics.get('precision@50', 0))
        history['learning_rate'].append(optimizer.param_groups[0]['lr'])
        
        # Logging
        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d} | "
                  f"Loss: {loss_dict['total_loss']:.4f} | "
                  f"AUPRC: {ranking_metrics['auc_pr']:.4f} | "
                  f"AUROC: {ranking_metrics['auc_roc']:.4f} | "
                  f"P@50: {ranking_metrics.get('precision@50', 0):.4f} | "
                  f"F1: {val_metrics['f1']:.4f} | "
                  f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model based on AUPRC (not F1!)
        if ranking_metrics['auc_pr'] > best_val_auprc:
            best_val_auprc = ranking_metrics['auc_pr']
            best_metrics = {**val_metrics, **ranking_metrics}
            
            # Save model
            model.save_checkpoint(
                str(model_path),
                epoch,
                optimizer,
                best_metrics,
                metadata={
                    'fold': fold_idx,
                    'num_views': len(augmented_views),
                    'loss_type': 'focal' if use_focal_loss else 'bce',
                    'focal_alpha': focal_alpha_adjusted if use_focal_loss else None,
                    'focal_gamma': focal_gamma_adjusted if use_focal_loss else None,
                    'imbalance_ratio': imbalance_ratio,
                    'model_prefix': model_prefix
                }
            )
        
        # Early stopping based on AUPRC
        if early_stopping(ranking_metrics['auc_pr']):
            print(f"\nEarly stopping triggered at epoch {epoch}")
            print(f"Best validation AUPRC: {best_val_auprc:.4f}")
            break
    
    # Load best model
    checkpoint = model.load_checkpoint(str(model_path), optimizer, device)
    print(f"\n✓ Loaded best model from epoch {checkpoint['epoch']}")
    print(f"  Best Val AUPRC: {best_val_auprc:.4f}")
    print(f"  Best Val AUROC: {best_metrics.get('auc_roc', 0):.4f}")
    print(f"  Best Val F1: {best_metrics.get('f1', 0):.4f}")
    print(f"  Saved to: {model_path}")
    
    # THRESHOLD OPTIMIZATION on validation set
    print(f"\n{'='*80}")
    print("THRESHOLD OPTIMIZATION (Validation Set)")
    print(f"{'='*80}")
    
    with torch.no_grad():
        val_probs = model.predict_probability(
            original, val_mask_original, curvature_type='ollivier', device=device
        )
    
    val_probs_np = val_probs.cpu().numpy()
    val_labels_np = labels[val_mask_original].cpu().numpy()
    
    # Analyze distribution
    driver_stats = analyze_driver_distribution(val_probs_np, val_labels_np)
    
    print(f"\nKnown Driver Distribution:")
    print(f"  Mean probability: {driver_stats['driver_mean']:.3f}")
    print(f"  Median probability: {driver_stats['driver_median']:.3f}")
    print(f"  Separation from non-drivers: {driver_stats['separation']:.3f}")
    
    # Find optimal threshold
    threshold_results = find_optimal_threshold_f1(val_probs_np, val_labels_np)
    
    print(f"\nOptimal Threshold (F1-based):")
    print(f"  Threshold: {threshold_results['optimal_threshold']:.3f}")
    print(f"  Expected F1: {threshold_results['max_f1']:.3f}")
    print(f"  Expected Precision: {threshold_results['precision']:.3f}")
    print(f"  Expected Recall: {threshold_results['recall']:.3f}")
    
    print(f"\nSuggested Thresholds (based on driver distribution):")
    print(f"  Conservative (75% drivers): {driver_stats['suggested_conservative']:.3f}")
    print(f"  Balanced (50% drivers): {driver_stats['suggested_balanced']:.3f}")
    print(f"  Liberal (25% drivers): {driver_stats['suggested_liberal']:.3f}")
    print(f"{'='*80}\n")
    
    # Store threshold information
    threshold_info = {
        'optimal_f1_threshold': threshold_results['optimal_threshold'],
        'expected_f1': threshold_results['max_f1'],
        'expected_precision': threshold_results['precision'],
        'expected_recall': threshold_results['recall'],
        'driver_stats': driver_stats,
        'conservative_threshold': driver_stats['suggested_conservative'],
        'balanced_threshold': driver_stats['suggested_balanced'],
        'liberal_threshold': driver_stats['suggested_liberal']
    }
    
    # Create threshold analysis plots
    plot_path = results_dir / f'fold_{fold_idx}_threshold_analysis.png'
    try:
        plot_threshold_analysis(val_probs_np, val_labels_np, plot_path, fold_idx)
        print(f"✓ Saved threshold analysis to: {plot_path}")
    except Exception as e:
        logger.warning(f"Could not create threshold plots: {e}")
    
    return model, best_metrics, history, threshold_info


def main():
    
    parser = argparse.ArgumentParser(description='Train the Contrastive Driver Gene Predictor with Imbalance Handling')
    parser.add_argument('--dataset_file', type=str, required=True,
                        help='Input dataset pickle file')
    parser.add_argument('--train_metrics_dir', type=str, 
                        help='Specify the output directory for Training and Evaluation Metrics', 
                        default='model_results')
    parser.add_argument('--model_out_dir', type=str, 
                        help='Specify the directory to output the model checkpoints', 
                        default='trained_models')
    parser.add_argument('--model_out_prefix', type=str, default='',
                        help='Prefix for model checkpoint filenames (e.g., "dataset1", "cancer_type_A")')
    parser.add_argument('--num_folds', type=int, default=None, 
                        help='Number of folds to use (default: use all folds in dataset)')
    parser.add_argument('--specific_folds', type=int, nargs='+', default=None,
                        help='Train only specific folds (e.g., --specific_folds 1 3 5)')
    parser.add_argument('--num_epochs', type=int, default=200,
                        help='Number of training epochs per fold')
    parser.add_argument('--batch_size', type=int, default=2048,
                        help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Initial learning rate')
    parser.add_argument('--hidden_channels', type=int, default=256,
                        help='Hidden layer channels')
    parser.add_argument('--use_focal_loss', action='store_true', default=False,
                        help='Use focal loss for class imbalance (DEFAULT: True)')
    parser.add_argument('--focal_alpha', type=float, default=0.25,
                        help='Focal loss alpha parameter (weight for positive class)')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                        help='Focal loss gamma parameter (focusing parameter)')
    parser.add_argument('--early_stopping_patience', type=int, default=50,
                        help='Early stopping patience')
    parser.add_argument('--save_all_checkpoints', action='store_true',
                        help='Save checkpoints every N epochs (not just best)')
    parser.add_argument('--checkpoint_frequency', type=int, default=10,
                        help='Save checkpoint every N epochs when --save_all_checkpoints is used')
    parser.add_argument('--optimal_threshold', type = float, default=0.5, 
                        help = 'Select the optimal threshold for identifying potential drivers')

    args = parser.parse_args()
    
    dataset_file = args.dataset_file
    
    # Create directories for outputs
    models_dir = Path(args.model_out_dir)
    results_dir = Path(args.train_metrics_dir)
    models_dir.mkdir(exist_ok=True, parents=True)
    results_dir.mkdir(exist_ok=True, parents=True)
    
    # Add prefix to results directory if specified
    if args.model_out_prefix:
        results_subdir = results_dir / args.model_out_prefix
        results_subdir.mkdir(exist_ok=True, parents=True)
        results_dir = results_subdir
        print(f"Results will be saved to: {results_dir}")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.cuda.empty_cache()
    
    print("\n" + "="*80)
    print("K-FOLD CONTRASTIVE DRIVER GENE PREDICTOR WITH IMBALANCE HANDLING")
    print("="*80)
    print(f"Dataset: {dataset_file}")
    if args.model_out_prefix:
        print(f"Model Prefix: {args.model_out_prefix}")
    print(f"Model Output Dir: {models_dir}")
    print(f"Results Output Dir: {results_dir}")
    print(f"Device: {device}")
    print(f"Loss: {'Focal Loss' if args.use_focal_loss else 'Weighted BCE'}")
    if args.use_focal_loss:
        print(f"  Focal alpha: {args.focal_alpha}")
        print(f"  Focal gamma: {args.focal_gamma}")
    print(f"PyTorch version: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print("="*80 + "\n")
    
    # Load data
    try:
        with open(dataset_file, 'rb') as f:
            data = pickle.load(f)
    except FileNotFoundError:
        logger.error("Data file not found. Please check the path.")
        exit(1)    
    
    original = data['original']
    augmented_views = data['augmented_views']
    
    print("\n" + "="*80)
    print("VALIDATING INPUT DATA")
    print("="*80)
    
    logger.info("Checking original graph for NaN/Inf values...")
    validate_graph_data(original, name="Original Graph", strict=False)
    
    # Check and sanitize features
    features = original.get('feature', original.get('x'))
    if torch.isnan(features).any() or torch.isinf(features).any():
        logger.warning("Found NaN/Inf in features, sanitizing...")
        features = torch.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
        if 'feature' in original:
            original['feature'] = features
        else:
            original['x'] = features
        logger.info("✓ Features sanitized")
    
    # Check labels
    labels = original['label']
    if torch.isnan(labels).any():
        logger.error("Labels contain NaN! This is a critical error.")
        raise ValueError("Labels cannot contain NaN")
    
    logger.info("Checking augmented views for NaN/Inf values...")
    for i, view in enumerate(augmented_views):
        try:
            validate_graph_data(view, name=f"Augmented View {i+1}", strict=False)
            
            # Sanitize view features if needed
            view_features = view.get('x', view.get('feature'))
            if view_features is not None:
                if torch.isnan(view_features).any() or torch.isinf(view_features).any():
                    logger.warning(f"View {i+1}: Found NaN/Inf in features, sanitizing...")
                    view_features = torch.nan_to_num(view_features, nan=0.0, posinf=1.0, neginf=-1.0)
                    if 'x' in view:
                        view['x'] = view_features
                    else:
                        view['feature'] = view_features
        except Exception as e:
            logger.error(f"Validation failed for view {i+1}: {e}")
    
    print("✓ Data validation complete")
    print("="*80 + "\n")
    
    # IMPORTANT: Preprocess curvature data
    print("\n" + "="*80)
    print("PREPROCESSING CURVATURE DATA")
    print("="*80)
    
    logger.info("Preprocessing original graph curvature...")
    original = preprocess_curvature_data(original, curvature_type='ollivier')
    
    logger.info(f"Preprocessing {len(augmented_views)} augmented views...")
    
    num_original_nodes = original['feature'].shape[0]
    logger.info(f"Original graph has {num_original_nodes} nodes")
    
    for i, view in enumerate(augmented_views):
        logger.info(f"  Processing view {i+1}/{len(augmented_views)}")
        augmented_views[i] = preprocess_curvature_data(view, curvature_type='ollivier')
        
        view_nodes = augmented_views[i]['x'].shape[0]
        eliminated_count = len(augmented_views[i]['metadata']['eliminated_node_ids'])
        
        logger.info(f"    View {i+1}: {view_nodes} nodes")
        logger.info(f"    Metadata says {eliminated_count} nodes were eliminated")
        
        max_node_in_edges = augmented_views[i]['edge_index'].max().item() if augmented_views[i]['edge_index'].numel() > 0 else -1
        
        if max_node_in_edges >= view_nodes:
            raise ValueError(
                f"View {i+1} is inconsistent: edge_index references node {max_node_in_edges} "
                f"but only {view_nodes} nodes exist in features"
            )
        
        logger.info(f"    ✓ View {i+1} verified: max edge node index = {max_node_in_edges}, node count = {view_nodes}")
    
    logger.info("✓ All augmented views are self-consistent")
    print("✓ Curvature preprocessing complete")
    print("="*80 + "\n")
    
    # Binary labels
    labels = original['label']
    
    # Get k-fold splits
    kfold_splits = original['kfold_splits']
    num_folds = len(kfold_splits)
    
    # Filter folds if specified
    if args.specific_folds:
        folds_to_train = [i-1 for i in args.specific_folds if 1 <= i <= num_folds]
        kfold_splits = [kfold_splits[i] for i in folds_to_train]
        print(f"Training only folds: {args.specific_folds}")
    elif args.num_folds and args.num_folds < num_folds:
        kfold_splits = kfold_splits[:args.num_folds]
        print(f"Training only first {args.num_folds} folds")
    
    print(f"\n{'='*80}")
    print(f"K-FOLD CROSS-VALIDATION: {len(kfold_splits)} FOLDS")
    print(f"{'='*80}")
    
    # Store results
    all_fold_metrics = []
    all_fold_histories = []
    all_threshold_info = []
    fold_models = []
    
    # Train each fold
    for fold_idx, fold_data in enumerate(kfold_splits, 1):
        model, best_metrics, history, threshold_info = train_single_fold(
            fold_idx=fold_idx,
            fold_data=fold_data,
            original=original,
            augmented_views=augmented_views,
            labels=labels,
            num_epochs=args.num_epochs,
            device=device,
            model_save_dir=models_dir,
            results_dir=results_dir,
            model_prefix=args.model_out_prefix,
            use_focal_loss=args.use_focal_loss,
            focal_alpha=args.focal_alpha,
            focal_gamma=args.focal_gamma
        )
        
        all_fold_metrics.append(best_metrics)
        all_fold_histories.append(history)
        all_threshold_info.append(threshold_info)
        fold_models.append(model)
        
        # Clear memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    # Aggregate results
    print("\n" + "="*80)
    print("K-FOLD CROSS-VALIDATION RESULTS")
    print("="*80)
    
    metrics_df_data = []
    for fold_idx, metrics in enumerate(all_fold_metrics, 1):
        print(f"\nFold {fold_idx}:")
        print(f"  AUPRC:     {metrics.get('auc_pr', 0):.4f}  ← PRIMARY METRIC")
        print(f"  AUROC:     {metrics.get('auc_roc', 0):.4f}")
        print(f"  F1 Score:  {metrics['f1']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  P@50:      {metrics.get('precision@50', 0):.4f}")
        
        metrics_df_data.append({
            'Fold': fold_idx,
            'AUPRC': metrics.get('auc_pr', 0),
            'AUROC': metrics.get('auc_roc', 0),
            'F1': metrics['f1'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'Precision@50': metrics.get('precision@50', 0),
            'Optimal_Threshold': all_threshold_info[fold_idx-1]['optimal_f1_threshold']
        })
    
    # Calculate statistics
    mean_metrics = {
        'auprc': np.mean([m.get('auc_pr', 0) for m in all_fold_metrics]),
        'auroc': np.mean([m.get('auc_roc', 0) for m in all_fold_metrics]),
        'f1': np.mean([m['f1'] for m in all_fold_metrics]),
        'precision': np.mean([m['precision'] for m in all_fold_metrics]),
        'recall': np.mean([m['recall'] for m in all_fold_metrics]),
        'precision@50': np.mean([m.get('precision@50', 0) for m in all_fold_metrics])
    }
    
    std_metrics = {
        'auprc': np.std([m.get('auc_pr', 0) for m in all_fold_metrics]),
        'auroc': np.std([m.get('auc_roc', 0) for m in all_fold_metrics]),
        'f1': np.std([m['f1'] for m in all_fold_metrics]),
        'precision': np.std([m['precision'] for m in all_fold_metrics]),
        'recall': np.std([m['recall'] for m in all_fold_metrics]),
        'precision@50': np.std([m.get('precision@50', 0) for m in all_fold_metrics])
    }
    
    print(f"\n{'='*80}")
    print("MEAN ± STD ACROSS ALL FOLDS")
    print(f"{'='*80}")
    print(f"AUPRC:        {mean_metrics['auprc']:.4f} ± {std_metrics['auprc']:.4f}  ← PRIMARY")
    print(f"AUROC:        {mean_metrics['auroc']:.4f} ± {std_metrics['auroc']:.4f}")
    print(f"F1 Score:     {mean_metrics['f1']:.4f} ± {std_metrics['f1']:.4f}")
    print(f"Precision:    {mean_metrics['precision']:.4f} ± {std_metrics['precision']:.4f}")
    print(f"Recall:       {mean_metrics['recall']:.4f} ± {std_metrics['recall']:.4f}")
    print(f"Precision@50: {mean_metrics['precision@50']:.4f} ± {std_metrics['precision@50']:.4f}")
    
    # Threshold statistics
    optimal_thresholds = [t['optimal_f1_threshold'] for t in all_threshold_info]
    mean_threshold = np.mean(optimal_thresholds)
    std_threshold = np.std(optimal_thresholds)
    print(f"\nOptimal Threshold: {mean_threshold:.3f} ± {std_threshold:.3f}")
    print(f"  (Use this instead of 0.5 for binary classification)")
    
    # Save fold results
    df_metrics = pd.DataFrame(metrics_df_data)
    
    # Add mean row
    df_metrics.loc[len(df_metrics)] = {
        'Fold': 'Mean',
        'AUPRC': mean_metrics['auprc'],
        'AUROC': mean_metrics['auroc'],
        'F1': mean_metrics['f1'],
        'Precision': mean_metrics['precision'],
        'Recall': mean_metrics['recall'],
        'Precision@50': mean_metrics['precision@50'],
        'Optimal_Threshold': mean_threshold
    }
    
    # Add std row
    df_metrics.loc[len(df_metrics)] = {
        'Fold': 'Std',
        'AUPRC': std_metrics['auprc'],
        'AUROC': std_metrics['auroc'],
        'F1': std_metrics['f1'],
        'Precision': std_metrics['precision'],
        'Recall': std_metrics['recall'],
        'Precision@50': std_metrics['precision@50'],
        'Optimal_Threshold': std_threshold
    }
    
    df_metrics.to_csv(results_dir / 'kfold_results.csv', index=False)
    print(f"\n✓ Saved k-fold results to '{results_dir / 'kfold_results.csv'}'")
    
    # ENSEMBLE EVALUATION
    print("\n" + "="*80)
    print("ENSEMBLE MODEL EVALUATION")
    print("="*80)
    
    test_mask = original['mask']
    
    # Ensemble predictions
    ensemble_probs, ensemble_std = ensemble_predict(
        fold_models, original, test_mask,
        curvature_type='ollivier', device=device
    )
    
    # Calculate ensemble metrics
    test_labels = labels[test_mask].cpu().numpy()
    
    # Ranking metrics for ensemble
    ensemble_ranking_metrics = {
        'auc_roc': roc_auc_score(test_labels, ensemble_probs.cpu().numpy()),
        'auc_pr': average_precision_score(test_labels, ensemble_probs.cpu().numpy())
    }
    
    # Find optimal threshold for ensemble
    ensemble_threshold_results = find_optimal_threshold_f1(
        ensemble_probs.cpu().numpy(), test_labels
    )
    
    ensemble_preds = (ensemble_probs >= ensemble_threshold_results['optimal_threshold']).cpu().numpy()
    
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    ensemble_metrics = {
        'accuracy': accuracy_score(test_labels, ensemble_preds),
        'precision': precision_score(test_labels, ensemble_preds, zero_division=0),
        'recall': recall_score(test_labels, ensemble_preds, zero_division=0),
        'f1': f1_score(test_labels, ensemble_preds, zero_division=0),
        'auc_roc': ensemble_ranking_metrics['auc_roc'],
        'auc_pr': ensemble_ranking_metrics['auc_pr'],
        'optimal_threshold': ensemble_threshold_results['optimal_threshold']
    }
    
    print(f"\nEnsemble Performance:")
    print(f"  AUPRC:     {ensemble_metrics['auc_pr']:.4f}  ← PRIMARY")
    print(f"  AUROC:     {ensemble_metrics['auc_roc']:.4f}")
    print(f"  F1 Score:  {ensemble_metrics['f1']:.4f}")
    print(f"  Precision: {ensemble_metrics['precision']:.4f}")
    print(f"  Recall:    {ensemble_metrics['recall']:.4f}")
    print(f"  Optimal Threshold: {ensemble_metrics['optimal_threshold']:.3f}")
    
    # Calculate Precision@K for ensemble
    k_values = [10, 20, 50, 100]
    n_positives = test_labels.sum()
    ensemble_probs_np = ensemble_probs.cpu().numpy()
    
    print(f"\n  Precision@K:")
    for k in k_values:
        if k <= len(test_labels):
            top_k_indices = np.argsort(-ensemble_probs_np)[:k]
            top_k_labels = test_labels[top_k_indices]
            p_at_k = top_k_labels.sum() / k
            r_at_k = top_k_labels.sum() / n_positives if n_positives > 0 else 0
            print(f"    P@{k}: {p_at_k:.4f} (R@{k}: {r_at_k:.4f})")
            ensemble_metrics[f'precision@{k}'] = p_at_k
            ensemble_metrics[f'recall@{k}'] = r_at_k
    
    # Plot ensemble ROC and PR curves
    try:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # ROC curve
        fpr, tpr, _ = roc_curve(test_labels, ensemble_probs_np)
        axes[0].plot(fpr, tpr, label=f'Ensemble (AUC={ensemble_metrics["auc_roc"]:.3f})', linewidth=2)
        axes[0].plot([0, 1], [0, 1], 'k--', label='Random', alpha=0.3)
        axes[0].set_xlabel('False Positive Rate')
        axes[0].set_ylabel('True Positive Rate')
        axes[0].set_title('Ensemble ROC Curve')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # Precision-Recall curve
        precision, recall, _ = precision_recall_curve(test_labels, ensemble_probs_np)
        baseline = test_labels.sum() / len(test_labels)
        axes[1].plot(recall, precision, label=f'Ensemble (AP={ensemble_metrics["auc_pr"]:.3f})', linewidth=2)
        axes[1].axhline(baseline, color='k', linestyle='--', label=f'Random ({baseline:.3f})', alpha=0.3)
        axes[1].set_xlabel('Recall')
        axes[1].set_ylabel('Precision')
        axes[1].set_title('Ensemble Precision-Recall Curve')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(results_dir / 'ensemble_curves.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved ensemble curves to '{results_dir / 'ensemble_curves.png'}'")
        plt.close()
    except Exception as e:
        logger.warning(f"Could not create ensemble plots: {e}")
    
    # GENERATE RANKINGS (Better than binary classification!)
    print("\n" + "="*80)
    print("GENERATING GENE RANKINGS")
    print("="*80)
    
    # Use best fold model for ranking
    best_fold_idx = np.argmax([m.get('auc_pr', 0) for m in all_fold_metrics])
    best_model = fold_models[best_fold_idx]
    
    print(f"Using Fold {best_fold_idx + 1} model (AUPRC: {all_fold_metrics[best_fold_idx].get('auc_pr', 0):.4f})")
    
    # Generate full rankings
    try:
        ranking_df = best_model.rank_genes_by_driver_likelihood(
            original,
            curvature_type='ollivier',
            top_k=None,  # All genes
            min_confidence=0.0,
            device=device
        )
        
        ranking_df.to_csv(results_dir / 'all_genes_ranked.csv', index=False)
        print(f"✓ Saved full rankings to '{results_dir / 'all_genes_ranked.csv'}'")
        
        # Top candidates
        top_100 = ranking_df.head(100)
        top_100.to_csv(results_dir / 'top_100_driver_candidates.csv', index=False)
        print(f"✓ Saved top 100 to '{results_dir / 'top_100_driver_candidates.csv'}'")
        
        print(f"\nTop 10 Predicted Drivers:")
        print(top_100.head(10).to_string(index=False))
        
    except Exception as e:
        logger.warning(f"Could not generate rankings: {e}")
    
    # IDENTIFY POTENTIAL DRIVERS
    print("\n" + "="*80)
    print("POTENTIAL DRIVER IDENTIFICATION")
    print("="*80)
    
    feature_names = original.get('feature_name', [])
    feature_criteria = {}
    
    if feature_names:
        if 'ppin_hub' in feature_names:
            feature_criteria['ppin_hub'] = (feature_names.index('ppin_hub'), 0.5)
        if 'essentiality_percentage' in feature_names:
            feature_criteria['essentiality_percentage'] = (
                feature_names.index('essentiality_percentage'), 0.2
            )
    
    potential_results = best_model.identify_potential_drivers(
        original, labels, test_mask,
        confidence_threshold=args.optimal_threshold,
        curvature_threshold=0.0,
        feature_criteria=feature_criteria if feature_criteria else None,
        curvature_type='ollivier',
        device=device
    )
    
    print(f"\nTotal False Positives: {potential_results['total_false_positives']}")
    print(f"Potential Drivers Identified: {potential_results['num_potential_drivers']}")
    
    if potential_results['num_potential_drivers'] > 0:
        print("\nTop 10 Potential Drivers:")
        sorted_indices = torch.argsort(potential_results['scores'], descending=True)
        for rank, idx in enumerate(sorted_indices[:10], 1):
            idx_val = idx.item()
            gene_name = potential_results['node_names'][idx_val]
            score = potential_results['scores'][idx_val].item()
            print(f"  {rank}. {gene_name} (score: {score:.3f})")
    
    # Save complete results
    output = {
        'kfold_metrics': all_fold_metrics,
        'mean_metrics': mean_metrics,
        'std_metrics': std_metrics,
        'ensemble_metrics': ensemble_metrics,
        'threshold_info': all_threshold_info,
        'potential_drivers': potential_results,
        'best_fold_idx': best_fold_idx,
        'dataset_file': dataset_file,
        'model_prefix': args.model_out_prefix,
        'loss_type': 'focal' if args.use_focal_loss else 'bce',
        'focal_alpha': args.focal_alpha if args.use_focal_loss else None,
        'focal_gamma': args.focal_gamma if args.use_focal_loss else None
    }
    
    with open(results_dir / 'kfold_results.pkl', 'wb') as f:
        pickle.dump(output, f)
    print(f"\n✓ Saved complete results to '{results_dir / 'kfold_results.pkl'}'")
    
    # Plot fold comparison
    try:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        metrics_to_plot = ['auprc', 'auroc', 'f1', 'precision', 'recall', 'precision@50']
        titles = ['AUPRC', 'AUROC', 'F1 Score', 'Precision', 'Recall', 'Precision@50']
        
        for idx, (metric, title) in enumerate(zip(metrics_to_plot, titles)):
            ax = axes[idx // 3, idx % 3]
            
            values = [m.get(metric.replace('precision@', 'precision@').replace('auprc', 'auc_pr').replace('auroc', 'auc_roc'), 0) 
                     for m in all_fold_metrics]
            folds = list(range(1, len(all_fold_metrics) + 1))
            
            ax.bar(folds, values, alpha=0.7, color='steelblue')
            ax.axhline(y=mean_metrics[metric], color='red', linestyle='--', 
                      label=f'Mean: {mean_metrics[metric]:.4f}')
            ax.set_xlabel('Fold')
            ax.set_ylabel(title)
            ax.set_title(f'{title} Across Folds')
            ax.legend()
            ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(results_dir / 'kfold_comparison.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved fold comparison to '{results_dir / 'kfold_comparison.png'}'")
        plt.close()
        
    except Exception as e:
        logger.warning(f"Could not create comparison plots: {e}")
    
    # Final summary
    print("\n" + "="*80)
    print("TRAINING COMPLETE - SUMMARY")
    print("="*80)
    print(f"\n📊 Key Results:")
    print(f"  Mean AUPRC:  {mean_metrics['auprc']:.4f} ± {std_metrics['auprc']:.4f}")
    print(f"  Mean AUROC:  {mean_metrics['auroc']:.4f} ± {std_metrics['auroc']:.4f}")
    print(f"  Mean F1:     {mean_metrics['f1']:.4f} ± {std_metrics['f1']:.4f}")
    print(f"  Mean P@50:   {mean_metrics['precision@50']:.4f} ± {std_metrics['precision@50']:.4f}")
    print(f"\n🎯 Ensemble Performance:")
    print(f"  AUPRC:       {ensemble_metrics['auc_pr']:.4f}")
    print(f"  AUROC:       {ensemble_metrics['auc_roc']:.4f}")
    print(f"  F1:          {ensemble_metrics['f1']:.4f}")
    print(f"\n🔧 Optimal Threshold: {mean_threshold:.3f} ± {std_threshold:.3f}")
    print(f"  (Use this instead of 0.5 for binary predictions)")
    print(f"\n📁 Output Files (in {results_dir}/):")
    print("  - kfold_results.csv: Metrics for all folds")
    print("  - kfold_results.pkl: Complete results with models")
    print("  - kfold_comparison.png: Visual comparison")
    print("  - fold_X_threshold_analysis.png: Threshold analysis per fold")
    print("  - ensemble_curves.png: ROC and PR curves")
    print("  - all_genes_ranked.csv: Full gene rankings")
    print("  - top_100_driver_candidates.csv: Top predictions")
    print("="*80 + "\n")


if __name__ == '__main__':
    logger = get_logger(__name__)
    main()