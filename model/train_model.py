import torch
import torch.nn as nn
from typing import Tuple, List, Optional, Dict
import pandas as pd
import gc
import numpy as np
from torch.utils.checkpoint import checkpoint

from utils.logging_manager import get_logger
from model.DriverGenePredictor import ContrastiveDriverGenePredictor
from model.support_models import WarmupScheduler, EarlyStopping

# At the beginning of your script, after imports
torch.cuda.empty_cache()
gc.collect()

# Enable memory efficient attention if using transformers
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

logger = get_logger(__name__)

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
        temperature=0.5,
        dropout=0.2,
        device=device
    ).to(device)

    logger.info(f"Created ContrastiveCancerDriverPredictor model:")
    logger.info(f"  - Input features: {num_features}")
    logger.info(f"  - Hidden channels: {hidden_channels}")
    logger.info(f"  - Device: {device}")
    logger.info(f"  - Binary classification: driver vs non-driver")
    logger.info(f"  - Potential driver identification from FPs")
    
    return model

def preprocess_curvature_data(data: Dict, curvature_type: str = 'ollivier') -> Dict:
    """
    Preprocess data to ensure curvature dimensions match edge_index.
    Call this once before training to fix all curvature issues upfront.
    
    Args:
        data: Dictionary containing 'edge_index' and curvature data
        curvature_type: Type of curvature to check ('ollivier' or 'forman')
    
    Returns:
        Updated data dictionary with properly sized curvatures
    """
    curv_key = f'{curvature_type}_curvature'
    
    if curv_key not in data:
        logger.warning(f"No '{curv_key}' found in data, skipping preprocessing")
        return data
    
    edge_index = data['edge_index']
    edge_curvature = data[curv_key]
    
    num_edges = edge_index.shape[1]
    num_curvatures = edge_curvature.shape[0]
    
    logger.info(f"Preprocessing curvature: {num_curvatures} curvatures for {num_edges} edges")
    
    if num_curvatures == num_edges:
        logger.info("Curvature dimensions already match, no preprocessing needed")
        return data
    
    if num_curvatures * 2 == num_edges:
        logger.info("Detected undirected curvature for directed edges, matching...")
        
        # Build edge mapping
        device = edge_curvature.device
        matched_curvature = torch.zeros(num_edges, device=device, dtype=edge_curvature.dtype)
        
        # Get unique edges
        edge_list = edge_index.t().cpu().numpy()
        edge_to_curv_idx = {}
        
        for src, dst in edge_list:
            src, dst = int(src), int(dst)
            canonical = (min(src, dst), max(src, dst))
            if canonical not in edge_to_curv_idx:
                edge_to_curv_idx[canonical] = len(edge_to_curv_idx)
        
        # Sort for consistent mapping
        sorted_edges = sorted(edge_to_curv_idx.keys())
        edge_to_curv_idx = {edge: i for i, edge in enumerate(sorted_edges)}
        
        # Match curvatures
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
                matched_curvature[i] = edge_curvature.mean()
        
        data[curv_key] = matched_curvature
        logger.info(f"Best-effort matching complete")
    
    return data


def create_compatible_mask(original_mask: torch.Tensor, target_size: int, eliminated_ids: List[int]) -> torch.Tensor:
    """
    DEPRECATED - Model handles mask mapping internally
    This function is kept for reference but should not be used
    """
    eliminated_set = set(eliminated_ids)
    aug_mask = torch.zeros(target_size, dtype=torch.bool, device=original_mask.device)
    
    aug_idx = 0
    for orig_idx in range(len(original_mask)):
        if orig_idx not in eliminated_set:
            if orig_idx < len(original_mask) and original_mask[orig_idx]:
                aug_mask[aug_idx] = True
            aug_idx += 1
    
    return aug_mask


def train_single_fold(
    fold_idx: int,
    fold_data: Dict,
    original: Dict,
    augmented_views: List[Dict],
    labels: torch.Tensor,
    num_epochs: int,
    device: torch.device,
    use_focal_loss: bool = True,
    pos_weight: Optional[torch.Tensor] = None
) -> Tuple[ContrastiveDriverGenePredictor, Dict, Dict]:
    """
    Train model on a single fold
    
    Returns:
        model: Trained model
        best_metrics: Best validation metrics
        history: Training history
    """
    print(f"\n{'='*80}")
    print(f"TRAINING FOLD {fold_idx}")
    print(f"{'='*80}")
    
    # Extract masks for this fold from ORIGINAL graph
    train_mask_original = torch.from_numpy(fold_data['train_mask'])
    val_mask_original = torch.from_numpy(fold_data['val_mask'])
    
    # CRITICAL: Ensure masks match the size of the original graph (not augmented)
    num_original_nodes = original['feature'].shape[0]
    if len(train_mask_original) != num_original_nodes:
        raise ValueError(
            f"Train mask size ({len(train_mask_original)}) doesn't match "
            f"original graph nodes ({num_original_nodes})"
        )
    
    print(f"Train samples: {train_mask_original.sum().item()}")
    print(f"Val samples: {val_mask_original.sum().item()}")
    
    # Calculate fold-specific positive weight
    num_pos = labels[train_mask_original].sum().item()
    num_neg = (train_mask_original.sum() - num_pos).item()
    fold_pos_weight = torch.tensor([num_neg / num_pos], device=device)
    
    print(f"Fold {fold_idx} - Drivers: {num_pos}, Non-drivers: {num_neg}")
    print(f"Positive class weight: {fold_pos_weight.item():.2f}")
    
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
        mode='max', 
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
    
    # Early stopping
    early_stopping = EarlyStopping(patience=50, min_delta=0.0001, mode='max')
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_f1': [],
        'val_precision': [],
        'val_recall': [],
        'learning_rate': []
    }
    
    best_val_f1 = 0.0
    best_metrics = None
    
    # Training loop
    for epoch in range(num_epochs):
        # Warmup learning rate
        if epoch < 10:
            warmup_scheduler.step()
        
        # Sample two different augmented views for contrastive learning
        view_indices = torch.randperm(len(augmented_views))[:2]
        augmented_view1 = augmented_views[view_indices[0]]
        augmented_view2 = augmented_views[view_indices[1]]
        
        # Pass ORIGINAL-sized mask to train_step
        # The model's train_step will handle mapping it to augmented space internally
        try:
            loss_dict = model.train_step(
                augmented_view1,
                augmented_view2,
                original,
                labels,
                train_mask_original,  # Original size mask - model maps internally
                optimizer,
                contrastive_weight=0.3,
                pos_weight=fold_pos_weight if not use_focal_loss else None,
                curvature_type='ollivier',
                device=device,
                batch_size=2048,
                use_focal_loss=use_focal_loss,
                focal_alpha=0.25,
                focal_gamma=2.0
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
                    focal_alpha=0.25,
                    focal_gamma=2.0
                )
            else:
                raise e
        
        # Validation on original graph (use original masks)
        val_metrics = model.evaluate(
            original, labels, val_mask_original, 
            curvature_type='ollivier', device=device
        )
        
        # Update learning rate (after warmup)
        if epoch >= 10:
            scheduler.step(val_metrics['f1'])
        
        # Store history
        history['train_loss'].append(loss_dict['total_loss'])
        history['train_acc'].append(loss_dict['train_accuracy'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_f1'].append(val_metrics['f1'])
        history['val_precision'].append(val_metrics['precision'])
        history['val_recall'].append(val_metrics['recall'])
        history['learning_rate'].append(optimizer.param_groups[0]['lr'])
        
        # Logging
        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d} | "
                  f"Loss: {loss_dict['total_loss']:.4f} | "
                  f"Train Acc: {loss_dict['train_accuracy']:.4f} | "
                  f"Val F1: {val_metrics['f1']:.4f} | "
                  f"Val Prec: {val_metrics['precision']:.4f} | "
                  f"Val Rec: {val_metrics['recall']:.4f} | "
                  f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            best_metrics = val_metrics.copy()
            # Save to trained_models directory
            model_path = Path('trained_models') / f'fold_{fold_idx}_best_model.pt'
            model.save_checkpoint(
                str(model_path),
                epoch,
                optimizer,
                val_metrics,
                metadata={
                    'fold': fold_idx,
                    'num_views': len(augmented_views),
                    'loss_type': 'focal' if use_focal_loss else 'bce',
                    'pos_weight': fold_pos_weight.item()
                }
            )
        
        # Early stopping check
        if early_stopping(val_metrics['f1']):
            print(f"\nEarly stopping triggered at epoch {epoch}")
            print(f"Best validation F1: {best_val_f1:.4f}")
            break
    
    # Load best model for this fold
    model_path = Path('trained_models') / f'fold_{fold_idx}_best_model.pt'
    checkpoint = model.load_checkpoint(str(model_path), optimizer, device)
    print(f"\n✓ Loaded best model from epoch {checkpoint['epoch']}")
    print(f"  Best Val F1: {checkpoint['metrics']['f1']:.4f}")
    
    return model, best_metrics, history


if __name__ == "__main__":
    import pickle
    import os
    from pathlib import Path
    from sklearn.metrics import classification_report, roc_auc_score, roc_curve
    
    # Create directories for outputs
    models_dir = Path('trained_models')
    results_dir = Path('model_results')
    models_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.cuda.empty_cache()
    
    print("\n" + "="*80)
    print("K-FOLD CROSS-VALIDATION CONTRASTIVE DRIVER GENE PREDICTOR")
    print("="*80)
    print(f"Device: {device}")
    print(f"PyTorch version: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print("="*80 + "\n")
    
    # Load data
    try:
        with open('curvature_output/GGNet_contrastive_v2_priority_r0.2.pkl', 'rb') as f:
            data = pickle.load(f)
    except FileNotFoundError:
        logger.error("Data file not found. Please check the path.")
        exit(1)
    
    original = data['original']
    augmented_views = data['augmented_views']
    
    # IMPORTANT: Preprocess curvature data to fix dimension mismatches
    print("\n" + "="*80)
    print("PREPROCESSING CURVATURE DATA")
    print("="*80)
    
    logger.info("Preprocessing original graph curvature...")
    original = preprocess_curvature_data(original, curvature_type='ollivier')
    
    logger.info(f"Preprocessing {len(augmented_views)} augmented views...")
    
    # Check the structure of augmented views
    num_original_nodes = original['feature'].shape[0]
    logger.info(f"Original graph has {num_original_nodes} nodes")
    
    for i, view in enumerate(augmented_views):
        logger.info(f"  Processing view {i+1}/{len(augmented_views)}")
        
        # Preprocess curvature
        augmented_views[i] = preprocess_curvature_data(view, curvature_type='ollivier')
        
        # Check node count
        view_nodes = augmented_views[i]['x'].shape[0]
        eliminated_count = len(augmented_views[i]['metadata']['eliminated_node_ids'])
        
        logger.info(f"    View {i+1}: {view_nodes} nodes")
        logger.info(f"    Metadata says {eliminated_count} nodes were eliminated")
        
        # The augmented views from your pipeline appear to already be correctly sized
        # The eliminated_node_ids in metadata may refer to original IDs from a larger graph
        # Just verify the view is self-consistent
        
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
    
    # Binary labels: 0 (non-driver), 1 (known driver)
    labels = original['label']
    
    # Get k-fold splits
    kfold_splits = original['kfold_splits']
    num_folds = len(kfold_splits)
    
    print(f"\n{'='*80}")
    print(f"K-FOLD CROSS-VALIDATION: {num_folds} FOLDS")
    print(f"{'='*80}")
    
    # Training configuration
    num_epochs = 200
    use_focal_loss = True
    
    # Store results for all folds
    all_fold_metrics = []
    all_fold_histories = []
    fold_models = []
    
    # Train each fold
    for fold_idx, fold_data in enumerate(kfold_splits, 1):
        model, best_metrics, history = train_single_fold(
            fold_idx=fold_idx,
            fold_data=fold_data,
            original=original,
            augmented_views=augmented_views,
            labels=labels,
            num_epochs=num_epochs,
            device=device,
            use_focal_loss=use_focal_loss
        )
        
        all_fold_metrics.append(best_metrics)
        all_fold_histories.append(history)
        fold_models.append(model)
        
        # Clear memory after each fold
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    # Aggregate results across folds
    print("\n" + "="*80)
    print("K-FOLD CROSS-VALIDATION RESULTS")
    print("="*80)
    
    metrics_df_data = []
    for fold_idx, metrics in enumerate(all_fold_metrics, 1):
        print(f"\nFold {fold_idx}:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1 Score:  {metrics['f1']:.4f}")
        
        metrics_df_data.append({
            'Fold': fold_idx,
            'Accuracy': metrics['accuracy'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1': metrics['f1']
        })
    
    # Calculate mean and std
    mean_metrics = {
        'accuracy': np.mean([m['accuracy'] for m in all_fold_metrics]),
        'precision': np.mean([m['precision'] for m in all_fold_metrics]),
        'recall': np.mean([m['recall'] for m in all_fold_metrics]),
        'f1': np.mean([m['f1'] for m in all_fold_metrics])
    }
    
    std_metrics = {
        'accuracy': np.std([m['accuracy'] for m in all_fold_metrics]),
        'precision': np.std([m['precision'] for m in all_fold_metrics]),
        'recall': np.std([m['recall'] for m in all_fold_metrics]),
        'f1': np.std([m['f1'] for m in all_fold_metrics])
    }
    
    print(f"\n{'='*80}")
    print("MEAN ± STD ACROSS ALL FOLDS")
    print(f"{'='*80}")
    print(f"Accuracy:  {mean_metrics['accuracy']:.4f} ± {std_metrics['accuracy']:.4f}")
    print(f"Precision: {mean_metrics['precision']:.4f} ± {std_metrics['precision']:.4f}")
    print(f"Recall:    {mean_metrics['recall']:.4f} ± {std_metrics['recall']:.4f}")
    print(f"F1 Score:  {mean_metrics['f1']:.4f} ± {std_metrics['f1']:.4f}")
    
    # Save fold results
    try:
        import pandas as pd
        df_metrics = pd.DataFrame(metrics_df_data)
        
        # Add mean row
        df_metrics.loc[len(df_metrics)] = {
            'Fold': 'Mean',
            'Accuracy': mean_metrics['accuracy'],
            'Precision': mean_metrics['precision'],
            'Recall': mean_metrics['recall'],
            'F1': mean_metrics['f1']
        }
        
        # Add std row
        df_metrics.loc[len(df_metrics)] = {
            'Fold': 'Std',
            'Accuracy': std_metrics['accuracy'],
            'Precision': std_metrics['precision'],
            'Recall': std_metrics['recall'],
            'F1': std_metrics['f1']
        }
        
        df_metrics.to_csv(results_dir / 'kfold_results.csv', index=False)
        print(f"\n✓ Saved k-fold results to '{results_dir / 'kfold_results.csv'}'")
    except ImportError:
        logger.warning("pandas not available for CSV export")
    
    # ENSEMBLE EVALUATION
    print("\n" + "="*80)
    print("ENSEMBLE MODEL EVALUATION")
    print("="*80)
    print("Using all fold models for ensemble prediction...")
    
    # Use original mask for ensemble evaluation (or create test set from all data)
    test_mask = original['mask']
    
    # Ensemble predictions
    ensemble_probs, ensemble_std = ensemble_predict(
        fold_models,
        original,
        test_mask,
        curvature_type='ollivier',
        device=device
    )
    
    # Calculate ensemble metrics
    test_labels = labels[test_mask].cpu().numpy()
    ensemble_preds = (ensemble_probs > 0.5).cpu().numpy()
    
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    ensemble_metrics = {
        'accuracy': accuracy_score(test_labels, ensemble_preds),
        'precision': precision_score(test_labels, ensemble_preds),
        'recall': recall_score(test_labels, ensemble_preds),
        'f1': f1_score(test_labels, ensemble_preds)
    }
    
    print(f"\nEnsemble Performance:")
    print(f"  Accuracy:  {ensemble_metrics['accuracy']:.4f}")
    print(f"  Precision: {ensemble_metrics['precision']:.4f}")
    print(f"  Recall:    {ensemble_metrics['recall']:.4f}")
    print(f"  F1 Score:  {ensemble_metrics['f1']:.4f}")
    
    # ROC-AUC for ensemble
    ensemble_roc_auc = roc_auc_score(test_labels, ensemble_probs.cpu().numpy())
    print(f"  ROC-AUC:   {ensemble_roc_auc:.4f}")
    
    # Plot ensemble ROC curve
    try:
        import matplotlib.pyplot as plt
        
        fpr, tpr, _ = roc_curve(test_labels, ensemble_probs.cpu().numpy())
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'Ensemble ROC (AUC = {ensemble_roc_auc:.4f})', linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Ensemble Model ROC Curve')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(results_dir / 'ensemble_roc_curve.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved ensemble ROC curve to '{results_dir / 'ensemble_roc_curve.png'}'")
    except ImportError:
        logger.warning("matplotlib not available")
    
    # IDENTIFY POTENTIAL DRIVERS USING BEST FOLD MODEL
    print("\n" + "="*80)
    print("POTENTIAL DRIVER GENE IDENTIFICATION")
    print("="*80)
    
    # Use the fold with best F1 score
    best_fold_idx = np.argmax([m['f1'] for m in all_fold_metrics])
    best_model = fold_models[best_fold_idx]
    
    print(f"Using Fold {best_fold_idx + 1} model (F1: {all_fold_metrics[best_fold_idx]['f1']:.4f})")
    
    # Define feature criteria
    feature_names = original.get('feature_name', [])
    feature_criteria = {}
    
    if feature_names:
        if 'ppin_hub' in feature_names:
            feature_criteria['ppin_hub'] = (feature_names.index('ppin_hub'), 0.5)
        if 'essentiality_percentage' in feature_names:
            feature_criteria['essentiality_percentage'] = (
                feature_names.index('essentiality_percentage'), 0.2
            )
        if 'ppin_betweenness' in feature_names:
            feature_criteria['ppin_betweenness'] = (
                feature_names.index('ppin_betweenness'), 0.1
            )
    
    potential_results = best_model.identify_potential_drivers(
        original,
        labels,
        test_mask,
        confidence_threshold=0.6,
        curvature_threshold=0.0,
        feature_criteria=feature_criteria if feature_criteria else None,
        curvature_type='ollivier',
        device=device
    )
    
    print(f"\nTotal False Positives: {potential_results['total_false_positives']}")
    print(f"Potential Drivers Identified: {potential_results['num_potential_drivers']}")
    if potential_results['total_false_positives'] > 0:
        filtering_rate = (potential_results['num_potential_drivers'] / 
                         potential_results['total_false_positives'] * 100)
        print(f"Filtering Rate: {filtering_rate:.1f}% of FPs retained as potential drivers")
    
    # Display top potential drivers
    if potential_results['num_potential_drivers'] > 0:
        print("\n" + "-"*80)
        print("TOP POTENTIAL DRIVER GENES")
        print("-"*80)
        
        sorted_indices = torch.argsort(potential_results['scores'], descending=True)
        top_k = min(20, len(sorted_indices))
        
        for rank, idx in enumerate(sorted_indices[:top_k], 1):
            idx_val = idx.item()
            gene_idx = potential_results['potential_driver_indices'][idx_val]
            gene_name = (potential_results['node_names'][idx_val] 
                        if potential_results['node_names'] else f"Node_{gene_idx}")
            score = potential_results['scores'][idx_val].item()
            reason = potential_results['reasons'][idx_val]
            details = potential_results['detailed_features'][idx_val]
            
            print(f"\n{rank}. {gene_name} (Score: {score:.3f})")
            print(f"   Reason: {reason}")
            print(f"   Curvature: mean={details['curvature']['mean_curvature']:.3f}, "
                  f"pos={details['curvature']['positive_ratio']:.2f}, "
                  f"neg={details['curvature']['negative_ratio']:.2f}")
    
    # Save results
    output = {
        'kfold_metrics': all_fold_metrics,
        'mean_metrics': mean_metrics,
        'std_metrics': std_metrics,
        'ensemble_metrics': ensemble_metrics,
        'ensemble_roc_auc': ensemble_roc_auc,
        'potential_drivers': potential_results,
        'best_fold_idx': best_fold_idx
    }
    
    with open(results_dir / 'kfold_results.pkl', 'wb') as f:
        pickle.dump(output, f)
    print(f"\n✓ Saved complete results to '{results_dir / 'kfold_results.pkl'}'")
    
    # Plot fold comparison
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1']
        titles = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
        
        for idx, (metric, title) in enumerate(zip(metrics_to_plot, titles)):
            ax = axes[idx // 2, idx % 2]
            
            values = [m[metric] for m in all_fold_metrics]
            folds = list(range(1, num_folds + 1))
            
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
        
        # Plot training history for best fold
        best_history = all_fold_histories[best_fold_idx]
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Loss curves
        axes[0, 0].plot(best_history['train_loss'], label='Train Loss', alpha=0.8)
        axes[0, 0].plot(best_history['val_loss'], label='Val Loss', alpha=0.8)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title(f'Best Fold {best_fold_idx + 1}: Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)
        
        # Accuracy and F1
        axes[0, 1].plot(best_history['train_acc'], label='Train Accuracy', alpha=0.8)
        axes[0, 1].plot(best_history['val_f1'], label='Val F1', alpha=0.8)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Score')
        axes[0, 1].set_title(f'Best Fold {best_fold_idx + 1}: Accuracy and F1')
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)
        
        # Precision and Recall
        axes[1, 0].plot(best_history['val_precision'], label='Precision', alpha=0.8)
        axes[1, 0].plot(best_history['val_recall'], label='Recall', alpha=0.8)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].set_title(f'Best Fold {best_fold_idx + 1}: Precision and Recall')
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)
        
        # Learning rate
        axes[1, 1].plot(best_history['learning_rate'], alpha=0.8)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_title(f'Best Fold {best_fold_idx + 1}: Learning Rate Schedule')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(results_dir / 'best_fold_training_history.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved best fold training history to '{results_dir / 'best_fold_training_history.png'}'")
        
    except ImportError:
        logger.warning("matplotlib not available for plotting")
    
    # Save potential drivers to CSV
    if potential_results['num_potential_drivers'] > 0:
        try:
            import pandas as pd
            df_data = []
            sorted_indices = torch.argsort(potential_results['scores'], descending=True)
            top_k = min(50, len(sorted_indices))
            
            for i, idx_val in enumerate(sorted_indices[:top_k]):
                idx_val = idx_val.item()
                details = potential_results['detailed_features'][idx_val]
                row = {
                    'Rank': i + 1,
                    'Gene': potential_results['node_names'][idx_val],
                    'Confidence': potential_results['scores'][idx_val].item(),
                    'Mean_Curvature': details['curvature']['mean_curvature'],
                    'Positive_Ratio': details['curvature']['positive_ratio'],
                    'Negative_Ratio': details['curvature']['negative_ratio'],
                    'Reason': potential_results['reasons'][idx_val]
                }
                # Add node features
                row.update(details['node_features'])
                df_data.append(row)
            
            df = pd.DataFrame(df_data)
            df.to_csv(results_dir / 'kfold_potential_driver_genes.csv', index=False)
            print(f"✓ Saved potential drivers to '{results_dir / 'kfold_potential_driver_genes.csv'}'")
        except ImportError:
            logger.warning("pandas not available for CSV export")
    
    print("\n" + "="*80)
    print("K-FOLD CROSS-VALIDATION COMPLETE")
    print("="*80)
    print("\nGenerated files:")
    print(f"\nModels (in {models_dir}/):")
    print("  - fold_X_best_model.pt: Best model for each fold")
    print(f"\nResults (in {results_dir}/):")
    print("  - kfold_results.pkl: Complete k-fold results")
    print("  - kfold_results.csv: Fold metrics comparison")
    print("  - kfold_comparison.png: Visual comparison across folds")
    print("  - best_fold_training_history.png: Training curves for best fold")
    print("  - ensemble_roc_curve.png: Ensemble model ROC curve")
    print("  - kfold_potential_driver_genes.csv: Potential drivers")
    print("\nKey Results:")
    print(f"  ✓ Mean F1 Score: {mean_metrics['f1']:.4f} ± {std_metrics['f1']:.4f}")
    print(f"  ✓ Ensemble F1 Score: {ensemble_metrics['f1']:.4f}")
    print(f"  ✓ Ensemble ROC-AUC: {ensemble_roc_auc:.4f}")
    print(f"  ✓ Best Fold: {best_fold_idx + 1} (F1: {all_fold_metrics[best_fold_idx]['f1']:.4f})")
    print(f"  ✓ Potential Drivers: {potential_results['num_potential_drivers']}")
    print("="*80 + "\n")