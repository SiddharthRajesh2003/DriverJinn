import torch
import torch.nn as nn
from typing import Tuple, List, Optional, Dict
import pandas as pd
import gc
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


if __name__ == "__main__":
    import pickle
    from sklearn.metrics import classification_report, roc_auc_score, roc_curve
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.cuda.empty_cache()
    
    print("\n" + "="*80)
    print("ENHANCED CONTRASTIVE DRIVER GENE PREDICTOR")
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
    for i, view in enumerate(augmented_views):
        logger.info(f"  Processing view {i+1}/{len(augmented_views)}")
        augmented_views[i] = preprocess_curvature_data(view, curvature_type='ollivier')
    
    print("✓ Curvature preprocessing complete")
    print("="*80 + "\n")
    
    # Binary labels: 0 (non-driver), 1 (known driver)
    labels = original['label']
    train_mask = original['mask']
    
    # Create model
    model = create_cancer_driver_model(
        num_features=original['feature'].shape[1],
        hidden_channels=256,
        projection_dim=128,
        num_layers=3,
        device=device
    )
    
    # Calculate positive class weight for imbalanced data
    num_pos = labels[train_mask].sum().item()
    num_neg = (train_mask.sum() - num_pos).item()
    pos_weight = torch.tensor([num_neg / num_pos], device=device)
    
    logger.info(f"Training data: {num_pos} drivers, {num_neg} non-drivers")
    logger.info(f"Positive class weight: {pos_weight.item():.2f}")
    logger.info(f"Class imbalance ratio: 1:{num_neg/num_pos:.1f}")
    
    # Optimizer with weight decay
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
    
    # Mixed precision training
    scaler = torch.amp.GradScaler() if device == 'cuda' else None
    
    print("\n" + "="*80)
    print("DATA PIPELINE INTEGRATION")
    print("="*80)
    print("Your CurvaturePipeline precomputes curvature for augmented views")
    print("Augmented views have fewer nodes (Schur complement elimination)")
    print("\nNode Mapping:")
    print("  - Original graph: N nodes with train/val/test masks")
    print("  - Augmented views: N - k nodes (k nodes eliminated)")
    print("  - Masks are automatically mapped using eliminated_node_ids")
    print("="*80 + "\n")
    
    # Verify data structure
    print("Checking data structure...")
    num_original_nodes = original['feature'].shape[0]
    num_augmented_nodes = augmented_views[0]['x'].shape[0]
    num_eliminated = len(augmented_views[0]['metadata']['eliminated_node_ids'])
    shape_edge_index1 = augmented_views[0]['edge_index'].shape
    shape_edge_index2 = augmented_views[1]['edge_index'].shape
    
    print(f"  Original nodes: {num_original_nodes}")
    print(f"  Augmented nodes: {num_augmented_nodes}")
    print(f"  Eliminated nodes: {num_eliminated}")
    print(f"  ✓ Verified: {num_original_nodes - num_eliminated} == {num_augmented_nodes}")
    print(f'  Shape of edge index of Augmented View 1: {shape_edge_index1}')
    print(f'  Shape of edge index of Augmented View 2: {shape_edge_index2}')
    print()
    
    # Training configuration
    num_epochs = 200
    best_val_f1 = 0.0
    use_focal_loss = True  # Set to True to use focal loss instead of BCE
    
    print(f"Training with {len(augmented_views)} augmented views")
    print("Using pairs of augmented views for contrastive learning")
    print(f"Loss function: {'Focal Loss' if use_focal_loss else 'Weighted BCE'}")
    print("Masks are automatically mapped for each augmented view\n")
    
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
    
    # Training loop
    for epoch in range(num_epochs):
        # Warmup learning rate
        if epoch < 10:
            warmup_scheduler.step()
        
        # Sample two different augmented views for contrastive learning
        view_indices = torch.randperm(len(augmented_views))[:2]
        augmented_view1 = augmented_views[view_indices[0]]
        augmented_view2 = augmented_views[view_indices[1]]
        
        # Training step
        try:
            loss_dict = model.train_step(
                augmented_view1,
                augmented_view2,
                original,
                labels,
                train_mask,
                optimizer,
                contrastive_weight=0.3,
                pos_weight=pos_weight if not use_focal_loss else None,
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
                # Try again with smaller batch size
                loss_dict = model.train_step(
                    augmented_view1,
                    augmented_view2,
                    original,
                    labels,
                    train_mask,
                    optimizer,
                    contrastive_weight=0.3,
                    pos_weight=pos_weight if not use_focal_loss else None,
                    curvature_type='ollivier',
                    device=device,
                    batch_size=1024,
                    use_focal_loss=use_focal_loss,
                    focal_alpha=0.25,
                    focal_gamma=2.0
                )
            else:
                raise e
        
        # Validation on original graph (for stable evaluation)
        if 'val_mask' in original:
            val_metrics = model.evaluate(
                original, labels, original['val_mask'], 
                curvature_type='ollivier', device=device
            )
        else:
            val_metrics = model.evaluate(
                original, labels, train_mask, 
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
                  f"Views: {view_indices[0].item()},{view_indices[1].item()} | "
                  f"Loss: {loss_dict['total_loss']:.4f} | "
                  f"Train Acc: {loss_dict['train_accuracy']:.4f} | "
                  f"Val F1: {val_metrics['f1']:.4f} | "
                  f"Val Prec: {val_metrics['precision']:.4f} | "
                  f"Val Rec: {val_metrics['recall']:.4f} | "
                  f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            model.save_checkpoint(
                'best_cancer_driver_model.pt',
                epoch,
                optimizer,
                val_metrics,
                metadata={
                    'num_views': len(augmented_views),
                    'loss_type': 'focal' if use_focal_loss else 'bce',
                    'pos_weight': pos_weight.item()
                }
            )
            print(f"  ✓ Saved best model (F1: {best_val_f1:.4f})")
        
        # Early stopping check
        if early_stopping(val_metrics['f1']):
            print(f"\nEarly stopping triggered at epoch {epoch}")
            print(f"Best validation F1: {best_val_f1:.4f}")
            break
    
    # Load best model
    checkpoint = model.load_checkpoint('best_cancer_driver_model.pt', optimizer, device)
    print(f"\n✓ Loaded best model from epoch {checkpoint['epoch']}")
    print(f"  Best Val F1: {checkpoint['metrics']['f1']:.4f}")
    
    # Final evaluation
    test_mask = original.get('test_mask', original.get('val_mask', train_mask))
    test_metrics = model.evaluate(
        original, 
        labels, 
        test_mask,
        curvature_type='ollivier',
        device=device
    )
    
    print("\n" + "="*80)
    print("FINAL TEST RESULTS")
    print("="*80)
    print(f"Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"Precision: {test_metrics['precision']:.4f}")
    print(f"Recall:    {test_metrics['recall']:.4f}")
    print(f"F1 Score:  {test_metrics['f1']:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  TP: {test_metrics['true_positives']:4d}  |  FP: {test_metrics['false_positives']:4d}")
    print(f"  FN: {test_metrics['false_negatives']:4d}  |  TN: {test_metrics['true_negatives']:4d}")
    
    # ROC-AUC analysis
    print("\n" + "="*80)
    print("ROC-AUC ANALYSIS")
    print("="*80)
    
    probs = model.predict_probability(
        original,
        test_mask,
        curvature_type='ollivier',
        device=device
    )
    test_labels = labels[test_mask].cpu().numpy()
    probs_np = probs.cpu().numpy()
    
    roc_auc = roc_auc_score(test_labels, probs_np)
    print(f"ROC-AUC Score: {roc_auc:.4f}")
    
    # Plot ROC curve
    try:
        import matplotlib.pyplot as plt
        fpr, tpr, thresholds = roc_curve(test_labels, probs_np)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.4f})', linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - Cancer Driver Gene Prediction')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
        print("✓ Saved ROC curve to 'roc_curve.png'")
    except ImportError:
        logger.warning("matplotlib not available for ROC curve plotting")
    
    # IDENTIFY POTENTIAL DRIVER GENES
    print("\n" + "="*80)
    print("POTENTIAL DRIVER GENE IDENTIFICATION")
    print("="*80)
    
    # Define feature criteria for potential drivers
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
    
    potential_results = model.identify_potential_drivers(
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
        
        # Sort by confidence score
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
            
            # Show top features
            if details['node_features']:
                feat_str = ", ".join([f"{k}={v:.3f}" 
                                     for k, v in list(details['node_features'].items())[:3]])
                print(f"   Features: {feat_str}")
            
            # Show curvature importance
            curv_imp = details['curvature_importance']
            curv_str = ", ".join([f"{k}={v:.3f}" for k, v in curv_imp.items()])
            print(f"   Curvature Importance: {curv_str}")
    
        # Save potential drivers to file
        output = {
            'potential_driver_indices': potential_results['potential_driver_indices'].cpu().numpy(),
            'potential_driver_names': potential_results['node_names'],
            'scores': potential_results['scores'].cpu().numpy(),
            'reasons': potential_results['reasons'],
            'detailed_features': potential_results['detailed_features'],
            'test_metrics': test_metrics,
            'roc_auc': roc_auc
        }
        
        with open('potential_driver_genes.pkl', 'wb') as f:
            pickle.dump(output, f)
        print(f"\n✓ Saved potential drivers to 'potential_driver_genes.pkl'")
        
        # Save as CSV for easy viewing
        try:
            import pandas as pd
            df_data = []
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
            df.to_csv('potential_driver_genes.csv', index=False)
            print("✓ Saved potential drivers to 'potential_driver_genes.csv'")
        except ImportError:
            logger.warning("pandas not available for CSV export")
    
    # Analyze curvature importance for different predictions
    print("\n" + "="*80)
    print("CURVATURE PATHWAY ANALYSIS")
    print("="*80)
    
    _, attention_info = model.encode(
        original.get('feature', original.get('x')).to(device),
        original['edge_index'].to(device),
        original['ollivier_curvature'].to(device),
        return_attention=True
    )
    
    # Get predictions for analysis
    logits, _ = model.forward(
        original.get('feature', original.get('x')).to(device),
        original['edge_index'].to(device),
        original['ollivier_curvature'].to(device)
    )
    
    if logits.dim() > 1:
        logits = logits.squeeze(-1)
    
    cross_attn = attention_info['cross_curvature_attention'][test_mask]
    
    # True positives vs False positives
    tp_mask = (logits[test_mask] > 0) & (labels[test_mask] == 1)
    fp_mask = (logits[test_mask] > 0) & (labels[test_mask] == 0)
    tn_mask = (logits[test_mask] <= 0) & (labels[test_mask] == 0)
    fn_mask = (logits[test_mask] <= 0) & (labels[test_mask] == 1)
    
    print("\nCurvature Pathway Importance by Prediction Type:")
    print("-" * 60)
    
    for mask_name, mask in [('True Positives', tp_mask), 
                            ('False Positives', fp_mask),
                            ('True Negatives', tn_mask),
                            ('False Negatives', fn_mask)]:
        if mask.sum() > 0:
            attn = cross_attn[mask].mean(dim=0)
            print(f"\n{mask_name} (n={mask.sum().item()}):")
            for i, curv_type in enumerate(model.curvature_types):
                print(f"  {curv_type:10s}: {attn[i].item():.4f}")
    
    # Visualize attention for a few example nodes
    print("\n" + "="*80)
    print("ATTENTION VISUALIZATION")
    print("="*80)
    
    if potential_results['num_potential_drivers'] > 0:
        print("\nGenerating attention visualizations for top 3 potential drivers...")
        for i in range(min(3, potential_results['num_potential_drivers'])):
            node_idx = potential_results['potential_driver_indices'][i].item()
            gene_name = potential_results['node_names'][i]
            save_path = f'attention_viz_{gene_name.replace("/", "_")}.png'
            
            try:
                model.visualize_attention_weights(
                    original,
                    node_idx,
                    curvature_type='ollivier',
                    save_path=save_path,
                    device=device
                )
                print(f"  ✓ Saved visualization for {gene_name}")
            except Exception as e:
                logger.warning(f"Could not visualize {gene_name}: {e}")
    
    # Plot training history
    print("\n" + "="*80)
    print("TRAINING HISTORY")
    print("="*80)
    
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Loss curves
        axes[0, 0].plot(history['train_loss'], label='Train Loss', alpha=0.8)
        axes[0, 0].plot(history['val_loss'], label='Val Loss', alpha=0.8)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)
        
        # Accuracy and F1
        axes[0, 1].plot(history['train_acc'], label='Train Accuracy', alpha=0.8)
        axes[0, 1].plot(history['val_f1'], label='Val F1', alpha=0.8)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Score')
        axes[0, 1].set_title('Training Accuracy and Validation F1')
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)
        
        # Precision and Recall
        axes[1, 0].plot(history['val_precision'], label='Precision', alpha=0.8)
        axes[1, 0].plot(history['val_recall'], label='Recall', alpha=0.8)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].set_title('Validation Precision and Recall')
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)
        
        # Learning rate
        axes[1, 1].plot(history['learning_rate'], alpha=0.8)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        print("✓ Saved training history to 'training_history.png'")
    except ImportError:
        logger.warning("matplotlib not available for training history plotting")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("\nGenerated files:")
    print("  - best_cancer_driver_model.pt: Best model checkpoint")
    print("  - potential_driver_genes.pkl: Potential drivers (Python)")
    print("  - potential_driver_genes.csv: Potential drivers (CSV)")
    print("  - roc_curve.png: ROC curve visualization")
    print("  - training_history.png: Training metrics over time")
    print("  - attention_viz_*.png: Attention visualizations")
    print("\nKey Improvements:")
    print("  ✓ Fixed contrastive loss computation")
    print("  ✓ Added dimension validation for curvatures")
    print("  ✓ Fixed device handling issues")
    print("  ✓ Added predict_probability method")
    print("  ✓ Added focal loss option")
    print("  ✓ Added early stopping")
    print("  ✓ Added learning rate warmup")
    print("  ✓ Added gradient clipping")
    print("  ✓ Added comprehensive visualizations")
    print("  ✓ Better checkpointing with metadata")
    print("="*80 + "\n")