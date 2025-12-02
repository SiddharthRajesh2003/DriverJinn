import math
import torch
import torch.nn as nn
from typing import Tuple, List, Optional, Dict
import pandas as pd
import gc
import numpy as np
import pickle
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.checkpoint import checkpoint as gradient_checkpoint

from utils.logging_manager import get_logger
from model.DriverGenePredictor import ContrastiveDriverGenePredictor, compute_ndcg
from model.support_models import WarmupScheduler, EarlyStopping, RankingLoss

# Memory optimization
torch.cuda.empty_cache()
gc.collect()

# Enable efficient operations
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def setup_multi_gpu(
    model,
    use_multi_gpu: bool = False,
    gpu_ids: List = None
):
    """Setup model for multi-GPU training
    
    Args:
        model: PyTorch model
        use_multi_gpu: Whether to use multiple GPUs
        gpu_ids: Specific GPU IDs (None = all available)
    
    Returns:
        model: Wrapped model
        device: Primary device
        is_multi_gpu: Whether multi-GPU is enabled
    """
    
    if not torch.cuda.is_available():
        print('CUDA not available, using CPU')
        return model, torch.device('cpu'), False
    
    num_gpus = torch.cuda.device_count()
    print('\nGPU Configuration:')
    print(f'    Available GPUs: {num_gpus}')
    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        mem_gb = props.total_memory / 1024**3
        print(f"  GPU {i}: {props.name} ({mem_gb:.1f} GB)")
        
    # Single GPU mode
    if not use_multi_gpu or num_gpus <= 1:
        device = torch.device('cuda:0')
        model = model.to(device)
        print(f"\n✓ Using single GPU: cuda:0")
        return model, device, False
    
    # Multi-GPU Mode
    if gpu_ids is None:
        gpu_ids = list(range(num_gpus))
        
    print(f"\n✓ Using DataParallel on {len(gpu_ids)} GPUs: {gpu_ids}")
    
    model = nn.DataParallel(model, device_ids=gpu_ids)
    primary_device = torch.device(f'cuda:{gpu_ids[0]}')
    model = model.to(primary_device)
    
    print(f"  Primary device: cuda:{gpu_ids[0]}")
    print(f"  Batch will be split across {len(gpu_ids)} GPUs")
    
    return model, primary_device, True


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
    
    nan_mask = torch.isnan(edge_curvature)
    inf_mask = torch.isinf(edge_curvature)
    
    if nan_mask.any() or inf_mask.any():
        logger.warning(f"{name}: Found {nan_mask.sum().item()} NaN and {inf_mask.sum().item()} Inf values")
        edge_curvature = edge_curvature.clone()
        edge_curvature[nan_mask] = 0.0
        edge_curvature[inf_mask] = 0.0
        logger.warning(f"{name}: Replaced problematic values with 0")
    
    edge_curvature = torch.clamp(edge_curvature, min=-10.0, max=10.0)
    return edge_curvature

def validate_graph_data(data, name="graph", strict=False):
    """Validate graph data structure for NaN/Inf values"""
    issues = []
    
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
    
    edge_index = data.get('edge_index')
    if edge_index is not None:
        num_nodes = features.shape[0] if features is not None else edge_index.max().item() + 1
        if edge_index.min() < 0:
            issues.append(f"{name}: Edge index contains negative values")
        if edge_index.max() >= num_nodes:
            issues.append(f"{name}: Edge index exceeds number of nodes")
    
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

def preprocess_curvature_data(data: Dict, curvature_type: str = 'ollivier') -> Dict:
    """Preprocess data to ensure curvature dimensions match edge_index with NaN/Inf sanitization"""
    curv_key = f'{curvature_type}_curvature'
    
    if curv_key not in data:
        logger.warning(f"No '{curv_key}' found in data, skipping preprocessing")
        return data
    
    edge_index = data['edge_index']
    edge_curvature = data[curv_key]
    
    # Sanitize curvature values
    logger.info(f"Sanitizing {curv_key} values...")
    edge_curvature = sanitize_curvature(edge_curvature, name=curv_key)
    
    num_edges = edge_index.shape[1]
    num_curvatures = edge_curvature.shape[0]
    
    logger.info(f"Preprocessing curvature: {num_curvatures} curvatures for {num_edges} edges")
    
    if num_curvatures == num_edges:
        logger.info("Curvature dimensions already match")
        data[curv_key] = edge_curvature
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
                matched_curvature[i] = 0.0
        
        data[curv_key] = matched_curvature
    
    # Final sanitization
    data[curv_key] = sanitize_curvature(data[curv_key], name=f"{curv_key}_matched")
    
    return data


def evaluate_with_ranking_metrics(
    model: nn.Module,
    data: Dict,
    labels: torch.Tensor,
    mask: torch.Tensor,
    curvature_type: str = 'ollivier',
    device: torch.device = None
) -> Dict[str, float]:
    """Evaluate model using ranking metrics (AUROC, AUPRC, NDCG, MRR, Precision@K)"""
    model.eval()
    
    try:
        with torch.no_grad():
            metrics = model.evaluate(  # <-- This should return a dict
                data, labels, mask,
                curvature_type=curvature_type,
                device=device,
                k_values=[10, 20, 50, 100]
            )
        
        # Ensure metrics is a dict
        if not isinstance(metrics, dict):
            logger.error(f"model.evaluate() returned {type(metrics)}, expected dict")
            return {
                'auroc': 0.0,
                'auprc': 0.0,
                'ndcg@50': 0.0,
                'precision@50': 0.0,
                'mrr': 0.0
            }
        
        # Check for NaN in metrics
        for key, value in metrics.items():
            if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
                logger.warning(f"Invalid metric {key}: {value}, setting to 0")
                metrics[key] = 0.0
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error in evaluate_with_ranking_metrics: {e}")
        import traceback
        logger.error(traceback.format_exc())
        
        return {
            'auroc': 0.0,
            'auprc': 0.0,
            'ndcg@10': 0.0,
            'ndcg@50': 0.0,
            'precision@10': 0.0,
            'precision@50': 0.0,
            'mrr': 0.0
        }

def create_cancer_driver_model(
    num_features: int = 74,
    hidden_channels: int = 256,
    projection_dim: int = 128,
    num_layers: int = 3,
    temperature: float = 0.4,
    attention_mode: str = 'hybrid',
    pathway_aggregator: str = 'hierarchical',
    num_heads: int = 4,
    dropout: float = 0.2,
    concat_heads: bool = True,
    device: torch.device = None
) -> ContrastiveDriverGenePredictor:
    """Factory function to create cancer driver prediction model"""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = ContrastiveDriverGenePredictor(
        in_channels=num_features,
        hidden_channels=hidden_channels,
        projection_dim=projection_dim,
        num_gnn_layers=num_layers,
        curvature_types=['positive', 'negative', 'both'],
        hop_types=['one_hop', 'two_hop'],
        num_attention_heads=num_heads,
        use_attention=True,
        temperature=temperature,
        attention_mode=attention_mode,
        pathway_aggregator=pathway_aggregator,
        concat=concat_heads,
        dropout=dropout,
        device=device
    ).to(device)

    logger.info(f"Created ContrastiveDriverGenePredictor model:")
    logger.info(f"  - Input features: {num_features}")
    logger.info(f"  - Hidden channels: {hidden_channels}")
    logger.info(f"  - Attention mode: {attention_mode}")
    logger.info(f"  - Attention heads: {num_heads} (concat={concat_heads})")
    logger.info(f"  - Device: {device}")
    logger.info(f"  - Training objective: Ranking (not binary classification)")
    
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
    attention_mode: str = 'hybrid',
    pathway_aggregator: str = 'hierarchical',
    num_heads: int = 4,
    num_layers: int = 3,
    temperature:float = 0.4,
    dropout: float = 0.2,
    concat_heads: bool = True,
    ranking_loss_type: str = 'pairwise',
    ranking_margin: float = 1.0,
    gradient_accumulation_steps: int = 4,
    mixed_precision: bool = True,
    reduce_model_size: bool = True,
    use_multi_gpu: bool = False,
    gpu_ids: Optional[List] = None
) -> Tuple[ContrastiveDriverGenePredictor, Dict, Dict]:
    
    """Train model on a single fold with ranking objective"""
    print(f"\n{'='*80}")
    print(f"TRAINING FOLD {fold_idx}")
    print(f"{'='*80}")
    
    # Extract masks
    train_mask_original = torch.from_numpy(fold_data['train_mask'])
    val_mask_original = torch.from_numpy(fold_data['val_mask'])
    
    # Get features - handle both 'feature' and 'x' keys
    features = original.get('feature', original.get('x'))
    if features is None:
        raise ValueError("No 'feature' or 'x' key found in original data")
    
    # Extract node names using node_id_to_name mapping from augmented_views
    num_nodes = features.shape[0]
    
    # Get node_id_to_name mapping from first augmented view (since it's not in original)
    if augmented_views and len(augmented_views) > 0 and 'metadata' in augmented_views[0]:
        node_id_to_name_aug = augmented_views[0]['metadata'].get('node_id_to_name', {})
        eliminated_node_id_to_name = augmented_views[0]['metadata'].get('eliminated_node_id_to_name', {})
        
        if node_id_to_name_aug or eliminated_node_id_to_name:
            # Combine augmented and eliminated mappings to reconstruct original full mapping
            node_id_to_name_full = {**node_id_to_name_aug, **eliminated_node_id_to_name}
            
            # Create array of names in order of node IDs
            # The dict maps node_id -> np.str_('GENE_NAME')
            node_names = np.array([str(node_id_to_name_full.get(i, f"Gene_{i}")) for i in range(num_nodes)])
            
            logger.info(f"Reconstructed full node mapping: {len(node_id_to_name_aug)} remaining + "
                    f"{len(eliminated_node_id_to_name)} eliminated = {len(node_id_to_name_full)} total")
            
            # Log eliminated nodes information
            if eliminated_node_id_to_name:
                logger.info(f"Eliminated nodes during augmentation:")
                # Show first few eliminated nodes as examples
                sample_eliminated = list(eliminated_node_id_to_name.items())[:5]
                for node_id, gene_name in sample_eliminated:
                    logger.info(f"  Node {node_id}: {gene_name}")
                if len(eliminated_node_id_to_name) > 5:
                    logger.info(f"  ... and {len(eliminated_node_id_to_name) - 5} more")
            else:
                logger.warning("eliminated_node_id_to_name mapping is empty in augmented_views metadata")
        else:
            logger.warning("Both node_id_to_name and eliminated_node_id_to_name mappings are empty")
            node_names = np.array([f"Gene_{i}" for i in range(num_nodes)])
    else:
        logger.warning("No augmented_views metadata found")
        node_names = np.array([f"Gene_{i}" for i in range(num_nodes)])
    
    num_original_nodes = features.shape[0]
    if len(train_mask_original) != num_original_nodes:
        raise ValueError(
            f"Train mask size ({len(train_mask_original)}) doesn't match "
            f"original graph nodes ({num_original_nodes})"
        )
    
    # Analyze class distribution
    train_labels = labels[train_mask_original]
    num_pos = (train_labels == 1).sum().item()
    num_neg = (train_labels == 0).sum().item()
    imbalance_ratio = num_neg / num_pos if num_pos > 0 else float('inf')
    
    print(f"\nClass Distribution:")
    print(f"  Training samples: {len(train_labels)}")
    print(f"  Positive (drivers): {num_pos} ({100*num_pos/len(train_labels):.2f}%)")
    print(f"  Negative: {num_neg} ({100*num_neg/len(train_labels):.2f}%)")
    print(f"  Imbalance ratio: 1:{imbalance_ratio:.1f}")
    print(f"  Val samples: {val_mask_original.sum().item()}")
    
    # EMERGENCY: Reduce model size if still OOM
    if reduce_model_size:
        logger.warning("Using reduced model size due to memory constraints")
        hidden_channels = 64
        projection_dim = 32
        num_layers = 2
        num_heads = 1
        concat_heads = False
    else:
        hidden_channels = 256
        projection_dim = 128
        num_layers = num_layers
    
    # Create model with reduced size
    model = create_cancer_driver_model(
        num_features=features.shape[1],  # Use features variable instead
        hidden_channels=hidden_channels,
        projection_dim=projection_dim,
        num_layers=num_layers,
        attention_mode=attention_mode,
        temperature=temperature,
        dropout=dropout,
        pathway_aggregator=pathway_aggregator,
        num_heads=num_heads,
        concat_heads=concat_heads,
        device=device
    )
    
    # Setup multi-GPU before moving to device
    model, primary_device, is_multi_gpu = setup_multi_gpu(
        model=model,
        use_multi_gpu=use_multi_gpu,
        gpu_ids=gpu_ids
    )
    
    # Update device to primary device for multi-GPU
    if is_multi_gpu:
        device = primary_device
    
    actual_model = model.module if is_multi_gpu else model
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.001,
        weight_decay=1e-5,
        betas=(0.9, 0.999)
    )
    
    # Learning rate scheduler (maximize NDCG)
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
    
    # Early stopping based on NDCG@50
    early_stopping = EarlyStopping(patience=50, min_delta=0.0001, mode='max')
    
    # Mixed precision scaler
    scaler = torch.amp.GradScaler('cuda') if mixed_precision and torch.cuda.is_available() else None
    
    # Pre-process original data ONCE and keep on device
    logger.info("Pre-processing original graph data...")
    original_device = {
        'x': features.to(device),  # Use features variable
        'edge_index': original['edge_index'].to(device),
        'ollivier_curvature': original['ollivier_curvature'].to(device),
    }
    
    # Don't delete from original - other folds need it!
    # Just clear CUDA cache
    gc.collect()
    torch.cuda.empty_cache()
    
    labels = labels.to(device)
    train_mask_original = train_mask_original.to(device)
    val_mask_original = val_mask_original.to(device)
    
    # Training history
    history = {
        'train_loss': [],
        'train_ndcg': [],
        'val_auroc': [],
        'val_auprc': [],
        'val_f1': [],
        'val_ndcg@50': [],
        'val_precision@50': [],
        'val_mrr': [],
        'learning_rate': []
    }
    
    best_val_ndcg = 0.0
    best_metrics = None
    
    # Construct model checkpoint path in a subfolder named after model_prefix
    if model_prefix:
        checkpoint_dir = model_save_dir / model_prefix
        checkpoint_dir.mkdir(exist_ok=True, parents=True)
        model_filename = f'fold_{fold_idx}_best_model.pt'
        model_path = checkpoint_dir / model_filename
    else:
        model_path = model_save_dir / f'fold_{fold_idx}_best_model.pt'
    
    print(f"\n{'='*80}")
    print("TRAINING WITH RANKING OBJECTIVE")
    print(f"{'='*80}")
    print(f"Loss type: Contrastive + {ranking_loss_type.capitalize()} Ranking")
    print(f"Ranking margin: {ranking_margin}")
    print(f"Early stopping metric: NDCG@50")
    print(f"Attention mode: {attention_mode}")
    print(f"{'='*80}\n")
    
    for epoch in range(num_epochs):
        # Clear cache at start of epoch
        torch.cuda.empty_cache()
        gc.collect()
        
        if epoch < 10:
            warmup_scheduler.step()
        
        # Zero gradients once
        optimizer.zero_grad()
        
        accumulated_loss = 0.0
        accumulated_metrics = {'train_ndcg': 0.0}
        successful_steps = 0
        
        # Process one view at a time with smaller chunks
        for accum_step in range(gradient_accumulation_steps):
            try:
                # Sample ONE view per step to minimize memory
                view_idx = torch.randint(0, len(augmented_views), (1,)).item()
                view = augmented_views[view_idx]
                
                # Move view to device ONLY when needed
                view_x = view['x'].to(device)
                view_edge_index = view['edge_index'].to(device)
                view_curvature = view.get('ollivier_curvature')
                
                if view_curvature is not None:
                    view_curvature = view_curvature.to(device)
                else:
                    # Compute curvature on-the-fly
                    view_curvature = actual_model.compute_augmented_curvature(
                        view_edge_index,
                        None,
                        view_x,
                        original_curvature=original_device['ollivier_curvature'],
                        original_edge_index=original_device['edge_index'],
                        method='transfer'
                    )
                
                # Validate curvature
                view_curvature = actual_model.validate_and_fix_curvature_dimensions(
                    view_edge_index, view_curvature, f"View{accum_step}"
                )
                
                # Map labels/masks
                eliminated_ids = set(view['metadata']['eliminated_node_ids'])
                aug_size = view_x.shape[0]
                num_original = labels.shape[0]
                
                orig_to_aug = {}
                aug_idx = 0
                for orig_idx in range(num_original):
                    if orig_idx not in eliminated_ids:
                        orig_to_aug[orig_idx] = aug_idx
                        aug_idx += 1
                
                aug_labels = torch.full((aug_size,), -100, dtype=torch.long, device=device)
                aug_mask = torch.zeros(aug_size, dtype=torch.bool, device=device)
                
                for orig_idx in range(num_original):
                    if orig_idx in orig_to_aug:
                        aug_idx = orig_to_aug[orig_idx]
                        if aug_idx < aug_size:
                            aug_labels[aug_idx] = labels[orig_idx]
                            aug_mask[aug_idx] = train_mask_original[orig_idx]
                
                if aug_mask.sum() == 0:
                    logger.warning(f"Step {accum_step}: No training samples after mapping")
                    continue
                
                # Forward pass with mixed precision
                with torch.amp.autocast('cuda') if scaler else torch.no_grad():
                    # Get scores
                    scores= gradient_checkpoint(
                        lambda x, e, c: model.forward(x, e, c)[0],  # Only need scores
                        view_x,
                        view_edge_index,
                        view_curvature,
                        use_reentrant=False
                    )
                    
                    # Compute ranking loss
                    ranking_criterion = RankingLoss(
                        margin=ranking_margin, 
                        loss_type=ranking_loss_type
                    )
                    loss = ranking_criterion(scores, aug_labels, aug_mask)
                    
                    # Scale for accumulation
                    loss = loss / gradient_accumulation_steps
                
                # Backward
                if scaler:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                accumulated_loss += loss.item() * gradient_accumulation_steps
                successful_steps += 1
                
                if successful_steps > 0:
                    # Compute training NDCG periodically
                    if epoch % 5 == 0:  # Don't compute every epoch to save time
                        with torch.no_grad():
                            train_scores, _ = model.forward(
                                original_device['x'], 
                                original_device['edge_index'], 
                                original_device['ollivier_curvature']
                            )
                            train_ndcg = compute_ndcg(train_scores[train_mask_original], 
                                                    labels[train_mask_original], k=50)
                            
                        accumulated_metrics['train_ndcg'] = train_ndcg
                    else:
                        # Reuse previous value
                        accumulated_metrics['train_ndcg'] = history['train_ndcg'][-1] if history['train_ndcg'] else 0.0
                
                # Clean up immediately
                del view_x, view_edge_index, view_curvature
                del aug_labels, aug_mask, scores, loss
                torch.cuda.empty_cache()
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    logger.warning(f"OOM at epoch {epoch}, step {accum_step}")
                    torch.cuda.empty_cache()
                    gc.collect()
                    continue
                else:
                    raise e
        
        # Check if we had any successful steps
        if successful_steps == 0:
            logger.error(f"Epoch {epoch}: All steps failed with OOM. Skipping epoch.")
            optimizer.zero_grad()
            torch.cuda.empty_cache()
            continue
        
        # Average loss
        avg_loss = accumulated_loss / successful_steps
        
        # Check for NaN
        if math.isnan(avg_loss):
            logger.error(f"Epoch {epoch}: NaN loss")
            optimizer.zero_grad()
            continue
        
        # Gradient clipping
        if scaler is not None:
            scaler.unscale_(optimizer)
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Optimizer step
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        
        optimizer.zero_grad()
        torch.cuda.empty_cache()
        
        # Replace your validation and early stopping section with this:

        # Validation (less frequently to save time)
        if epoch % 5 == 0 or epoch == num_epochs - 1:
            val_metrics = evaluate_with_ranking_metrics(
                actual_model, original_device, labels, val_mask_original,
                curvature_type='ollivier', device=device
            )
            
            # SAFETY CHECK: Ensure val_metrics is a dict
            if not isinstance(val_metrics, dict):
                logger.error(f"val_metrics is {type(val_metrics)}, expected dict. Creating empty dict.")
                val_metrics = {
                    'auroc': 0.0,
                    'auprc': 0.0,
                    'f1': 0.0,
                    'ndcg@50': 0.0,
                    'precision@50': 0.0,
                    'mrr': 0.0
                }
            
            # Update scheduler ONLY when we have fresh validation metrics
            if epoch >= 10:
                scheduler.step(val_metrics.get('ndcg@50', 0))
            
            # Check early stopping ONLY when we have fresh validation metrics
            current_ndcg = val_metrics.get('ndcg@50', 0)
            if early_stopping(current_ndcg):
                logger.info(f"Early stopping triggered at epoch {epoch}")
                logger.info(f"Best NDCG@50: {early_stopping.best_score:.4f} at epoch {early_stopping.best_epoch}")
                break
            
        else:
            # Skip validation - reuse previous metrics as dict
            if history['val_ndcg@50']:
                val_metrics = {
                    'ndcg@50': history['val_ndcg@50'][-1],
                    'auroc': history['val_auroc'][-1] if history['val_auroc'] else 0,
                    'auprc': history['val_auprc'][-1] if history['val_auprc'] else 0,
                    'f1': history['val_f1'][-1] if history['val_f1'] else 0,
                    'precision@50': history['val_precision@50'][-1] if history['val_precision@50'] else 0,
                    'mrr': history['val_mrr'][-1] if history['val_mrr'] else 0
                }
            else:
                val_metrics = {'ndcg@50': 0, 'auroc': 0, 'auprc': 0, 'f1': 0, 'precision@50': 0, 'mrr': 0}
            
            # DO NOT check early stopping on reused metrics
        
        # Store history
        history['train_loss'].append(avg_loss)
        history['val_auroc'].append(val_metrics.get('auroc', 0))
        history['val_auprc'].append(val_metrics.get('auprc', 0))
        history['val_f1'].append(val_metrics.get('f1', 0))
        history['val_ndcg@50'].append(val_metrics.get('ndcg@50', 0))
        history['val_precision@50'].append(val_metrics.get('precision@50', 0))
        history['val_mrr'].append(val_metrics.get('mrr', 0))
        history['learning_rate'].append(optimizer.param_groups[0]['lr'])
        
        # Logging
        if epoch % 10 == 0:
            print(
                f"Epoch {epoch:3d} | "
                f"Loss: {avg_loss:.4f} | "
                f"Steps: {successful_steps}/{gradient_accumulation_steps} | "
                f"NDCG@50: {val_metrics.get('ndcg@50', 0):.4f} | "
                f"LR: {optimizer.param_groups[0]['lr']:.6f}"
            )
        
        # Save best model
        if val_metrics.get('ndcg@50', 0) > best_val_ndcg:
            best_val_ndcg = val_metrics.get('ndcg@50', 0)
            best_metrics = val_metrics
            
            if is_multi_gpu:
                torch.save(
                    {
                        'epoch': epoch,
                        'model_state_dict': model.module.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'best_metrics': best_metrics,
                        'metadata': {'fold': fold_idx}
                    },
                    model_path
                )
            else:
                actual_model.save_checkpoint(
                    str(model_path), epoch, optimizer, best_metrics,
                    metadata={'fold': fold_idx}
                )
        
        # Early stopping
        if early_stopping(val_metrics.get('ndcg@50', 0)):
            logger.info(f"Early stopping at epoch {epoch}")
            break
    
    # Load best model
    # Load best model - FIX 2: Handle multi-GPU loading
    if is_multi_gpu:
        checkpoint = torch.load(model_path, map_location=device)
        # Load into the underlying model (module)
        model.module.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        best_metrics = checkpoint.get('best_metrics', best_metrics)
        epoch_loaded = checkpoint['epoch']
    else:
        checkpoint = actual_model.load_checkpoint(str(model_path), optimizer, device)
        epoch_loaded = checkpoint['epoch']
        
    print(f"\n✓ Loaded best model from epoch {epoch_loaded}")
    print(f"  Best Val NDCG@50: {best_val_ndcg:.4f}")
    print(f"  Best Val AUROC: {best_metrics.get('auroc', 0):.4f}")
    print(f"  Best Val MRR: {best_metrics.get('mrr', 0):.4f}")
    print(f"  Saved to: {model_path}")
    
    # Score all genes with statistical significance
    print(f"\n{'='*80}")
    print(f"SCORING ALL GENES - FOLD {fold_idx}")
    print(f"{'='*80}")
    
    scored_genes_df = actual_model.score_genes_with_statistics(
        data=original_device,
        labels=labels,
        node_names=node_names,  # <-- Pass node names here
        curvature_type='ollivier',
        num_permutations=1000,
        device=device,
        save_path=results_dir,
        save_prefix=f"{model_prefix}_fold_{fold_idx}" if model_prefix else f"fold_{fold_idx}"
    )
    
    return model, best_metrics, history, scored_genes_df

def plot_training_curves(histories: List[Dict], save_dir: Path, prefix: str = ""):
    """Plot training curves for all folds"""
    try:
        num_folds = len(histories)
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Training Curves Across Folds', fontsize=16)
        
        metrics_to_plot = [
            ('train_loss', 'Training Loss', 'lower'),
            ('train_ndcg', 'Training NDCG', 'upper'),
            ('val_auroc', 'Validation AUROC', 'upper'),
            ('val_ndcg@50', 'Validation NDCG@50', 'upper'),
            ('val_precision@50', 'Validation Precision@50', 'upper'),
            ('learning_rate', 'Learning Rate', 'log')
        ]
        
        for idx, (metric_key, title, scale) in enumerate(metrics_to_plot):
            ax = axes[idx // 3, idx % 3]
            
            for fold_idx, history in enumerate(histories, 1):
                if metric_key in history and len(history[metric_key]) > 0:
                    epochs = range(1, len(history[metric_key]) + 1)
                    ax.plot(epochs, history[metric_key], label=f'Fold {fold_idx}', alpha=0.7)
            
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric_key)
            ax.set_title(title)
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
            
            if scale == 'log':
                ax.set_yscale('log')
        
        plt.tight_layout()
        
        output_file = save_dir / f"{prefix}training_curves.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved training curves to: {output_file}")
        
    except Exception as e:
        logger.error(f"Error plotting training curves: {e}")

def plot_metrics_comparison(metrics_data: List[Dict], save_dir: Path, prefix: str = ""):
    """Plot comparison of metrics across folds"""
    try:
        df = pd.DataFrame(metrics_data)
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Performance Metrics Across Folds', fontsize=16)
        
        # Bar plot
        metrics_cols = ['NDCG@50', 'AUROC', 'AUPRC', 'Precision@50', 'MRR']
        x = np.arange(len(metrics_cols))
        width = 0.15
        
        for idx, fold_idx in enumerate(df['Fold']):
            offset = (idx - len(df) / 2) * width
            values = [df.loc[idx, col] for col in metrics_cols]
            axes[0].bar(x + offset, values, width, label=f'Fold {fold_idx}', alpha=0.8)
        
        axes[0].set_xlabel('Metrics')
        axes[0].set_ylabel('Score')
        axes[0].set_title('Metrics by Fold')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(metrics_cols, rotation=45, ha='right')
        axes[0].legend(loc='best')
        axes[0].grid(True, alpha=0.3, axis='y')
        axes[0].set_ylim([0, 1])
        
        # Box plot
        box_data = [df[col].values for col in metrics_cols]
        bp = axes[1].boxplot(box_data, labels=metrics_cols, patch_artist=True)
        
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
            patch.set_alpha(0.7)
        
        axes[1].set_xlabel('Metrics')
        axes[1].set_ylabel('Score')
        axes[1].set_title('Metrics Distribution')
        axes[1].grid(True, alpha=0.3, axis='y')
        axes[1].set_xticklabels(metrics_cols, rotation=45, ha='right')
        axes[1].set_ylim([0, 1])
        
        plt.tight_layout()
        
        output_file = save_dir / f"{prefix}metrics_comparison.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved metrics comparison to: {output_file}")
        
    except Exception as e:
        logger.error(f"Error plotting metrics comparison: {e}")

def aggregate_gene_scores(all_fold_dfs: List[pd.DataFrame], save_dir: Path, prefix: str = ""):
    """
    Aggregate gene scores across all folds.
    
    For each gene, compute:
    - Mean score across folds
    - Std score across folds
    - Number of folds where gene was significant
    - Consensus significance (significant in majority of folds)
    """
    try:
        print(f"\nAggregating scores from {len(all_fold_dfs)} folds...")
        
        # Get gene IDs from first fold (should be same across all folds)
        gene_ids = all_fold_dfs[0]['gene_id'].values
        gene_names = all_fold_dfs[0]['gene_name'].values
        true_labels = all_fold_dfs[0]['true_label'].values
        
        # Collect scores from each fold
        scores_matrix = np.array([df['driver_score'].values for df in all_fold_dfs])
        pvalues_matrix = np.array([df['adjusted_pvalue'].values for df in all_fold_dfs])
        ranks_matrix = np.array([df['rank'].values for df in all_fold_dfs])
        
        # Compute aggregate statistics
        mean_scores = scores_matrix.mean(axis=0)
        std_scores = scores_matrix.std(axis=0)
        median_scores = np.median(scores_matrix, axis=0)
        
        mean_pvalues = pvalues_matrix.mean(axis=0)
        median_ranks = np.median(ranks_matrix, axis=0)
        
        # Count how many folds each gene was significant in
        significant_counts = (pvalues_matrix < 0.05).sum(axis=0)
        consensus_significant = significant_counts >= (len(all_fold_dfs) / 2)
        
        # Create aggregate DataFrame
        aggregate_df = pd.DataFrame({
            'gene_id': gene_ids,
            'gene_name': gene_names,
            'true_label': true_labels,
            'is_known_driver': true_labels == 1,
            'mean_score': mean_scores,
            'std_score': std_scores,
            'median_score': median_scores,
            'mean_adjusted_pvalue': mean_pvalues,
            'median_rank': median_ranks,
            'folds_significant': significant_counts,
            'total_folds': len(all_fold_dfs),
            'consensus_significant': consensus_significant & (true_labels == 0)  # Only unknowns
        })
        
        # Sort by mean score
        aggregate_df = aggregate_df.sort_values('mean_score', ascending=False)
        aggregate_df['aggregate_rank'] = range(1, len(aggregate_df) + 1)
        
        # Reorder columns
        aggregate_df = aggregate_df[[
            'aggregate_rank', 'gene_id', 'gene_name', 'is_known_driver',
            'mean_score', 'std_score', 'median_score', 
            'mean_adjusted_pvalue', 'median_rank',
            'folds_significant', 'total_folds', 'consensus_significant'
        ]]
        
        # Get consensus significant genes
        consensus_genes = aggregate_df[aggregate_df['consensus_significant']]
        
        print(f"\n{'='*80}")
        print("AGGREGATE RESULTS")
        print(f"{'='*80}")
        print(f"\nTotal genes: {len(aggregate_df)}")
        print(f"Known drivers: {(aggregate_df['is_known_driver']).sum()}")
        print(f"Consensus significant genes: {len(consensus_genes)}")
        
        if len(consensus_genes) > 0:
            print(f"\nTop 20 Consensus Significant Genes:")
            print("-" * 80)
            display_cols = ['aggregate_rank', 'gene_name', 'mean_score', 'std_score',
                            'folds_significant', 'total_folds', 'mean_adjusted_pvalue']
            print(consensus_genes[display_cols].head(20).to_string(index=False))
        
        # Save aggregate results
        output_prefix = f"{prefix}_" if prefix else ""
        
        # Save all aggregated scores
        all_file = save_dir / f"{output_prefix}aggregated_all_genes.csv"
        aggregate_df.to_csv(all_file, index=False)
        print(f"\n✓ Saved aggregated scores to: {all_file}")
        
        # Save consensus significant genes
        if len(consensus_genes) > 0:
            consensus_file = save_dir / f"{output_prefix}aggregated_consensus_significant.csv"
            consensus_genes.to_csv(consensus_file, index=False)
            print(f"✓ Saved {len(consensus_genes)} consensus genes to: {consensus_file}")
        
        # Save summary
        summary_file = save_dir / f"{output_prefix}aggregated_summary.txt"
        with open(summary_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("AGGREGATED GENE SCORING SUMMARY (ACROSS ALL FOLDS)\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Number of folds: {len(all_fold_dfs)}\n")
            f.write(f"Total genes: {len(aggregate_df)}\n")
            f.write(f"Known driver genes: {(aggregate_df['is_known_driver']).sum()}\n")
            f.write(f"Unknown genes: {(~aggregate_df['is_known_driver']).sum()}\n\n")
            
            f.write(f"Consensus significant genes (sig in ≥{len(all_fold_dfs)/2:.0f} folds): {len(consensus_genes)}\n\n")
            
            if len(consensus_genes) > 0:
                f.write("Score Statistics for Consensus Genes:\n")
                f.write(f"  Mean score: {consensus_genes['mean_score'].mean():.4f}\n")
                f.write(f"  Median score: {consensus_genes['mean_score'].median():.4f}\n")
                f.write(f"  Min rank: {consensus_genes['aggregate_rank'].min()}\n")
                f.write(f"  Max rank: {consensus_genes['aggregate_rank'].max()}\n\n")
                
                f.write("Top 20 Consensus Significant Genes:\n")
                f.write("-"*80 + "\n")
                top_20 = consensus_genes.head(20)[
                    ['aggregate_rank', 'gene_name', 'mean_score', 'folds_significant', 'mean_adjusted_pvalue']
                ]
                f.write(top_20.to_string(index=False))
        
        print(f"✓ Saved aggregate summary to: {summary_file}")
        print(f"\n{'='*80}\n")
        
        return aggregate_df
        
    except Exception as e:
        logger.error(f"Error aggregating gene scores: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def main():
    parser = argparse.ArgumentParser(
        description='Train Contrastive Driver Gene Predictor with Advanced Attention'
    )
    parser.add_argument('--dataset_file', type=str, required=True,
                        help='Input dataset pickle file')
    parser.add_argument('--train_metrics_dir', type=str,
                        help='Output directory for training metrics', 
                        default='model_results')
    parser.add_argument('--model_out_dir', type=str, 
                        help='Directory to output model checkpoints', 
                        default='trained_models')
    parser.add_argument('--model_out_prefix', type=str, default='',
                        help='Prefix for model checkpoint filenames')
    parser.add_argument('--num_folds', type=int, default=5, 
                        help='Number of folds to use')
    parser.add_argument('--specific_folds', type=int, nargs='+', default=None,
                        help='Train only specific folds')
    parser.add_argument('--num_epochs', type=int, default=200,
                        help='Number of training epochs per fold')
    parser.add_argument('--hidden_channels', type=int, default=256,
                        help='Hidden layer channels')
    parser.add_argument('--temperature', type=float, default=0.4,
                        help = 'Choose the temperature for contrastive loss.')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Choose the proportion of neurons to drop')
    parser.add_argument('--attention_mode', type=str, default='hybrid',
                        choices=['standard', 'edge_feature', 'bias', 'gated', 'hybrid'],
                        help='Attention mode for message passing')
    parser.add_argument('--pathway_aggregator', type = str, default='attention',
                        choices= ['attention','concat', 'hierarchical', 'mean'])
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Set the learning rate')
    parser.add_argument('--num_heads', type=int, default=4,
                        help='Number of attention heads')
    parser.add_argument('--num_layers', type = int, default = 3,
                        help = 'Choose the number of GNN layers')
    parser.add_argument('--concat_heads', action='store_true', default=False,
                        help='Concatenate attention heads (vs average)')
    parser.add_argument('--ranking_loss_type', type=str, default='pairwise',
                        choices=['pairwise', 'listwise', 'pointwise'],
                        help='Type of ranking loss')
    parser.add_argument('--ranking_margin', type=float, default=1.0,
                        help='Margin for pairwise ranking loss')
    parser.add_argument('--early_stopping_patience', type=int, default=50,
                        help='Early stopping patience')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4,
                    help='Number of gradient accumulation steps')
    parser.add_argument('--mixed_precision', action='store_true',
                        help='Use mixed precision training')
    parser.add_argument('--max_views_per_step', type=int, default=2,
                        help='Maximum augmented views to use per training step')
    parser.add_argument('--checkpoint_layers', action='store_true',
                        help='Use gradient checkpointing for each layer')
    
    # Add command-line argument for emergency mode
    parser.add_argument('--emergency_mode', action='store_true',
                        help='Use drastically reduced model for OOM issues')
    parser.add_argument('--max_augmented_views', type=int, default=3,
                        help='Maximum number of augmented views to keep in memory')
    parser.add_argument('--reduce_model_size', action='store_true', default = False,
                        help = 'For OOM issues, use this arg to reduce the model size')
    
    # Multi-GPU Arguments
    parser.add_argument('--use_multi_gpu', action = 'store_true', default=False,
                        help = 'Use Multiple GPUs with DataParallel')
    parser.add_argument('--gpu_ids', type = int, nargs='+', default=None,
                        help='Specific GPU IDs to use (default=None)')
    args = parser.parse_args()
    
    # Create directories
    models_dir = Path(args.model_out_dir)
    results_dir = Path(args.train_metrics_dir)
    models_dir.mkdir(exist_ok=True, parents=True)
    results_dir.mkdir(exist_ok=True, parents=True)
    
    if args.model_out_prefix:
        results_subdir = results_dir / args.model_out_prefix
        results_subdir.mkdir(exist_ok=True, parents=True)
        results_dir = results_subdir
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()
    
    print("\n" + "="*80)
    print("K-FOLD DRIVER GENE PREDICTOR WITH ADVANCED ATTENTION")
    print("="*80)
    print(f"Dataset: {args.dataset_file}")
    print(f"Attention Mode: {args.attention_mode}")
    print(f"Multi-head: {args.num_heads} heads (concat={args.concat_heads})")
    print(f"Ranking Loss: {args.ranking_loss_type}")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("="*80 + "\n")
    
    # Load data
    with open(args.dataset_file, 'rb') as f:
        data = pickle.load(f)
    
    original = data['original']
    augmented_views = data['augmented_views']
    
    # Validate and preprocess
    print("\n" + "="*80)
    print("VALIDATING AND PREPROCESSING DATA")
    print("="*80)
    
    validate_graph_data(original, "Original Graph")
    
    # Sanitize features
    features = original.get('feature', original.get('x'))
    if torch.isnan(features).any() or torch.isinf(features).any():
        features = torch.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
        if 'feature' in original:
            original['feature'] = features
        else:
            original['x'] = features
    
    # Preprocess curvature
    original = preprocess_curvature_data(original, curvature_type='ollivier')
    
    for i, view in enumerate(augmented_views):
        augmented_views[i] = preprocess_curvature_data(view, curvature_type='ollivier')
    
    print("✓ Data validation and preprocessing complete")
    print("="*80 + "\n")
    
    labels = original['label']
    kfold_splits = original['kfold_splits']
    
    # Filter folds if specified
    if args.specific_folds:
        num_folds = len(kfold_splits)
        folds_to_train = [i-1 for i in args.specific_folds if 1 <= i <= num_folds]
        kfold_splits = [kfold_splits[i] for i in folds_to_train]
    elif args.num_folds:
        kfold_splits = kfold_splits[:args.num_folds]
    
    print(f"\n{'='*80}")
    print(f"K-FOLD CROSS-VALIDATION: {len(kfold_splits)} FOLDS")
    print(f"{'='*80}")
    
    # Train each fold
    all_fold_metrics = []
    all_fold_histories = []
    fold_models = []
    all_fold_scored_genes = []
    
    for fold_idx, fold_data in enumerate(kfold_splits, 1):
        model, best_metrics, history, scored_genes_df = train_single_fold(
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
            attention_mode=args.attention_mode,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            concat_heads=args.concat_heads,
            ranking_loss_type=args.ranking_loss_type,
            ranking_margin=args.ranking_margin,
            reduce_model_size=args.reduce_model_size,
            use_multi_gpu=args.use_multi_gpu,
            gpu_ids=args.gpu_ids
        )
        
        all_fold_metrics.append(best_metrics)
        all_fold_histories.append(history)
        fold_models.append(model)
        all_fold_scored_genes.append(scored_genes_df)
        
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
        print(f"  NDCG@50:   {metrics.get('ndcg@50', 0):.4f}  ← PRIMARY")
        print(f"  AUROC:     {metrics.get('auroc', 0):.4f}")
        print(f"  AUPRC:     {metrics.get('auprc', 0):.4f}")
        print(f"  P@50:      {metrics.get('precision@50', 0):.4f}")
        print(f"  MRR:       {metrics.get('mrr', 0):.4f}")
        
        metrics_df_data.append({
            'Fold': fold_idx,
            'NDCG@50': metrics.get('ndcg@50', 0),
            'AUROC': metrics.get('auroc', 0),
            'AUPRC': metrics.get('auprc', 0),
            'Precision@50': metrics.get('precision@50', 0),
            'MRR': metrics.get('mrr', 0)
        })
    
    # Calculate statistics
    mean_metrics = {
        key: np.mean([m.get(key, 0) for m in all_fold_metrics])
        for key in ['ndcg@50', 'auroc', 'auprc', 'precision@50', 'mrr']
    }
    
    std_metrics = {
        key: np.std([m.get(key, 0) for m in all_fold_metrics])
        for key in ['ndcg@50', 'auroc', 'auprc', 'precision@50', 'mrr']
    }
    
    print(f"\n{'='*80}")
    print("AGGREGATE STATISTICS")
    print(f"{'='*80}")
    print(f"\nMean ± Std across {len(all_fold_metrics)} folds:")
    print(f"  NDCG@50:      {mean_metrics['ndcg@50']:.4f} ± {std_metrics['ndcg@50']:.4f}")
    print(f"  AUROC:        {mean_metrics['auroc']:.4f} ± {std_metrics['auroc']:.4f}")
    print(f"  AUPRC:        {mean_metrics['auprc']:.4f} ± {std_metrics['auprc']:.4f}")
    print(f"  Precision@50: {mean_metrics['precision@50']:.4f} ± {std_metrics['precision@50']:.4f}")
    print(f"  MRR:          {mean_metrics['mrr']:.4f} ± {std_metrics['mrr']:.4f}")
    print(f"{'='*80}\n")
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame(metrics_df_data)
    
    # Add mean and std rows
    mean_row = {
        'Fold': 'Mean',
        'NDCG@50': mean_metrics['ndcg@50'],
        'AUROC': mean_metrics['auroc'],
        'AUPRC': mean_metrics['auprc'],
        'Precision@50': mean_metrics['precision@50'],
        'MRR': mean_metrics['mrr']
    }
    
    std_row = {
        'Fold': 'Std',
        'NDCG@50': std_metrics['ndcg@50'],
        'AUROC': std_metrics['auroc'],
        'AUPRC': std_metrics['auprc'],
        'Precision@50': std_metrics['precision@50'],
        'MRR': std_metrics['mrr']
    }
    
    metrics_df = pd.concat([
        metrics_df,
        pd.DataFrame([mean_row, std_row])
    ], ignore_index=True)
    
    # Save metrics
    prefix = f"{args.model_out_prefix}_" if args.model_out_prefix else ""
    metrics_file = results_dir / f"{prefix}kfold_metrics.csv"
    metrics_df.to_csv(metrics_file, index=False)
    print(f"✓ Saved metrics to: {metrics_file}")
    
    # Save training histories
    history_file = results_dir / f"{prefix}training_history.pkl"
    with open(history_file, 'wb') as f:
        pickle.dump(all_fold_histories, f)
    print(f"✓ Saved training histories to: {history_file}")
    
    # Plot training curves
    plot_training_curves(all_fold_histories, results_dir, prefix)
    
    # Plot metrics comparison
    plot_metrics_comparison(metrics_df_data, results_dir, prefix)
    
    # Aggregate gene scores across folds
    print(f"\n{'='*80}")
    print("AGGREGATING GENE SCORES ACROSS FOLDS")
    print(f"{'='*80}")
    
    aggregate_gene_scores(all_fold_scored_genes, results_dir, prefix)
    
    print(f"\n{'='*80}")
    print("TRAINING COMPLETE")
    print(f"{'='*80}")
    print(f"Models saved to: {models_dir}")
    print(f"Results saved to: {results_dir}")
    print(f"{'='*80}\n")
    

if __name__ == '__main__':
    logger = get_logger(__name__)
    main()