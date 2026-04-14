#!/usr/bin/env python

import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Dict
import pandas as pd
import gc
import numpy as np
import pickle
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.checkpoint import checkpoint as gradient_checkpoint
from sklearn.preprocessing import StandardScaler

from utils.logging_manager import get_logger
from model.DriverGenePredictor import ContrastiveDriverGenePredictor, compute_ndcg
from model.support_models import WarmupScheduler, EarlyStopping, RankingLoss, EMA

# Memory optimization
torch.cuda.empty_cache()                                                        # Flush any GPU memory allocated from previous runs before training starts
gc.collect()                                                                    # Trigger python garbage collection to free unreferenced CPU objects

# Enable efficient operations
torch.backends.cuda.matmul.allow_tf32 = True                                    # Use TF32 for matrix multiplications; faster on Ampere+ GPUs with negligible precision loss
torch.backends.cudnn.allow_tf32 = True                                          # Use TF32 for cuDNN convolutions; same trade-off as above

logger = get_logger(__name__)

def set_seed(seed: int = 42):
    """Set random seed for reproducibility across all libraries."""
    random.seed(seed)                                                           # Seed python's built in random module
    np.random.seed(seed)                                                        # Seed NumPy's RNG
    torch.manual_seed(seed)                                                     # Seed PyTorch CPU RNG
    torch.cuda.manual_seed_all(seed)                                            # Seed all GPU RNG (covers multi-GPU setups)
    torch.backends.cudnn.deterministic = True                                   # Force cuDNN to use deterministic algorithms; prevents non-reproducible results from algorithm selection
    torch.backends.cudnn.benchmark = False                                      # Disable cuDNN auto-tuner; tuner picks fastest kernel per input shape but is non-deterministic
    logger.info(f"Random seed set to {seed}")

def setup_device(model):
    """Setup model on GPU or CPU.

    Returns:
        model: Model moved to device
        device: The device being used
    """
    if not torch.cuda.is_available():
        print('CUDA not available, using CPU')
        return model, torch.device('cpu')                                       # Graceful fallback to CPU

    device = torch.device('cuda:0')                                             # use the First available GPU
    props = torch.cuda.get_device_properties(0)
    mem_gb = props.total_memory / 1024**3                                       # Convert bytes -> GB for human-readable display
    print(f"\n✓ Using GPU: {props.name} ({mem_gb:.1f} GB)")

    model = model.to(device)                                                    # Move all model parameters and buffers to the GPU
    return model, device


def check_for_nans(tensor, name="tensor", raise_error=True):
    """Check if tensor contains NaN or Inf values"""
    if torch.isnan(tensor).any():
        msg = f"{name} contains NaN values"
        if raise_error:
            raise ValueError(msg)                                               # Hardstop: NaN in a tensor usually means a  broken forward pass
        else:
            logger.error(msg)                                                   # Soft warning: caller decides how to handle
            return True
    if torch.isinf(tensor).any():
        msg = f"{name} contains Inf values"
        if raise_error:
            raise ValueError(msg)                                               # Hardstop: Inf propagates through computations silently
        else:
            logger.error(msg)
            return True                                                         # Tensor is clean
    return False

def sanitize_curvature(edge_curvature, name="curvature"):
    """Replace NaN/Inf values in curvature with safe defaults"""
    if edge_curvature is None or edge_curvature.numel() == 0:
        return edge_curvature                                                   # Nothing to sanitize; return as is
    
    nan_mask = torch.isnan(edge_curvature)                                      # Locate NaN entries
    inf_mask = torch.isinf(edge_curvature)                                      # Locate Inf entries
    
    if nan_mask.any() or inf_mask.any():
        logger.warning(f"{name}: Found {nan_mask.sum().item()} NaN and {inf_mask.sum().item()} Inf values")
        edge_curvature = edge_curvature.clone()                                 # Clone before in-place modification to avoid mutating the original
        edge_curvature[nan_mask] = 0.0                                          # Replace NaN with neutral curvature value
        edge_curvature[inf_mask] = 0.0                                          # Replace NaN with neutral curvature value
        logger.warning(f"{name}: Replaced problematic values with 0")
    
    edge_curvature = torch.clamp(edge_curvature, min=-10.0, max=10.0)           # Hard clip; prevents extreme curvature from dominating attention scores
    return edge_curvature

def validate_graph_data(data, name="graph", strict=False):
    """Validate graph data structure for NaN/Inf values"""
    issues = []                                                                 # Accumulates all validation errors found; lets caller see all problems at once instead of stopping at the first
    
    features = data.get('feature', data.get('x'))                               # Support both naming conventions
    if features is not None:
        if torch.isnan(features).any():
            issues.append(f"{name}: Features contain NaN")
            if strict:
                raise ValueError(issues[-1])                                    # In strict mode, abort immediately on first issue
        if torch.isinf(features).any():
            issues.append(f"{name}: Features contain Inf")
            if strict:
                raise ValueError(issues[-1])
    
    edge_index = data.get('edge_index')
    if edge_index is not None:
        num_nodes = features.shape[0] if features is not None else edge_index.max().item() + 1
        if edge_index.min() < 0:
            issues.append(f"{name}: Edge index contains negative values")       # Negative node indices are invalid
        if edge_index.max() >= num_nodes:
            issues.append(f"{name}: Edge index exceeds number of nodes")        # Out-of-bounds index would cause index errors during message passing
    
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
    
    return len(issues) == 0                                                     # True = clean data, False = issues found (logged but not raised in non-strict mode)

def preprocess_curvature_data(data: Dict, curvature_type: str = 'ollivier') -> Dict:
    """Preprocess data to ensure curvature dimensions match edge_index with NaN/Inf sanitization"""
    curv_key = f'{curvature_type}_curvature'
    
    if curv_key not in data:
        logger.warning(f"No '{curv_key}' found in data, skipping preprocessing")
        return data                                                             # Nothing to preprocess, return unchanged
    
    edge_index = data['edge_index']
    edge_curvature = data[curv_key]
    
    # Sanitize curvature values
    logger.info(f"Sanitizing {curv_key} values...")
    edge_curvature = sanitize_curvature(edge_curvature, name=curv_key)          # Replace NaN/Inf values before any dimension checjs
    
    num_edges = edge_index.shape[1]
    num_curvatures = edge_curvature.shape[0]
    
    logger.info(f"Preprocessing curvature: {num_curvatures} curvatures for {num_edges} edges")
    
    if num_curvatures == num_edges:
        logger.info("Curvature dimensions already match")
        data[curv_key] = edge_curvature                                         # Dimensions already aligned; just write back the sanitized values
        return data
    
    if num_curvatures * 2 == num_edges:
        logger.info("Detected undirected curvature for directed edges, matching...")
        
        device = edge_curvature.device
        matched_curvature = torch.zeros(num_edges, device=device, dtype=edge_curvature.dtype)
        
        edge_list = edge_index.t().cpu().numpy()                                # [num_edges, 2] for row-wise iteration
        edge_to_curv_idx = {}
        
        for src, dst in edge_list:
            src, dst = int(src), int(dst)
            canonical = (min(src, dst), max(src, dst))                          # Canonical form so both directions share the same key
            if canonical not in edge_to_curv_idx:
                edge_to_curv_idx[canonical] = len(edge_to_curv_idx)             # Assign next available index on first encounter
        
        sorted_edges = sorted(edge_to_curv_idx.keys())
        edge_to_curv_idx = {edge: i for i, edge in enumerate(sorted_edges)}     # Re-index by sorted order for determination
        
        for i, (src, dst) in enumerate(edge_list):
            src, dst = int(src), int(dst)
            canonical = (min(src, dst), max(src, dst))
            curv_idx = edge_to_curv_idx.get(canonical, 0)                       # Look up the curvature index for this canonical edge
            if curv_idx < num_curvatures:
                matched_curvature[i] = edge_curvature[curv_idx]                 # Copy undirected curvature to this directed edge
        
        data[curv_key] = matched_curvature
        logger.info(f"Successfully matched {num_curvatures} curvatures to {num_edges} edges")
    
    elif num_edges * 2 == num_curvatures:
        logger.info("Detected directed curvature for undirected edges, averaging...")
        data[curv_key] = edge_curvature.reshape(-1, 2).mean(dim=1)              # Average forward and backward curvature per undirected edge
    
    else:
        logger.warning(f"Unusual ratio: {num_curvatures} curvatures for {num_edges} edges")
        device = edge_curvature.device
        
        # non-standard ratio: fallback to best-effort canonical matching
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
        
        # same canonical matching logic as the 2:1 case above...
        for i, (src, dst) in enumerate(edge_list):
            src, dst = int(src), int(dst)
            canonical = (min(src, dst), max(src, dst))
            curv_idx = edge_to_curv_idx.get(canonical, 0)
            if curv_idx < num_curvatures:
                matched_curvature[i] = edge_curvature[curv_idx]
            else:
                matched_curvature[i] = 0.0                                      # No match; use neutral curvature
        
        data[curv_key] = matched_curvature
    
    # Final sanitization
    data[curv_key] = sanitize_curvature(data[curv_key], name=f"{curv_key}_matched") # Re-sanitize after matching in case any edge had no match and was left as 0
    
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
    model.eval()                                                                # Disable dropout and use inference-mode batch norm
    
    try:
        with torch.no_grad():                                                   # No gradients needed during evaluation
            metrics = model.evaluate(  # <-- This should return a dict
                data, labels, mask,
                curvature_type=curvature_type,
                device=device,
                k_values=[10, 20, 50, 100]                                      # Compute Precision@K and NDCG@K for these cutoffs
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
            }                                                                   # Safe default metrics dict to prevent downstream KeyError crashes
        
        # Check for NaN in metrics
        for key, value in metrics.items():
            if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
                logger.warning(f"Invalid metric {key}: {value}, setting to 0")
                metrics[key] = 0.0                                              # Replace degenerate metric values so they don't corrupt averages
        
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
        }                                                                       # Return zero metrics on any exception so training loop continues

def create_cancer_driver_model(
    num_features: int = 74,
    hidden_channels: int = 256,
    projection_dim: int = 128,
    num_layers: int = 3,
    temperature: float = 0.4,
    attention_mode: str = 'hybrid',
    pathway_aggregator: str = 'hierarchical',
    aggregation: str = 'add',
    num_heads: int = 4,
    dropout: float = 0.2,
    negative_slope: float = 0.2,
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
        curvature_types=['positive', 'negative', 'both'],                  # Fixed set of curvature pathways
        hop_types=['one_hop', 'two_hop'],                                  # Fixed set of hop distance
        num_attention_heads=num_heads,
        use_attention=True,
        aggregation=aggregation,
        temperature=temperature,
        attention_mode=attention_mode,
        pathway_aggregator=pathway_aggregator,
        concat=concat_heads,
        negative_slope=negative_slope,
        dropout=dropout,
        device=device
    ).to(device)                                                           # Move all parameters to target device immediately after construction

    logger.info(f"Created ContrastiveDriverGenePredictor model:")
    logger.info(f"  - Input features: {num_features}")
    logger.info(f"  - Hidden channels: {hidden_channels}")
    logger.info(f"  - Attention mode: {attention_mode}")
    logger.info(f"  - Attention heads: {num_heads} (concat={concat_heads})")
    logger.info(f"  - Device: {device}")
    logger.info(f"  - Training objective: Ranking")
    
    return model


def reconstruct_node_mapping(augmented_views, num_nodes):
    """Reconstruct full node_id_to_name mapping by cross-validating across all views.

    Each view's remaining + eliminated nodes should reconstruct the same full mapping.
    Validates consistency across all views to catch metadata corruption.

    Returns:
        node_names: np.array of gene names ordered by node ID
        node_id_to_name_full: dict mapping node ID -> gene name
    """
    if not augmented_views or len(augmented_views) == 0:
        logger.warning("No augmented_views found, using placeholder names")
        return np.array([f"Gene_{i}" for i in range(num_nodes)]), {}       # No views = no metadata; fallback to index names

    reference_mapping = None                                               # Will hold the mapping from view 0 for cross-validation

    for view_idx, view in enumerate(augmented_views):
        metadata = view.get('metadata', {})
        remaining = metadata.get('node_id_to_name', {})                    # Nodes kept after Schur elimination
        eliminated = metadata.get('eliminated_node_id_to_name', {})        # Nodes removed by Schur elimination

        if not remaining and not eliminated:
            logger.warning(f"View {view_idx}: both node_id_to_name and eliminated_node_id_to_name are empty")
            continue

        full_mapping = {**remaining, **eliminated}                         # Union: Every node should appear in exactly one of the 2 dicts

        # Check for overlap (a node shouldn't be both remaining and eliminated)
        overlap = set(remaining.keys()) & set(eliminated.keys())
        if overlap:
            logger.error(f"View {view_idx}: {len(overlap)} nodes appear in both remaining and eliminated: "
                        f"{list(overlap)[:5]}")                            # Data corruption: a node can't be both kept and removed

        if reference_mapping is None:
            reference_mapping = full_mapping                               # Use view 0 as the authoritative mapping
            logger.info(f"View {view_idx}: {len(remaining)} remaining + {len(eliminated)} eliminated "
                       f"= {len(full_mapping)} total nodes")               # Inconsistent metadata across views signals a preprocessing bug
        else:
            # Cross-validate against reference
            if full_mapping != reference_mapping:
                missing = set(reference_mapping.keys()) - set(full_mapping.keys())
                extra = set(full_mapping.keys()) - set(reference_mapping.keys())
                mismatched = {k for k in set(reference_mapping.keys()) & set(full_mapping.keys())
                             if reference_mapping[k] != full_mapping[k]}
                logger.error(f"View {view_idx} mapping DISAGREES with view 0: "
                           f"{len(missing)} missing, {len(extra)} extra, {len(mismatched)} name mismatches")
            else:
                logger.info(f"View {view_idx}: mapping consistent with view 0 "
                           f"({len(remaining)} remaining + {len(eliminated)} eliminated)")

    if reference_mapping is None:
        logger.warning("No valid mappings found in any view, using placeholder names")
        return np.array([f"Gene_{i}" for i in range(num_nodes)]), {}       # All views had empty metadata

    node_names = np.array([str(reference_mapping.get(i, f"Gene_{i}")) for i in range(num_nodes)])   # Build ordered name array indexed by Node ID
    return node_names, reference_mapping


def align_and_verify_labels(node_id_to_name_full, original_labels, num_nodes):
    """Align labels using the full node mapping and verify known drivers are correct.

    Returns:
        labels: aligned label tensor
    """
    aligned_labels = torch.zeros(num_nodes, dtype=original_labels.dtype)   # Start with all-zero labels
    missing_count = 0

    for i in range(num_nodes):
        if i in node_id_to_name_full:
            aligned_labels[i] = original_labels[i]                         # Copy label from original tensor at the same node index
        else:
            missing_count += 1
            aligned_labels[i] = 0                                          # Node not in mapping; conservatively treat as non-driver

    if missing_count > 0:
        logger.warning(f"{missing_count} nodes not found in node_id_to_name_full mapping")

    logger.info(f"Aligned labels using node_id_to_name_full mapping")

    # Verify known driver genes
    node_names = np.array([str(node_id_to_name_full.get(i, f"Gene_{i}")) for i in range(num_nodes)])
    known_drivers = ['TP53', 'KRAS', 'EGFR', 'PIK3CA', 'BRAF', 'PTEN']     # Well-established cancer drivers used as alignment sanity check
    all_correct = True

    for gene in known_drivers:
        if gene in node_names:
            idx = np.where(node_names == gene)[0][0]
            label = aligned_labels[idx].item()
            if label == 1:
                logger.info(f"  {gene} (index {idx}): label=1 (driver)")
            else:
                logger.error(f"  {gene} (index {idx}): label={label} (INCORRECT!)") # Known driver labeled as non-driver = alignment bug
                all_correct = False

    if not all_correct:
        logger.error("Label alignment verification FAILED - some known drivers are mislabeled")
    else:
        logger.info("All checked known drivers are correctly labeled")

    return aligned_labels


def normalize_features_per_fold(features, augmented_views, train_mask, fold_idx):
    """Fit StandardScaler on this fold's training data and normalize all features.

    Prevents data leakage by fitting only on training nodes for the current fold.

    Args:
        features: Original feature tensor (or raw features)
        augmented_views: List of augmented view dicts
        train_mask: Boolean training mask for this fold
        fold_idx: Current fold index (for logging)

    Returns:
        features: Normalized original feature tensor
        augmented_views: Shallow-copied views with normalized features
    """
    raw_features = features
    if isinstance(raw_features, torch.Tensor):
        raw_features_np = raw_features.cpu().numpy()                       # Move to CPU numpy for sklearn
    else:
        raw_features_np = np.array(raw_features)

    train_mask_np = (train_mask.cpu().numpy()
                    if isinstance(train_mask, torch.Tensor)
                    else train_mask)                                       # Convert mask to numpy for boolean indexing

    logger.info(f"Fold {fold_idx}: Fitting StandardScaler on {train_mask_np.sum()} training nodes")
    fold_scaler = StandardScaler()
    fold_scaler.fit(raw_features_np[train_mask_np])                        # Fit ONLY on training nodes; prevents val/test data from influencing normalization (leakage fix)

    # Guard against zero-variance features
    zero_var = fold_scaler.scale_ == 0
    if zero_var.any():
        logger.warning(f"Found {zero_var.sum()} zero-variance features, setting scale to 1.0")
        fold_scaler.scale_[zero_var] = 1.0

    # Normalize original graph features
    normalized_np = fold_scaler.transform(raw_features_np)                 # Apply scalar fitted on training set to all nodes
    normalized_np = np.nan_to_num(normalized_np, nan=0.0, posinf=1.0, neginf=-1.0)  # Final cleanup in case transform produces edge-case values
    features = torch.tensor(normalized_np, dtype=torch.float32)

    # Normalize augmented view features (shallow-copy so raw data stays untouched for next fold)
    logger.info(f"Normalizing {len(augmented_views)} augmented views for fold {fold_idx}...")
    fold_augmented_views = []
    for view in augmented_views:
        raw_x = view.get('x_raw', view['x'])                               # Prefer x_raw if available; falls back to x (which may already be normalized from a previous fold)
        if isinstance(raw_x, torch.Tensor):
            raw_x_np = raw_x.cpu().numpy()
        else:
            raw_x_np = np.array(raw_x)
        norm_x_np = fold_scaler.transform(raw_x_np)                        # Apply same fold scaler to augmented view features
        norm_x_np = np.nan_to_num(norm_x_np, nan=0.0, posinf=1.0, neginf=-1.0)
        fold_augmented_views.append({**view, 'x': torch.tensor(norm_x_np, dtype=torch.float32)})    # Shallow copy with normalized 'x'; original view dict is not mutated

    logger.info(f"Fold {fold_idx}: Per-fold normalization complete")
    return features, fold_augmented_views


def precompute_contrastive_alignment(augmented_views, labels, train_mask):
    """Precompute label mappings and contrastive alignment indices for all views.

    For each view, maps original node indices to augmented indices and builds
    label/mask tensors. For each view pair, finds common nodes and builds
    aligned index tensors for contrastive loss.

    Args:
        augmented_views: List of augmented view dicts with metadata
        labels: Original label tensor
        train_mask: Boolean training mask

    Returns:
        precomputed_mappings: List of dicts with aug_labels, aug_mask, orig_to_aug per view
        contrastive_alignment: Dict mapping (i, j) -> (idx_i, idx_j) index tensors
    """
    num_original = labels.shape[0]
    precomputed_mappings = []

    for _, view in enumerate(augmented_views):
        eliminated_ids = set(view['metadata']['eliminated_node_ids'])      # Set for O(1) membership checks
        aug_size = view['x'].shape[0]                                      # Number of nodes remaining after Schur elimination

        orig_to_aug = {}                                                   # Maps original node index -> augmented node index for this view 
        aug_idx = 0
        for orig_idx in range(num_original):
            if orig_idx not in eliminated_ids:
                orig_to_aug[orig_idx] = aug_idx                            # Surviving nodes get compacted indices
                aug_idx += 1

        aug_labels_cpu = torch.full((aug_size,), -100, dtype=torch.long)   # -100 = 'no label'; will be overwritten for known nodes
        aug_mask_cpu = torch.zeros(aug_size, dtype=torch.bool)             # training mask for the augmented graph

        for orig_idx in range(num_original):
            if orig_idx in orig_to_aug:
                aug_idx = orig_to_aug[orig_idx]
                if aug_idx < aug_size:
                    aug_labels_cpu[aug_idx] = labels.cpu()[orig_idx]       # Transfer label to augmented index
                    aug_mask_cpu[aug_idx] = train_mask.cpu()[orig_idx]     # Transfer training mask for the augmented graph

        precomputed_mappings.append({
            'aug_labels': aug_labels_cpu,                                  # Label tensor aligned to augmented graph
            'aug_mask': aug_mask_cpu,                                      # Training mask aligned to augmented graph
            'aug_size': aug_size,
            'orig_to_aug': orig_to_aug                                     # Lookup dict for contrastive alignment
        })

    logger.info(f"Precomputed mappings for {len(augmented_views)} views")

    # Precompute aligned index tensors for all view pairs
    contrastive_alignment = {}
    for i in range(len(augmented_views)):
        for j in range(len(augmented_views)):
            if i == j:
                continue
            orig_to_aug_i = precomputed_mappings[i]['orig_to_aug']
            orig_to_aug_j = precomputed_mappings[j]['orig_to_aug']
            common_orig_ids = sorted(set(orig_to_aug_i.keys()) & set(orig_to_aug_j.keys()))                 # Nodes present in both views after their respective eliminations
            idx_i = torch.tensor([orig_to_aug_i[oid] for oid in common_orig_ids], dtype=torch.long)         # Augmented indices in view 1 for common nodes
            idx_j = torch.tensor([orig_to_aug_j[oid] for oid in common_orig_ids], dtype=torch.long)         # Augmented indices in view 2 for common nodes
            contrastive_alignment[(i, j)] = (idx_i, idx_j)                                                  # Pre-computed once; reused every epoch to align embeddings for NT-Xent loss
            logger.info(f"  View pair ({i},{j}): {len(common_orig_ids)} common nodes for contrastive loss")

    logger.info(f"Precomputed contrastive alignment for {len(contrastive_alignment)} view pairs")
    return precomputed_mappings, contrastive_alignment


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
    model_prefix: str = None,
    attention_mode: str = 'hybrid',
    pathway_aggregator: str = 'hierarchical',
    num_heads: int = 4,
    num_layers: int = 3,
    hidden_channels: int = 256,
    projection_dim: int = 128,
    temperature:float = 0.4,
    dropout: float = 0.2,
    return_all_layers: float = False,
    concat_heads: bool = True,
    scheduler_patience: int = 50,
    early_stopping_patience: int = 50,
    ranking_loss_samples: int = 256,
    aggregation: str = 'add',
    ranking_loss_type: str = 'bpr',
    ranking_loss_scale: float = 10.0,
    ranking_margin: float = 0.5,
    focal_gamma: float = 2.0,
    use_focal: bool = True,
    contrastive_weight: float = 0.3,
    gradient_accumulation_steps: int = 4,
    mixed_precision: bool = True,
    reduce_model_size: bool = True,
    validation_frequency: int = 10,
    decay: float = 0.999,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    negative_slope: float = 0.2,
    scheduler_factor: float = 0.7
) -> Tuple[ContrastiveDriverGenePredictor, Dict, Dict]:
    
    """Train model on a single fold with ranking objective"""
    print(f"\n{'='*80}")
    print(f"TRAINING FOLD {fold_idx}")
    print(f"{'='*80}")
    
    # Extract masks
    train_mask_original = torch.from_numpy(fold_data['train_mask'])                 # Convert numpy bool array to torch tensor
    val_mask_original = torch.from_numpy(fold_data['val_mask'])
    
    # Get features - handle both 'feature' and 'x' keys
    features = original.get('feature', original.get('x'))                           # Support both key naming conventions
    if features is None:
        raise ValueError("No 'feature' or 'x' key found in original data")
    
    # Reconstruct node mapping (cross-validated across all views)
    num_nodes = features.shape[0]
    node_names, node_id_to_name_full = reconstruct_node_mapping(augmented_views, num_nodes) # Cross-validate gene name mapping across all views

    # Align labels using the validated mapping
    if node_id_to_name_full:
        labels = align_and_verify_labels(node_id_to_name_full, original['label'], num_nodes)# Re-align labels to verified mapping; spot-checks known drivers

    if len(node_names) != len(labels):
        raise ValueError(f"FATAL: After alignment, {len(node_names)} names vs {len(labels)} labels")    # Misalignment would silently corrupt all downstream scoring
    logger.info(f"Node names and labels are aligned: {len(node_names)} genes")
    
    num_original_nodes = features.shape[0]
    if len(train_mask_original) != num_original_nodes:
        raise ValueError(
            f"Train mask size ({len(train_mask_original)}) doesn't match "
            f"original graph nodes ({num_original_nodes})"
        )                                                                           # Mask must cover exactly the original graph's nodes

    # Per-fold normalization: fit scaler on this fold's training data only
    raw_features = original.get('x_raw', features)                                  # Prefer x_raw (un-normalized) to avoid double normalization
    features, augmented_views = normalize_features_per_fold(
        raw_features, augmented_views, train_mask_original, fold_idx
    )

    # Analyze class distribution
    train_labels = labels[train_mask_original]
    num_pos = (train_labels == 1).sum().item()
    num_neg = (train_labels == 0).sum().item()
    imbalance_ratio = num_neg / num_pos if num_pos > 0 else float('inf')            # Driver:Non-driver ratio; informs focal loss importance
    
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
        hidden_channels = hidden_channels
        projection_dim = projection_dim
        num_layers = num_layers
    
    # Create model
    model = create_cancer_driver_model(
        num_features=features.shape[1],  # Use features variable instead
        hidden_channels=hidden_channels,
        projection_dim=projection_dim,
        num_layers=num_layers,
        attention_mode=attention_mode,
        temperature=temperature,
        dropout=dropout,
        aggregation=aggregation,
        pathway_aggregator=pathway_aggregator,
        num_heads=num_heads,
        negative_slope=negative_slope,
        concat_heads=concat_heads,
        device=device
    )                                                           # Instantiate model with resolved architecture params
    
    # Setup device (GPU or CPU)
    model, device = setup_device(model)                         # Move Model to GPU; returns actual device in case of fallback
    actual_model = model                                        # Alias used throughout to access the un-wrapped model directly

    # Print model architecture and parameter count
    print(f"\n{'='*60}")
    print("MODEL ARCHITECTURE")
    print(f"{'='*60}")
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"{'='*60}\n")
    
    # Initialize EMA 
    ema = EMA(
        model = actual_model,
        decay = decay,  # High decay  = very smooth
        device = torch.device('cpu')
    )                                                           # Shadow params on CPU to preserve GPU memory
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.999)                                      # Standard Adam momentum; beta2 = 0.999 smooths second moment
    )

    # Learning rate scheduler (maximize NDCG) - Less aggressive for stability
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',                                             # Maximize NDCG@50
        factor=scheduler_factor,
        patience=scheduler_patience,  # Increased from 20 (more patience)
        min_lr=1e-6                                             # Floor to prevent LR from reaching 0
    )

    # Warmup scheduler
    warmup_scheduler = WarmupScheduler(
        optimizer,
        warmup_epochs=10,
        initial_lr=learning_rate / 100,                         # Start at 1% of target LR to stabilize early GNN gradients
        target_lr=learning_rate
    )
    
    # Early stopping based on NDCG@50
    early_stopping = EarlyStopping(patience = early_stopping_patience, min_delta=0.0001, mode='max')    # Monitors smoothed NDCG@50
    
    # Mixed precision scaler
    scaler = torch.amp.GradScaler('cuda') if mixed_precision and torch.cuda.is_available() else None    # FP16 gradient scaler; None disables mixed precision
    
    # Pre-process original data ONCE and keep on device
    logger.info("Pre-processing original graph data...")
    original_device = {
        'x': features.to(device),  # Use features variable
        'edge_index': original['edge_index'].to(device),
        'ollivier_curvature': original['ollivier_curvature'].to(device),        # Keep original graph permanently on GPU; avoids repeated CPU->GPU transfers each epoch
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
    }                                                                   # Accumulates per-epoch metrics for post-training plotting
    
    best_val_ndcg = 0.0                                                 # Tracks the best validation NDCG@50 seen so far for each checkpoint decisions
    best_metrics = None                                                 # Stores the full metrics dict at the best checkpoint epoch
    smoothed_ndcg = 0.0                                                 # EMA-smoothed NDCG@50; used for scheduler and early stopping to reduce noise
    metric_ema_alpha = 0.3                                              # EMD weight for new observations; lower=smoother signal, slower to react
    
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
    print(f"Loss type: Contrastive ({contrastive_weight:.0%}) + {ranking_loss_type.capitalize()} Ranking ({1-contrastive_weight:.0%})")
    print(f"Ranking margin: {ranking_margin} | Focal gamma: {focal_gamma} | Use focal: {use_focal}")
    print(f"Early stopping metric: NDCG@50")
    print(f"Attention mode: {attention_mode}")
    print(f"{'='*80}\n")

    # Precompute label mappings and contrastive alignment for all view pairs
    precomputed_mappings, contrastive_alignment = precompute_contrastive_alignment(
        augmented_views, labels, train_mask_original
    )

    for epoch in range(num_epochs):
        # Clear cache at start of epoch
        torch.cuda.empty_cache()                                                # Free fragmented GPU memory at the start of each epoch
        gc.collect()

        # Ensure model is in training mode (validation sets eval mode)
        model.train()                                                           # Re-enable dropout and batch norm training mode (validation sets eval mode)

        if epoch < 10:
            warmup_scheduler.step()                                             # Linearly ramp LR from initial_lr -> target_lr over the first 10 epochs

        # Zero gradients once
        optimizer.zero_grad()                                                   # Clear gradients once at the start of the accumulation window
        
        accumulated_loss = 0.0
        accumulated_contrastive_loss = 0.0
        accumulated_orig_ranking_loss = 0.0
        accumulated_metrics = {'train_ndcg': 0.0}
        successful_steps = 0                                                    # Counts accumulation steps that completed without OOM

        # Create ranking criterion once per epoch (more efficient)
        ranking_criterion = RankingLoss(
            margin=ranking_margin,
            loss_type=ranking_loss_type,
            num_samples=ranking_loss_samples,
            focal_gamma=focal_gamma,
            use_focal=use_focal
        )                                                                       # Create once per epoch; cheaper than creating inside the inner

        # Process views with contrastive + ranking loss
        for accum_step in range(gradient_accumulation_steps):
            try:
                # Sample TWO views for contrastive learning
                # Use deterministic pairing based on epoch for stability
                num_views = len(augmented_views)
                if num_views >= 2:
                    # Deterministic pairing: (epoch * steps + step) % num_views ensures each pair is used 
                    view1_idx = (epoch * gradient_accumulation_steps + accum_step) % num_views
                    view2_idx = (epoch * gradient_accumulation_steps + accum_step + 1) % num_views
                else:
                    # If only one view, use it for both
                    view1_idx, view2_idx = 0, 0

                # === Load View 1 ===
                view1 = augmented_views[view1_idx]
                view1_x = view1['x'].to(device)
                
                # Adding Gaussian noise to reduce learning rate
                if model.training:
                    view1_x = view1_x + torch.randn_like(view1_x) * 0.01        # Small Gaussian noise augments the view further; acts as stochastic regularization
                
                view1_edge_index = view1['edge_index'].to(device)
                view1_curvature = view1.get('ollivier_curvature')

                if view1_curvature is not None:
                    view1_curvature = view1_curvature.to(device)
                else:
                    view1_curvature = actual_model.compute_augmented_curvature(
                        view1_edge_index, None, view1_x,
                        original_curvature=original_device['ollivier_curvature'],
                        original_edge_index=original_device['edge_index'],
                        method='transfer'
                    )                                                           # Fallback: Transfer curvature from the original graph when view dict doesn't store its own

                view1_curvature = actual_model.validate_and_fix_curvature_dimensions(
                    view1_edge_index, view1_curvature, f"View1_Step{accum_step}"
                )                                                               # Ensure curvature tensor aligns to this view's directed edge index

                # === Load View 2 ===
                view2 = augmented_views[view2_idx]
                view2_x = view2['x'].to(device)
                
                # Adding Gaussian noise to reduce the learning rate
                if model.training:
                    view2_x = view2_x + torch.randn_like(view2_x) * 0.01
                
                view2_edge_index = view2['edge_index'].to(device)
                view2_curvature = view2.get('ollivier_curvature')

                if view2_curvature is not None:
                    view2_curvature = view2_curvature.to(device)
                else:
                    view2_curvature = actual_model.compute_augmented_curvature(
                        view2_edge_index, None, view2_x,
                        original_curvature=original_device['ollivier_curvature'],
                        original_edge_index=original_device['edge_index'],
                        method='transfer'
                    )

                view2_curvature = actual_model.validate_and_fix_curvature_dimensions(
                    view2_edge_index, view2_curvature, f"View2_Step{accum_step}"
                )

                # Forward pass with mixed precision
                if scaler:
                    with torch.amp.autocast('cuda'):                            # FP16 autocast region; reduces memory and speed up tensor ops
                        # === Encode both views ===
                        embeddings1, _ = gradient_checkpoint(
                            actual_model._encoder_wrapper,
                            view1_x, view1_edge_index, view1_curvature,
                            return_all_layers, False,
                            use_reentrant=False
                        )                                                       # Recompute activations on backward to save memory
                        # Aggregate pathway outputs
                        h1, _ = actual_model.aggregator(embeddings1, return_attention=False)

                        embeddings2, _ = gradient_checkpoint(
                            actual_model._encoder_wrapper,
                            view2_x, view2_edge_index, view2_curvature,
                            return_all_layers, False,
                            use_reentrant=False
                        )
                        h2, _ = actual_model.aggregator(embeddings2, return_attention=False)

                        # === Contrastive Loss ===
                        # Project to contrastive space
                        z1 = F.normalize(actual_model.projection(h1), dim=-1)   # L2-normalize: dot-product = cosine similarity
                        z2 = F.normalize(actual_model.projection(h2), dim=-1)

                        # Align common nodes between views for contrastive loss
                        if view1_idx != view2_idx and (view1_idx, view2_idx) in contrastive_alignment:
                            align_idx1, align_idx2 = contrastive_alignment[(view1_idx, view2_idx)]  # Select only nodes present in both views
                            align_idx1 = align_idx1.to(device)
                            align_idx2 = align_idx2.to(device)
                            z1_common = z1[align_idx1]
                            z2_common = z2[align_idx2]
                        else:
                            # Same view used for both (single view fallback)
                            min_nodes = min(z1.size(0), z2.size(0))                         # Same view fallback: truncate to shared size
                            z1_common = z1[:min_nodes]
                            z2_common = z2[:min_nodes]

                        contrastive_loss = actual_model.compute_contrastive_loss(z1_common, z2_common)

                        # Contrastive-only loss (ranking is on original graph)
                        loss = contrastive_weight * contrastive_loss                        # Contrastive component of the total loss
                        loss = torch.clamp(loss, max=10.0)                                  # Hardcap: Prevents a single bad batch from causing gradient explosion
                        loss = loss / gradient_accumulation_steps                           # Normalize gradient by number of accumulation steps
                else:
                    # No mixed precision - same logic
                    embeddings1, _ = gradient_checkpoint(
                        actual_model._encoder_wrapper,
                        view1_x, view1_edge_index, view1_curvature,
                        return_all_layers, False,
                        use_reentrant=False
                    )
                    h1, _ = actual_model.aggregator(embeddings1, return_attention=False)

                    embeddings2, _ = gradient_checkpoint(
                        actual_model._encoder_wrapper,
                        view2_x, view2_edge_index, view2_curvature,
                        return_all_layers, False,
                        use_reentrant=False
                    )
                    h2, _ = actual_model.aggregator(embeddings2, return_attention=False)

                    # Contrastive Loss
                    z1 = F.normalize(actual_model.projection(h1), dim=-1)
                    z2 = F.normalize(actual_model.projection(h2), dim=-1)

                    # Align common nodes between views for contrastive loss
                    if view1_idx != view2_idx and (view1_idx, view2_idx) in contrastive_alignment:
                        align_idx1, align_idx2 = contrastive_alignment[(view1_idx, view2_idx)]
                        align_idx1 = align_idx1.to(device)
                        align_idx2 = align_idx2.to(device)
                        z1_common = z1[align_idx1]
                        z2_common = z2[align_idx2]
                    else:
                        # Same view used for both (single view fallback)
                        min_nodes = min(z1.size(0), z2.size(0))
                        z1_common = z1[:min_nodes]
                        z2_common = z2[:min_nodes]

                    contrastive_loss = actual_model.compute_contrastive_loss(z1_common, z2_common)

                    # Contrastive-only loss (ranking is on original graph)
                    loss = contrastive_weight * contrastive_loss
                    loss = torch.clamp(loss, max=10.0)
                    loss = loss / gradient_accumulation_steps

                # Backward
                if scaler:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                accumulated_loss += loss.item() * gradient_accumulation_steps
                accumulated_contrastive_loss += contrastive_loss.item()
                successful_steps += 1

                # Clean up immediately (CRITICAL for OOM)
                del view1_x, view1_edge_index, view1_curvature                      # Explicitly free GPU tensors before the next accumulation step
                del view2_x, view2_edge_index, view2_curvature
                del embeddings1, embeddings2, h1, h2, z1, z2
                del loss, contrastive_loss
                del view1, view2, z1_common, z2_common

                # Aggressive CUDA cache clearing
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()  # Ensures all pending CUDA ops finish before checking summary
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    logger.warning(f"OOM at epoch {epoch}, step {accum_step}")
                    torch.cuda.empty_cache()
                    gc.collect()
                    continue                                                        # Skip this accumulation step and try the next one+
                else:
                    raise e
        
        # === Ranking loss on original graph ===
        # Bridge train-evaluate gap: train ranking on the full graph used for evaluation
        try:
            if scaler:
                with torch.amp.autocast('cuda'):
                    
                    # Adding gaussian noise to the feature matrix
                    noisy_x = original_device['x'] + torch.randn_like(original_device['x']) * 0.01          # Gaussian noise on original features; prevents the ranking head from memorizing exact feature values
                    
                    orig_scores, _ = model.forward(
                        noisy_x,
                        original_device['edge_index'],
                        original_device['ollivier_curvature'],
                        return_all_layers=return_all_layers
                    )
                    orig_ranking_loss = ranking_criterion(orig_scores, labels, train_mask_original)
                    orig_loss = (1 - contrastive_weight) * ranking_loss_scale * orig_ranking_loss           # Ranking component; scale compensates for magnitude difference vs contrastive loss
                scaler.scale(orig_loss).backward()
            else:
                
                # Adding gaussian noise to the feature matrix
                noisy_x = original_device['x'] + torch.randn_like(original_device['x']) * 0.01
                
                orig_scores, _ = model.forward(
                    noisy_x,
                    original_device['edge_index'],
                    original_device['ollivier_curvature'],
                    return_all_layers=return_all_layers
                )
                orig_ranking_loss = ranking_criterion(orig_scores, labels, train_mask_original)
                orig_loss = (1 - contrastive_weight) * ranking_loss_scale * orig_ranking_loss
                orig_loss.backward()

            accumulated_orig_ranking_loss += orig_ranking_loss.item()

            del orig_scores, orig_ranking_loss, orig_loss
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except RuntimeError as e:
            if "out of memory" in str(e):
                logger.warning(f"OOM during original graph ranking at epoch {epoch}")
                torch.cuda.empty_cache()
                gc.collect()
            else:
                raise e

        # Check if we had any successful steps
        if successful_steps == 0:
            logger.error(f"Epoch {epoch}: All steps failed with OOM. Skipping epoch.")
            optimizer.zero_grad()
            torch.cuda.empty_cache()
            continue                                        # All accumulation steps failed with OOM; skip gradient update entirely
        
        # Average loss (combined contrastive + ranking for accurate reporting)
        avg_contrastive_raw = accumulated_contrastive_loss / max(successful_steps, 1)
        avg_loss = (contrastive_weight * avg_contrastive_raw +
                    (1 - contrastive_weight) * ranking_loss_scale * accumulated_orig_ranking_loss)      # Combined loss for logging
        
        # Check for NaN
        if math.isnan(avg_loss):
            logger.error(f"Epoch {epoch}: NaN loss")
            optimizer.zero_grad()                           # NaN loss means something exploded; skip this epoch's update
            continue
        
        # Gradient clipping
        if scaler is not None:
            scaler.unscale_(optimizer)                      # Unscale FP16 gradients before clipping so clip_grad_norm_ sees the true magnitudes
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)    # Gradient clipping; prevents single large update from destabilizing training
        
        # Optimizer step
        if scaler is not None:
            scaler.step(optimizer)                          # FP16-aware optimizer step; skips update if gradients contain Inf/NaN values
            scaler.update()                                 # Adjust the loss scaling factor for the next step
        else:
            optimizer.step()
        
        ema.update(actual_model)                            # Blend current weights into shadow params: shadow = decay*shadow + (1-decay)*current
        
        optimizer.zero_grad()
        torch.cuda.empty_cache()

        # Compute training NDCG using EMA weights (matches validation evaluation)
        if epoch % validation_frequency == 0 or epoch == num_epochs - 1:
            ema.apply_shadow(actual_model)                  # Swap live weights for EMA shadow weights before evaluation
            with torch.no_grad():
                train_scores, _ = model.forward(
                    original_device['x'],
                    original_device['edge_index'],
                    original_device['ollivier_curvature'],
                    return_all_layers=return_all_layers
                )
                accumulated_metrics['train_ndcg'] = compute_ndcg(
                    train_scores[train_mask_original],
                    labels[train_mask_original], k=50
                )                                           # NDCG@50 on training set using EMA weights; tracks training progress without overfitting signal
            ema.restore(actual_model)                       # Restore live weights after training metric computation
            torch.cuda.empty_cache()
        else:
            accumulated_metrics['train_ndcg'] = history['train_ndcg'][-1] if history['train_ndcg'] else 0.0

        # Validation (Memory Optimization: configurable frequency to save time and memory)
        if epoch % validation_frequency == 0 or epoch == num_epochs - 1:
            
            # Apply EMA weights before validation
            ema.apply_shadow(actual_model)                  # Swap to EMA weights again for validation
            
            val_metrics = evaluate_with_ranking_metrics(
                actual_model, original_device, labels, val_mask_original,
                curvature_type='ollivier', device=device
            )
            
            # Restore original weights after validation
            ema.restore(actual_model)                       # Restore live weights; EMA weights only used for evaluation

            # CRITICAL: Clear validation activations immediately
            torch.cuda.empty_cache()
            gc.collect()

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

            # Update smoothed NDCG (EMA of validation metric)
            current_ndcg = val_metrics.get('ndcg@50', 0)
            if smoothed_ndcg == 0.0:
                smoothed_ndcg = current_ndcg  # Initialize with first real observation
            else:
                smoothed_ndcg = metric_ema_alpha * current_ndcg + (1 - metric_ema_alpha) * smoothed_ndcg        # EMA smoothing: reduces noise from single-epoch fluctuation

            # Use smoothed NDCG for scheduler and early stopping decisions
            if epoch >= 10:
                scheduler.step(smoothed_ndcg)                   # Feed smoothed metric to ReduceLROnPlateau; skip warmup epochs to avoid premature LR reduction

            if early_stopping(smoothed_ndcg, epoch):
                logger.info(f"Early stopping triggered at epoch {epoch}")
                logger.info(f"Best smoothed NDCG@50: {early_stopping.best_score:.4f} at epoch {early_stopping.best_epoch}")
                break                                           # Exit training loop; best model already saved to disk
            
        else:
            # Skip validation - reuse previous metrics as dict
            # Reuse previous validation metrics dict to keep history arrays the same length
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
            

        
        # Store history
        history['train_loss'].append(avg_loss)
        history['val_auroc'].append(val_metrics.get('auroc', 0))
        history['train_ndcg'].append(accumulated_metrics['train_ndcg'])
        history['val_auprc'].append(val_metrics.get('auprc', 0))
        history['val_f1'].append(val_metrics.get('f1', 0))
        history['val_ndcg@50'].append(val_metrics.get('ndcg@50', 0))
        history['val_precision@50'].append(val_metrics.get('precision@50', 0))
        history['val_mrr'].append(val_metrics.get('mrr', 0))
        history['learning_rate'].append(optimizer.param_groups[0]['lr'])        # Track LR to visualize scheduler behaviour
        
        # Logging
        if epoch % 10 == 0:
            print(
                f"Epoch {epoch:3d} | "
                f"Loss: {avg_loss:.4f} (C:{avg_contrastive_raw:.3f} R_orig:{accumulated_orig_ranking_loss:.3f}) | "
                f"Steps: {successful_steps}/{gradient_accumulation_steps} | "
                f"NDCG@50: {val_metrics.get('ndcg@50', 0):.4f} (smooth: {smoothed_ndcg:.4f}) | "
                f"LR: {optimizer.param_groups[0]['lr']:.6f}"
            )
        
        # Save best model
        if val_metrics.get('ndcg@50', 0) > best_val_ndcg:
            best_val_ndcg = val_metrics.get('ndcg@50', 0)
            best_metrics = val_metrics

            actual_model.save_checkpoint(
                str(model_path), epoch, optimizer, best_metrics,
                metadata={
                    'fold': fold_idx,
                    'ema_state': ema.state_dict()                       # Save EMA state alongside weights for exact reproducibility
                    }
            )

            # CRITICAL: Clear checkpoint tensors from GPU memory
            torch.cuda.empty_cache()
            gc.collect()

        # Note: early stopping is checked inside the validation block above (line 1161).
        # Do NOT check again here — it would double-increment the counter on
        # validation epochs and increment on stale metrics for non-validation epochs.
    
    # Load best model
    checkpoint = actual_model.load_checkpoint(str(model_path), optimizer, device)           # Reload best checkpoint; discards any improvement-free epochs after the best 
    epoch_loaded = checkpoint['epoch']
    
    # Restore EMA state if available
    if 'metadata' in checkpoint and 'ema_state' in checkpoint['metadata']:
        ema.load_state_dict(checkpoint['metadata']['ema_state'])                            # Restore EMA shadow params to match the best epoch's state
    
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
        num_permutations=1000,                                                              # Permutation test: 1000 shuffles give stable empirical p-values
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
            ('learning_rate', 'Learning Rate', 'log')                                               # Log scale reveals LR decay steps clearly
        ]
        
        for idx, (metric_key, title, scale) in enumerate(metrics_to_plot):
            ax = axes[idx // 3, idx % 3]                                                            # Fill a 2x3 grid row by row
            
            for fold_idx, history in enumerate(histories, 1):
                if metric_key in history and len(history[metric_key]) > 0:
                    epochs = range(1, len(history[metric_key]) + 1)
                    ax.plot(epochs, history[metric_key], label=f'Fold {fold_idx}', alpha=0.7)       # Alpha reduces visual clutter when folds overlap
            
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric_key)
            ax.set_title(title)
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
            
            if scale == 'log':
                ax.set_yscale('log')                                                                # Log scale for learning rate so plateau drops are visible
        
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
        x = np.arange(len(metrics_cols))                                                    # x-axis positions for metric groups
        width = 0.15                                                                        # Bar width; 5 folds x 0.15 = 0.75 total width per group
        
        for idx, fold_idx in enumerate(df['Fold']):
            offset = (idx - len(df) / 2) * width                                            # Center the group of bars around each x position
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
        bp = axes[1].boxplot(box_data, tick_labels=metrics_cols, patch_artist=True)
        
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
            patch.set_alpha(0.7)                                                            # Semi-transparent boxes show underlying data distribution better
        
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
        gene_ids = all_fold_dfs[0]['gene_id'].values                                            # Gene IDs are the same cross all folds; use fold 0 as reference
        gene_names = all_fold_dfs[0]['gene_name'].values                                        # Gene names likewise identifcal across folds
        true_labels = all_fold_dfs[0]['true_label'].values                                      # Ground truth labels do not change across folds
        
        # Collect scores from each fold
        scores_matrix = np.array([df['driver_score'].values for df in all_fold_dfs])            # [num_folds, num_genes] raw driver scores
        pvalues_matrix = np.array([df['adjusted_pvalue'].values for df in all_fold_dfs])        # [num_folds, num_genes] BH-adjusted p-values
        ranks_matrix = np.array([df['rank'].values for df in all_fold_dfs])                     # [num_folds, num_genes] within-fold ranks
        
        # Compute aggregate statistics
        mean_scores = scores_matrix.mean(axis=0)                                                # Average driver score across all folds; primary ranking criterion
        std_scores = scores_matrix.std(axis=0)                                                  # Spread across folds; high std = inconsistent prediction
        median_scores = np.median(scores_matrix, axis=0)                                        # Robust central tendency; less sensitive to outlier folds
        
        mean_pvalues = pvalues_matrix.mean(axis=0)                                              # Mean adjusted p-value; supplementary significance measure
        median_ranks = np.median(ranks_matrix, axis=0)                                          # Borda count-stype rank aggregation; median rank is more robust than mean
        
        # Count how many folds each gene was significant in
        significant_counts = (pvalues_matrix < 0.05).sum(axis=0)                                # Number of folds where this gene was FDR-significant
        consensus_significant = significant_counts >= (len(all_fold_dfs) / 2)                   # Gene must be significant in atleast half the folds
        
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
        aggregate_df = aggregate_df.sort_values('mean_score', ascending=False)                  # Rank genes by mean score across folds
        aggregate_df['aggregate_rank'] = range(1, len(aggregate_df) + 1)                        # 1-indexed global rank
        
        # Reorder columns
        aggregate_df = aggregate_df[[
            'aggregate_rank', 'gene_id', 'gene_name', 'is_known_driver',
            'mean_score', 'std_score', 'median_score', 
            'mean_adjusted_pvalue', 'median_rank',
            'folds_significant', 'total_folds', 'consensus_significant'
        ]]
        
        # Get consensus significant genes
        consensus_genes = aggregate_df[aggregate_df['consensus_significant']]                   # Subset: unknown genes significant in majority of folds = most robust novel predictions
        
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
    parser.add_argument('--num_epochs', type=int, default=1200,
                        help='Number of training epochs per fold')
    parser.add_argument('--hidden_channels', type=int, default=128,
                        help='Hidden layer channels')
    parser.add_argument('--projection_dim', type=int, default=96,
                        help = 'Projection Dimensions')
    parser.add_argument('--temperature', type=float, default=0.3,
                        help = 'Choose the temperature for contrastive loss.')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Choose the proportion of neurons to drop')
    parser.add_argument('--attention_mode', type=str, default='hybrid',
                        choices=['standard', 'edge_feature', 'bias', 'gated', 'hybrid'],
                        help='Attention mode for message passing')
    parser.add_argument('--pathway_aggregator', type = str, default='attention',
                        choices= ['attention','concat', 'hierarchical', 'mean'])
    parser.add_argument('--aggregation', type=str, default='max',
                        help = 'Aggregation Method')
    parser.add_argument('--learning_rate', type=float, default=5e-4,
                        help='Set the learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay for AdamW optimizer')
    parser.add_argument('--num_heads', type=int, default=4,
                        help='Number of attention heads')
    parser.add_argument('--num_layers', type = int, default = 2,
                        help = 'Choose the number of GNN layers')
    parser.add_argument('--concat_heads', action='store_true', default=False,
                        help='Concatenate attention heads (vs average)')
    parser.add_argument('--ranking_loss_type', type=str, default='bpr',
                        choices=['pairwise', 'sampled_pairwise', 'bpr', 'listwise', 'approxndcg'],
                        help='Type of ranking loss (bpr is most stable)')
    parser.add_argument('--ranking_margin', type=float, default=0.5,
                        help='Margin for pairwise ranking loss')
    parser.add_argument('--ranking_loss_samples', type=int, default=512,
                        help='Number of pairs to sample for calculating ranking loss')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                        help='Focal loss gamma (higher = more focus on hard examples)')
    parser.add_argument('--use_focal', action='store_true', default=False,
                        help='Use focal weighting in ranking loss')
    parser.add_argument('--contrastive_weight', type=float, default=0.2,
                        help='Weight for contrastive loss (0-1, reduced to 0.1 for better balance). Ranking weight = 1 - contrastive_weight')
    parser.add_argument('--scheduler_patience', type=int, default=5,
                        help = 'Patience for scheduler to stop training on learning rate plateau')
    parser.add_argument('--early_stopping_patience', type=int, default=15,
                        help='Early stopping patience')
    parser.add_argument('--ranking_loss_scale', type=float, default=5.0,
                        help = 'Scale the ranking loss to prevent sharp increase due to contrastive loss')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8,
                    help='Number of gradient accumulation steps (increased for stability)')
    parser.add_argument('--mixed_precision', action='store_true', default=False,
                        help='Use mixed precision training')
    parser.add_argument('--max_views_per_step', type=int, default=2,
                        help='Maximum augmented views to use per training step')
    parser.add_argument('--attention_chunk_size', type=int, default=1000,
                    help='Process nodes in chunks for attention (lower = less memory)')
    parser.add_argument('--return_all_layers', action='store_true', default = False,
                        help = 'Return all layers')
    parser.add_argument('--decay', type=float, default=0.999,
                        help = 'Decay rate for Exponential Moving Average for model weights')
    # ============================================================================
    # MEMORY OPTIMIZATION ARGUMENTS
    # ============================================================================
    parser.add_argument('--emergency_mode', action='store_true',
                        help='Use drastically reduced model for OOM issues (auto-sets optimal params)')
    parser.add_argument('--max_augmented_views', type=int, default=2,
                        help='Maximum number of augmented views to keep in memory (default: 3, recommended: 2)')
    parser.add_argument('--reduce_model_size', action='store_true', default=False,
                        help='Reduce model dimensions (hidden=64, proj=32, layers=2, heads=1)')
    parser.add_argument('--validation_frequency', type=int, default=10,
                        help='Validate every N epochs (default: 10, lower saves memory)')
    parser.add_argument('--negative_slope', type=float, default=0.2,
                        help='Negative slope for LeakyReLU in GNN layers')
    parser.add_argument('--scheduler_factor', type=float, default=0.7,
                        help='Factor by which LR is reduced on plateau')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    args = parser.parse_args()

    # ============================================================================
    # EMERGENCY MODE AUTO-CONFIGURATION
    # ============================================================================
    if args.emergency_mode:
        logger.warning("⚠️  EMERGENCY MODE ACTIVATED - Using aggressive memory optimizations")
        args.reduce_model_size = True                                                       # Cut hidden=64, proj=32, layers=2, heads=1
        args.max_augmented_views = 2                                                        # Keep only 2 views
        args.gradient_accumulation_steps = 8                                                # Accumulate more steps to compensate for smaller effective batch
        args.attention_chunk_size = 500                                                     # Process fewer nodes per chunk in MultiLayerAttention
        args.mixed_precision = True                                                         # FP16 halves activation memory
        args.validation_frequency = 20                                                      # Validate less often to reduce peak memory during eval
        logger.warning(f"  → Hidden channels: 64, Projection: 32, Layers: 2, Heads: 1")
        logger.warning(f"  → Max views: 2, Chunk size: 500, Grad accum: 8")
        logger.warning(f"  → Mixed precision: ON, Validation frequency: 20")
    # ============================================================================

    # Create directories
    models_dir = Path(args.model_out_dir)
    results_dir = Path(args.train_metrics_dir)
    models_dir.mkdir(exist_ok=True, parents=True)                                           # Create output directories if they don't exist
    results_dir.mkdir(exist_ok=True, parents=True)
    
    if args.model_out_prefix:
        results_subdir = results_dir / args.model_out_prefix
        results_subdir.mkdir(exist_ok=True, parents=True)
        results_dir = results_subdir                                                        # Namespace results under the prefix subdirectory to avoid collisions between runs
    
    set_seed(args.seed)

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
        data = pickle.load(f)                                                               # Load the pre-processed pickle produced by curvature_pipeline.py
    
    original = data['original']                                                             # Original graph with features, edge_index, labels, curvature, kfold_splits
    augmented_views = data['augmented_views']                                               # List of Schur Complement augmented graph dicts

    # ============================================================================
    # LIMIT AUGMENTED VIEWS (Memory Optimization)
    # ============================================================================
    if len(augmented_views) > args.max_augmented_views:
        logger.warning(f"Limiting augmented views from {len(augmented_views)} to {args.max_augmented_views} to save memory")
        augmented_views = augmented_views[:args.max_augmented_views]                        # Trim to memory budget; earlier views are kept
    logger.info(f"Using {len(augmented_views)} augmented views for training")
    # ============================================================================

    # Validate and preprocess
    print("\n" + "="*80)
    print("VALIDATING AND PREPROCESSING DATA")
    print("="*80)
    
    validate_graph_data(original, "Original Graph")                                         # Check for Inf/NaN/invalid edge indices before any training
    
    # Sanitize features
    features = original.get('feature', original.get('x'))
    if torch.isnan(features).any() or torch.isinf(features).any():
        features = torch.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)             # Clean raw features in place before scaler fitting
        if 'feature' in original:
            original['feature'] = features
        else:
            original['x'] = features
    
    # Preprocess curvature
    original = preprocess_curvature_data(original, curvature_type='ollivier')               # Align curvature tensor dimensions to edge_index
    
    for i, view in enumerate(augmented_views):
        augmented_views[i] = preprocess_curvature_data(view, curvature_type='ollivier')     # Same alignment for each augmented view
    
    print("✓ Data validation and preprocessing complete")
    print("="*80 + "\n")
    
    labels = original['label']
    kfold_splits = original['kfold_splits']
    
    # Filter folds if specified
    if args.specific_folds:
        num_folds = len(kfold_splits)
        folds_to_train = [i-1 for i in args.specific_folds if 1 <= i <= num_folds]          # Convert 1-indexed CLI input to 0-indexed list
        kfold_splits = [kfold_splits[i] for i in folds_to_train]                            # Select only the requested folds
    elif args.num_folds:
        kfold_splits = kfold_splits[:args.num_folds]                                        # Take the first N folds
    
    print(f"\n{'='*80}")
    print(f"K-FOLD CROSS-VALIDATION: {len(kfold_splits)} FOLDS")
    print(f"{'='*80}")
    
    # Train each fold
    all_fold_metrics = []                                                                   # Collect best_metrics dict from each fold
    all_fold_histories = []                                                                 # Collects training history dict from each fold for plotting    
    fold_models = []                                                                        # Collects trained model objects (kept in memory for potential ensemble use)
    all_fold_scored_genes = []                                                              # Collects scored gene DataFrames for cross-fold aggregation
    
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
            scheduler_patience=args.scheduler_patience,
            early_stopping_patience=args.early_stopping_patience,
            projection_dim=args.projection_dim,
            hidden_channels=args.hidden_channels,
            concat_heads=args.concat_heads,
            ranking_loss_type=args.ranking_loss_type,
            ranking_loss_scale=args.ranking_loss_scale,
            ranking_margin=args.ranking_margin,
            focal_gamma=args.focal_gamma,
            use_focal=args.use_focal,
            contrastive_weight=args.contrastive_weight,
            reduce_model_size=args.reduce_model_size,
            validation_frequency=args.validation_frequency,
            decay=args.decay,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            dropout=args.dropout,
            temperature=args.temperature,
            pathway_aggregator=args.pathway_aggregator,
            negative_slope=args.negative_slope,
            scheduler_factor=args.scheduler_factor
        )                                                                                   # Train one fold end-to-end
        
        all_fold_metrics.append(best_metrics)
        all_fold_histories.append(history)
        fold_models.append(model)
        all_fold_scored_genes.append(scored_genes_df)
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()                                                        # Release GPU memory between folds; previous fold's model stays reference by fold_models
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
    }                                                                                       # Mean of each metric across folds
    
    std_metrics = {
        key: np.std([m.get(key, 0) for m in all_fold_metrics])
        for key in ['ndcg@50', 'auroc', 'auprc', 'precision@50', 'mrr']
    }                                                                                       # Std of each metric across folds
    
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
    metrics_df = pd.DataFrame(metrics_df_data)                                              # Append mean and std as summary rows at the bottom of the per-fold table
    
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
    metrics_df.to_csv(metrics_file, index=False)                                            # Save per-fold metrics + summary rows
    print(f"✓ Saved metrics to: {metrics_file}")
    
    # Save training histories
    history_file = results_dir / f"{prefix}training_history.pkl"
    with open(history_file, 'wb') as f:
        pickle.dump(all_fold_histories, f)                                                  # Save training curves for offline plotting
    print(f"✓ Saved training histories to: {history_file}")
    
    # Plot training curves
    plot_training_curves(all_fold_histories, results_dir, prefix)                           # Generate loss/metric curve plots
    
    # Plot metrics comparison
    plot_metrics_comparison(metrics_df_data, results_dir, prefix)                           # Generate bar and box plots across folds
    
    # Aggregate gene scores across folds
    print(f"\n{'='*80}")
    print("AGGREGATING GENE SCORES ACROSS FOLDS")
    print(f"{'='*80}")
    
    aggregate_gene_scores(all_fold_scored_genes, results_dir, prefix)                       # Compute cross-fold consensus driver predictions
    
    print(f"\n{'='*80}")
    print("TRAINING COMPLETE")
    print(f"{'='*80}")
    print(f"Models saved to: {models_dir}")
    print(f"Results saved to: {results_dir}")
    print(f"{'='*80}\n")
    

if __name__ == '__main__':
    logger = get_logger(__name__)
    main()