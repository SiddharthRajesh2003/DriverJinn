#!/usr/bin/env python

import torch
import torch.nn.functional as F
from typing import Tuple, Dict
import gc
import pickle
import argparse
import json
import random
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import optuna
from optuna.pruners import HyperbandPruner
from optuna.samplers import TPESampler

from model.DriverGenePredictor import ContrastiveDriverGenePredictor, compute_ndcg
from utils.logging_manager import get_logger
from model.support_models import RankingLoss, EMA

logger = get_logger(__name__)


def set_seed(seed: int):
    """Set random seed for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Random seed set to {seed}")


# Memory optimization
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def sanitize_curvature(curvature: torch.Tensor) -> torch.Tensor:
    """Sanitize curvature values"""
    curvature = torch.nan_to_num(curvature, nan=0.0, posinf=0.0, neginf=0.0)
    curvature = torch.clamp(curvature, min=-10.0, max=10.0)
    return curvature


def preprocess_curvature_data(data: Dict, curvature_type: str = 'ollivier') -> Dict:
    """Preprocess curvature data to match edge dimensions"""
    curv_key = f'{curvature_type}_curvature'

    if curv_key not in data:
        logger.warning(f"No '{curv_key}' found in data, skipping preprocessing")
        return data

    edge_index = data['edge_index']
    edge_curvature = data[curv_key]

    edge_curvature = sanitize_curvature(edge_curvature)

    num_edges = edge_index.shape[1]
    num_curvatures = edge_curvature.shape[0]

    if num_curvatures == num_edges:
        data[curv_key] = edge_curvature
        return data

    if num_curvatures * 2 == num_edges:
        # Undirected curvature for directed edges - duplicate for both directions
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

    return data


class HyperparameterSearch:
    """
    Hyperparameter search for ContrastiveDriverGenePredictor using Optuna.

    Uses NDCG@50 as the primary optimization metric for ranking performance.
    Memory-constrained to max 128 hidden channels for H100 compatibility.

    Recommended workflow:
    1. Run hyperparameter search with n_folds=1 (fast, ~50 trials)
    2. Take best hyperparameters
    3. Run full training with train_model.py using all 5 folds
    """

    def __init__(
        self,
        data_path: str,
        output_dir: str,
        n_trials: int = 50,
        n_folds: int = 1,  # Default to 1 fold for faster search
        epochs_per_trial: int = 100,
        patience: int = 20,
        device: torch.device = None,
        study_name: str = None,
        seed: int = 42
    ):
        self.seed = seed
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.n_trials = n_trials
        self.n_folds = n_folds
        self.epochs_per_trial = epochs_per_trial
        self.patience = patience
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.study_name = study_name or f"driver_gene_search_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Load data
        logger.info(f"Loading data from {self.data_path}")
        with open(self.data_path, 'rb') as f:
            data = pickle.load(f)

        self.original = data['original']
        self.augmented_views = data['augmented_views'][:2]  # Limit to 2 views for memory
        self.labels = self.original['label']
        self.kfold_splits = self.original['kfold_splits'][:n_folds]

        # Preprocess data
        self._preprocess_data()

        # Get feature dimension
        self.num_features = self.original['feature'].shape[1] if 'feature' in self.original else self.original['x'].shape[1]

        logger.info(f"Data loaded: {self.original['feature'].shape[0]} nodes, {self.num_features} features")
        logger.info(f"Using {len(self.augmented_views)} augmented views")
        logger.info(f"Device: {self.device}")
        logger.info(f"Trials: {n_trials}, Folds: {n_folds}, Epochs/trial: {epochs_per_trial}")

        self.trial_results = []
        self.best_params = None
        self.best_score = 0.0

    def _preprocess_data(self):
        """Preprocess curvature data for original and augmented views"""
        logger.info("Preprocessing curvature data...")

        # Sanitize features
        feature_key = 'feature' if 'feature' in self.original else 'x'
        features = self.original[feature_key]
        if torch.isnan(features).any() or torch.isinf(features).any():
            features = torch.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
            self.original[feature_key] = features

        # Preprocess original curvature
        self.original = preprocess_curvature_data(self.original, curvature_type='ollivier')

        # Preprocess augmented views
        for i, view in enumerate(self.augmented_views):
            self.augmented_views[i] = preprocess_curvature_data(view, curvature_type='ollivier')

        # Precompute label mappings and contrastive alignment for augmented views
        self._precompute_view_mappings()

        logger.info("✓ Data preprocessing complete")

    def _precompute_view_mappings(self):
        """Precompute orig-to-aug mappings and contrastive alignment for all views."""
        num_original = self.labels.shape[0]
        self.precomputed_mappings = []

        for view in self.augmented_views:
            eliminated_ids = set(view['metadata']['eliminated_node_ids'])
            feature_key = 'feature' if 'feature' in view else 'x'
            aug_size = view[feature_key].shape[0]

            orig_to_aug = {}
            aug_idx = 0
            for orig_idx in range(num_original):
                if orig_idx not in eliminated_ids:
                    orig_to_aug[orig_idx] = aug_idx
                    aug_idx += 1

            self.precomputed_mappings.append({
                'aug_size': aug_size,
                'orig_to_aug': orig_to_aug
            })

        # Precompute contrastive alignment for all view pairs
        self.contrastive_alignment = {}
        for i in range(len(self.augmented_views)):
            for j in range(len(self.augmented_views)):
                if i == j:
                    continue
                orig_to_aug_i = self.precomputed_mappings[i]['orig_to_aug']
                orig_to_aug_j = self.precomputed_mappings[j]['orig_to_aug']
                common_orig_ids = sorted(set(orig_to_aug_i.keys()) & set(orig_to_aug_j.keys()))
                idx_i = torch.tensor([orig_to_aug_i[oid] for oid in common_orig_ids], dtype=torch.long)
                idx_j = torch.tensor([orig_to_aug_j[oid] for oid in common_orig_ids], dtype=torch.long)
                self.contrastive_alignment[(i, j)] = (idx_i, idx_j)

        logger.info(f"Precomputed mappings for {len(self.augmented_views)} views, "
                    f"{len(self.contrastive_alignment)} view pairs")

    def suggest_hyperparameters(self, trial: optuna.Trial) -> Dict:
        """
        Define hyperparameter search space.

        Constrained to max 128 hidden channels for memory efficiency.
        """
        params = {
            # Model architecture
            'hidden_channels': trial.suggest_categorical('hidden_channels', [128, 144]),
            'projection_dim': trial.suggest_categorical('projection_dim', [72, 96]),
            'num_gnn_layers': trial.suggest_int('num_gnn_layers', 2, 4),
            'num_attention_heads': trial.suggest_categorical('num_attention_heads', [1, 2, 4]),
            'attention_mode': trial.suggest_categorical('attention_mode', ['gated', 'hybrid', 'standard', 'edge_feature', 'bias']),
            'pathway_aggregator': trial.suggest_categorical('pathway_aggregator', ['attention', 'concat', 'mean', 'hierarchical']),
            'concat_heads': trial.suggest_categorical('concat_heads', [True, False]),

            # Regularization
            'dropout': trial.suggest_float('dropout', 0.1, 0.4),
            'temperature': trial.suggest_float('temperature', 0.3, 1.0),
            'negative_slope': trial.suggest_float('negative_slope', 0.1, 0.3),

            # Training
            'learning_rate': trial.suggest_float('learning_rate', 5e-5, 5e-3, log=True),
            'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-4, log=True),
            'contrastive_weight': trial.suggest_float('contrastive_weight', 0.1, 0.4),
            'ranking_loss_scale': trial.suggest_float('ranking_loss_scale', 5.0, 15.0),

            # Ranking loss
            'ranking_loss_type': trial.suggest_categorical('ranking_loss_type', ['bpr', 'pairwise']),
            'ranking_margin': trial.suggest_float('ranking_margin', 0.3, 1.0),
            'use_focal': trial.suggest_categorical('use_focal', [True, False]),
            'focal_gamma': trial.suggest_float('focal_gamma', 1.0, 3.0),

            # Message passing
            'aggregation': trial.suggest_categorical('aggregation', ['add', 'mean', 'max']),

            # Optimization
            'gradient_accumulation_steps': trial.suggest_categorical('gradient_accumulation_steps', [8, 12, 16]),
            'use_ema': trial.suggest_categorical('use_ema', [True, False]),
            'ema_decay': trial.suggest_float('ema_decay', 0.99, 0.999),

            # Scheduler
            'scheduler_patience': trial.suggest_int('scheduler_patience', 20, 150),
            'scheduler_factor': trial.suggest_float('scheduler_factor', 0.5, 0.8),
        }

        return params

    def create_model(self, params: Dict) -> ContrastiveDriverGenePredictor:
        """Create model from hyperparameters"""
        model = ContrastiveDriverGenePredictor(
            in_channels=self.num_features,
            hidden_channels=params['hidden_channels'],
            projection_dim=params['projection_dim'],
            num_gnn_layers=params['num_gnn_layers'],
            curvature_types=['positive', 'negative', 'both'],
            hop_types=['one_hop', 'two_hop'],
            num_attention_heads=params['num_attention_heads'],
            use_attention=True,
            attention_mode=params['attention_mode'],
            pathway_aggregator=params['pathway_aggregator'],
            aggregation=params['aggregation'],
            concat=params['concat_heads'],
            negative_slope=params['negative_slope'],
            temperature=params['temperature'],
            dropout=params['dropout'],
            device=self.device
        ).to(self.device)

        return model

    def train_single_fold(
        self,
        model: ContrastiveDriverGenePredictor,
        params: Dict,
        fold_idx: int,
        trial: optuna.Trial
    ) -> Tuple[float, Dict]:
        """Train model on one fold and return validation NDCG@50"""

        fold_data = self.kfold_splits[fold_idx]
        train_mask = torch.tensor(fold_data['train_mask'], dtype=torch.bool)
        val_mask = torch.tensor(fold_data['val_mask'], dtype=torch.bool)

        # Prepare data on device
        feature_key = 'feature' if 'feature' in self.original else 'x'
        original_device = {
            'x': self.original[feature_key].to(self.device),
            'edge_index': self.original['edge_index'].to(self.device),
            'ollivier_curvature': self.original['ollivier_curvature'].to(self.device),
        }
        labels = self.labels.to(self.device)

        # Optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=params['learning_rate'],
            weight_decay=params['weight_decay']
        )

        # Scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=params['scheduler_factor'],
            patience=params['scheduler_patience'],
            min_lr=1e-6
        )

        # Ranking loss
        ranking_criterion = RankingLoss(
            loss_type=params['ranking_loss_type'],
            margin=params['ranking_margin'],
            use_focal=params['use_focal'],
            focal_gamma=params['focal_gamma']
        )

        # EMA
        ema = None
        if params['use_ema']:
            ema = EMA(model, decay=params['ema_decay'], device=torch.device('cpu'))

        # Mixed precision
        scaler = torch.amp.GradScaler('cuda') if self.device.type == 'cuda' else None

        best_val_ndcg = 0.0
        best_metrics = {}
        epochs_without_improvement = 0
        smoothed_ndcg = 0.0  # EMA of validation NDCG for scheduler/early stopping
        metric_ema_alpha = 0.3  # Weight for new observation (lower = smoother)

        gradient_accumulation_steps = params['gradient_accumulation_steps']
        contrastive_weight = params['contrastive_weight']
        ranking_loss_scale = params['ranking_loss_scale']

        for epoch in range(self.epochs_per_trial):
            model.train()
            optimizer.zero_grad()

            accumulated_loss = 0.0
            successful_steps = 0

            num_views = len(self.augmented_views)

            for accum_step in range(gradient_accumulation_steps):
                try:
                    # Deterministic view pairing
                    view1_idx = (epoch * gradient_accumulation_steps + accum_step) % num_views
                    view2_idx = (epoch * gradient_accumulation_steps + accum_step + 1) % num_views

                    view1 = self.augmented_views[view1_idx]
                    view2 = self.augmented_views[view2_idx]

                    # Get view data
                    feature_key_view = 'feature' if 'feature' in view1 else 'x'
                    view1_x = view1[feature_key_view].to(self.device)
                    view1_edge_index = view1['edge_index'].to(self.device)
                    view1_curvature = view1['ollivier_curvature'].to(self.device)

                    view2_x = view2[feature_key_view].to(self.device)
                    view2_edge_index = view2['edge_index'].to(self.device)
                    view2_curvature = view2['ollivier_curvature'].to(self.device)

                    with torch.amp.autocast('cuda', enabled=scaler is not None):
                        # Forward pass - encode both views
                        h1, _ = model.encode(
                            view1_x, view1_edge_index, view1_curvature,
                            return_attention=False, return_all_layers=True
                        )

                        h2, _ = model.encode(
                            view2_x, view2_edge_index, view2_curvature,
                            return_attention=False, return_all_layers=True
                        )

                        # Project for contrastive loss
                        z1 = F.normalize(model.projection(h1), dim=-1)
                        z2 = F.normalize(model.projection(h2), dim=-1)

                        # Contrastive loss with proper node alignment
                        if view1_idx != view2_idx and (view1_idx, view2_idx) in self.contrastive_alignment:
                            align_idx1, align_idx2 = self.contrastive_alignment[(view1_idx, view2_idx)]
                            align_idx1 = align_idx1.to(self.device)
                            align_idx2 = align_idx2.to(self.device)
                            z1_common = z1[align_idx1]
                            z2_common = z2[align_idx2]
                        else:
                            min_nodes = min(z1.shape[0], z2.shape[0])
                            z1_common = z1[:min_nodes]
                            z2_common = z2[:min_nodes]
                        contrastive_loss = model.compute_contrastive_loss(z1_common, z2_common)

                        # Contrastive-only loss (ranking is on original graph below)
                        loss = contrastive_weight * contrastive_loss
                        loss = torch.clamp(loss, max=10.0)
                        loss = loss / gradient_accumulation_steps

                    # Backward
                    if scaler:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()

                    accumulated_loss += loss.item() * gradient_accumulation_steps
                    successful_steps += 1

                    # Cleanup
                    del view1_x, view1_edge_index, view1_curvature
                    del view2_x, view2_edge_index, view2_curvature
                    del h1, h2, z1, z2
                    del loss, contrastive_loss

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                except RuntimeError as e:
                    if "out of memory" in str(e):
                        logger.warning(f"OOM at fold {fold_idx}, epoch {epoch}, step {accum_step}")
                        torch.cuda.empty_cache()
                        gc.collect()
                        continue
                    else:
                        raise e

            if successful_steps == 0:
                logger.error(f"All steps failed with OOM at epoch {epoch}")
                continue

            # === Ranking loss on original graph ===
            try:
                train_mask_device = train_mask.to(self.device)
                with torch.amp.autocast('cuda', enabled=scaler is not None):
                    orig_scores, _ = model.forward(
                        original_device['x'],
                        original_device['edge_index'],
                        original_device['ollivier_curvature'],
                        return_all_layers=True
                    )
                    orig_ranking_loss = ranking_criterion(orig_scores, labels, train_mask_device)
                    orig_loss = (1 - contrastive_weight) * ranking_loss_scale * orig_ranking_loss
                    orig_loss = orig_loss / gradient_accumulation_steps

                if scaler:
                    scaler.scale(orig_loss).backward()
                else:
                    orig_loss.backward()

                del orig_scores, orig_ranking_loss, orig_loss, train_mask_device
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except RuntimeError as e:
                if "out of memory" in str(e):
                    logger.warning(f"OOM during original graph ranking at epoch {epoch}")
                    torch.cuda.empty_cache()
                    gc.collect()
                else:
                    raise e

            # Optimizer step
            if scaler:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            optimizer.zero_grad()

            # Update EMA
            if ema:
                ema.update(model)

            # Validation
            if epoch % 10 == 0 or epoch == self.epochs_per_trial - 1:
                model.eval()

                if ema:
                    ema.apply_shadow(model)

                with torch.no_grad():
                    val_scores, _ = model.forward(
                        original_device['x'],
                        original_device['edge_index'],
                        original_device['ollivier_curvature']
                    )

                    val_ndcg = compute_ndcg(
                        val_scores[val_mask],
                        labels[val_mask],
                        k=50
                    )

                if ema:
                    ema.restore(model)

                # Update smoothed NDCG (EMA of validation metric)
                if smoothed_ndcg == 0.0:
                    smoothed_ndcg = val_ndcg
                else:
                    smoothed_ndcg = metric_ema_alpha * val_ndcg + (1 - metric_ema_alpha) * smoothed_ndcg

                # Use smoothed NDCG for scheduler
                scheduler.step(smoothed_ndcg)

                # Track best (using raw NDCG for best model selection)
                if val_ndcg > best_val_ndcg:
                    best_val_ndcg = val_ndcg
                    best_metrics = {
                        'ndcg@50': val_ndcg,
                        'epoch': epoch,
                        'loss': accumulated_loss / successful_steps
                    }
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 10

                # Report raw NDCG to Optuna for pruning
                trial.report(val_ndcg, epoch)

                if trial.should_prune():
                    raise optuna.TrialPruned()

                # Early stopping (using smoothed NDCG)
                if epochs_without_improvement >= self.patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break

        # Cleanup
        del model, optimizer, scheduler
        if ema:
            del ema
        torch.cuda.empty_cache()
        gc.collect()

        return best_val_ndcg, best_metrics

    def objective(self, trial: optuna.Trial) -> float:
        """Optuna objective function - returns mean NDCG@50 across folds"""

        # Seed each trial deterministically based on trial number
        trial_seed = self.seed + trial.number
        set_seed(trial_seed)

        params = self.suggest_hyperparameters(trial)

        logger.info(f"{'='*60}")
        logger.info(f"Trial {trial.number}: Testing hyperparameters")
        logger.info(f"{'='*60}")
        for key, value in params.items():
            logger.info(f"  {key}: {value}")

        fold_scores = []

        for fold_idx in range(self.n_folds):
            logger.info(f"\n--- Fold {fold_idx + 1}/{self.n_folds} ---")

            try:
                model = self.create_model(params)
                fold_ndcg, _ = self.train_single_fold(
                    model, params, fold_idx, trial
                )
                fold_scores.append(fold_ndcg)

                logger.info(f"Fold {fold_idx + 1} NDCG@50: {fold_ndcg:.4f}")

                del model
                torch.cuda.empty_cache()
                gc.collect()

            except optuna.TrialPruned:
                raise
            except Exception as e:
                logger.error(f"Error in fold {fold_idx}: {e}")
                fold_scores.append(0.0)

        mean_ndcg = np.mean(fold_scores)
        std_ndcg = np.std(fold_scores)

        logger.info(f"\nTrial {trial.number} Results:")
        logger.info(f"  Mean NDCG@50: {mean_ndcg:.4f} ± {std_ndcg:.4f}")
        logger.info(f"  Fold scores: {fold_scores}")

        # Store results
        result = {
            'trial': trial.number,
            'params': params,
            'mean_ndcg': mean_ndcg,
            'std_ndcg': std_ndcg,
            'fold_scores': fold_scores
        }
        self.trial_results.append(result)

        # Update best
        if mean_ndcg > self.best_score:
            self.best_score = mean_ndcg
            self.best_params = params.copy()
            logger.info(f"  *** New best score! ***")

        return mean_ndcg

    def run(self) -> Dict:
        """Run hyperparameter search"""

        logger.info(f"\n{'='*80}")
        logger.info(f"STARTING HYPERPARAMETER SEARCH")
        logger.info(f"{'='*80}")
        logger.info(f"Study name: {self.study_name}")
        logger.info(f"Trials: {self.n_trials}")
        logger.info(f"Folds per trial: {self.n_folds}")
        logger.info(f"Epochs per trial: {self.epochs_per_trial}")

        # Create Optuna study
        sampler = TPESampler(seed=self.seed)
        pruner = HyperbandPruner(
            min_resource=10,
            max_resource=self.epochs_per_trial,
            reduction_factor=3
        )

        study = optuna.create_study(
            study_name=self.study_name,
            direction='maximize',
            sampler=sampler,
            pruner=pruner
        )

        # Run optimization
        study.optimize(
            self.objective,
            n_trials=self.n_trials,
            show_progress_bar=True,
            gc_after_trial=True
        )

        # Results
        logger.info(f"\n{'='*80}")
        logger.info(f"HYPERPARAMETER SEARCH COMPLETE")
        logger.info(f"{'='*80}")

        logger.info(f"\nBest trial: {study.best_trial.number}")
        logger.info(f"Best NDCG@50: {study.best_value:.4f}")
        logger.info(f"\nBest hyperparameters:")
        for key, value in study.best_params.items():
            logger.info(f"  {key}: {value}")

        # Save results
        self._save_results(study)

        return {
            'best_params': study.best_params,
            'best_value': study.best_value,
            'best_trial': study.best_trial.number,
            'all_results': self.trial_results
        }

    def _save_results(self, study: optuna.Study):
        """Save search results to files"""

        # Save best params as JSON
        best_params_path = self.output_dir / f"{self.study_name}_best_params.json"
        with open(best_params_path, 'w') as f:
            json.dump(study.best_params, f, indent=2)
        logger.info(f"Saved best params to {best_params_path}")

        # Save all trial results
        results_path = self.output_dir / f"{self.study_name}_all_results.json"
        with open(results_path, 'w') as f:
            json.dump(self.trial_results, f, indent=2, default=str)
        logger.info(f"Saved all results to {results_path}")

        # Save as DataFrame
        df_results = pd.DataFrame([
            {
                'trial': r['trial'],
                'mean_ndcg': r['mean_ndcg'],
                'std_ndcg': r['std_ndcg'],
                **r['params']
            }
            for r in self.trial_results
        ])
        csv_path = self.output_dir / f"{self.study_name}_results.csv"
        df_results.to_csv(csv_path, index=False)
        logger.info(f"Saved results CSV to {csv_path}")

        # Create visualization
        self._create_plots(study)

    def _create_plots(self, study: optuna.Study):
        """Create visualization plots"""
        try:
            import matplotlib.pyplot as plt

            _, axes = plt.subplots(2, 2, figsize=(14, 10))

            # Plot 1: Optimization history
            ax1 = axes[0, 0]
            trials = [t.number for t in study.trials if t.value is not None]
            values = [t.value for t in study.trials if t.value is not None]
            ax1.plot(trials, values, 'b-', alpha=0.5, label='Trial value')
            ax1.plot(trials, np.maximum.accumulate(values), 'r-', linewidth=2, label='Best so far')
            ax1.set_xlabel('Trial')
            ax1.set_ylabel('NDCG@50')
            ax1.set_title('Optimization History')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Plot 2: Parameter importance (top parameters)
            ax2 = axes[0, 1]
            if len(study.trials) > 5:
                try:
                    importance = optuna.importance.get_param_importances(study)
                    top_params = dict(list(importance.items())[:10])
                    ax2.barh(list(top_params.keys()), list(top_params.values()))
                    ax2.set_xlabel('Importance')
                    ax2.set_title('Parameter Importance (Top 10)')
                except:
                    ax2.text(0.5, 0.5, 'Not enough trials\nfor importance analysis',
                            ha='center', va='center', transform=ax2.transAxes)
            else:
                ax2.text(0.5, 0.5, 'Not enough trials\nfor importance analysis',
                        ha='center', va='center', transform=ax2.transAxes)

            # Plot 3: Hidden channels vs NDCG
            ax3 = axes[1, 0]
            hidden_channels = []
            ndcg_values = []
            for t in study.trials:
                if t.value is not None and 'hidden_channels' in t.params:
                    hidden_channels.append(t.params['hidden_channels'])
                    ndcg_values.append(t.value)
            if hidden_channels:
                ax3.scatter(hidden_channels, ndcg_values, alpha=0.6)
                ax3.set_xlabel('Hidden Channels')
                ax3.set_ylabel('NDCG@50')
                ax3.set_title('Hidden Channels vs Performance')
                ax3.grid(True, alpha=0.3)

            # Plot 4: Contrastive weight vs NDCG
            ax4 = axes[1, 1]
            cont_weights = []
            ndcg_values = []
            for t in study.trials:
                if t.value is not None and 'contrastive_weight' in t.params:
                    cont_weights.append(t.params['contrastive_weight'])
                    ndcg_values.append(t.value)
            if cont_weights:
                ax4.scatter(cont_weights, ndcg_values, alpha=0.6)
                ax4.set_xlabel('Contrastive Weight')
                ax4.set_ylabel('NDCG@50')
                ax4.set_title('Contrastive Weight vs Performance')
                ax4.grid(True, alpha=0.3)

            plt.tight_layout()
            plot_path = self.output_dir / f"{self.study_name}_plots.png"
            plt.savefig(plot_path, dpi=150)
            plt.close()
            logger.info(f"Saved plots to {plot_path}")

        except Exception as e:
            logger.warning(f"Could not create plots: {e}")


def main():
    parser = argparse.ArgumentParser(description='Hyperparameter search for Driver Gene Predictor')

    parser.add_argument('--dataset', type=str, required=True,
                        help='Path to curvature-enhanced dataset')
    parser.add_argument('--output_dir', type=str, default='hyperparam_results',
                        help='Output directory for results')
    parser.add_argument('--n_trials', type=int, default=50,
                        help='Number of Optuna trials')
    parser.add_argument('--n_folds', type=int, default=1,
                        help='Number of folds for cross-validation per trial (1 is faster for search)')
    parser.add_argument('--epochs_per_trial', type=int, default=500,
                        help='Maximum epochs per trial')
    parser.add_argument('--patience', type=int, default=30,
                        help='Early stopping patience')
    parser.add_argument('--study_name', type=str, default=None,
                        help='Name for Optuna study')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')

    args = parser.parse_args()

    # Run search
    # Set global seed before anything else
    set_seed(args.seed)

    searcher = HyperparameterSearch(
        data_path=args.dataset,
        output_dir=args.output_dir,
        n_trials=args.n_trials,
        n_folds=args.n_folds,
        epochs_per_trial=args.epochs_per_trial,
        patience=args.patience,
        study_name=args.study_name,
        seed=args.seed
    )

    results = searcher.run()

    print(f"\n{'='*60}")
    print("SEARCH COMPLETE")
    print(f"{'='*60}")
    print(f"Best NDCG@50: {results['best_value']:.4f}")
    print(f"\nBest hyperparameters:")
    for key, value in results['best_params'].items():
        print(f"  {key}: {value}")


if __name__ == '__main__':
    main()
