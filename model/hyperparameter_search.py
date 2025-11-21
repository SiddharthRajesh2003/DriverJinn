import torch
import torch.nn
import torch.nn.functional as F
from typing import Tuple, List, Optional, Dict
import os
import gc
import pickle
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import matplotlib.pyplot as plt
import seaborn as sns

from .DriverGenePredictor import ContrastiveDriverGenePredictor
from utils.logging_manager import get_logger
from .support_models import WarmupScheduler, RankingLoss

logger = get_logger(__name__)

# Memory optimization
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

class HyperparameterSearch:
    """
    Hyperparameter search for ContrastiveDriverGenePredictor using Optuna
    """
    def __init__(
        self,
        data_path: str,
        output_dir: str,
        n_trials:int = 50,
        n_folds:int = 3,
        epochs_per_trial: int = 50,
        device: torch.device = None
    ) :
        self.data_path = data_path
        self.output_dir = output_dir
        self.n_trials = n_trials
        self.n_folds = n_folds
        self.epochs = epochs_per_trial
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"Loading Data from path {self.data_path}")
        with open(self.data_path, 'rb') as f:
            data = pickle.load(f)
            
        self.original = data['original']
        self.augmented_views = data['augmented_views']
        self.labels = self.original['label']
        self.kfold_splits = self.original['kfold_splits'][:n_folds]
        
        # Preprocess data
        self.preprocess_data()
        
        logger.info(f"Data loaded: {self.original['feature'].shape[0]} nodes, {n_folds} folds")
        logger.info(f"Device: {self.device}")
        
        self.trial_results = []
        
    def preprocess_data(self):
        """Preprocess Curvature Data"""
        logger.info("Preprocessing curvature data...")
        
        # Sanitize features
        features = self.original.get('feature', self.original.get('x'))
        if torch.isnan(features).any() or torch.isinf(features).any():
            features = torch.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
            if 'feature' in self.original:
                self.original['feature'] = features
            else:
                self.original['x'] = features
        
        # Sanitize curvature
        for curv_key in ['ollivier_curvature', 'forman_curvature']:
            if curv_key in self.original:
                curv = self.original[curv_key]
                curv = torch.nan_to_num(curv, nan=0.0, posinf=0.0, neginf=0.0)
                curv = torch.clamp(curv, min=-10.0, max=10.0)
                self.original[curv_key] = curv
        
        for view in self.augmented_views:
            for curv_key in ['ollivier_curvature', 'forman_curvature']:
                if curv_key in view:
                    curv = view[curv_key]
                    curv = torch.nan_to_num(curv, nan=0.0, posinf=0.0, neginf=0.0)
                    curv = torch.clamp(curv, min=-10.0, max=10.0)
                    view[curv_key] = curv
        
        logger.info("✓ Data preprocessing complete")
        
    def suggest_hyperparameters(
        self, trial: optuna.Trial
    ) -> Dict:
        """Define Hyperparameter search space"""
        
        params = {
            # Model architecture
            'hidden_channels': trial.suggest_categorical('hidden_channels', [64, 128, 256]),
            'projection_dim': trial.suggest_categorical('projection_dim', [32, 64, 128]),
            'num_gnn_layers': trial.suggest_int('num_gnn_layers', 2, 4),
            'num_attention_heads': trial.suggest_int('num_attention_heads', [1, 2, 4, 8]),
            'attention_mode': trial.suggest_categorical('attention_mode', ['standard', 'gated', 'hybrid']),
            'pathway_aggregator': trial.suggest_categorical('pathway_aggregator', ['concat','attention','mean', 'hierarchical']),
            'concat_heads':trial.suggest_categorical('concat_heads', ['True',"False"]),
            
            # Regularization
            'dropout': trial.suggest_float('dropout', 0.1, 0.5),
            'temperature': trial.suggest_float('temperature', 0.3, 0.9),
            'min_edge_ratio': trial.suggest_float('min_edge_ratio', 0.05, 0.25),
            'negative_slope': trial.suggest_float('negative_slop,e', 0.1, 0.3),
            
            # Training
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
            'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
            'contrastive_weight': trial.suggest_float('contrastive_weight', 0.1, 0.5),
            
            'ranking_loss_type': trial.suggest_float('ranking_loss_type', ['pairwise', 'listwise']),
            'ranking_margin': trial.suggest_float('ranking_margin', 0.5, 2.0),
            
            # Optimization
            'gradient_accumulation_steps': trial.suggest_int('gradient_accumulation_steps', [2, 4, 8])
        }
        
        return params
    
    def create_model(
        self,
        params: Dict
    ) -> ContrastiveDriverGenePredictor:
        """Create model from hyperparameters"""
        num_features = self.original['x'].shape
        
        model = ContrastiveDriverGenePredictor(
            in_channels=num_features,
            hidden_channels=params['hidden_channels'],
            projection_dim=params['projection_dim'],
            num_gnn_layers=params['num_gnn_layers'],
            curvature_types=['positive','negative','both'],
            hop_types=['one_hop', 'two_hop'],
            num_attention_heads=params['num_attention_heads'],
            use_attention=True,
            attention_mode=params['attention_mode'],
            pathway_aggregator=params['pathway_aggregator'],
            concat = params['concat'],
            min_edge_ratio=params['min_edge_ratio'],
            negative_slope=params['negative_slope'],
            temperature=params['temperature'],
            dropout=params['dropout'],
            device = self.device
        ).to(self.device)
        
        return model
    
    def train_and_evaluate(
        self,
        model: ContrastiveDriverGenePredictor,
        params: Dict,
        fold_data: Dict,
        trial: optuna.Trial
    ) -> float:
        """Train model on one fold and return validation metric"""
        
        # Extract masks
        train_mask = torch.from_numpy(fold_data['train_mask']).to(self.device)
        val_mask = torch.from_numpy(fold_data['val_mask']).to(self.device)
        
        # Prepare data
        original_device = {
            'x': self.original['x'].to(self.device),
            'edge_index': self.original['edge_index'].to(self.device),
            'ollivier_curvature': self.original['ollivier_curvature'].to(self.device),
        }
        
        labels = self.labels.to(self.device)