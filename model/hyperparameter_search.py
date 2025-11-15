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