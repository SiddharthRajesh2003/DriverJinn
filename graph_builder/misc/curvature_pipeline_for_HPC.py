import multiprocessing as mp
import os
original_get_context = mp.get_context
def hpc_compatible_get_context(method=None):
    """Patch for HPC/cluster environments where multiprocessing can be problematic"""

    # Check if we're in a problematic environment
    is_hpc = any([
        os.environ.get('SLURM_JOB_ID'),  # SLURM scheduler
        os.environ.get('PBS_JOBID'),     # PBS scheduler
        os.environ.get('LSB_JOBID'),     # LSF scheduler
        'conda' in os.environ.get('PATH', '').lower(),  # Conda environment
    ])

    if method is None:
        if is_hpc:
            # In HPC environments, spawn is often more reliable
            try:
                return original_get_context('spawn')
            except (ValueError, RuntimeError):
                return original_get_context('fork')
        else:
            # Regular Linux systems prefer fork
            try:
                return original_get_context('fork')
            except (ValueError, RuntimeError):
                return original_get_context('spawn')

    # Handle specific method requests with fallbacks
    try:
        return original_get_context(method)
    except (ValueError, RuntimeError) as e:
        print(f"Warning: Method '{method}' failed ({e}), using fallback")
        if method == 'fork':
            return original_get_context('spawn')
        else:
            return original_get_context('fork')
# Apply the patch
mp.get_context = hpc_compatible_get_context

from graph_builder.build_network import Network
from graph_builder.curvature_calculator import EdgeCurvature
from graph_builder.curvature_integration import CurvatureFeatureIntegrator
from graph_builder.schur_complement import SchurComplementAugmentation
from utils.logging_manager import get_logger

import pickle
import torch
import pandas as pd
import networkx as nx
import os
import argparse
from sklearn.model_selection import train_test_split, StratifiedKFold
import numpy as np

class CurvaturePipeline:
    """
    Complete pipeline for integrating curvature features with gene networks
    """
    
    def __init__(self, dataset_file):
        """
        Initialize pipeline with dataset file
        
        Parameters:
        dataset_file: str, path to your dataset pickle file (e.g., 'dataset_GGNet.pkl')
        """
        
        self.dataset_file = dataset_file
        self.data_dict = None
        self.network = None
        self.edge_curvature = None
        self.enhanced_data_dict = None
        self.schur_augmenter = None
        self.split_data = None
    
    def load_data(self):
        """Load data from pickle file"""
        
        logger.info(f'Loading data from pickle file: {self.dataset_file}')
        
        with open(self.dataset_file, 'rb') as f:
            self.data_dict = pickle.load(f)
            
        if self.data_dict is not None:
            logger.info('Data loaded successfully')
            logger.info(f"Features shape: {self.data_dict['feature'].shape}")
            logger.info(f"Edge index shape: {self.data_dict['edge_index'].shape}")
            logger.info(f"Number of nodes: {len(self.data_dict['node_name'])}")
            
        return self.data_dict
    
    def create_train_test_split(self, test_size = 0.2, val_size = 0.1, stratify = True,
                                random_seed = 42, use_existing_mask = False, train_ratio_from_mask=0.8):
        """
        Create train/validation/test split for the dataset
        
        Parameters:
        test_size: float, proportion of dataset for test set (default: 0.2)
        val_size: float, proportion of dataset for validation set (default: 0.1)
        stratify: bool, whether to use stratified splitting based on labels
        random_seed: int, random seed for reproducibility
        use_existing_mask: bool, whether to use existing mask in data_dict (boolean array)
        train_ratio_from_mask: float, when using existing mask, ratio to split into train/val (default: 0.8)
        
        Returns:
        dict: Dictionary with train/val/test indices and masks
        """
        logger.info("Creating train/validation/test split...")
        
        if self.enhanced_data_dict is None:
            raise ValueError("Must integrate features first using integrate_features()")
        
        num_nodes = self.enhanced_data_dict['feature'].shape[0]
        labels = self.enhanced_data_dict['label']
        
        if isinstance(labels, torch.Tensor):
            labels_np = labels.numpy()
        else:
            labels_np = labels
        
        if use_existing_mask and 'mask' in self.enhanced_data_dict:
            logger.info("Using existing boolean mask from dataset")
            existing_mask = self.enhanced_data_dict['mask']
            
            # Convert to numpy boolean array if needed
            if isinstance(existing_mask, torch.Tensor):
                existing_mask = existing_mask.numpy()
                
            
            unique_values = np.unique(existing_mask)
            logger.info(f"Existing mask unique values: {unique_values}")
            
            if len(unique_values) == 2 and set(unique_values).issubset({0, 1, True, False}):
                # Boolean mask: True/False or 1/0
                logger.info("Detected boolean mask - splitting True values into train/val, False as test")
                
                # Get indices where mask is True (labeled/known nodes)
                true_indices = np.where(existing_mask)[0]
                # Get indices where mask is False (unlabeled/test nodes)
                test_idx = np.where(~existing_mask)[0]
                
                # Split true_indices into train and val
                if val_size > 0:
                    stratify_labels = labels_np[true_indices] if stratify else None
                    
                    train_idx, val_idx = train_test_split(
                        true_indices,
                        test_size = val_size,
                        random_state=random_seed,
                        stratify=stratify_labels
                    )
                else:
                    train_idx = true_indices
                    val_idx = np.array([])
            
                logger.info(f"Split based on existing mask: True nodes split into train/val, False nodes as test")
            
            elif len(unique_values) <= 4 and all(v in [0, 1, 2, 3] for v in unique_values):
                # Integer mask: assume 1=train, 2=val, 3=test, 0=unused
                logger.info("Detected integer mask with encoding (1=train, 2=val, 3=test)")
                train_idx = np.where(existing_mask==1)[0]
                val_idx = np.where(existing_mask==2)[0]
                test_idx = np.where(existing_mask == 3)[0]
                
            else:
                logger.warning(f"Unrecognized mask format with values {unique_values}. Creating new split.")
                use_existing_mask = False
                
        # Option 2: Create new split
        if not use_existing_mask:
            all_indices = np.arange(num_nodes)
            
            # Stratify labels
            stratify_labels = labels_np if stratify else None
            
            train_val_idx, test_idx = train_test_split(
                all_indices,
                test_size=test_size,
                random_state=random_seed,
                stratify=stratify_labels if stratify else None
            )
            
            if val_size > 0:
                # Calculate validation size relative to train_val set
                val_size_adjusted = val_size / (1 - test_size)
                
                stratify_train_val = labels_np[train_val_idx] if stratify else None
                
                train_idx, val_idx = train_test_split(
                    train_val_idx,
                    test_size=val_size_adjusted,
                    random_state=random_seed,
                    stratify= stratify_train_val if stratify else None
                )
            else:
                train_idx = train_val_idx
                val_idx = np.array([])
                
        # Create boolean mask
        train_mask = np.zeros(num_nodes, dtype=bool)
        val_mask = np.zeros(num_nodes, dtype=bool)
        test_mask = np.zeros(num_nodes, dtype=bool)
        
        train_mask[train_idx] = True
        if len(val_idx) > 0:
            val_mask[val_idx] = True
        test_mask[test_idx] = True
        
        # Create integer mask (1=train, 2=val, 3=test, 0=unused)
        mask = np.zeros(num_nodes, dtype = int)
        mask[train_idx] = 1
        if len(val_idx) > 0:
            mask[val_idx] = 2
        mask[test_idx] = 3
        
        self.split_data = {
            'train_idx': train_idx,
            'val_idx': val_idx,
            'test_idx': test_idx,
            'train_mask': train_mask,
            'val_mask': val_mask,
            'test_mask': test_mask,
            'mask': mask,
            'split_config': {
                'test_size': test_size,
                'val_size': val_size,
                'stratify': stratify,
                'random_seed': random_seed,
                'use_existing_mask': use_existing_mask
            }
        }
        
        # Add to enhanced data dict
        self.enhanced_data_dict['train_mask'] = torch.tensor(train_mask)
        self.enhanced_data_dict['val_mask'] = torch.tensor(val_mask)
        self.enhanced_data_dict['test_mask'] = torch.tensor(test_mask)
        self.enhanced_data_dict['mask'] = torch.tensor(mask)
        
        # Log statistics
        logger.info(f"Split created successfully:")
        logger.info(f"  Train: {len(train_idx)} nodes ({len(train_idx)/num_nodes*100:.1f}%)")
        if len(val_idx) > 0:
            logger.info(f"  Val:   {len(val_idx)} nodes ({len(val_idx)/num_nodes*100:.1f}%)")
        logger.info(f"  Test:  {len(test_idx)} nodes ({len(test_idx)/num_nodes*100:.1f}%)")
        
        # Log class distribution if stratified
        if stratify:
            self.log_class_distribution(labels_np, train_idx, val_idx, test_idx)
        
        return self.split_data
    
    def log_class_distribution(self, labels: np.ndarray,
                                train_idx, val_idx, test_idx):
        """Log class distribution across splits"""
        unique_labels = np.unique(labels)
        
        logger.info("\nClass distribution:")
        for label in unique_labels:
            train_count = np.sum(labels[train_idx] == label)
            val_count = np.sum(labels[val_idx] == label) if len(val_idx) > 0 else 0
            test_count = np.sum(labels[test_idx] == label)
            
            logger.info(f"  Class {label}: Train={train_count}, Val={val_count}, Test={test_count}")
    
    def create_kfold_splits(self, n_splits=5, stratify=True, random_seed=42):
        """
        Create K-fold cross-validation splits
        
        Parameters:
        n_splits: int, number of folds
        stratify: bool, whether to use stratified K-fold
        random_seed: int, random seed for reproducibility
        
        Returns:
        list: List of dictionaries, each containing train/val indices for one fold
        """
        logger.info(f"Creating {n_splits}-fold cross-validation splits...")
        
        # K-fold can work on original data_dict (before enhancement)
        data_source = self.enhanced_data_dict if self.enhanced_data_dict is not None else self.data_dict
        
        if data_source is None:
            raise ValueError("Must load data first using load_data()")
        
        num_nodes = data_source['feature'].shape[0]
        labels = data_source['label']
        
        if isinstance(labels, torch.Tensor):
            labels_np = labels.numpy()
        else:
            labels_np = labels
        
        all_indices = np.arange(num_nodes)
        
        # Use StratifiedKFold or regular KFold
        if stratify:
            kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
            splits = list(kfold.split(all_indices, labels_np))
        else:
            from sklearn.model_selection import KFold
            kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
            splits = list(kfold.split(all_indices))
        
        fold_data = []
        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            # Create masks for this fold
            train_mask = np.zeros(num_nodes, dtype=bool)
            val_mask = np.zeros(num_nodes, dtype=bool)
            
            train_mask[train_idx] = True
            val_mask[val_idx] = True
            
            fold_data.append({
                'fold': fold_idx + 1,
                'train_idx': train_idx,
                'val_idx': val_idx,
                'train_mask': train_mask,
                'val_mask': val_mask
            })
            
            logger.info(f"Fold {fold_idx + 1}: Train={len(train_idx)}, Val={len(val_idx)}")
        
        # Store in whichever dict is available
        if self.enhanced_data_dict is not None:
            self.enhanced_data_dict['kfold_splits'] = fold_data
        else:
            self.data_dict['kfold_splits'] = fold_data
        
        return fold_data
    
    def apply_fold_normalization(self, fold_idx, normalize=True):
        """
        Apply normalization for a specific k-fold split
        
        This should be called during k-fold cross-validation to normalize each fold separately
        
        Parameters:
        fold_idx: int, index of the fold (0-based)
        normalize: bool, whether to normalize
        
        Returns:
        dict: Data dict with features normalized for this fold
        """
        if 'kfold_splits' not in self.enhanced_data_dict:
            raise ValueError("No k-fold splits found. Run create_kfold_splits() first.")
        
        if fold_idx >= len(self.enhanced_data_dict['kfold_splits']):
            raise ValueError(f"Fold index {fold_idx} out of range")
        
        fold_info = self.enhanced_data_dict['kfold_splits'][fold_idx]
        train_mask = fold_info['train_mask']
        
        logger.info(f"Applying normalization for fold {fold_idx + 1}...")
        
        # Re-integrate features with this fold's training mask
        if self.edge_curvature is None:
            raise ValueError("Must calculate curvatures first")
        
        integrator = CurvatureFeatureIntegrator(self.edge_curvature, self.data_dict)
        
        fold_data_dict = integrator.create_enhanced_features(
            normalize=normalize,
            train_mask=train_mask,
            val_mask=fold_info['val_mask'],
            test_mask=None  # K-fold doesn't have separate test set
        )
        
        # Add curvature edge features
        curvature_dict = integrator.create_edge_features_dict()
        fold_data_dict['ollivier_curvature'] = curvature_dict['edge_ollivier_curvature']
        fold_data_dict['forman_curvature'] = curvature_dict['edge_forman_curvature']
        fold_data_dict['edge_features'] = curvature_dict['edge_features']
        fold_data_dict['edge_feature_names'] = curvature_dict['edge_feature_names']
        
        # Add fold information
        fold_data_dict['current_fold'] = fold_idx + 1
        fold_data_dict['train_mask'] = torch.tensor(train_mask)
        fold_data_dict['val_mask'] = torch.tensor(fold_info['val_mask'])
        fold_data_dict['train_idx'] = fold_info['train_idx']
        fold_data_dict['val_idx'] = fold_info['val_idx']
        
        return fold_data_dict
    
    def build_network(self):
        """Build NetworkX graph using Network class"""
        logger.info('Building NetworkX graph...')
        
        self.network = Network(self.data_dict['edge_index'], 
                               self.data_dict['node_name'],
                               len(self.data_dict['node_name'])
                        )
        
        if self.network is not None:
            logger.info('Graph built successfully')
            logger.info(f'Nodes: {self.network.G.number_of_nodes()}')
            logger.info(f'Edges: {self.network.G.number_of_edges()}')
            
        return self.network
    
    def calculate_curvatures(self, method = 'both', feature_df = None):
        """
        Calculate edge curvatures using your EdgeCurvature class
        
        Parameters:
        method: str, 'ollivier', 'forman', or 'both'
        feature_df: pd.DataFrame, optional feature dataframe
        """
        try:
            logger.info(f'Calculating edge curvature using {method} method')
            
            if feature_df is None:
                # CHANGED: Get node list from the graph structure
                graph_nodes = list(self.network.G.nodes())
                
                # Create feature dataframe using graph nodes
                feature_df = pd.DataFrame(
                    data=self.data_dict['feature'].numpy(),
                    columns=self.data_dict['feature_name'],
                    index=graph_nodes
                )
            
            self.edge_curvature = EdgeCurvature(self.network.G, feature_df)
            
            if self.edge_curvature is not None:
                curvatures = self.edge_curvature.calculate_edge_curvature(method=method)
                stats = self.edge_curvature.get_curvature_statistics()

                logger.info('Curvature Calculation complete')
                for curvature_type, stat_dict in stats.items():
                    print(f"{curvature_type}: {stat_dict['count']} edges, "
                    f"mean={stat_dict['mean']:.4f}, "
                    f"positive={stat_dict['positive_edges']}, "
                    f"negative={stat_dict['negative_edges']}")
            
            return curvatures
        
        except Exception as e:
            logger.error(f'Error in calculating edge curvatures: {e}')
    
    def integrate_features(self, normalize = True):
        """
        Integrate curvature features with existing features
        
        Parameters:
        normalize: bool, whether to normalize curvature features
        
        Note: If train/test split has been created, normalization will be done
        using training data only (fit on train, transform on all)
        """
        try:
            logger.info("Integrating curvature features with existing features...")
            
            if self.edge_curvature is None:
                raise ValueError("Must calculate curvatures first using calculate_curvatures()")
            
            integrator = CurvatureFeatureIntegrator(self.edge_curvature, self.data_dict)
            integrator.analyze_curvature_distribution()
            
            # Check if we have splits already
            train_mask = None
            val_mask = None
            test_mask = None
            
            if self.split_data is not None:
                train_mask = self.split_data['train_mask']
                val_mask = self.split_data['val_mask']
                test_mask = self.split_data['test_mask']
                logger.info("Using existing split for normalization (fit on train only)")
            else:
                logger.info("No split found - will create enhanced features without split-aware normalization")
            
            # Create enhanced features with proper train/test normalization
            self.enhanced_data_dict = integrator.create_enhanced_features(
                normalize=normalize,
                train_mask=train_mask,
                val_mask=val_mask,
                test_mask=test_mask
            )
            
            self.curvature_dict = integrator.create_edge_features_dict()
            
            self.enhanced_data_dict['ollivier_curvature'] = self.curvature_dict['edge_ollivier_curvature']
            self.enhanced_data_dict['forman_curvature'] = self.curvature_dict['edge_forman_curvature']
            self.enhanced_data_dict['edge_features'] = self.curvature_dict['edge_features']
            self.enhanced_data_dict['edge_feature_names'] = self.curvature_dict['edge_feature_names']
            
            # Store the integrator for potential later use
            self.integrator = integrator
            # Save normalization parameters (means and stds) used by integrator if available
            try:
                orig_scaler = getattr(integrator, 'original_scaler', None)
                curv_scaler = getattr(integrator, 'curvature_scaler', None)
                if orig_scaler is not None:
                    self.enhanced_data_dict['normalization_params'] = self.enhanced_data_dict.get('normalization_params', {})
                    self.enhanced_data_dict['normalization_params']['original_mean'] = getattr(orig_scaler, 'mean_', None)
                    self.enhanced_data_dict['normalization_params']['original_std'] = getattr(orig_scaler, 'scale_', None)
                if curv_scaler is not None:
                    self.enhanced_data_dict['normalization_params'] = self.enhanced_data_dict.get('normalization_params', {})
                    self.enhanced_data_dict['normalization_params']['curvature_mean'] = getattr(curv_scaler, 'mean_', None)
                    self.enhanced_data_dict['normalization_params']['curvature_std'] = getattr(curv_scaler, 'scale_', None)
            except Exception:
                logger.warning('Could not extract normalization parameters from integrator')
            
            logger.info("Feature integration completed!")
            return self.enhanced_data_dict
        
        except Exception as e:
            logger.error(f'Error occurred during integration of edge curvature into features: {e}')
        
    def initialize_schur_augmentation(self, elimination_ratio: float = 0.2, neighbor_sort_method: str = 'weight',
                                        elimination_strategy: str = 'priority', random_seed = None):
        
        """
        Initialize Schur complement augmentation
        
        Parameters:
        elimination_ratio: float, ratio of nodes to eliminate (0.1-0.3 recommended)
        neighbor_sort_method: str, 'weight', 'degree', 'random', 'asc', 'desc'
        elimination_strategy: str, 'priority', 'random', or 'coarsening'
        random_seed: int, for reproducibility
        """
        logger.info("Initializing Schur complement augmentation...")
        
        self.schur_augmenter = SchurComplementAugmentation(
            elimination_ratio=elimination_ratio,
            neighbor_sort_method=neighbor_sort_method,
            random_seed=random_seed,
            elimination_strategy=elimination_strategy
        )
        
        logger.info(f"Schur augmenter initialized with strategy={elimination_strategy}, "
                   f"elimination_ratio={elimination_ratio}")
        
        return self.schur_augmenter
    
    def generate_augmented_views(self, num_views: int = 2,  use_curvature_weights: bool = True,
                                 curvature_type: str = 'ollivier', compute_aug_curvature: bool = True,
                                 curvature_method: str = 'both'):
        """
        Generate augmented graph views using Schur complement with curvature calculation
        
        Parameters:
        num_views: int, number of augmented views to generate
        use_curvature_weights: bool, whether to use curvature as edge weights
        curvature_type: str, 'ollivier' or 'forman' or 'both' (average)
        compute_aug_curvature: bool, whether to compute curvature for augmented graphs
        curvature_method: str, 'ollivier', 'forman', or 'both' for augmented graphs
        
        Returns:
        list: List of (augmented_graph, augmented_features, metadata, aug_curvature_dict) tuples
        """
        
        if self.schur_augmenter is None:
            logger.info("Schur augmenter not initialized, initializing with defaults...")
            self.initialize_schur_augmentation()
        
        if self.network is None:
            raise ValueError("Must build network first using build_network()")
        
        logger.info(f"Generating {num_views} augmented views...")
        
        # Prepare edge weights from your enhanced dataset
        edge_weights = None
        if use_curvature_weights and self.enhanced_data_dict is not None:
            logger.info(f"Using {curvature_type} curvature as edge weights for augmentation")
            edge_weights = self.extract_weights_from_curvature(curvature_type)
        
        node_features = self.enhanced_data_dict['feature'] if self.enhanced_data_dict is not None else None
        
        augmented_views = []
        for i in range(num_views):
            logger.info(f'Generating view {i+1}/{num_views}')
            
            aug_graph, aug_features, metadata = self.schur_augmenter.augment(
                self.network.G,
                node_features=node_features,
                edge_weights=edge_weights
            )
            
                        # CRITICAL FIX: Verify and repair feature-graph consistency before curvature calculation
            num_graph_nodes = aug_graph.number_of_nodes()
            if aug_features is not None:
                # Convert to numpy if it's a tensor
                if isinstance(aug_features, torch.Tensor):
                    aug_features_np = aug_features.numpy()
                    was_tensor = True
                else:
                    aug_features_np = aug_features
                    was_tensor = False
                
                num_feature_rows = aug_features_np.shape[0]
                
                if num_feature_rows != num_graph_nodes:
                    logger.warning(f"Repairing feature-graph mismatch: {num_feature_rows} features vs {num_graph_nodes} graph nodes")
                    
                    # Get sorted graph nodes as the source of truth
                    graph_nodes = sorted(aug_graph.nodes())
                    original_nodes = sorted(self.network.G.nodes())
                    
                    # Create proper feature matrix aligned to graph nodes
                    feature_dim = aug_features_np.shape[1]
                    repaired_features = np.zeros((num_graph_nodes, feature_dim))
                    
                    # Get source features (use enhanced if available and matches dimension)
                    if self.enhanced_data_dict is not None:
                        source_features = self.enhanced_data_dict['feature']
                    else:
                        source_features = self.data_dict['feature']
                    
                    # Convert source to numpy
                    if isinstance(source_features, torch.Tensor):
                        source_features_np = source_features.numpy()
                    else:
                        source_features_np = source_features
                    
                    # Compute mean vector once (for missing nodes)
                    mean_feature = np.mean(source_features_np, axis=0)
                    
                    # Map original nodes to indices and build mask mapping
                    node_to_idx = {node: idx for idx, node in enumerate(original_nodes)}
                    
                    # Get masks if they exist in enhanced_data_dict
                    train_mask = self.enhanced_data_dict.get('train_mask', None)
                    test_mask = self.enhanced_data_dict.get('test_mask', None)
                    val_mask = self.enhanced_data_dict.get('val_mask', None)
                    
                    # Convert masks to numpy if they're tensors
                    if isinstance(train_mask, torch.Tensor):
                        train_mask = train_mask.numpy()
                    if isinstance(test_mask, torch.Tensor):
                        test_mask = test_mask.numpy()
                    if isinstance(val_mask, torch.Tensor):
                        val_mask = val_mask.numpy()
                    
                    # Initialize new masks for augmented graph
                    new_train_mask = np.zeros(num_graph_nodes, dtype=bool)
                    new_test_mask = np.zeros(num_graph_nodes, dtype=bool)
                    new_val_mask = np.zeros(num_graph_nodes, dtype=bool)
                    
                    # Fill in features and propagate masks for all graph nodes
                    missing_count = 0
                    for idx, node in enumerate(graph_nodes):
                        if node in node_to_idx:
                            orig_idx = node_to_idx[node]
                            if orig_idx < source_features_np.shape[0]:
                                repaired_features[idx] = source_features_np[orig_idx]
                                # Propagate masks
                                if train_mask is not None:
                                    new_train_mask[idx] = train_mask[orig_idx]
                                if test_mask is not None:
                                    new_test_mask[idx] = test_mask[orig_idx]
                                if val_mask is not None:
                                    new_val_mask[idx] = val_mask[orig_idx]
                            else:
                                missing_count += 1
                                repaired_features[idx] = mean_feature
                        else:
                            missing_count += 1
                            repaired_features[idx] = mean_feature
                    
                    # Update masks in metadata
                    metadata['train_mask'] = torch.from_numpy(new_train_mask)
                    metadata['test_mask'] = torch.from_numpy(new_test_mask)
                    metadata['val_mask'] = torch.from_numpy(new_val_mask)
                    
                    aug_features = torch.from_numpy(repaired_features).float()
                    logger.info(f"Repaired features: {repaired_features.shape}, filled {missing_count} missing nodes")
                elif was_tensor:
                    # No repair needed but ensure it's a tensor
                    aug_features = torch.from_numpy(aug_features_np).float()
            
            aug_curvature_dict = None
            if compute_aug_curvature:
                logger.info(f"Computing curvature for augmented view {i+1}...")
                aug_curvature_dict = self.compute_augmented_curvature(
                    aug_graph,
                    aug_features,
                    method = curvature_method
                )
                
                if aug_curvature_dict:
                    metadata['curvature_stats'] = {
                    curv_type: {
                        'mean': float(curvatures.mean()),
                        'std': float(curvatures.std()),
                        'min': float(curvatures.min()),
                        'max': float(curvatures.max())
                    }
                    for curv_type, curvatures in aug_curvature_dict.items()
                }
            
            augmented_views.append((aug_graph, aug_features, metadata, aug_curvature_dict))
        
        logger.info(f"View {i+1}: {metadata['augmented_nodes']} nodes, "
                    f"{metadata.get('augmented_edges', aug_graph.number_of_edges())} edges")
        
        return augmented_views
    
    def compute_augmented_curvature(self, aug_G: nx.Graph, features: torch.Tensor, method='both'):
        """
        Calculate curvature for an augmented graph
        
        Parameters:
        aug_graph: nx.Graph, augmented graph
        aug_features: torch.Tensor or None, node features
        method: str, 'ollivier', 'forman', or 'both'
        
        Returns:
        dict: Dictionary with curvature tensors for each edge
        """
        
        try:
            # Use SORTED nodes from augmented graph for consistency
            aug_node_list = sorted(aug_G.nodes())
            num_aug_nodes = len(aug_node_list)
            
            if features is not None:
                if isinstance(features, torch.Tensor):
                    features_np = features.numpy()
                else:
                    features_np = features

                # Handle feature-graph mismatch by re-indexing
                if features_np.shape[0] != num_aug_nodes:
                    logger.warning(f"Feature shape mismatch: {features_np.shape[0]} features but {num_aug_nodes} graph nodes")
                    logger.warning(f"Re-aligning features to match graph node order...")
                    
                    # CRITICAL FIX: Use enhanced features (with curvature), not original features
                    # Get the feature source that matches the dimension we're working with
                    feature_dim = features_np.shape[1]
                    
                    # Determine which feature source to use based on dimension
                    if self.enhanced_data_dict is not None:
                        enhanced_feat_dim = self.enhanced_data_dict['feature'].shape[1]
                        if feature_dim == enhanced_feat_dim:
                            # Use enhanced features (original + curvature)
                            source_features = self.enhanced_data_dict['feature']
                            source_nodes = sorted(self.network.G.nodes())
                            logger.info(f"Using enhanced features ({enhanced_feat_dim}D) for alignment")
                        else:
                            # Use original features
                            source_features = self.data_dict['feature']
                            source_nodes = sorted(self.network.G.nodes())
                            logger.info(f"Using original features ({self.data_dict['feature'].shape[1]}D) for alignment")
                    else:
                        # Fallback to original features
                        source_features = self.data_dict['feature']
                        source_nodes = sorted(self.network.G.nodes())
                    
                    # Convert to numpy if needed
                    if isinstance(source_features, torch.Tensor):
                        source_features_np = source_features.numpy()
                    else:
                        source_features_np = np.array(source_features) if not isinstance(source_features, np.ndarray) else source_features
                    
                    # Create mapping from node to feature index
                    node_to_feat_idx = {node: idx for idx, node in enumerate(source_nodes)}
                    
                    # Pre-compute mean vector for missing nodes
                    mean_vec = np.mean(source_features_np, axis=0)
                    
                    # Create new feature matrix aligned with aug_node_list
                    aligned_features = np.zeros((num_aug_nodes, feature_dim))
                    
                    missing_count = 0
                    for new_idx, node in enumerate(aug_node_list):
                        if node in node_to_feat_idx:
                            old_idx = node_to_feat_idx[node]
                            aligned_features[new_idx] = source_features_np[old_idx]
                        else:
                            # Node not in original graph - use mean of existing features
                            missing_count += 1
                            aligned_features[new_idx] = mean_vec
                    
                    if missing_count > 0:
                        logger.warning(f"Filled {missing_count} missing nodes with mean feature vector")
                    
                    features_np = aligned_features
                    logger.info(f"Feature re-alignment complete: {features_np.shape}")
                
                # Verify alignment
                if features_np.shape[0] != num_aug_nodes:
                    logger.error(f"CRITICAL: Still have mismatch after alignment: {features_np.shape[0]} != {num_aug_nodes}")
                    # Last resort: pad or trim
                    if features_np.shape[0] > num_aug_nodes:
                        features_np = features_np[:num_aug_nodes]
                        logger.warning(f"Trimmed features to {num_aug_nodes}")
                    else:
                        # Need to pad - use the same source we used above
                        if self.enhanced_data_dict is not None and feature_dim == self.enhanced_data_dict['feature'].shape[1]:
                            pad_source = self.enhanced_data_dict['feature']
                        else:
                            pad_source = self.data_dict['feature']
                        
                        # Ensure pad_source is numpy
                        if isinstance(pad_source, torch.Tensor):
                            pad_source = pad_source.numpy()
                        else:
                            pad_source = np.array(pad_source) if not isinstance(pad_source, np.ndarray) else pad_source
                        
                        mean_vec = np.mean(pad_source, axis=0)
                        padding = np.tile(mean_vec, (num_aug_nodes - features_np.shape[0], 1))
                        features_np = np.vstack([features_np, padding])
                        logger.warning(f"Padded features to {num_aug_nodes}")
                
                # Create DataFrame using SORTED graph nodes as index
                feature_df = pd.DataFrame(
                    data=features_np,
                    index=aug_node_list
                )
            
            else:
                # Use original features indexed by graph nodes
                graph_nodes = list(self.network.G.nodes())
                feature_df = pd.DataFrame(
                    data=self.data_dict['feature'].numpy(),
                    columns=self.data_dict['feature_name'],
                    index=graph_nodes
                )
                
                # Filter to only nodes present in augmented graph
                common_nodes = [n for n in aug_node_list if n in feature_df.index]
                if len(common_nodes) < num_aug_nodes:
                    logger.warning(f"Only {len(common_nodes)}/{num_aug_nodes} augmented nodes have features")
                feature_df = feature_df.loc[common_nodes]
            
            edge_curv_calculator = EdgeCurvature(aug_G, feature_df)
            curvatures = edge_curv_calculator.calculate_edge_curvature(method='both')
            
            # Convert to tensors organized by edge
            edge_list = list(aug_G.edges())
            num_edges = len(edge_list)
            
            curvature_dict = {}    
            
            if method in ['ollivier', 'both']:
                ollivier_curv = torch.zeros(num_edges)
                ollivier_dict = edge_curv_calculator.edge_curvature.get('OllivierRicci', {})
                for i, (u, v) in enumerate(edge_list):
                    curv = ollivier_dict.get((u, v), ollivier_dict.get((v, u), 0.0))
                    ollivier_curv[i] = curv
                curvature_dict['ollivier_curvature'] = ollivier_curv
            
            if method in ['forman', 'both']:
                forman_curv = torch.zeros(num_edges)
                forman_dict = edge_curv_calculator.edge_curvature.get('FormanRicci', {})
                for i, (u, v) in enumerate(edge_list):
                    curv = forman_dict.get((u, v), forman_dict.get((v, u), 0.0))
                    forman_curv[i] = curv
                curvature_dict['forman_curvature'] = forman_curv
            
            logger.info(f"Calculated {method} curvature for {num_edges} edges in augmented graph")
        
            return curvature_dict
            
        except Exception as e:
            logger.error(f"Error calculating augmented curvature: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def extract_weights_from_curvature(self, curvature_type: str = 'ollivier'):
        """
        Extract edge weights from enhanced dataset based on curvature values
        
        Your dataset structure:
        - edge_index: [2, num_edges] tensor
        - ollivier_curvature: [num_edges] tensor
        - forman_curvature: [num_edges] tensor
        
        Parameters:
        curvature_type: str, 'ollivier', 'forman', or 'both' (average)
        
        Returns:
        dict: {(node_i, node_j): weight} dictionary
        """
        edge_index = self.enhanced_data_dict['edge_index']
        edge_weights = {}
        curvature_type = curvature_type.lower()
        
        if curvature_type == 'ollivier':
            curvatures = self.enhanced_data_dict['ollivier_curvature']
        elif curvature_type == 'forman':
            curvatures = self.enhanced_data_dict['forman_curvature']
        elif curvature_type == 'both':
            ollivier = self.enhanced_data_dict['ollivier_curvature']
            forman = self.enhanced_data_dict['forman_curvature']
            curvatures = (ollivier + forman)/2.0
        else:
            raise ValueError(f"Unknown curvature_type: {curvature_type}")
        
        # Convert to numpy for easier indexing
        if isinstance(edge_index, torch.Tensor):
            edge_index = edge_index.numpy()
        if isinstance(curvatures, torch.Tensor):
            curvatures = curvatures.numpy()
        
        # Get nodes from graph instead of node_name
        graph_nodes = list(self.network.G.nodes())
        
        num_edges = edge_index.shape[1]
        for i in range(num_edges):
            src_idx = edge_index[0, i]
            dst_idx = edge_index[1, i]
            
            # Get curvature value and convert to positive weight
            curv = float(curvatures[i])
            
            # Transform curvature to positive weight
            # Ollivier curvature typically ranges from -1 to 1
            # We map it to [0.1, 2.0] for edge weights
            weight = 1.0 + curv # Maps [-1, 1] to [0, 2]
            weight = max(0.1, min(weight, 2.0)) # Clamp to reasonable range
            
            edge_weights[(src_idx, dst_idx)] = weight
            edge_weights[(dst_idx, src_idx)] = weight
            
        logger.info(f"Extracted {len(edge_weights)} edge weights from {curvature_type} curvature")
        logger.info(f"Weight range: [{min(edge_weights.values()):.4f}, {max(edge_weights.values()):.4f}]")
        
        return edge_weights
    
    def save_augmented_views(self, augmented_views, output_dir, prefix = 'augmented'):
        """
        Save augmented graph views to files
        
        Parameters:
        augmented_views: list, output from generate_augmented_views()
        output_dir: str, directory to save outputs
        prefix: str, prefix for saved files
        """
        os.makedirs(output_dir, exist_ok=True)
        # Extract base name from dataset file for consistent naming
        if 'GGNet' in self.dataset_file:
            base_name = 'GGNet'
        elif 'PathNet' in self.dataset_file:
            base_name = 'PathNet'
        elif 'PPNet' in self.dataset_file:
            base_name = 'PPNet'
        else:
            base_name = 'network'
        
        for i, (aug_graph, aug_features, metadata) in enumerate(augmented_views):
            # Create temporary Network object for this augmented graph
            temp_network = Network.__new__(Network)
            temp_network.G = aug_graph
            # Get node names from graph
            temp_network.node_names = list(aug_graph.nodes())
            temp_network.num_nodes = aug_graph.number_of_nodes()
            
            # Save graph using Network.save_graph method
            graph_name = f'{base_name}_{prefix}_view_{i+1}'
            temp_network.save_graph(graph_name, output_dir)
            
            # Save features if available
            if aug_features is not None:
                features_path = os.path.join(output_dir, f'{graph_name}_features.pt')
                torch.save(aug_features, features_path)
            
            # Save metadata
            metadata_path = os.path.join(output_dir, f'{prefix}_metadata_{i+1}.pkl')
            with open(metadata_path, 'wb') as f:
                pickle.dump(metadata, f)
            
            logger.info(f"View {i+1} saved: {graph_name} in {output_dir}")
    
    def create_contrastive_network(self, num_views=2, use_curvature_weights=True,
                                compute_aug_curvature = True, curvature_method = 'both', 
                                propagate_splits = True):
        """
        Create a complete contrastive learning dataset with augmented views and curvatures
        
        Parameters:
        num_views: int, number of augmented views
        use_curvature_weights: bool, use curvature as edge weights
        compute_aug_curvature: bool, compute curvature for augmented graphs
        curvature_method: str, 'ollivier', 'forman', or 'both' for augmented graphs
        propagate_splits: bool, whether to propagate train/val/test splits to augmented views
        
        Returns:
        dict: Dictionary containing original and augmented data for contrastive learning
        """
        if self.enhanced_data_dict is None:
            raise ValueError("Must integrate features first using integrate_features()")
        
        logger.info("Creating contrastive learning dataset...")
        
        # Check if we have k-fold splits or regular splits
        has_kfold = 'kfold_splits' in self.enhanced_data_dict
        has_regular_split = self.split_data is not None
        
        # Generate augmented views
        augmented_views = self.generate_augmented_views(
            num_views=num_views, 
            use_curvature_weights=use_curvature_weights,
            compute_aug_curvature=compute_aug_curvature,
            curvature_method=curvature_method
        )
        
        # Convert to PyTorch Geometric format
        pyg_views = []
        for view_idx, (aug_graph, aug_features, metadata, curvature_dict) in enumerate(augmented_views):
            edge_index, edge_weight, x = self.schur_augmenter.to_pytorch_geometric(
                aug_graph, 
                aug_features.numpy() if aug_features is not None else None
            )
            
            view_dict = {
                'edge_index': edge_index,
                'edge_weight': edge_weight,
                'x': x,
                'metadata': metadata
            }
        
            # Add curvature data if computed
            if curvature_dict is not None:
                view_dict.update(curvature_dict)
            
            # Propagate regular train/val/test splits if available
            if propagate_splits and has_regular_split:
                view_dict = self.propagate_splits_to_view(
                    view_dict,
                    aug_graph,
                    metadata,
                    view_idx
                )
            
            # Propagate k-fold splits if available
            if propagate_splits and has_kfold:
                view_dict = self.propagate_kfold_splits_to_view(
                    view_dict,
                    aug_graph,
                    metadata,
                    view_idx
                )

            # Apply saved Fold-1 normalization to the view so data is ready before training
            try:
                self.apply_fold1_normalization_to_view(view_dict)
            except Exception as e:
                logger.warning(f"Could not apply Fold-1 normalization to view {view_idx}: {e}")

            pyg_views.append(view_dict)
        
        contrastive_dataset = {
            'original': self.enhanced_data_dict,
            'augmented_views': pyg_views,
            'num_views': num_views,
            'augmentation_config': {
                'elimination_ratio': self.schur_augmenter.elimination_ratio,
                'strategy': self.schur_augmenter.elimination_strategy,
                'use_curvature_weights': use_curvature_weights,
                'compute_aug_curvature': compute_aug_curvature,
                'curvature_method': curvature_method,
                'propagate_splits': propagate_splits
            }
        }
        
        # Add split information at dataset level if available
        if has_regular_split:
            contrastive_dataset['split_info'] = {
                'split_type': 'regular',
                'split_config': self.split_data['split_config'],
                'original_train_size': len(self.split_data['train_idx']),
                'original_val_size': len(self.split_data['val_idx']),
                'original_test_size': len(self.split_data['test_idx'])
            }
        elif has_kfold:
            contrastive_dataset['split_info'] = {
                'split_type': 'kfold',
                'n_folds': len(self.enhanced_data_dict['kfold_splits']),
                'total_nodes': self.enhanced_data_dict['feature'].shape[0]
            }
        
        logger.info(f"Contrastive dataset created with {num_views} augmented views")
        if compute_aug_curvature:
            logger.info("Augmented views include pre-computed curvature features")
        if propagate_splits:
            if has_regular_split:
                logger.info("Train/val/test splits propagated to all augmented views")
            elif has_kfold:
                logger.info(f"K-fold splits ({len(self.enhanced_data_dict['kfold_splits'])} folds) propagated to all augmented views")
        
        return contrastive_dataset
        
    def propagate_splits_to_view(self, view_dict, aug_graph, metadata, view_idx):
        """
        Propagate train/val/test splits from original graph to augmented view
        
        Parameters:
        view_dict: dict, augmented view data
        aug_graph: nx.Graph, augmented graph
        metadata: dict, augmentation metadata
        view_idx: int, index of the view
        
        Returns:
        dict: view_dict with added split masks
        """
        # Use SORTED nodes for consistency
        original_graph_nodes = sorted(self.network.G.nodes())
        aug_node_list = sorted(aug_graph.nodes())
        
        # Create mapping from augmented node indices to original node indices
        aug_to_orig_idx = {}
        orig_name_to_idx = {name: i for i, name in enumerate(original_graph_nodes)}
        
        for aug_idx, node_name in enumerate(aug_node_list):
            if node_name in orig_name_to_idx:
                aug_to_orig_idx[aug_idx] = orig_name_to_idx[node_name]
        
        num_aug_nodes = len(aug_node_list)
        
        # Initialize mask for augmented view
        train_mask = torch.zeros(num_aug_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_aug_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_aug_nodes, dtype=torch.bool)
        mask = np.zeros(num_aug_nodes, dtype = int)
        
        # Get original mask
        orig_train_mask = self.enhanced_data_dict['train_mask']
        orig_val_mask = self.enhanced_data_dict['val_mask']
        orig_test_mask = self.enhanced_data_dict['test_mask']
        
        # Propagate masks to augmented nodes
        train_count = 0
        val_count = 0
        test_count = 0
        
        for aug_idx, orig_idx in aug_to_orig_idx.items():
            if orig_train_mask[orig_idx]:
                train_mask[aug_idx] = True
                mask[aug_idx] = 1
                train_count += 1
            elif orig_val_mask[orig_idx]:
                val_mask[aug_idx] = True
                mask[aug_idx] = 2
                val_count += 1
            elif orig_test_mask[orig_idx]:
                test_mask[aug_idx] = True
                mask[aug_idx] = 3
                test_count += 1
            
        
        view_dict['train_mask'] = train_mask
        view_dict['test_mask'] = test_mask
        view_dict['val_mask'] = val_mask
        view_dict['mask'] = mask
        
        # Create index lists for convenience
        view_dict['train_idx'] = torch.where(train_mask)[0]
        view_dict['val_idx'] = torch.where(val_mask)[0]
        view_dict['test_idx'] = torch.where(test_mask)[0]
        
        if 'eliminated_node_ids' not in metadata:
            # Compute eliminated nodes
            all_orig_indices = set(range(len(original_graph_nodes)))
            remaining_orig_indices = set(aug_to_orig_idx.values())
            eliminated_indices = sorted(list(all_orig_indices - remaining_orig_indices))
            metadata['eliminated_node_ids'] = eliminated_indices
        
        # Add split statistics to metadata
        metadata['split_stats'] = {
            'train_nodes': train_count,
            'val_nodes': val_count,
            'test_nodes': test_count,
            'total_nodes': num_aug_nodes,
            'unmapped_nodes': num_aug_nodes - (train_count + val_count + test_count)
        }
        
        # Add node name list for model compatibility
        view_dict['node_name'] = aug_node_list
        
        # Try to apply normalization to the view features (non-destructive: write to 'x_norm')
        try:
            norm_params = self.enhanced_data_dict.get('normalization_params', None)
            x = view_dict.get('x')
            if x is not None:
                is_torch = isinstance(x, torch.Tensor)
                x_np = x.numpy() if is_torch else (np.array(x) if not isinstance(x, np.ndarray) else x)

                # Determine original feature count and curvature feature count
                orig_feat_count = self.data_dict['feature'].shape[1] if isinstance(self.data_dict['feature'], torch.Tensor) else self.data_dict['feature'].shape[1]
                total_feat_count = x_np.shape[1]
                curv_feat_count = total_feat_count - orig_feat_count

                applied = False
                if norm_params is not None:
                    orig_mean = norm_params.get('original_mean')
                    orig_std = norm_params.get('original_std')
                    curv_mean = norm_params.get('curvature_mean')
                    curv_std = norm_params.get('curvature_std')

                    if orig_mean is not None and orig_std is not None and curv_mean is not None and curv_std is not None:
                        # Ensure shapes align
                        orig_std = np.where(orig_std == 0, 1.0, orig_std)
                        curv_std = np.where(curv_std == 0, 1.0, curv_std)

                        orig_part = (x_np[:, :orig_feat_count] - orig_mean) / orig_std
                        curv_part = (x_np[:, orig_feat_count:] - curv_mean) / curv_std
                        x_norm = np.hstack([orig_part, curv_part])
                        view_dict['x_norm'] = torch.from_numpy(x_norm).to(dtype=x.dtype) if is_torch else x_norm
                        logger.info('Applied stored normalization_params to augmented view and saved x_norm')
                        applied = True

                # Fallback: use fold-computed normalization if available
                if (not applied) and ('fold_normalization' in view_dict):
                    norm = view_dict['fold_normalization']
                    mean = norm['mean']
                    std = norm['std'].copy()
                    std[std == 0] = 1.0
                    x_norm = (x_np - mean) / std
                    view_dict['x_norm'] = torch.from_numpy(x_norm).to(dtype=x.dtype) if is_torch else x_norm
                    logger.info('Applied fold-computed normalization to augmented view and saved x_norm')
        except Exception as e:
            logger.warning(f'Failed to apply normalization to augmented view: {e}')

        logger.info(f"View {view_idx + 1} splits - Train: {train_count}, Val: {val_count}, Test: {test_count}")
        
        return view_dict
    
    def propagate_kfold_splits_to_view(self, view_dict, aug_graph, metadata, view_idx):
        """
        Propagate k-fold cross-validation splits from original graph to augmented view
        
        Parameters:
        view_dict: dict, augmented view data
        aug_graph: nx.Graph, augmented graph
        metadata: dict, augmentation metadata
        view_idx: int, index of the view
        
        Returns:
        dict: view_dict with added k-fold split information
        """
        # Use SORTED nodes for consistency
        original_graph_nodes = sorted(self.network.G.nodes())
        aug_node_list = sorted(aug_graph.nodes())
        
        # Create mapping from augmented node indices to original node indices
        aug_to_orig_idx = {}
        orig_name_to_idx = {name: i for i, name in enumerate(original_graph_nodes)}
        
        for aug_idx, node_name in enumerate(aug_node_list):
            if node_name in orig_name_to_idx:
                aug_to_orig_idx[aug_idx] = orig_name_to_idx[node_name]
        
        num_aug_nodes = len(aug_node_list)
        
            # Get original k-fold splits and normalization info
        original_kfold_splits = self.enhanced_data_dict['kfold_splits']
        normalization_info = self.enhanced_data_dict.get('normalization_info', {})
    
        # Check if we're using shared normalization (fitted on Fold 1)
        is_shared_norm = normalization_info.get('mode') == 'kfold_shared'
        fitted_on_fold = normalization_info.get('fitted_on_fold', 1)
    
        # Get the features we'll normalize (if needed)
        features = view_dict.get('x')
        
        # Convert to numpy if it's a tensor
        if features is not None and isinstance(features, torch.Tensor):
            features_np = features.numpy()
        elif features is not None:
            features_np = np.array(features) if not isinstance(features, np.ndarray) else features
        else:
            features_np = None
        
        # Propagate each fold
        augmented_kfold_splits = []
        
        for fold_data in original_kfold_splits:
            fold_idx = fold_data['fold']
            orig_train_mask = fold_data['train_mask']
            orig_val_mask = fold_data['val_mask']
            
            # Initialize masks for augmented view
            aug_train_mask = np.zeros(num_aug_nodes, dtype=bool)
            aug_val_mask = np.zeros(num_aug_nodes, dtype=bool)
            
            train_count = 0
            val_count = 0
            
            # Propagate masks to augmented nodes
            for aug_idx, orig_idx in aug_to_orig_idx.items():
                if orig_train_mask[orig_idx]:
                    aug_train_mask[aug_idx] = True
                    train_count += 1
                elif orig_val_mask[orig_idx]:
                    aug_val_mask[aug_idx] = True
                    val_count += 1
            
                # If we're using shared normalization and features exist
                if is_shared_norm and features_np is not None and fold_idx == fitted_on_fold:
                    # Store normalization parameters from this fold's training data
                    train_features_np = features_np[aug_train_mask]
                    
                    if len(train_features_np) > 0:
                        feature_mean = np.mean(train_features_np, axis=0)
                        feature_std = np.std(train_features_np, axis=0)
                        feature_std[feature_std == 0] = 1.0  # Avoid division by zero
                    
                        view_dict['fold_normalization'] = {
                            'mean': feature_mean,
                            'std': feature_std,
                            'fitted_on_fold': fitted_on_fold
                        }
            
            augmented_fold_data = {
                'fold': fold_idx,
                'train_idx': np.where(aug_train_mask)[0],
                'val_idx': np.where(aug_val_mask)[0],
                'train_mask': aug_train_mask,
                'val_mask': aug_val_mask,
                'train_count': train_count,
                'val_count': val_count
            }
            
            # Add normalization reference if we're using shared normalization
            if is_shared_norm:
                augmented_fold_data['normalization_reference'] = {
                    'use_fold': fitted_on_fold,
                    'mode': 'shared'
                }
            
            augmented_kfold_splits.append(augmented_fold_data)
            
            logger.info(f"View {view_idx + 1}, Fold {fold_idx}: Train={train_count}, Val={val_count}")
        
        # Add k-fold splits to view_dict
        view_dict['kfold_splits'] = augmented_kfold_splits
        
        # Add normalization mode to metadata
        metadata['normalization'] = {
            'mode': 'kfold_shared' if is_shared_norm else 'per_fold',
            'fitted_on_fold': fitted_on_fold if is_shared_norm else None
        }
        
        # Add node name list for model compatibility
        view_dict['node_name'] = aug_node_list
        
        # Add summary statistics to metadata
        metadata['kfold_stats'] = {
            'n_folds': len(augmented_kfold_splits),
            'total_nodes': num_aug_nodes,
            'avg_train_per_fold': np.mean([f['train_count'] for f in augmented_kfold_splits]),
            'avg_val_per_fold': np.mean([f['val_count'] for f in augmented_kfold_splits])
        }
        
        return view_dict

    def apply_fold1_normalization_to_view(self, view_dict):
        """
        Overwrite view_dict['x'] with normalization fitted on Fold 1 (if available).
        Keeps original features in 'x_raw'.

        Uses stored params in self.enhanced_data_dict['normalization_params'] if present;
        otherwise, will try to use view_dict['fold_normalization'] as a fallback.
        """
        if 'x' not in view_dict or view_dict['x'] is None:
            return

        x = view_dict['x']
        is_torch = isinstance(x, torch.Tensor)
        x_np = x.numpy() if is_torch else (np.array(x) if not isinstance(x, np.ndarray) else x)

        norm_params = self.enhanced_data_dict.get('normalization_params', None)
        applied = False

        # Determine original feature counts
        orig_feat_count = self.data_dict['feature'].shape[1] if isinstance(self.data_dict['feature'], torch.Tensor) else self.data_dict['feature'].shape[1]

        if norm_params is not None:
            orig_mean = norm_params.get('original_mean')
            orig_std = norm_params.get('original_std')
            curv_mean = norm_params.get('curvature_mean')
            curv_std = norm_params.get('curvature_std')

            if orig_mean is not None and orig_std is not None and curv_mean is not None and curv_std is not None:
                orig_std = np.where(orig_std == 0, 1.0, orig_std)
                curv_std = np.where(curv_std == 0, 1.0, curv_std)

                orig_part = (x_np[:, :orig_feat_count] - orig_mean) / orig_std
                curv_part = (x_np[:, orig_feat_count:] - curv_mean) / curv_std
                x_norm = np.hstack([orig_part, curv_part])

                # Keep raw copy
                view_dict['x_raw'] = x if is_torch else x_np.copy()
                view_dict['x'] = torch.from_numpy(x_norm).to(dtype=x.dtype) if is_torch else x_norm
                applied = True

        # Fallback to fold-specific normalization stored in the view
        if not applied and 'fold_normalization' in view_dict:
            norm = view_dict['fold_normalization']
            mean = norm['mean']
            std = norm['std'].copy()
            std[std == 0] = 1.0
            x_norm = (x_np - mean) / std
            view_dict['x_raw'] = x if is_torch else x_np.copy()
            view_dict['x'] = torch.from_numpy(x_norm).to(dtype=x.dtype) if is_torch else x_norm
            applied = True

        if applied:
            logger.info("Applied Fold-1 normalization to view (x overwritten, x_raw preserved)")

    def save_enhanced_dataset(self, output_file: str):
        """
        Save enhanced dataset with curvature features
        
        Parameters:
        output_file: str, path to save enhanced dataset
        """
        if self.enhanced_data_dict is None:
            raise ValueError("Must integrate features first using integrate_features()")
        
        logger.info(f"Saving enhanced dataset to {output_file}...")
        
        with open(output_file, 'wb') as f:
            pickle.dump(self.enhanced_data_dict, f)
        
        logger.info("Enhanced dataset saved successfully!")
        
    def save_network(self, output_path: str, graph_name: str = None):
        """
        Save NetworkX graph using your Network class method
        
        Parameters:
        output_path: str, path to save graph
        graph_name: str, name for saved graph file
        """
        if self.network is None:
            raise ValueError("Must build network first using build_network()")
        
        if graph_name is None:
            # Extract name from dataset file
            if 'GGNet' in self.dataset_file:
                graph_name = 'GGNet_graph_with_curvatures'
            elif 'PathNet' in self.dataset_file:
                graph_name = 'PathNet_graph_with_curvatures'
            elif 'PPNet' in self.dataset_file:
                graph_name = 'PPNet_graph_with_curvatures'
            else:
                graph_name = 'network_graph_with_curvatures'
        
        self.network.save_graph(graph_name, output_path)
        logger.info(f"Network saved as {graph_name} in {output_path}")
    
    def save_contrastive_dataset(self, contrastive_dataset, output_dir):
        """
        Save contrastive dataset with dataset-specific naming
        
        Parameters:
        contrastive_dataset: dict, output from create_contrastive_network()
        output_dir: str, directory to save the file
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract dataset name from file
        if 'GGNet' in self.dataset_file:
            dataset_name = 'GGNet'
        elif 'PathNet' in self.dataset_file:
            dataset_name = 'PathNet'
        elif 'PPNet' in self.dataset_file:
            dataset_name = 'PPNet'
        else:
            dataset_name = 'network'
        
        # Create filename with dataset name and augmentation config
        strategy = contrastive_dataset['augmentation_config']['strategy']
        num_views = contrastive_dataset['num_views']
        ratio = contrastive_dataset['augmentation_config']['elimination_ratio']
        
        filename = f'{dataset_name}_contrastive_v{num_views}_{strategy}_r{ratio}.pkl'
        output_path = os.path.join(output_dir, filename)
        
        with open(output_path, 'wb') as f:
            pickle.dump(contrastive_dataset, f)
        
        logger.info(f"Contrastive dataset saved to {output_path}")
        logger.info(f"  - Original nodes: {contrastive_dataset['original']['feature'].shape[0]}")
        logger.info(f"  - Augmented views: {num_views}")
        logger.info(f"  - Strategy: {strategy}, Elimination ratio: {ratio}")
    
        return output_path
        
    def run_pipeline(self, 
                    output_dir='./output', 
                    method='both', 
                    normalize=True,
                    use_augmentation=False,
                    num_augmented_views=2,
                    elimination_ratio=0.2,
                    elimination_strategy='priority',
                    compute_aug_curvature = True,
                    create_split = True,
                    test_size = 0.2,
                    val_size = 0.1,
                    stratify = True,
                    random_seed = 42,
                    use_existing_mask = False,
                    train_ratio_from_mask = 0.8,
                    use_kfold = True,
                    n_folds = 5):
        """
        Run the complete curvature integration pipeline with optional augmentation
        
        Parameters:
        output_dir: str, directory to save outputs
        method: str, curvature calculation method
        normalize: bool, whether to normalize features
        use_augmentation: bool, whether to generate augmented views
        num_augmented_views: int, number of augmented views to generate
        elimination_ratio: float, ratio of nodes to eliminate in augmentation
        elimination_strategy: str, 'priority', 'random', or 'coarsening'
        create_split: bool, whether to create train/val/test split
        test_size: float, proportion for test set
        val_size: float, proportion for validation set
        stratify: bool, whether to stratify splits by labels
        random_seed: int, random seed for reproducibility
        use_existing_mask: bool, whether to use existing mask from dataset
        use_kfold: bool, whether to create k-fold splits instead
        train_ratio_from_mask: int
        n_folds: int, number of folds for k-fold cross-validation
        
        Returns:
        dict: Enhanced data dictionary with curvature features (and augmentations if requested)
        """
        try:
            logger.info("=== Starting Complete Curvature Integration Pipeline ===")
            
            # Standard pipeline steps
            self.load_data()
            self.build_network()
            self.calculate_curvatures(method=method)
            
            if create_split:
                if use_kfold:
                    logger.info("\n=== Creating K-Fold Cross-Validation Splits ===")
                    # K-fold splits can be created before feature integration
                    kfold_data = self.create_kfold_splits(
                        n_splits=n_folds,
                        stratify=stratify,
                        random_seed=random_seed
                    )
                    
                    # Use fold 0's training mask for normalization
                    logger.info("Note: K-fold mode - using Fold 1 training data for normalization")
                    logger.info("All folds will share the same normalization (fitted on Fold 1 train data)")
                    
                    # Create a temporary split_data using first fold
                    first_fold = kfold_data[0]
                    self.split_data = {
                        'train_mask': first_fold['train_mask'],
                        'val_mask': first_fold['val_mask'],
                        'test_mask': np.zeros(len(first_fold['train_mask']), dtype=bool),
                        'train_idx': first_fold['train_idx'],
                        'val_idx': first_fold['val_idx'],
                        'test_idx': np.array([]),
                        'split_config': {
                            'mode': 'kfold_shared_normalization',
                            'note': 'All folds use Fold 1 training data for normalization'
                        }
                    }
                    
                    # Integrate features ONCE with fold 0's training mask
                    self.integrate_features(normalize=normalize)
                    
                    # Copy k-fold splits to enhanced data dict
                    self.enhanced_data_dict['kfold_splits'] = kfold_data
                    
                    # Add normalization info
                    self.enhanced_data_dict['normalization_info'] = {
                        'mode': 'kfold_shared',
                        'fitted_on_fold': 1,
                        'n_folds': n_folds,
                        'note': 'All folds use same normalization for consistency'
                    }
                    
                    logger.info("✅ All folds ready with consistent normalization")
                    
                    # Clear the temporary split_data
                    self.split_data = None
                    
                else:
                    # Regular train/val/test split
                    logger.info("\n=== Creating Train/Validation/Test Split ===")
                    # Create split first for split-aware normalization
                    self.create_train_test_split(
                        test_size=test_size,
                        val_size=val_size,
                        stratify=stratify,
                        random_seed=random_seed,
                        use_existing_mask=use_existing_mask
                    )
                    # Then integrate features (will use splits for normalization)
                    self.integrate_features(normalize=normalize)
            else:
                # No splits - integrate features normally
                self.integrate_features(normalize=normalize)
            
            os.makedirs(output_dir, exist_ok=True)
            
            # Save network
            self.save_network(output_dir)
            
            # Save enhanced dataset with splits
            # dataset_name = self.extract_dataset_name()
            # enhanced_path = os.path.join(output_dir, f'{dataset_name}_enhanced_with_splits.pkl')
            # self.save_enhanced_dataset(enhanced_path)
            
            result = self.enhanced_data_dict
            
            # Optional: Generate augmented views
            if use_augmentation:
                logger.info("\n=== Generating Schur Complement Augmentations ===")
                
                self.initialize_schur_augmentation(
                    elimination_ratio=elimination_ratio,
                    elimination_strategy=elimination_strategy,
                    neighbor_sort_method='weight'
                )
                
                # Create contrastive dataset
                contrastive_data = self.create_contrastive_network(
                    num_views=num_augmented_views,
                    use_curvature_weights=True,
                    compute_aug_curvature=compute_aug_curvature,
                    curvature_method=method
                )
                
                # Save contrastive dataset
                self.save_contrastive_dataset(contrastive_dataset=contrastive_data, output_dir=output_dir)
                
                result = contrastive_data
            
            logger.info("\n=== Pipeline Completed Successfully ===")
            logger.info(f"Original features: {self.data_dict['feature'].shape}")
            logger.info(f"Enhanced features: {self.enhanced_data_dict['feature'].shape}")
            logger.info(f"Outputs saved in: {output_dir}")
            
            if create_split and not use_kfold:
                logger.info(f"Train/Val/Test split created")
            elif create_split and use_kfold:
                logger.info(f"{n_folds}-fold cross-validation splits created")
                logger.info(f"All folds use shared normalization (fitted on Fold 1)")
            
            logger.info(f"Outputs saved in: {output_dir}")
            
            return result

        except Exception as e:
            logger.error(f'Error occurred while processing the dataset: {e}')
            import traceback
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
            raise
        
    def extract_dataset_name(self):
        """Extract dataset name from file path"""
        if 'GGNet' in self.dataset_file:
            return 'GGNet'
        elif 'PathNet' in self.dataset_file:
            return 'PathNet'
        elif 'PPNet' in self.dataset_file:
            return 'PPNet'
        else:
            return 'network'

def main():
    
    """
    Example usage of the complete pipeline
    """
    parser = argparse.ArgumentParser(description='Integrate curvature features with gene network data')
    
    # Input/Output
    parser.add_argument('--dataset_file', type=str, help='Input dataset pickle file')
    parser.add_argument('--output_dir', type=str, default='./curvature_output', 
                        help='Output directory for enhanced dataset and graph')
    
    # Curvature calculation
    parser.add_argument('--method', type=str, choices=['ollivier', 'forman', 'both'], 
                        default='both', help='Curvature calculation method')
    parser.add_argument('--no_normalize', action='store_true', 
                        help='Skip normalization of curvature features')
    
    # Augmentation
    parser.add_argument('--augment', action='store_true',
                        help='Generate augmented views using Schur complement')
    parser.add_argument('--num_views', type=int, default=2,
                        help='Number of augmented views to generate')
    parser.add_argument('--elimination_ratio', type=float, default=0.2,
                        help='Ratio of nodes to eliminate in augmentation (0.1-0.3 recommended)')
    parser.add_argument('--strategy', type=str, choices=['priority', 'random', 'coarsening'],
                        default='priority', help='Elimination strategy for augmentation')
    parser.add_argument('--no_aug_curvature', action='store_true',
                        help='Skip curvature calculation for augmented graphs')
    
    # Train/Test Split
    parser.add_argument('--no_split', action='store_true',
                        help='Skip creating train/val/test split')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Proportion of data for test set (default: 0.2)')
    parser.add_argument('--val_size', type=float, default=0.1,
                        help='Proportion of data for validation set (default: 0.1)')
    parser.add_argument('--no_stratify', action='store_true',
                        help='Disable stratified splitting by labels')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--use_existing_mask', action='store_true',
                        help='Use existing boolean mask from dataset (True values split into train/val, False as test)')
    parser.add_argument('--train_ratio_from_mask', type=float, default=0.8,
                        help='When using existing mask, ratio of True values to use for training vs validation (default: 0.8)')
    
    # K-Fold Cross-Validation
    parser.add_argument('--use_kfold', action='store_true',
                        help='Use k-fold cross-validation instead of single train/test split')
    parser.add_argument('--n_folds', type=int, default=5,
                        help='Number of folds for cross-validation (default: 5)')
    
    args = parser.parse_args()
    
    try:
        pipeline = CurvaturePipeline(args.dataset_file)
        
        result = pipeline.run_pipeline(
            output_dir=args.output_dir,
            method=args.method,
            normalize=not args.no_normalize,
            use_augmentation=args.augment,
            num_augmented_views=args.num_views,
            elimination_ratio=args.elimination_ratio,
            elimination_strategy=args.strategy,
            compute_aug_curvature=not args.no_aug_curvature,
            create_split=not args.no_split,
            test_size=args.test_size,
            val_size=args.val_size,
            stratify=not args.no_stratify,
            random_seed=args.random_seed,
            use_existing_mask=args.use_existing_mask,
            train_ratio_from_mask=args.train_ratio_from_mask,
            use_kfold=args.use_kfold,
            n_folds=args.n_folds
        )
        
        print("\n=== Final Statistics ===")
        if args.augment:
            print(f"Generated {args.num_views} augmented views")
            print(f"Augmentation strategy: {args.strategy}")
            print(f"Elimination ratio: {args.elimination_ratio}")
        else:
            enhanced_data_dict = result
            print(f"Enhanced feature names: {len(enhanced_data_dict['feature_name'])}")
            print("New curvature features added:")
            original_features = len(pipeline.data_dict['feature_name'])
            for i, name in enumerate(enhanced_data_dict['feature_name'][original_features:], 1):
                print(f"  {i}. {name}")
        
        if not args.no_split:
            if args.use_kfold:
                print(f"\nK-Fold Cross-Validation: {args.n_folds} folds created")
            else:
                print(f"\nTrain/Val/Test Split:")
                if pipeline.split_data:
                    print(f"  Train: {len(pipeline.split_data['train_idx'])} samples")
                    if len(pipeline.split_data['val_idx']) > 0:
                        print(f"  Val:   {len(pipeline.split_data['val_idx'])} samples")
                    print(f"  Test:  {len(pipeline.split_data['test_idx'])} samples")
        
    except Exception as e:
        logger.info(f"Error in pipeline: {e}")
        exit(1)

if __name__ == '__main__':
    logger=get_logger(__name__)
    main()