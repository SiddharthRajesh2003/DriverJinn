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
                                seed = 42, use_existing_mask = False, train_ratio_from_mask=0.8):
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
                        random_state=seed,
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
                random_state=seed,
                stratify=stratify_labels if stratify else None
            )
            
            if val_size > 0:
                # Calculate validation size relative to train_val set
                val_size_adjusted = val_size / (1 - test_size)
                
                stratify_train_val = labels_np[train_val_idx] if stratify else None
                
                train_idx, val_idx = train_test_split(
                    train_val_idx,
                    test_size=val_size_adjusted,
                    random_state=seed,
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
                'random_seed': seed,
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
    
    def create_kfold_splits(self, n_splits = 5, stratify = True, seed = 42):
        """
        Create K-fold cross-validation splits
        
        Parameters:
        n_splits: int, number of folds
        stratify: bool, whether to use stratified K-fold
        seed: int, random seed for reproducibility
        
        Returns:
        list: List of dictionaries, each containing train/val indices for one fold
        """
        logger.info(f"Creating {n_splits}-fold cross-validation splits...")
        
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
            kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            splits = list(kfold.split(all_indices, labels_np))
        else:
            from sklearn.model_selection import KFold
            kfold = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
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
                feature_df = pd.DataFrame(
                    data=self.data_dict['feature'].numpy(),
                    columns = self.data_dict['feature_name'],
                    index = self.data_dict['node_name']
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
            if features is not None:
                if isinstance(features, torch.Tensor):
                    features_np = features.numpy()
                else:
                    features_np = features

                node_names = list(aug_G.nodes())
                
                # CRITICAL FIX: Verify shapes match before creating DataFrame
                if len(node_names) != features_np.shape[0]:
                    logger.warning(f"Shape mismatch: {len(node_names)} nodes but {features_np.shape[0]} feature rows")
                    
                    # Check if we have more nodes than features or vice versa
                    if len(node_names) > features_np.shape[0]:
                        logger.warning(f"Trimming {len(node_names) - features_np.shape[0]} nodes without features")
                        # Keep only nodes that have features
                        node_names = node_names[:features_np.shape[0]]
                    else:
                        logger.warning(f"Trimming {features_np.shape[0] - len(node_names)} extra features")
                        # Keep only features for existing nodes
                        features_np = features_np[:len(node_names)]
                
                feature_df = pd.DataFrame(
                    data=features_np,
                    index=node_names
                )
            
            else:
                # Use original features if augmented features not available
                feature_df = pd.DataFrame(
                    data=self.data_dict['feature'].numpy(),
                    columns=self.data_dict['feature_name'],
                    index=self.data_dict['node_name']
                )
            
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
        
        node_names = self.enhanced_data_dict['node_name']
        
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
        
        Returns:
        dict: Dictionary containing original and augmented data for contrastive learning
        """
        if self.enhanced_data_dict is None:
            raise ValueError("Must integrate features first using integrate_features()")
        
        logger.info("Creating contrastive learning dataset...")
        
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
                
            
            if propagate_splits and self.split_data is not None:
                view_dict = self.propagate_splits_to_view(
                    view_dict,
                    aug_graph,
                    metadata,
                    view_idx
                )
        
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
        if self.split_data is not None:
            contrastive_dataset['split_info'] = {
                'split_config': self.split_data['split_config'],
                'original_train_size': len(self.split_data['train_idx']),
                'original_val_size': len(self.split_data['val_idx']),
                'original_test_size': len(self.split_data['test_idx'])
            }
        
        logger.info(f"Contrastive dataset created with {num_views} augmented views")
        if compute_aug_curvature:
            logger.info("Augmented views include pre-computed curvature features")
        if propagate_splits and self.split_data is not None:
            logger.info("Train/val/test splits propagated to all augmented views")
        
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
        # Get original node names and augmented node names
        original_nodes = self.enhanced_data_dict['node_name']
        aug_node_list = list(aug_graph.nodes())
        
        # Create mapping from augmented node indices to original node indices
        aug_to_orig_idx = {}
        orig_name_to_idx = {name: i for i, name in enumerate(original_nodes)}
        
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
            all_orig_indices = set(range(len(original_nodes)))
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
        
        logger.info(f"View {view_idx + 1} splits - Train: {train_count}, Val: {val_count}, Test: {test_count}")
        
        return view_dict

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
            
            # For K-fold: Create splits on raw data, then integrate features for each fold
            # For regular split: Create split first, then integrate with split-aware normalization
            if create_split:
                if use_kfold:
                    logger.info("\n=== Creating K-Fold Cross-Validation Splits ===")
                    # K-fold splits can be created before feature integration
                    self.create_kfold_splits(
                        n_splits=n_folds,
                        stratify=stratify,
                        seed=random_seed
                    )
                    # For k-fold, integrate features without split-aware normalization
                    # Each fold will handle normalization separately during training
                    logger.info("Note: K-fold mode - normalization will be per-fold during training")
                    self.integrate_features(normalize=normalize)
                    
                    # Copy k-fold splits to enhanced data dict
                    if 'kfold_splits' in self.data_dict:
                        self.enhanced_data_dict['kfold_splits'] = self.data_dict['kfold_splits']
                    
                else:
                    # Regular train/val/test split
                    logger.info("\n=== Creating Train/Validation/Test Split ===")
                    # Create split first for split-aware normalization
                    self.create_train_test_split(
                        test_size=test_size,
                        val_size=val_size,
                        stratify=stratify,
                        seed=random_seed,
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
            dataset_name = self.extract_dataset_name()
            enhanced_path = os.path.join(output_dir, f'{dataset_name}_enhanced_with_splits.pkl')
            self.save_enhanced_dataset(enhanced_path)
            
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
            
            logger.info(f"Outputs saved in: {output_dir}")
            
            return result

        except Exception as e:
            logger.error(f'Error occurred while processing the dataset: {e}')
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