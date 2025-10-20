from graph_builder.curvature_calculator import EdgeCurvature
from utils.logging_manager import get_logger

from collections import defaultdict
import numpy as np
import torch
import pandas as pd

from scipy.spatial.distance import cosine
from sklearn.preprocessing import StandardScaler

logger=get_logger(__name__)


class CurvatureFeatureIntegrator:
    """
    Enhanced integrator with proper train/test split normalization
    """
    
    def __init__(self, edge_calc, data_dict):
        """
        Initialize integrator
        
        Parameters:
        edge_calc: EdgeCurvature object with computed curvatures
        data_dict: Dictionary containing graph data
        """
        self.edge_calc = edge_calc
        self.data_dict = data_dict
        self.features = data_dict['feature']
        self.feature_names = data_dict['feature_name']
        self.node_names = data_dict['node_name']
        
        # Check if masks exist in data_dict
        self.has_splits = 'train_mask' in data_dict or 'mask' in data_dict
        
    def create_enhanced_features(self, normalize=True, train_mask=None, val_mask=None, test_mask=None):
        """
        Create enhanced feature matrix with curvature features
        
        Parameters:
        normalize: bool, whether to normalize features
        train_mask: Optional boolean mask for training nodes (fit scaler on these)
        val_mask: Optional boolean mask for validation nodes
        test_mask: Optional boolean mask for test nodes
        
        If masks not provided, will try to use from data_dict
        If no masks available, will fit on all data (old behavior)
        
        Returns:
        dict: Updated data_dict with enhanced features
        """
        
        logger.info("Creating enhanced features with proper curvature aggregation...")
        
        # Get masks from data_dict if not provided
        if train_mask is None and 'train_mask' in self.data_dict:
            train_mask = self.data_dict['train_mask']
            if isinstance(train_mask, torch.Tensor):
                train_mask = train_mask.numpy()
        
        if val_mask is None and 'val_mask' in self.data_dict:
            val_mask = self.data_dict['val_mask']
            if isinstance(val_mask, torch.Tensor):
                val_mask = val_mask.numpy()
        
        if test_mask is None and 'test_mask' in self.data_dict:
            test_mask = self.data_dict['test_mask']
            if isinstance(test_mask, torch.Tensor):
                test_mask = test_mask.numpy()
        
        # Alternative: Try to extract from integer mask
        if train_mask is None and 'mask' in self.data_dict:
            mask = self.data_dict['mask']
            if isinstance(mask, torch.Tensor):
                mask = mask.numpy()
            train_mask = (mask == 1)
            val_mask = (mask == 2)
            test_mask = (mask == 3)
        
        # Create curvature features using EdgeCurvature class
        curvature_df = self.edge_calc.create_node_curvature_features(node_names=self.node_names)
        
        # Debug info
        logger.info(f"Curvature DataFrame shape: {curvature_df.shape}")
        logger.info(f"Index matches node_names: {curvature_df.index.equals(pd.Index(self.node_names))}")
        
        # Check for non-zero values
        for col in ['ollivier_mean', 'ollivier_degree']:
            if col in curvature_df.columns:
                non_zero = (curvature_df[col] != 0).sum()
                logger.info(f"{col}: {non_zero} non-zero values")
        
        # Extract curvature features in order
        curvature_feature_names = [
            'ollivier_mean', 'ollivier_std', 'ollivier_min', 'ollivier_max', 'ollivier_median', 'ollivier_degree',
            'forman_mean', 'forman_std', 'forman_min', 'forman_max', 'forman_median', 'forman_degree'
        ]
        
        curvature_feature_list = []
        for name in self.node_names:
            if name in curvature_df.index:
                node_curvature_features = curvature_df.loc[name, curvature_feature_names].values
            else:
                logger.warning(f"Node {name} not found in curvature_df")
                node_curvature_features = np.zeros(len(curvature_feature_names))
            
            curvature_feature_list.append(node_curvature_features)
        
        curvature_features = np.array(curvature_feature_list)
        
        logger.info(f"Curvature features array shape: {curvature_features.shape}")
        logger.info(f"Ollivier_degree non-zero count: {np.sum(curvature_features[:, 5] != 0)}")
        
        # Calculate additional curvature-based features
        logger.info("Calculating additional curvature-based features...")
        pos_curve_deg, neg_curve_deg, curve_homophily = self.calculate_curvature_based_features()
        
        # Debug the additional features
        logger.info(f"Positive curvature degree - non-zero: {np.sum(pos_curve_deg != 0)}, mean: {np.mean(pos_curve_deg):.4f}")
        logger.info(f"Negative curvature degree - non-zero: {np.sum(neg_curve_deg != 0)}, mean: {np.mean(neg_curve_deg):.4f}")
        logger.info(f"Curvature homophily - non-zero: {np.sum(curve_homophily != 0)}, mean: {np.mean(curve_homophily):.4f}")
        
        additional_features = np.column_stack([pos_curve_deg, neg_curve_deg, curve_homophily])
        all_curvature_features = np.hstack([curvature_features, additional_features])
        
        # Get original features as numpy
        original_features = self.features.numpy() if isinstance(self.features, torch.Tensor) else self.features
        
        # Normalization with proper train/test split
        if normalize:
            logger.info("Normalizing features with train/test split awareness...")
            
            if train_mask is not None and np.any(train_mask):
                # FIT on training data only
                logger.info("Fitting scalers on TRAINING data only")
                
                # Scaler for original features
                original_scaler = StandardScaler()
                original_scaler.fit(original_features[train_mask])
                
                # Scaler for curvature features with handling for sparse/zero features
                curvature_scaler = StandardScaler()
                train_curvatures = all_curvature_features[train_mask]
                
                # Check for columns that are all zeros or have zero variance
                train_std = train_curvatures.std(axis=0)
                zero_var_cols = train_std == 0
                
                if np.any(zero_var_cols):
                    logger.warning(f"Found {zero_var_cols.sum()} curvature features with zero variance in training data")
                    logger.warning("These features will not be scaled to prevent division by zero")
                    
                    # Create a custom scaler that handles zero-variance columns
                    curvature_scaler.fit(train_curvatures)
                    
                    # Replace zero standard deviations with 1.0 to prevent division by zero
                    curvature_scaler.scale_ = np.where(
                        curvature_scaler.scale_ == 0, 
                        1.0, 
                        curvature_scaler.scale_
                    )
                    curvature_scaler.var_ = curvature_scaler.scale_ ** 2
                else:
                    curvature_scaler.fit(train_curvatures)
                
                # TRANSFORM all data (train, val, test)
                original_features_scaled = original_scaler.transform(original_features)
                curvature_features_scaled = curvature_scaler.transform(all_curvature_features)
                
                # Log training statistics
                train_samples = int(train_mask.sum())
                logger.info(f"Train samples used for fitting: {train_samples}")
                
                if train_samples > 0:
                    train_orig_mean = original_features_scaled[train_mask].mean()
                    train_orig_std = original_features_scaled[train_mask].std()
                    train_curv_mean = curvature_features_scaled[train_mask].mean()
                    train_curv_std = curvature_features_scaled[train_mask].std()
                    
                    logger.info(f"Original features - Train mean: {train_orig_mean:.4f}, std: {train_orig_std:.4f}")
                    logger.info(f"Curvature features - Train mean: {train_curv_mean:.4f}, std: {train_curv_std:.4f}")
                
                # Log test statistics if test mask is provided and valid
                if test_mask is not None:
                    test_samples = int(test_mask.sum())
                    if test_samples > 0:
                        test_orig_mean = original_features_scaled[test_mask].mean()
                        test_orig_std = original_features_scaled[test_mask].std()
                        test_curv_mean = curvature_features_scaled[test_mask].mean()
                        test_curv_std = curvature_features_scaled[test_mask].std()
                        
                        logger.info(f"Test samples: {test_samples}")
                        logger.info(f"Original features - Test mean: {test_orig_mean:.4f}, std: {test_orig_std:.4f}")
                        logger.info(f"Curvature features - Test mean: {test_curv_mean:.4f}, std: {test_curv_std:.4f}")
                    else:
                        logger.warning("Test mask provided but contains no True values")
                
            else:
                # Fallback: fit on all data (old behavior)
                logger.warning("No training mask provided - fitting scalers on ALL data (may cause data leakage)")
                
                original_scaler = StandardScaler()
                original_features_scaled = original_scaler.fit_transform(original_features)
                
                # Handle sparse curvature features
                curvature_scaler = StandardScaler()
                curvature_scaler.fit(all_curvature_features)
                
                # Check for zero variance
                zero_var_mask = curvature_scaler.scale_ == 0
                if np.any(zero_var_mask):
                    logger.warning(f"Found {zero_var_mask.sum()} curvature features with zero variance")
                    curvature_scaler.scale_ = np.where(
                        curvature_scaler.scale_ == 0,
                        1.0,
                        curvature_scaler.scale_
                    )
                    curvature_scaler.var_ = curvature_scaler.scale_ ** 2
                
                curvature_features_scaled = curvature_scaler.transform(all_curvature_features)
            
            # Combine scaled features
            enhanced_features = np.hstack([original_features_scaled, curvature_features_scaled])
            
            # Store scalers for later use
            self.original_scaler = original_scaler
            self.curvature_scaler = curvature_scaler
            
        else:
            # No normalization
            enhanced_features = np.hstack([original_features, all_curvature_features])
            self.original_scaler = None
            self.curvature_scaler = None
        
        # Create feature names
        additional_feature_names = ['positive_curvature_degree', 'negative_curvature_degree', 'curvature_homophily']
        complete_curvature_names = curvature_feature_names + additional_feature_names
        enhanced_feature_names = self.feature_names + complete_curvature_names
        
        # Create enhanced data dict
        enhanced_data_dict = self.data_dict.copy()
        enhanced_data_dict['feature'] = torch.tensor(enhanced_features, dtype=torch.float32)
        enhanced_data_dict['feature_name'] = enhanced_feature_names
        
        logger.info(f"Original features: {original_features.shape}")
        logger.info(f"Enhanced features: {enhanced_features.shape}")
        logger.info(f"Added {len(complete_curvature_names)} curvature-based features")
        
        return enhanced_data_dict
    
    def calculate_curvature_based_features(self):
        """
        Calculate additional curvature-based features:
        - Positive curvature degree (number of edges with positive curvature)
        - Negative curvature degree (number of edges with negative curvature)
        - Curvature homophily (tendency to connect to similar curvature nodes)
        
        Returns:
        tuple: (pos_curve_deg, neg_curve_deg, curve_homophily) numpy arrays
        """
        num_nodes = len(self.node_names)
        pos_curve_deg = np.zeros(num_nodes)
        neg_curve_deg = np.zeros(num_nodes)
        curve_homophily = np.zeros(num_nodes)
        
        # Get edge curvatures
        ollivier_curv = self.edge_calc.edge_curvature.get('OllivierRicci', {})
        
        if not ollivier_curv:
            logger.warning("No Ollivier curvature found. Returning zero features.")
            return pos_curve_deg, neg_curve_deg, curve_homophily
        
        logger.info(f"Processing {len(ollivier_curv)} edges for curvature-based features")
        
        # Calculate node-level curvature statistics
        node_curvatures = {i: [] for i in range(num_nodes)}
        
        # First pass: collect curvatures and count positive/negative
        edge_count = 0
        skipped_edges = 0
        
        for (u, v), curv in ollivier_curv.items():
            # u and v are ALREADY indices (integers), not names
            # Check if indices are valid
            if u >= num_nodes or v >= num_nodes or u < 0 or v < 0:
                skipped_edges += 1
                continue
            
            # Use indices directly
            u_idx = u
            v_idx = v
            
            node_curvatures[u_idx].append(curv)
            node_curvatures[v_idx].append(curv)
            
            # Count positive/negative curvature edges
            if curv > 0:
                pos_curve_deg[u_idx] += 1
                pos_curve_deg[v_idx] += 1
            elif curv < 0:
                neg_curve_deg[u_idx] += 1
                neg_curve_deg[v_idx] += 1
            
            edge_count += 1
        
        logger.info(f"Processed {edge_count} edges, skipped {skipped_edges} invalid edges")
        
        # Calculate mean curvature per node
        node_mean_curvatures = np.zeros(num_nodes)
        for i in range(num_nodes):
            if len(node_curvatures[i]) > 0:
                node_mean_curvatures[i] = np.mean(node_curvatures[i])
        
        # Second pass: Calculate curvature homophily
        for (u, v), curv in ollivier_curv.items():
            if u >= num_nodes or v >= num_nodes or u < 0 or v < 0:
                continue
            
            u_idx = u
            v_idx = v
            
            # Homophily: similarity of node's mean curvature to neighbor's mean curvature
            u_mean = node_mean_curvatures[u_idx]
            v_mean = node_mean_curvatures[v_idx]
            
            # Similarity measure (negative absolute difference, normalized)
            similarity = 1.0 / (1.0 + abs(u_mean - v_mean))
            
            curve_homophily[u_idx] += similarity
            curve_homophily[v_idx] += similarity
        
        # Normalize homophily by degree
        for i in range(num_nodes):
            degree = len(node_curvatures[i])
            if degree > 0:
                curve_homophily[i] /= degree
        
        # Log statistics
        logger.info(f"Pos curve degree: mean={pos_curve_deg.mean():.2f}, max={pos_curve_deg.max():.2f}, non-zero={np.sum(pos_curve_deg > 0)}")
        logger.info(f"Neg curve degree: mean={neg_curve_deg.mean():.2f}, max={neg_curve_deg.max():.2f}, non-zero={np.sum(neg_curve_deg > 0)}")
        logger.info(f"Curve homophily: mean={curve_homophily.mean():.4f}, max={curve_homophily.max():.4f}, non-zero={np.sum(curve_homophily > 0)}")
        
        return pos_curve_deg, neg_curve_deg, curve_homophily
    
    def analyze_curvature_distribution(self):
        """
        Analyze and log curvature distribution statistics
        """
        logger.info("\n=== Curvature Distribution Analysis ===")
        
        for curv_type in ['OllivierRicci', 'FormanRicci']:
            if curv_type in self.edge_calc.edge_curvature:
                curvatures = list(self.edge_calc.edge_curvature[curv_type].values())
                curvatures = np.array(curvatures)
                
                logger.info(f"\n{curv_type}:")
                logger.info(f"  Mean: {curvatures.mean():.4f}")
                logger.info(f"  Std: {curvatures.std():.4f}")
                logger.info(f"  Min: {curvatures.min():.4f}")
                logger.info(f"  Max: {curvatures.max():.4f}")
                logger.info(f"  Positive edges: {(curvatures > 0).sum()}")
                logger.info(f"  Negative edges: {(curvatures < 0).sum()}")
                logger.info(f"  Zero edges: {(curvatures == 0).sum()}")
    
    def create_edge_features_dict(self):
        """
        Create dictionary of edge-level curvature features
        
        Returns:
        dict: Dictionary containing edge curvature tensors
        """
        edge_index = self.data_dict['edge_index']
        num_edges = edge_index.shape[1]
        
        ollivier_curvatures = torch.zeros(num_edges)
        forman_curvatures = torch.zeros(num_edges)
        
        ollivier_dict = self.edge_calc.edge_curvature.get('OllivierRicci', {})
        forman_dict = self.edge_calc.edge_curvature.get('FormanRicci', {})
        
        # Map curvatures to edges
        for i in range(num_edges):
            src = edge_index[0, i].item()
            dst = edge_index[1, i].item()
            
            # Get node names
            src_name = self.node_names[src]
            dst_name = self.node_names[dst]
            
            # Get curvatures (try both directions)
            ollivier_curvatures[i] = ollivier_dict.get(
                (src_name, dst_name), 
                ollivier_dict.get((dst_name, src_name), 0.0)
            )
            
            forman_curvatures[i] = forman_dict.get(
                (src_name, dst_name),
                forman_dict.get((dst_name, src_name), 0.0)
            )
        
        # Create combined edge features
        edge_features = torch.stack([ollivier_curvatures, forman_curvatures], dim=1)
        
        return {
            'edge_ollivier_curvature': ollivier_curvatures,
            'edge_forman_curvature': forman_curvatures,
            'edge_features': edge_features,
            'edge_feature_names': ['ollivier_curvature', 'forman_curvature']
        }
    
    def transform_new_features(self, new_original_features, new_curvature_features):
        """
        Transform new features using fitted scalers (for inference on new data)
        
        Parameters:
        new_original_features: numpy array of original features
        new_curvature_features: numpy array of curvature features
        
        Returns:
        numpy array: Scaled and concatenated features
        """
        if self.original_scaler is None or self.curvature_scaler is None:
            logger.warning("Scalers not fitted. Returning unscaled features.")
            return np.hstack([new_original_features, new_curvature_features])
        
        original_scaled = self.original_scaler.transform(new_original_features)
        curvature_scaled = self.curvature_scaler.transform(new_curvature_features)
        
        return np.hstack([original_scaled, curvature_scaled])