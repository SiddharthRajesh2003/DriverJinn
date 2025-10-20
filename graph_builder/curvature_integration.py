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
        Fixed normalization handling for zero-variance features
        """
        
        logger.info("Creating enhanced features with proper curvature aggregation...")
        
        # [... existing mask extraction code ...]
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
        
        if train_mask is None and 'mask' in self.data_dict:
            mask = self.data_dict['mask']
            if isinstance(mask, torch.Tensor):
                mask = mask.numpy()
            train_mask = (mask == 1)
            val_mask = (mask == 2)
            test_mask = (mask == 3)
        
        # Create curvature features
        curvature_df = self.edge_calc.create_node_curvature_features(node_names=self.node_names)
        
        # Extract curvature features
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
        
        # Calculate additional features
        logger.info("Calculating additional curvature-based features...")
        pos_curve_deg, neg_curve_deg, curve_homophily = self.calculate_curvature_based_features()
        
        additional_features = np.column_stack([pos_curve_deg, neg_curve_deg, curve_homophily])
        all_curvature_features = np.hstack([curvature_features, additional_features])
        
        # Get original features
        original_features = self.features.numpy() if isinstance(self.features, torch.Tensor) else self.features
        
        # Normalization with proper zero-variance handling
        if normalize:
            logger.info("Normalizing features with train/test split awareness...")
            
            if train_mask is not None and np.any(train_mask):
                logger.info("Fitting scalers on TRAINING data only")
                
                # Original features scaler
                original_scaler = StandardScaler()
                original_scaler.fit(original_features[train_mask])
                
                # Curvature features scaler with PROPER zero-variance handling
                train_curvatures = all_curvature_features[train_mask]
                
                # Identify zero-variance columns
                train_std = train_curvatures.std(axis=0)
                train_mean = train_curvatures.mean(axis=0)
                zero_var_mask = train_std == 0
                
                if np.any(zero_var_mask):
                    zero_var_count = zero_var_mask.sum()
                    logger.warning(f"Found {zero_var_count} curvature features with zero variance")
                    logger.info(f"Zero-variance feature indices: {np.where(zero_var_mask)[0]}")
                    
                    # For zero-variance features, check if they're all zeros
                    for idx in np.where(zero_var_mask)[0]:
                        col_values = train_curvatures[:, idx]
                        unique_vals = np.unique(col_values)
                        logger.info(f"  Feature {idx}: unique values = {unique_vals}")
                    
                    # Create custom scaling
                    # For zero-variance features: center at 0, no scaling
                    # For normal features: standard scaling
                    curvature_scaler = StandardScaler()
                    curvature_scaler.fit(train_curvatures)
                    
                    # Fix zero-variance features
                    curvature_scaler.mean_ = np.where(zero_var_mask, train_mean, curvature_scaler.mean_)
                    curvature_scaler.scale_ = np.where(zero_var_mask, 1.0, curvature_scaler.scale_)
                    curvature_scaler.var_ = curvature_scaler.scale_ ** 2
                    
                else:
                    # Normal scaling
                    curvature_scaler = StandardScaler()
                    curvature_scaler.fit(train_curvatures)
                
                # Transform all data
                original_features_scaled = original_scaler.transform(original_features)
                curvature_features_scaled = curvature_scaler.transform(all_curvature_features)
                
                # CRITICAL: Verify normalization worked
                if train_mask is not None:
                    train_curv_mean = curvature_features_scaled[train_mask].mean(axis=0)
                    train_curv_std = curvature_features_scaled[train_mask].std(axis=0)
                    
                    logger.info(f"Train curvature features after scaling:")
                    logger.info(f"  Overall mean: {train_curv_mean.mean():.6f} (should be ~0)")
                    logger.info(f"  Overall std: {train_curv_std.mean():.6f} (should be ~1)")
                    logger.info(f"  Mean range: [{train_curv_mean.min():.6f}, {train_curv_mean.max():.6f}]")
                    
                    # Check for problematic features
                    bad_mean_mask = np.abs(train_curv_mean) > 0.1
                    if np.any(bad_mean_mask):
                        logger.warning(f"Found {bad_mean_mask.sum()} features with |mean| > 0.1:")
                        for idx in np.where(bad_mean_mask)[0]:
                            logger.warning(f"  Feature {idx}: mean={train_curv_mean[idx]:.4f}, std={train_curv_std[idx]:.4f}")
                
                # Store scalers
                self.original_scaler = original_scaler
                self.curvature_scaler = curvature_scaler
                
            else:
                # Fallback: fit on all data
                logger.warning("No training mask - fitting on ALL data (may cause leakage)")
                
                original_scaler = StandardScaler()
                original_features_scaled = original_scaler.fit_transform(original_features)
                
                # Handle curvature features
                all_std = all_curvature_features.std(axis=0)
                all_mean = all_curvature_features.mean(axis=0)
                zero_var_mask = all_std == 0
                
                curvature_scaler = StandardScaler()
                curvature_scaler.fit(all_curvature_features)
                
                if np.any(zero_var_mask):
                    logger.warning(f"Found {zero_var_mask.sum()} zero-variance features")
                    curvature_scaler.mean_ = np.where(zero_var_mask, all_mean, curvature_scaler.mean_)
                    curvature_scaler.scale_ = np.where(zero_var_mask, 1.0, curvature_scaler.scale_)
                    curvature_scaler.var_ = curvature_scaler.scale_ ** 2
                
                curvature_features_scaled = curvature_scaler.transform(all_curvature_features)
                
                self.original_scaler = original_scaler
                self.curvature_scaler = curvature_scaler
            
            # Combine features
            enhanced_features = np.hstack([original_features_scaled, curvature_features_scaled])
            
            # Final verification
            logger.info(f"Enhanced features - mean: {enhanced_features.mean():.6f}, std: {enhanced_features.std():.6f}")
            
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