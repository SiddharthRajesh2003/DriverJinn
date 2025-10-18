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
                
                # Scaler for curvature features
                curvature_scaler = StandardScaler()
                curvature_scaler.fit(all_curvature_features[train_mask])
                
                # TRANSFORM all data (train, val, test)
                original_features_scaled = original_scaler.transform(original_features)
                curvature_features_scaled = curvature_scaler.transform(all_curvature_features)
                
                logger.info(f"Train samples used for fitting: {train_mask.sum()}")
                logger.info(f"Original features - Train mean: {original_features_scaled[train_mask].mean():.4f}")
                logger.info(f"Original features - Test mean: {original_features_scaled[test_mask].mean():.4f}" if test_mask is not None else "")
                
            else:
                # Fallback: fit on all data (old behavior)
                logger.warning("No training mask provided - fitting scalers on ALL data (may cause data leakage)")
                
                original_scaler = StandardScaler()
                original_features_scaled = original_scaler.fit_transform(original_features)
                
                curvature_scaler = StandardScaler()
                curvature_features_scaled = curvature_scaler.fit_transform(all_curvature_features)
            
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
        
        # Build node index mapping
        node_to_idx = {name: idx for idx, name in enumerate(self.node_names)}
        
        # Calculate node-level curvature statistics
        node_curvatures = {i: [] for i in range(num_nodes)}
        
        for (u, v), curv in ollivier_curv.items():
            if u in node_to_idx and v in node_to_idx:
                u_idx = node_to_idx[u]
                v_idx = node_to_idx[v]
                
                node_curvatures[u_idx].append(curv)
                node_curvatures[v_idx].append(curv)
                
                # Count positive/negative curvature edges
                if curv > 0:
                    pos_curve_deg[u_idx] += 1
                    pos_curve_deg[v_idx] += 1
                elif curv < 0:
                    neg_curve_deg[u_idx] += 1
                    neg_curve_deg[v_idx] += 1
        
        # Calculate curvature homophily
        # (similarity of node's curvature to its neighbors' curvatures)
        for (u, v), curv in ollivier_curv.items():
            if u in node_to_idx and v in node_to_idx:
                u_idx = node_to_idx[u]
                v_idx = node_to_idx[v]
                
                if len(node_curvatures[u_idx]) > 0 and len(node_curvatures[v_idx]) > 0:
                    u_mean = np.mean(node_curvatures[u_idx])
                    v_mean = np.mean(node_curvatures[v_idx])
                    
                    # Similarity measure (negative absolute difference, normalized)
                    similarity = 1.0 / (1.0 + abs(u_mean - v_mean))
                    
                    curve_homophily[u_idx] += similarity
                    curve_homophily[v_idx] += similarity
        
        # Normalize homophily by degree
        for i in range(num_nodes):
            degree = len(node_curvatures[i])
            if degree > 0:
                curve_homophily[i] /= degree
        
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
    
    def create_edge_feature_vector(self, src_features, dst_features, ollivier_curv, forman_curv):
        """Create comprehensive edge feature vector"""
        
        edge_features = np.concatenate([src_features, dst_features])
        
        diff_l1 = np.sum(np.abs(src_features - dst_features))
        diff_l2 = np.linalg.norm(src_features - dst_features)
        cosine_sim = 1 - cosine(src_features, dst_features)
        
        curvature_ratio = forman_curv / (abs(ollivier_curv) + 1e-8)
        
        additional_features = np.array([
            diff_l1, diff_l2, cosine_sim,
            ollivier_curv, forman_curv, curvature_ratio
        ])
        
        return np.concatenate([edge_features, additional_features])