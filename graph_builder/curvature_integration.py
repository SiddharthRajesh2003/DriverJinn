from curvature_calculator import EdgeCurvature
from collections import defaultdict
import numpy as np
import torch
import logging
from scipy.spatial.distance import cosine
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger("my_logger")
logger.setLevel(logging.INFO)  


class CurvatureFeatureIntegrator:
    def __init__(self, edge_curvature_calculator: EdgeCurvature, data_dict: dict):
        """
        Integrate edge curvature with existing node features
        
        Parameters:
        edge_curvature_calculator: EdgeCurvature instance with calculated curvatures
        feature_df: DataFrame with node features (columns you showed)
        node_names: List of node names corresponding to feature_df rows
        """
        
        self.edge_calc = edge_curvature_calculator
        self.data_dict = data_dict.copy()
        
        self.features = self.data_dict['feature']
        self.node_names = self.data_dict['node_name']
        self.edge_index = self.data_dict['edge_index']
        self.feature_names = self.data_dict['feature_name']
        self.num_nodes = len(self.node_names)
        
        self.name_to_idx = {name: i for i, name in enumerate(self.node_names)}
        
        if not hasattr(self.edge_calc, 'edge_curvature'):
            logger.info('Calculating edge curvature')
            self.edge_calc.calculate_edge_curvature('both')
            
    def calculate_node_curvature_features(self):
        """
        Calculate node-level curvature features from edge curvatures
        """
        
        node_ollivier_stats = np.zeros((self.num_nodes, 4))
        node_forman_stats = np.zeros((self.num_nodes, 4))
        
        # Aggregate edge curvatures per node
        node_ollivier_values = defaultdict(list)
        node_forman_values = defaultdict(list)
        
        for i in range(self.edge_index.shape[1]):
            src_idx = self.edge_index[0, i].item()
            dst_idx = self.edge_index[1, i].item()
            
            ollivier_curv = self.get_edge_curvature('OllivierRicci', src_idx, dst_idx)
            forman_curv = self.get_edge_curvature('FormanRicci', src_idx, dst_idx)
            
            node_ollivier_values[src_idx].append(ollivier_curv)
            node_ollivier_values[dst_idx].append(ollivier_curv)
            
            node_forman_values[src_idx].append(forman_curv)
            node_forman_values[dst_idx].append(forman_curv)
            
        for node_idx in range(self.num_nodes):
            if node_idx in node_ollivier_values:
                ollivier_vals = node_ollivier_values[node_idx]
                node_ollivier_stats[node_idx] = [
                    np.mean(ollivier_vals),
                    np.std(ollivier_vals) if len(ollivier_vals) > 1 else 0,
                    np.min(ollivier_vals),
                    np.max(ollivier_vals)
                ]
            
            
            if node_idx in node_forman_values:
                forman_vals = node_forman_values[node_idx]
                node_forman_stats[node_idx] = [
                    np.mean(forman_vals),
                    np.std(forman_vals) if len(forman_vals) > 1 else 0,
                    np.min(forman_vals),
                    np.max(forman_vals)
                ]
        
        return node_ollivier_stats, node_forman_stats
    
    def get_edge_curvature(self, curvature_type, src_idx, dst_idx):
        """Get Edge curvature values"""
        
        if not hasattr(self.edge_calc, 'edge_curvature'):
            return 0.0
        
        edge_tuple = (src_idx, dst_idx)
        reversed_tuple = (src_idx, dst_idx)
        
        if curvature_type in self.edge_calc.edge_curvature:
            return self.edge_calc.edge_curvature[curvature_type].get(
                edge_tuple,
                self.edge_calc.edge_curvature[curvature_type].get(reversed_tuple, 0.0)
            )
        return 0.0
    
    
    def calculate_curvature_based_features(self):
        """
        Calculate additional curvature-based features
        """
        
        positive_curvature_degree = np.zeros(self.num_nodes)
        negative_curvature_degree = np.zeros(self.num_nodes)
        curvature_homophily = np.zeros(self.num_nodes)
        
        for i in range(self.edge_index.shape[1]):
            src_idx = self.edge_index[0, i].item()
            dst_idx = self.edge_index[1, i].item()
            
            ollivier_curv = self.get_edge_curvature('OllivierRicci', src_idx, dst_idx)
            
            if ollivier_curv > 0:
                positive_curvature_degree[src_idx] += 1
                positive_curvature_degree[dst_idx] += 1
            elif ollivier_curv < 0:
                negative_curvature_degree[src_idx] += 1
                negative_curvature_degree[src_idx] += 1
                
            src_features = self.features[src_idx].numpy()
            dst_features = self.features[dst_idx].numpy()
            
            similarity = 1 - cosine(src_features, dst_features)
            curvature_weighted_sim = similarity * abs(ollivier_curv)
            
            
            curvature_homophily[src_idx] += curvature_weighted_sim
            curvature_homophily[dst_idx] += curvature_weighted_sim
            
        total_degree = positive_curvature_degree + negative_curvature_degree
        curvature_homophily = np.divide(
            curvature_homophily,
            total_degree,
            out = np.zeros_like(curvature_homophily),
            where = total_degree != 0
        )
        
        return positive_curvature_degree, negative_curvature_degree, curvature_homophily

    def create_enhanced_features(self, normalize = True):
        """
        Create enhanced feature matrix with curvature features using your EdgeCurvature class
        
        Returns:
        dict: Updated data_dict with enhanced features
        """
        
        logger.info("Using existing EdgeCurvature class to create node curvature features...")
        
        curvature_df = self.edge_calc.create_node_curvature_features(node_names = self.node_names)
        
        curvature_feature_list = []
        curvature_feature_names = [
            'ollivier_mean', 'ollivier_std', 'ollivier_min', 'ollivier_max', 'ollivier_median', 'ollivier_degree',
            'forman_mean', 'forman_std', 'forman_min', 'forman_max', 'forman_median', 'forman_degree'
        ]
        
        for name in self.node_names:
            if name in curvature_df.index:
                node_curvature_features = curvature_df.loc[name, curvature_feature_names].values
            else:
                node_curvature_features = np.zeros(len(curvature_feature_names))
            
            curvature_feature_list.append(node_curvature_features)
        
        curvature_features = np.array(curvature_feature_list)
        
        logger.info("Calculating additional curvature-based features...")
        
        pos_curve_deg, neg_curve_deg, curve_homophily = self.calculate_curvature_based_features()
        
        additional_features = np.column_stack([
            pos_curve_deg,
            neg_curve_deg,
            curve_homophily
        ])
        
        all_curvature_features = np.hstack([curvature_features, additional_features])
        
        if normalize:
            scaler = StandardScaler()
            all_curvature_features = scaler.fit_transform(all_curvature_features)
            
        original_features = self.features.numpy()
        enhanced_features =  np.hstack([original_features, all_curvature_features])
        
        additional_feature_names = ['positive_curvature_degree', 'negative_curvature_degree', 'curvature_homophily']
        complete_curvature_names = curvature_feature_names + additional_feature_names
        enhanced_feature_names = self.feature_names + complete_curvature_names
        
        enhanced_data_dict = self.data_dict.copy()
        enhanced_data_dict['feature'] = torch.tensor(enhanced_features, dtype = torch.float32)
        enhanced_data_dict['feature_name'] = enhanced_feature_names
        
        logger.info(f"Original features: {original_features.shape}")
        logger.info(f"Enhanced features: {enhanced_features.shape}")
        logger.info(f"Added {len(complete_curvature_names)} curvature-based features")
        
        return enhanced_data_dict
    
    def create_edge_features_dict(self):
        """
        Create edge features dictionary for edge-level predictions
        """
        
        num_edges = self.edge_index.shape[1]
        edge_features_list = []
        
        for i in range(num_edges):
            src_idx = self.edge_index[0, i].item()
            dst_idx = self.edge_index[1, i].item()
            
            ollivier_curv = self.get_edge_curvature('OllivierRicci', src_idx, dst_idx)
            forman_curv = self.get_edge_curvature('FormanRicci', src_idx, dst_idx)
            
            src_features = self.features[src_idx].numpy()
            dst_features = self.features[dst_idx].numpy()
            
            edge_feature_vector = self.create_edge_feature_vector(
                src_features, dst_features, ollivier_curv, forman_curv
            )
            
            edge_features_list.append(edge_feature_vector)
        
        edge_featues = np.array(edge_features_list)
        
        edge_feature_names = (
            [f'src_{name}' for name in self.feature_names] +
            [f'dst_{name}' for name in self.feature_names] +
            ['node_feature_diff_l1', 'node_feature_diff_l2', 'cosine_similarity'] +
            ['ollivier_curvature', 'forman_curvature', 'curvature_ratio']
        )
        
        edge_data_dict = {
            'edge_features': torch.tensor(edge_featues, dtype = torch.float32),
            'edge_feature_name': edge_feature_names,
            'edge_index': self.edge_index
        }
        
        return edge_data_dict
    
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
    
    
    def analyze_curvature_distribution(self):
        """Analyze the distribution of curvature values using your EdgeCurvature class"""
        
        print("\n=== Curvature Distribution Analysis ===")
        
        # Use your class's get_curvature_statistics method
        stats = self.edge_calc.get_curvature_statistics()
        
        for curvature_type, stat_dict in stats.items():
            print(f"\n{curvature_type}:")
            print(f"  Total edges: {stat_dict['count']}")
            print(f"  Mean: {stat_dict['mean']:.4f}")
            print(f"  Std: {stat_dict['std']:.4f}")
            print(f"  Min: {stat_dict['min']:.4f}")
            print(f"  Max: {stat_dict['max']:.4f}")
            print(f"  Median: {stat_dict['median']:.4f}")
            print(f"  Positive edges: {stat_dict['positive_edges']} ({stat_dict['positive_edges']/stat_dict['count']*100:.1f}%)")
            print(f"  Negative edges: {stat_dict['negative_edges']} ({stat_dict['negative_edges']/stat_dict['count']*100:.1f}%)")
            print(f"  Zero edges: {stat_dict['zero_edges']} ({stat_dict['zero_edges']/stat_dict['count']*100:.1f}%)")


def integrate_curvature_features(edge_curvature_calculator, data_dict):
    """
    Main function to integrate curvature features with existing data using your EdgeCurvature class
    
    Parameters:
    edge_curvature_calculator: Your EdgeCurvature instance
    data_dict: Your existing data dictionary
    
    Returns:
    enhanced_data_dict: Data dictionary with curvature features added
    """
    
    integrator = CurvatureFeatureIntegrator(edge_curvature_calculator, data_dict)
    
    # Analyze curvature distribution using your class methods
    integrator.analyze_curvature_distribution()
    
    # Create enhanced node features
    enhanced_data_dict = integrator.create_enhanced_features(normalize=True)
    
    return enhanced_data_dict