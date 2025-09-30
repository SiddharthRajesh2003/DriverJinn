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
from utils.logging_manager import get_logger

import pickle
import torch
import pandas as pd
import numpy as np
import os
import argparse

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
        """
        try:
            logger.info("Integrating curvature features with existing features...")
            
            if self.edge_curvature is None:
                raise ValueError("Must calculate curvatures first using calculate_curvatures()")
            
            integrator = CurvatureFeatureIntegrator(self.edge_curvature, self.data_dict)
            integrator.analyze_curvature_distribution()
            self.enhanced_data_dict = integrator.create_enhanced_features(normalize=normalize)
            self.curvature_dict = integrator.create_edge_features_dict()
            
            self.enhanced_data_dict['ollivier_curvature'] = self.curvature_dict['edge_ollivier_curvature']
            self.enhanced_data_dict['forman_curvature'] = self.curvature_dict['edge_forman_curvature']
            self.enhanced_data_dict['edge_features'] = self.curvature_dict['edge_features']
            self.enhanced_data_dict['edge_feature_names'] = self.curvature_dict['edge_feature_names']
            
            print("Feature integration completed!")
            return self.enhanced_data_dict, self.curvature_dict
        
        except Exception as e:
            logger.error(f'Error occurred during integration of edge curvature into feautures: {e}')
    
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
        
    def run_pipeline(self, output_dir = './output', method = 'both', normalize = True):
        """
        Run the complete curvature integration pipeline
        
        Parameters:
        output_dir: str, directory to save outputs
        method: str, curvature calculation method
        normalize: bool, whether to normalize features
        
        Returns:
        dict: Enhanced data dictionary with curvature features
        """
        try:
            logger.info("=== Starting Complete Curvature Integration Pipeline ===")
            
            self.load_data()
            
            self.build_network()
            
            self.calculate_curvatures(method = method)
            
            self.integrate_features(normalize=normalize)
            
            os.makedirs(output_dir, exist_ok=True)
            
            dataset_name = os.path.basename(self.dataset_file).replace('.pkl', '_enhanced.pkl')
            enhanced_dataset_path = os.path.join(output_dir, dataset_name)
            self.save_enhanced_dataset(enhanced_dataset_path)
            
            self.save_network(output_dir)
            
            logger.info("=== Pipeline Completed Successfully ===")
            logger.info(f"Original features: {self.data_dict['feature'].shape}")
            logger.info(f"Enhanced features: {self.enhanced_data_dict['feature'].shape}")
            logger.info(f"Outputs saved in: {output_dir}")
            
            return self.enhanced_data_dict

        except Exception as e:
            logger.error(f'Error occurred while processing the dataset: {e}')
            exit(1)
            
def main():
    """
    Example usage of the complete pipeline
    """
    parser = argparse.ArgumentParser(description = 'Integrate curvature features with gene network data')
    
    parser.add_argument('dataset_file', type=str, help='Input dataset pickle file')
    parser.add_argument('--output_dir', type=str, default='./curvature_output', 
                       help='Output directory for enhanced dataset and graph')
    parser.add_argument('--method', type=str, choices=['ollivier', 'forman', 'both'], 
                       default='both', help='Curvature calculation method')
    parser.add_argument('--no_normalize', action='store_true', 
                       help='Skip normalization of curvature features')
    
    args = parser.parse_args()
    
    try:
        pipeline = CurvaturePipeline(args.dataset_file)
        
        enhanced_data_dict = pipeline.run_pipeline(
            output_dir = args.output_dir,
            method = args.method,
            normalize = not args.no_normalize
        )
        print("\n=== Final Statistics ===")
        print(f"Enhanced feature names: {len(enhanced_data_dict['feature_name'])}")
        print("New curvature features added:")
        original_features = len(pipeline.data_dict['feature_name'])
        for i, name in enumerate(enhanced_data_dict['feature_name'][original_features:], 1):
            print(f"  {i}. {name}")
        
    except Exception as e:
        logger.info(f"Error in pipeline: {e}")
        exit(1)

if __name__ == '__main__':
    logger=get_logger(__name__)
    main()