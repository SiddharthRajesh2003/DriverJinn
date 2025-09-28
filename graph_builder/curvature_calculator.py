from GraphRicciCurvature import OllivierRicci, FormanRicci
import networkx as nx
import torch
import numpy as np
import pandas as pd
from utils.logging_manager import get_logger

logger=get_logger(__name__)

class EdgeCurvature:
    def __init__(self, G: nx.Graph, feature_df:pd.DataFrame = None):
        self.G = G
        self.df = feature_df.copy()
        self.edge_curvature = {}
    
    def calculate_edge_curvature(self, method:str = 'both') -> dict:
        """
        Calculate edge curvature using specified method(s)
        
        Arguments:
            method: str, 'Ollivier', 'Forman', 'both'
        """
        method = method.lower()
        
        if not hasattr(self, 'edge_curvature'):
            self.edge_curvature = {}
        
        if method =='Ollivier' or method == 'ollivier' or method =='both':
            orc = OllivierRicci(self.G, alpha = 0.5, verbose = 'INFO')
            orc.compute_ricci_curvature()
            edge_curvature = {}
            for edge in self.G.edges():
                edge_curvature[edge] = orc.G[edge[0]][edge[1]]['ricciCurvature']
            self.edge_curvature['OllivierRicci'] = edge_curvature 
        
        
        if method == 'Forman' or method == 'forman' or method == 'both':
            frc = FormanRicci(self.G, verbose = 'INFO')
            frc.compute_ricci_curvature()
            edge_curvature = {}
            
            for edge in self.G.edges():
                edge_curvature[edge] = frc.G[edge[0]][edge[1]]['formanCurvature']
            self.edge_curvature['FormanRicci'] = edge_curvature
        
        return self.edge_curvature 
    
    def add_curvature_to_features(self, method:str = 'both') -> pd.DataFrame:
        
        """
        Add curvature values to the feature DataFrame
        
        Parameters:
        method: str, 'ollivier', 'forman', or 'both'
        
        Returns:
        pd.DataFrame: Updated DataFrame with curvature features
        """
        
        if self.df is None:
            logger.info('No dataframe was provided. Creating edge list with curvatures.')
            self.df = pd.DataFrame(list(self.G.edges()), columns=['source', 'target'])
            
            if hasattr(self, 'node_names') and self.node_names is not None:
                self.df['source_name'] = self.df['source'].map(
                    lambda x: self.node_names[x] if x < len(self.node_names) else f"Node_{x}"
                )
                self.df['target_name'] = self.df['target'].map(
                    lambda x: self.node_names[x] if x < len(self.node_names) else f"Node_{x}"
                )
            
        if 'OllivierRicci' in self.edge_curvature:
            ollivier_results = []
            
            for _, row in self.df.iterrows():
                edge = (row['source'], row['target'])
                reversed_edge = (row['target'], row['source'])
                
                if edge in self.edge_curvature['OlivierRicci']:
                    ollivier_results.append(self.edge_curvature['OllivierRicci'][edge])
                elif reversed_edge in self.edge_curvature['OlivierRicci']:
                    ollivier_results.append(self.edge_curvature['OllivierRicci'][reversed_edge])
                else:
                    ollivier_results.append(np.nan)
            
            self.df['OllivierRicciCurvature'] = ollivier_results
            
        
        if 'FormanRicci' in self.edge_curvature:
            forman_results = []
            
            for _, row in self.df.iterrows():
                edge = (row['source'], row['target'])
                reversed_edge = (row['target'], row['source'])
                
                if edge in self.edge_curvature['FormanRicci']:
                    forman_results.append(self.edge_curvature['FormanRicci'][edge])
                elif reversed_edge in self.edge_curvature['FormanRicci']:
                    forman_results.append(self.edge_curvature['FormanRicci'][reversed_edge])
                else:
                    forman_results.append(np.nan)

            self.df['FormanRicciCurvature'] = forman_results

        return self.df
    
    def create_node_curvature_features(self, node_names:list = None) -> pd.DataFrame:
        """
        Create node-level curvature features from edge curvatures
        This aggregates edge curvatures to create node features for your GGNet feature DataFrame
        
        Parameters:
        node_names: List of node names (genes) - if None, uses self.node_names
        
        Returns:
        pd.DataFrame: Node features with curvature statistics
        """
        
        if node_names is None:
            if node_names is not None:
                self.node_names = node_names
            else:
                node_names = [data['name'] for _, data in G.nodes(data=True)]
                print(node_names[:10])
        
        node_curvatures = {
            'ollivier': {i: [] for i in range(len(node_names))},
            'forman': {i: [] for i in range(len(node_names))}
        }
        
        if hasattr(self, 'edge_curvature'):
            if 'OlivierRicci' in self.edge_curvature:
                for (src_idx, dst_idx), curvature in self.edge_curvature['OllivierRicci'].items():
                    if src_idx < len(node_names):
                        node_curvatures['ollivier'][src_idx].append(curvature)
                    if dst_idx < len(node_names):
                        node_curvatures['ollivier'][dst_idx].append(curvature)
                
            if 'FormanRicci' in self.edge_curvature:
                for (src_idx, dst_idx), curvature in self.edge_curvature['FormanRicci'].items():
                    if src_idx < len(node_names):
                        node_curvatures['forman'][src_idx].append(curvature)
                    if dst_idx < len(node_names):
                        node_curvatures['forman'][dst_idx].append(curvature)
                        

        curvature_data = []
        
        for i, node in enumerate(node_names):
            node_stats = {'gene': node}
            
            ollivier_values = node_curvatures['ollivier'][i]
            if ollivier_values:
                node_stats.update({
                    'ollivier_mean': np.mean(ollivier_values),
                    'ollivier_std': np.std(ollivier_values),
                    'ollivier_min': np.min(ollivier_values),
                    'ollivier_max': np.max(ollivier_values),
                    'ollivier_median': np.median(ollivier_values),
                    'ollivier_degree': len(ollivier_values)
                })
            else:
                node_stats.update({
                    'ollivier_mean': 0.0,
                    'ollivier_std': 0.0,
                    'ollivier_min': 0.0,
                    'ollivier_max': 0.0,
                    'ollivier_median': 0.0,
                    'ollivier_degree': 0.0
                })
            
            forman_values = node_curvatures['forman'][i]
            if forman_values:
                node_stats.update({
                    'forman_mean': np.mean(forman_values),
                    'forman_std': np.std(forman_values),
                    'forman_min': np.min(forman_values),
                    'forman_max': np.max(forman_values),
                    'forman_median': np.median(forman_values),
                    'forman_degree': len(forman_values)
                })
            else:
                node_stats.update({
                    'forman_mean': 0.0,
                    'forman_std': 0.0,
                    'forman_min': 0.0,
                    'forman_max': 0.0,
                    'forman_median': 0.0,
                    'forman_degree': 0.0
                })
        
            curvature_data.append(node_stats)
        
        curvature_df = pd.DataFrame(curvature_data)
        curvature_df.set_index('gene', inplace = True)
        
        return curvature_df
    
    def get_curvature_statistics(self):
        """Get statistics for calculated curvatures"""
        stats = {}
        
        for curvature_type, edge_curvatures in self.edge_curvature.items():
            values = list(edge_curvatures.values())
            stats[curvature_type] = {
                'count': len(values),
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'median': np.median(values),
                'positive_edges': sum(1 for v in values if v > 0),
                'negative_edges': sum(1 for v in values if v < 0),
                'zero_edges': sum(1 for v in values if v == 0)
            }
        
        return stats