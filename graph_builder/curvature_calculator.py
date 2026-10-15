#!/usr/bin/env python

from GraphRicciCurvature.OllivierRicci import OllivierRicci
from GraphRicciCurvature.FormanRicci import FormanRicci
import networkx as nx
import numpy as np
import pandas as pd
from utils.logging_manager import get_logger

logger=get_logger(__name__)

class EdgeCurvature:
    """
    Main class that computes and stores Ricci curvature for all edges in a PPI graph
    """
    
    def __init__(self, G: nx.Graph, feature_df:pd.DataFrame = None):
        self.G = G                                                                                  # Store the input graph for use across all methods
        self.df = feature_df.copy()                                                                 # Copy the feature DataFrame to avoid mutating the caller's data
        self.edge_curvature = {}                                                                    # Initialize empty dict to hold computed curvatures keyed by method name
    
    def calculate_edge_curvature(self, method:str = 'both') -> dict:
        """
        Calculate edge curvature using specified method(s)
        
        Arguments:
            method: str, 'Ollivier', 'Forman', 'both'
        """
        method = method.lower()                                                                     # Normalize input to lowercase so comparisons are case-insensitive
        
        if not hasattr(self, 'edge_curvature'):
            self.edge_curvature = {}                                                                # Guard: Initialize the dict if it somehow doesn't exist (defensive check)
        
        if method =='Ollivier' or method == 'ollivier' or method =='both':                          # Run Ollivier-Ricci computation if requested
            orc = OllivierRicci(self.G, alpha = 0.5, verbose = 'TRACE', proc = 4, weight='weight')  # Build the ORC solver; alpha = 0.5 balances mass between source and neighbors, proc = 4 uses 4 parallel processes, weight = 'weight' uses edge weights
            orc.compute_ricci_curvature()                                                           # Execute the Ollivier-Ricci computation and annotate orc.G edges in-place
            edge_curvature = {}                                                                     # Temp dict to collect results before storing on self
            for edge in self.G.edges():
                edge_curvature[edge] = orc.G[edge[0]][edge[1]]['ricciCurvature']                    # For each edge, extract the 'ricciCurvature' value written by the library
            self.edge_curvature['OllivierRicci'] = edge_curvature                                   # Store the full edge->curvature mapping under the 'OllivierRicci' key
        
        
        if method == 'Forman' or method == 'forman' or method == 'both':                            # Run Forman-Ricci computation if requested
            frc = FormanRicci(self.G, verbose = 'TRACE')                                            # Build the FRC solver (Forman doesn't use mass transport, so no alpha)
            frc.compute_ricci_curvature()                                                           # Execute the Forman-Ricci computation and annotate frc.G edges in-place
            edge_curvature = {}                                                                     # Temp dict to collect results before storing on self
            
            for edge in self.G.edges():
                edge_curvature[edge] = frc.G[edge[0]][edge[1]]['formanCurvature']                   # For each edge, extract the 'formanCurvature' value written by the library
            self.edge_curvature['FormanRicci'] = edge_curvature                                     # Store the full edge->curvature mapping under the 'FormanRicci' key
        
        return self.edge_curvature                                                                  # Return the full dict of {method_name: {edge: curvature}} for caller inspection
    
    def add_curvature_to_features(self, method:str = 'both') -> pd.DataFrame:
        
        """
        Add curvature values to the feature DataFrame
        
        Parameters:
        method: str, 'ollivier', 'forman', or 'both'
        
        Returns:
        pd.DataFrame: Updated DataFrame with curvature features
        """
        
        if self.df is None:
            logger.info('No dataframe was provided. Creating edge list with curvatures.')           # Warn that we're building a DataFrame from scratch rather than extending one
            self.df = pd.DataFrame(list(self.G.edges()), columns=['source', 'target'])              # Create a minimal edge-list DataFrame with source/target node indices
            
            if hasattr(self, 'node_names') and self.node_names is not None:                         # If gene names were attached to the instance, map indices to names
                self.df['source_name'] = self.df['source'].map(
                    lambda x: self.node_names[x] if x < len(self.node_names) else f"Node_{x}"       # Add human-readable source gene name, fallback to "Node_<idx>" if out of range
                )
                self.df['target_name'] = self.df['target'].map(
                    lambda x: self.node_names[x] if x < len(self.node_names) else f"Node_{x}"       # Add human-readable target gene name, same fallback logic
                )
            
        if 'OllivierRicci' in self.edge_curvature:                                                  # Only append Ollivier column if that curvature was computed
            ollivier_results = []                                                                   # Accumulate per-row curvature values before assigning as a column
            
            for _, row in self.df.iterrows():
                edge = (row['source'], row['target'])                                               # Build the canonical (source, target) edge tuple for lookup
                reversed_edge = (row['target'], row['source'])                                      # Also try the reversed direction since the graph may store edges either way
                
                if edge in self.edge_curvature['OllivierRicci']:
                    ollivier_results.append(self.edge_curvature['OllivierRicci'][edge])             # Found the edge in its stored direction - use its curvature directly
                elif reversed_edge in self.edge_curvature['OllivierRicci']:
                    ollivier_results.append(self.edge_curvature['OllivierRicci'][reversed_edge])    # Edge was stored in reverse direction - curvature is symmetric so value is valid
                else:
                    ollivier_results.append(np.nan)                                                 # Edge not found in curvature dict - fill with NaN to preserve row alignment
            
            self.df['OllivierRicciCurvature'] = ollivier_results                                    # Attach the collected curvature values as a new column on the DataFrame
            
        
        if 'FormanRicci' in self.edge_curvature:                                                    # Only append Forman column if that curvature was computed
            forman_results = []                                                                     # Accumulate per-row curvature values before assigning as a column
            
            for _, row in self.df.iterrows():
                edge = (row['source'], row['target'])                                               # Same forward/reverse edge lookup pattern as the Ollivier block above
                reversed_edge = (row['target'], row['source'])
                
                if edge in self.edge_curvature['FormanRicci']:
                    forman_results.append(self.edge_curvature['FormanRicci'][edge])
                elif reversed_edge in self.edge_curvature['FormanRicci']:
                    forman_results.append(self.edge_curvature['FormanRicci'][reversed_edge])        
                else:
                    forman_results.append(np.nan)

            self.df['FormanRicciCurvature'] = forman_results                                        # Attach Forman curvature as a new column

        return self.df                                                                              # Return the updated DataFrame with new curvature column(s) appended
    
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
            if hasattr(self, 'node_names') and self.node_names is not None:
                node_names = self.node_names                                                        # Use gene names already attached to the instance if available
            else:
                # Get node names from graph - assuming nodes are indexed 0 to n-1
                node_names = [str(i) for i in range(self.G.number_of_nodes())]                      # Fallback to string indices when no gene names are provided
                logger.warning(f"No node names provided, using indices: {node_names[:10]}...")      # Warn that features will be keyed by index rather than gene symbol
        
        node_curvatures = {
            'ollivier': {i: [] for i in range(len(node_names))},
            'forman': {i: [] for i in range(len(node_names))}
        }                                                                                           # Per-node accumulator: Each node index maps to a list of its incident edge curvatures
        
        logger.info(f"Available curvature types: {list(self.edge_curvature.keys())}")               # Log which curvature methods were computed before aggregating
        
        if hasattr(self, 'edge_curvature'):                                                         # Proceed only if curvature data exists on the instance
            if 'OllivierRicci' in self.edge_curvature:
                logger.info(f'Processing Ollivier curvatures for {len(node_names)} nodes')
                ollivier_count = 0                                                                  # Tracks total node-curvature assignments made
                total_positive = 0                                                                  # Tracks edges with positive Ollivier curvature
                total_negative = 0                                                                  # Tracks edges with negative Ollivier curvature
                
                # Debug info
                logger.info(f"Number of Ollivier curvatures: {len(self.edge_curvature['OllivierRicci'])}")  # Log total number of edge curvatures before iterating
                
                for (src_idx, dst_idx), curvature in self.edge_curvature['OllivierRicci'].items():  # Iterate over each edge and assign its curvature to both endpoint nodes
                    try:
                        if src_idx < len(node_names):
                            node_curvatures['ollivier'][src_idx].append(float(curvature))           # Add this edge's curvature to the source node's list
                            ollivier_count += 1
                            if float(curvature) > 0:
                                total_positive += 1
                            elif float(curvature) < 0:
                                total_negative += 1                                                 # Tally sign of curvature for the source node assignment
                            
                        if dst_idx < len(node_names):
                            node_curvatures['ollivier'][dst_idx].append(float(curvature))
                            ollivier_count += 1
                            if float(curvature) > 0:
                                total_positive += 1
                            elif float(curvature) < 0:
                                total_negative += 1                                                 # Tally sign of curvature for the destination node assignment
                                
                    except Exception as e:
                        logger.error(f"Error processing curvature {curvature} for edge ({src_idx}, {dst_idx}): {e}")    # Skip malformed curvature values rather than crashing the whole run
                
                logger.info(f'Processed {ollivier_count} Ollivier Curvature assignments to nodes')
                logger.info(f'Found {total_positive} positive and {total_negative} negative curvatures')
            
            if 'FormanRicci' in self.edge_curvature:
                logger.info(f'Processing Forman curvatures for {len(node_names)} nodes')
                forman_count = 0                                                                    # Track total assignments for post-processing log
                for (src_idx, dst_idx), curvature in self.edge_curvature['FormanRicci'].items():    # Assign Forman curvature to both endpoints of each edge
                    
                    if src_idx < len(node_names):
                        node_curvatures['forman'][src_idx].append(curvature)
                        forman_count += 1
                            
                    if dst_idx < len(node_names):
                        node_curvatures['forman'][dst_idx].append(curvature)
                        forman_count += 1
                
                logger.info(f"Processed {forman_count} Forman curvature assignments to nodes")
            
        
        curvature_data = []                                                                         # Will hold one stats-dict per node before converting to DataFrame
        
        for i, node in enumerate(node_names):                                                       # Iterate over all genes in order to build one row per gene
            node_stats = {'gene': node}                                                             # Start each row with the gene name as identifier
            
            # Ollivier statistics
            ollivier_values = node_curvatures['ollivier'][i]                                        # Retrieve list of all Ollivier curvatures from edges incident to this node
            if ollivier_values:
                node_stats.update({
                    'ollivier_mean': np.mean(ollivier_values),                                      # Average curvature across incident edges
                    'ollivier_std': np.std(ollivier_values),                                        # Spread of curvature values
                    'ollivier_min': np.min(ollivier_values),                                        # Most negatively curved incident edge
                    'ollivier_max': np.max(ollivier_values),                                        # Most positively curved incident edge
                    'ollivier_median': np.median(ollivier_values),                                  # Robust central tendency
                    'ollivier_degree': len(ollivier_values)                                         # Number of incident edges with curvature (edge count, not graph degree)
                })
            else:
                node_stats.update({
                    'ollivier_mean': 0.0,
                    'ollivier_std': 0.0,
                    'ollivier_min': 0.0,
                    'ollivier_max': 0.0,
                    'ollivier_median': 0.0,
                    'ollivier_degree': 0.0
                })                                                                                  # Zero-fill isolated nodes or nodes with no computed curvature
            
            # Forman statistics
            forman_values = node_curvatures['forman'][i]                                            # Retrieve all Forman curvatures from edges incident to this node
            if forman_values:
                forman_values = [float(v) for v in forman_values]                                   # Ensure all values are float
                pos_curv = sum(1 for v in forman_values if v > 0)                                   # Count edges with positive Forman curvature (locally dense/cycle-rich topology)
                neg_curv = sum(1 for v in forman_values if v < 0)                                   # Count edges with negative Forman curvature (locally tree-like topology)
                
                # Calculate curvature homophily (similarity of curvature with neighbors)
                neighbor_indices = list(self.G.neighbors(i))
                if neighbor_indices:
                    neighbor_curvatures = []
                    for n_idx in neighbor_indices:
                        if n_idx in node_curvatures['forman'] and node_curvatures['forman'][n_idx]:
                            n_curv = np.mean([float(v) for v in node_curvatures['forman'][n_idx]])
                            neighbor_curvatures.append(n_curv)                                      ## Compute the mean Forman curvature for each neighbour node
                    
                    if neighbor_curvatures:
                        node_curv = np.mean(forman_values)                                          # This node's own mean Forman curvature
                        curv_diff = np.mean([abs(node_curv - n_curv) for n_curv in neighbor_curvatures])    # Average absolute curvature difference between this node and its neighbours
                        homophily = 1.0 / (1.0 + curv_diff)  # Convert difference to [0, 1] similarity score; 1 = perfectly homophilic curvature neighbourhood
                    else:
                        homophily = 0.0                                                             # No neighbours with computed curvature - can't measure homophily
                else:
                    homophily = 0.0                                                                 # Isolate node - no neighbours at all
                
                node_stats.update({
                    'forman_mean': np.mean(forman_values),
                    'forman_std': np.std(forman_values),
                    'forman_min': np.min(forman_values),
                    'forman_max': np.max(forman_values),
                    'forman_median': np.median(forman_values),
                    'forman_degree': len(forman_values),                                            # Incident edge count with curvature
                    'positive_curvature_degree': pos_curv,                                          # Edges with positive Forman curvature
                    'negative_curvature_degree': neg_curv,                                          # Edges with negative Forman curvature
                    'curvature_homophily': homophily                                                # Similar to neighbour's curvature profile
                })
            else:
                node_stats.update({
                    'forman_mean': 0.0,
                    'forman_std': 0.0,
                    'forman_min': 0.0,
                    'forman_max': 0.0,
                    'forman_median': 0.0,
                    'forman_degree': 0.0
                })                                                                                  # Zero-fill nodes with no Forman curvature data
            
            curvature_data.append(node_stats)                                                       # Append this gene's feature dict to the accumulator list
        
        curvature_df = pd.DataFrame(curvature_data)                                                 # Convert list of per-node dicts into a dataframe (one row per gene)
        curvature_df.set_index('gene', inplace=True)                                                # Use gene name as the row index for easy alignment with other feature tables
        
        logger.info("Node curvature feature summary:")
        for col in ['ollivier_mean', 'ollivier_degree', 'forman_mean', 'forman_degree']:
            if col in curvature_df.columns:
                non_zero = (curvature_df[col] != 0).sum()
                logger.info(f"  {col}: {non_zero} non-zero values out of {len(curvature_df)}")      # Log how many genes have non-zero curvature per key column to spot data issues
        
        return curvature_df                                                                         # Return the final node-feature DataFrame indexed by gene name
        
    def get_curvature_statistics(self):
        """Get statistics for calculated curvatures"""
        stats = {}                                                                                  # Output dict keyed by method name
        
        for curvature_type, edge_curvatures in self.edge_curvature.items():
            values = list(edge_curvatures.values())                                                 # Flattent the edge->curvature mapping into a plain list for numpy operations
            stats[curvature_type] = {
                'count': len(values),                                                               # Total number of edges
                'mean': np.mean(values),                                                            # Average curvature across all edges
                'std': np.std(values),                                                              # Standard deviation of curvature
                'min': np.min(values),                                                              # Most negatively curved edge
                'max': np.max(values),                                                              # Most positively curved edge
                'median': np.median(values),                                                        # Robust central tendency
                'positive_edges': sum(1 for v in values if v > 0),                                  # Edges with positive curvature (cycle-rich)
                'negative_edges': sum(1 for v in values if v < 0),                                  # Edges with negative curvature (tree-like)
                'zero_edges': sum(1 for v in values if v == 0)                                      # Flat edges
            }
        
        return stats                                                                                # Return the statistics dict for logging or downstream analysis