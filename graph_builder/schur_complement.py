#!/usr/bin/env python

import torch
import networkx as nx
import numpy as np
from typing import List, Dict, Optional, Tuple
import pandas as pd
from utils.logging_manager import get_logger

logger = get_logger(__name__)

class SchurComplementAugmentation:
    """
    Random Schur Complement-based Graph Augmentation
    
    Implements Algorithm 1 from the paper: generates augmented views of gene networks
    using Schur complement and Gaussian elimination with clique approximation.
    
    This augmentation strategy preserves essential network structures while
    introducing controlled randomness for contrastive learning.
    
    Args:
        elimination_ratio: Ratio of nodes to eliminate (ρ)
        neighbor_sort_method: Method to sort neighbors ('degree', 'weight', 'random', 'asc', 'desc')
        preserve_features: Whether to update node features during augmentation
        random_seed: Random seed for reproducibility
        elimination_strategy: 'priority' (degree-based), 'random', or 'coarsening'
    """
    
    def __init__(
        self,
        elimination_ratio: float = 0.2,
        neighbor_sort_method: str = 'weight',
        preserve_features: bool = True,
        random_seed: Optional[int] = None,
        elimination_strategy: str = 'priority'
    ): 
        self.elimination_ratio = elimination_ratio
        self.neighbor_sort_method = neighbor_sort_method.lower()
        self.preserve_features = preserve_features
        self.random_seed = random_seed
        self.elimination_strategy = elimination_strategy.lower()
        
        logger.info(f'Initialized SchurComplementAugmentation with elimination_ratio={elimination_ratio}, strategy={elimination_strategy}')
        
    def augment(self, 
                G: nx.Graph, 
                node_features: Optional[torch.Tensor] = None, 
                edge_weights: Optional[Dict] = None,
                node_names: Optional[List] = None,
                labels: Optional[torch.Tensor] = None
            ):
        """
    Generate augmented view of the graph using Schur complement method
    
    Implements three elimination strategies from C++ implementation:
    1. 'priority': Eliminate nodes by degree (default)
    2. 'random': Random elimination order
    3. 'coarsening': Collapse nodes onto neighbors
    
    Args:
        G: NetworkX graph
        node_features: Node feature matrix [num_nodes, feature_dim]
        edge_weights: Dictionary of edge weights {(i,j): weight}
    
    Returns:
        augmented_graph: New graph with modified structure
        augmented_features: Updated node features (if provided)
        metadata: Dictionary with augmentation statistics
    """
    
        if self.elimination_strategy == 'priority':
            return self.augment_priority(G, node_features, edge_weights, node_names, labels)
        
        elif self.elimination_strategy == 'random':
            return self.augment_random(G, node_features, edge_weights, node_names, labels)
        
        elif self.elimination_strategy == 'coarsening':
            return self.augment_coarsening(G, node_features, edge_weights, node_names, labels)
        else:
            raise ValueError(f'Unknown elimination strategy: {self.elimination_strategy}')
        
    def augment_priority(
        self,
        G: nx.Graph,
        node_features: Optional[torch.Tensor] = None,
        edge_weights: Optional[Dict] = None,
        node_names: Optional[List] = None,  
        labels: Optional[torch.Tensor] = None 
    ) -> Tuple[nx.Graph, Optional[torch.Tensor], Optional[torch.Tensor], Dict]:
        """
        Priority-based elimination (PriorityPreconditioner from C++)
        Eliminates nodes in order of degree (lowest first)
        """
        
        G_aug = G.copy()
        num_nodes = G.number_of_nodes()
        elimination_count = int(self.elimination_ratio * num_nodes)
        
        # Convert to numpy if needed
        if node_features is not None:
            node_features = node_features.numpy() if isinstance(node_features, torch.Tensor) else node_features
        if labels is not None:
            labels = labels.numpy() if isinstance(labels, torch.Tensor) else labels
        if node_names is not None:
            node_names = node_names if isinstance(node_names, np.ndarray) else np.array(node_names)
        
        logger.info(f"Priority augmentation: {num_nodes} nodes, eliminating {elimination_count} nodes")
        
        node_id_to_name_original = self.create_node_name_mapping(G, node_names, self.elimination_strategy)
        
        if edge_weights is None:
            edge_weights = {(u, v): G_aug[u][v].get('weight', 1.0) for u, v in G_aug.edges()}
            
        degree_pq = self.create_degree_priority_queue(G_aug)
        
        eliminated_nodes = []
        added_edges = []
        clique_sizes = []
        
        for i in range(elimination_count):
            if len(degree_pq) == 0:
                logger.warning(f"Priority queue empty at iteration {i}")
                break
            
            v_i = self.pop_min_degree(degree_pq)
            eliminated_nodes.append(v_i)
            
            neighbours = list(G_aug.neighbors(v_i))
            if len(neighbours) == 0:
                continue
            
            sorted_neighbours = self.sort_neighbours_compressed(
                G_aug, v_i, neighbours, edge_weights
            )
            
            clique_edges, clique_weights = self.build_probabilistic_clique(
                G_aug, v_i, sorted_neighbours, edge_weights, degree_pq
            )
            
            clique_sizes.append(len(clique_edges))
            
            for neighbour in neighbours:
                if neighbour in degree_pq:
                    self.decrement_degree(degree_pq, neighbour)
                    
            
            G_aug.remove_node(v_i)
            
            for (u, v), weight in zip(clique_edges, clique_weights):
                # CRITICAL: Verify both nodes exist before adding edge
                if not G_aug.has_node(u) or not G_aug.has_node(v):
                    logger.warning(f"Skipping edge ({u}, {v}) - node doesn't exist in graph")
                    continue
                
                if G_aug.has_edge(u, v):
                    G_aug[u][v]['weight'] = G_aug[u][v].get('weight', 0) + weight
                    
                else:
                    G_aug.add_edge(u, v, weight = weight)
                    added_edges.append((u, v))
                    if u in degree_pq:
                        self.increment_degree(degree_pq, u)
                    
                    if v in degree_pq:
                        self.increment_degree(degree_pq, v)
                        
        for u, v in G_aug.edges():
            if 'weight' not in G_aug[u][v]:
                G_aug[u][v]['weight'] = 1.0
        
        # CRITICAL: Force cleanup any remaining eliminated nodes
        G_aug, cleaned_count = self.force_cleanup_eliminated_nodes(G_aug, eliminated_nodes, None)
        if cleaned_count > 0:
            logger.warning(f"Priority: Force cleaned {cleaned_count} nodes that weren't properly eliminated")
        
        # Verify Node Name mapping after augmentation
        node_id_to_name_aug = self.verify_node_name_mapping(
            node_id_to_name_original, G_aug, eliminated_nodes, self.elimination_strategy
        )
        
        # Get final node mapping (use graph as source of truth)
        nodes = list(G.nodes())
        final_node_list, node_mapping = self.get_final_node_mapping(
            G, G_aug, nodes, eliminated_nodes, self.elimination_strategy
        )
        
        # CRITICAL: Restore edges for isolated nodes from original graph
        G_aug, edges_restored = self.restore_edges_for_isolated_nodes(
            G, G_aug, final_node_list, self.elimination_strategy
        )
        
        augmented_features = None
        augmented_labels = None
        augmented_node_names = None
        
        if node_features is not None and node_names is not None:
            if labels is None:
                logger.warning("Labels not provided - features will be aligned but labels won't match")
                labels = np.zeros(len(node_names), dtype = np.int64)
                
            augmented_features, augmented_labels, augmented_node_names = self.align_features_labels_with_names(
                features=node_features, labels=labels, 
                node_names = node_names, node_id_to_name_aug=node_id_to_name_aug, final_node_list=final_node_list, 
                strategy=self.elimination_strategy
            )
            
            augmented_features = torch.from_numpy(augmented_features)
            augmented_labels = torch.from_numpy(augmented_labels)
        
        
        eliminated_node_id_to_name = {
            node_id: node_id_to_name_original[node_id]
            for node_id in eliminated_nodes
            if node_id in node_id_to_name_original
        }
        
        metadata = {
            'original_nodes': num_nodes,
            'augmented_nodes': G_aug.number_of_nodes(),
            'augmented_node_ids': list(G_aug.nodes()),
            'final_node_list': final_node_list,
            'node_mapping': node_mapping,
            'eliminated_nodes': len(eliminated_nodes),
            'original_edges': G.number_of_edges(),
            'added_edges': len(added_edges),
            'avg_clique_size': np.mean(clique_sizes) if clique_sizes else 0,
            'eliminated_node_ids': eliminated_nodes,
            'eliminated_node_id_to_name': eliminated_node_id_to_name,
            'strategy': 'priority',
            'node_id_to_name': node_id_to_name_aug,
            'augmented_node_names': augmented_node_names,
            'augmented_labels': augmented_labels
        }
        
        self.verify_no_synthetic_nodes(G, G_aug, eliminated_nodes,self.elimination_strategy)
        
        logger.info(f"Priority augmentation complete: {metadata['original_nodes']} → "
                    f"{metadata['augmented_nodes']} nodes")
        
        return G_aug, augmented_features, augmented_labels, metadata
    
    def augment_random(
        self, 
        G: nx.Graph,
        node_features: Optional[torch.Tensor],
        edge_weights: Optional[Dict],
        node_names: Optional[List] = None,  # ← ADD
        labels: Optional[torch.Tensor] = None  # ← ADD
    ) -> Tuple[nx.Graph, Optional[torch.Tensor], Optional[torch.Tensor], Dict]:
        """
        Random elimination (RandomPreconditioner from C++)
        Eliminates nodes in random order
        """
        
        G_aug = G.copy()
        num_nodes = G_aug.number_of_nodes()
        elimination_count = int(self.elimination_ratio * num_nodes)
        # Convert to numpy if needed
        if node_features is not None:
            node_features = node_features.numpy() if isinstance(node_features, torch.Tensor) else node_features
        if labels is not None:
            labels = labels.numpy() if isinstance(labels, torch.Tensor) else labels
        if node_names is not None:
            node_names = node_names if isinstance(node_names, np.ndarray) else np.array(node_names)
        
        logger.info(f"Random augmentation: {num_nodes} nodes, eliminating {elimination_count} nodes")
        
        node_id_to_name_original = self.create_node_name_mapping(G, node_names, self.elimination_strategy)
        
        if edge_weights is None:
            edge_weights = {(u, v): G_aug[u][v].get('weight', 1.0) for u, v in G_aug.edges()}
        
        nodes = list(G_aug.nodes())
        np.random.shuffle(nodes)
        
        eliminated_nodes = []
        added_edges = []
        clique_sizes = []
        
        for i in range(elimination_count):
            if i >= len(nodes):
                break
            
            v_i = nodes[i]
            eliminated_nodes.append(v_i)
            
            neighbours = list(G_aug.neighbors(v_i))
            if len(neighbours) == 0:
                continue
            
            sorted_neighbours = self.sort_neighbours_compressed(
                G_aug, v_i, neighbours, edge_weights
            )
            
            clique_edges, clique_weights = self.build_probabilistic_clique(
                G_aug, v_i, sorted_neighbours, edge_weights, None
            )
            
            clique_sizes.append(len(clique_edges))
            G_aug.remove_node(v_i)
            
            for (u, v), weight in zip(clique_edges, clique_weights):
                # CRITICAL: Verify both nodes exist before adding edge
                if not G_aug.has_node(u) or not G_aug.has_node(v):
                    logger.warning(f"Skipping edge ({u}, {v}) - node doesn't exist in graph")
                    continue
                
                if G_aug.has_edge(u, v):
                    G_aug[u][v]['weight'] = G_aug[u][v].get('weight', 0) + weight
                else:
                    G_aug.add_edge(u, v, weight = weight)
                    added_edges.append((u, v))
                    
        for u, v in G_aug.edges():
            if 'weight' not in G_aug[u][v]:
                G_aug[u][v]['weight'] = 1.0
                
        # CRITICAL: Force cleanup any remaining eliminated nodes
        G_aug, cleaned_count = self.force_cleanup_eliminated_nodes(G_aug, eliminated_nodes, None)
        if cleaned_count > 0:
            logger.warning(f"Random: Force cleaned {cleaned_count} nodes that weren't properly eliminated")
        
        # Verify Node Name mapping after augmentation
        node_id_to_name_aug = self.verify_node_name_mapping(
            node_id_to_name_original, G_aug, eliminated_nodes, self.elimination_strategy
        )
        
        # Get final node mapping (use graph as source of truth)
        nodes = list(G.nodes())
        final_node_list, node_mapping = self.get_final_node_mapping(
            G, G_aug, nodes, eliminated_nodes, self.elimination_strategy
        )
        
        # CRITICAL: Restore edges for isolated nodes from original graph
        G_aug, edges_restored = self.restore_edges_for_isolated_nodes(
            G, G_aug, final_node_list, self.elimination_strategy
        )
        
        augmented_features = None
        augmented_labels = None
        augmented_node_names = None
        
        if node_features is not None and node_names is not None:
            if labels is None:
                logger.warning("Labels not provided - features will be aligned but labels won't match")
                labels = np.zeros(len(node_names), dtype = np.int64)
                
            augmented_features, augmented_labels, augmented_node_names = self.align_features_labels_with_names(
                features=node_features, labels=labels, 
                node_names = node_names, node_id_to_name_aug=node_id_to_name_aug, final_node_list=final_node_list, 
                strategy=self.elimination_strategy
            )
            
            augmented_features = torch.from_numpy(augmented_features)
            augmented_labels = torch.from_numpy(augmented_labels)
        
        eliminated_node_id_to_name = {
            node_id: node_id_to_name_original[node_id]
            for node_id in eliminated_nodes
            if node_id in node_id_to_name_original
        }
        
        metadata = {
            'original_nodes': num_nodes,
            'augmented_nodes': G_aug.number_of_nodes(),
            'augmented_node_ids': list(G_aug.nodes()),
            'final_node_list': final_node_list,
            'node_mapping': node_mapping,
            'eliminated_nodes': len(eliminated_nodes),
            'original_edges': G.number_of_edges(),
            'added_edges': len(added_edges),
            'avg_clique_size': np.mean(clique_sizes) if clique_sizes else 0,
            'eliminated_node_ids': eliminated_nodes,
            'eliminated_node_id_to_name': eliminated_node_id_to_name,
            'strategy': 'random',
            'node_id_to_name': node_id_to_name_aug,
            'augmented_node_names': augmented_node_names,
            'augmented_labels': augmented_labels
        }
        
        self.verify_no_synthetic_nodes(G, G_aug, eliminated_nodes, 'random')
        
        logger.info(f'Random Augmentation Complete')
        return G_aug, augmented_features, augmented_labels, metadata
    
    def augment_coarsening(
        self,
        G: nx.Graph,
        node_features: Optional[torch.Tensor] = None,
        edge_weights: Optional[Dict] = None,
        node_names: Optional[List] = None,  # ← ADD
        labels: Optional[torch.Tensor] = None  # ← ADD
    ) -> Tuple[nx.Graph, Optional[torch.Tensor], Dict]:
        """
        Coarsening-based elimination (CoarseningPreconditioner from C++)
        Collapses nodes onto neighbors instead of creating full cliques
        """
        
        G_aug = G.copy()
        num_nodes = G_aug.number_of_nodes()
        elimination_count = int(self.elimination_ratio * num_nodes)        
        # Convert to numpy if needed
        if node_features is not None:
            node_features = node_features.numpy() if isinstance(node_features, torch.Tensor) else node_features
        if labels is not None:
            labels = labels.numpy() if isinstance(labels, torch.Tensor) else labels
        if node_names is not None:
            node_names = node_names if isinstance(node_names, np.ndarray) else np.array(node_names)
        
        logger.info(f"Coarsening augmentation: {num_nodes} nodes, eliminating {elimination_count} nodes")
        
        node_id_to_name_original = self.create_node_name_mapping(G, node_names, self.elimination_strategy)
        
        if edge_weights is None:
            edge_weights = {(u, v): G_aug[u][v].get('weight', 1.0) for u, v in G_aug.edges()}
            
        degree_pq = self.create_degree_priority_queue(G_aug)
        eliminated_nodes = []
        collapsed_mapping = {}
        
        for i in range(elimination_count):
            if len(degree_pq) == 0:
                break
            
            v_i = self.pop_min_degree(degree_pq)
            eliminated_nodes.append(v_i)
            
            neighbours = list(G_aug.neighbors(v_i))
            if len(neighbours) == 0:
                continue
            
            neighbour_weights = [edge_weights.get((v_i, n), edge_weights.get((v_i, n), 1.0)) for n in neighbours]
            
            total_weight = sum(neighbour_weights)
            
            if total_weight == 0:
                continue
            
            probs = [w/total_weight for w in neighbour_weights]
            k = np.random.choice(neighbours, p = probs)
            w_k = edge_weights.get((v_i, k), edge_weights.get((v_i, k), 1.0))
            
            # Collapse v_i onto k
            collapsed_mapping[v_i] = k
            
            if G_aug.has_edge(v_i, k):
                G_aug.remove_edge(v_i, k)
            
            if k in degree_pq:
                self.decrement_degree(degree_pq, k)
            
            for j in neighbours:
                if j == k:
                    continue
                
                # CRITICAL: Verify nodes still exist
                if not G_aug.has_node(k) or not G_aug.has_node(j):
                    logger.warning(f"Skipping edge update ({k}, {j}) - node doesn't exist")
                    continue
                
                w_j = edge_weights.get((v_i, j), edge_weights.get((v_i, j), 1.0))
                
                new_weight = (w_k * w_j) /(w_j + w_k)
                
                if G_aug.has_edge(k, j):
                    G_aug[k][j]['weight'] = G_aug[k][j].get('weight', 0) + new_weight
                else:
                    G_aug.add_edge(k, j, weight = new_weight)
                    if k in degree_pq:
                        self.increment_degree(degree_pq, k)
                        
                if j in degree_pq:
                    self.decrement_degree(degree_pq, j)
                    
            G_aug.remove_node(v_i)
        
        for u, v in G_aug.edges():
            if 'weight' not in G_aug[u][v]:
                G_aug[u][v]['weight'] = 1.0
        
        # CRITICAL: Force cleanup any remaining eliminated nodes
        G_aug, cleaned_count = self.force_cleanup_eliminated_nodes(G_aug, eliminated_nodes, None)
        if cleaned_count > 0:
            logger.warning(f"Coarsening: Force cleaned {cleaned_count} nodes that weren't properly eliminated")
        
        # Verify Node Name mapping after augmentation
        node_id_to_name_aug = self.verify_node_name_mapping(
            node_id_to_name_original, G_aug, eliminated_nodes, self.elimination_strategy
        )
        
        # Get final node mapping (use graph as source of truth)
        nodes = list(G.nodes())
        final_node_list, node_mapping = self.get_final_node_mapping(
            G, G_aug, nodes, eliminated_nodes, self.elimination_strategy
        )
        
        # CRITICAL: Restore edges for isolated nodes from original graph
        G_aug, edges_restored = self.restore_edges_for_isolated_nodes(
            G, G_aug, final_node_list, self.elimination_strategy
        )
        
        augmented_features = None
        augmented_labels = None
        augmented_node_names = None
        
        if node_features is not None and node_names is not None:
            if labels is None:
                logger.warning("Labels not provided - features will be aligned but labels won't match")
                labels = np.zeros(len(node_names), dtype = np.int64)
                
            augmented_features, augmented_labels, augmented_node_names = self.align_features_labels_with_names(
                features=node_features, labels=labels, 
                node_names = node_names, node_id_to_name_aug=node_id_to_name_aug, final_node_list=final_node_list, 
                strategy=self.elimination_strategy
            )
            
            augmented_features = torch.from_numpy(augmented_features)
            augmented_labels = torch.from_numpy(augmented_labels)
        
        eliminated_node_id_to_name = {
            node_id: node_id_to_name_original[node_id]
            for node_id in eliminated_nodes
            if node_id in node_id_to_name_original
        }
        
        metadata = {
            'original_nodes': num_nodes,
            'augmented_nodes': G_aug.number_of_nodes(),
            'augmented_node_ids': list(G_aug.nodes()),
            'final_node_list': final_node_list,
            'node_mapping': node_mapping,
            'eliminated_node_ids': eliminated_nodes,
            'eliminated_node_id_to_name': eliminated_node_id_to_name,
            'eliminated_nodes': len(eliminated_nodes),
            'original_edges': G.number_of_edges(),
            'augmented_edges': G_aug.number_of_edges(),
            'collapsed_mapping': collapsed_mapping,
            'strategy': 'coarsening',
            'node_id_to_name': node_id_to_name_aug,
            'augmented_node_names': augmented_node_names,
            'augmented_labels': augmented_labels
        }
        
        self.verify_no_synthetic_nodes(G, G_aug, eliminated_nodes, self.elimination_strategy)
        
        logger.info(f'Coarsening augmentation complete!')
        return G_aug, augmented_features, augmented_labels, metadata
    
    def create_degree_priority_queue(self, G: nx.Graph) -> Dict:
        """
        Create priority queue based on node degrees (from C++ DegreePQ)
        Returns dict with degree as key, list of nodes as value
        """
        degree_dict = {}
        for node in G.nodes():
            deg = G.degree(node)
            if deg not in degree_dict:
                degree_dict[deg] = []
            degree_dict[deg].append(node)
        return degree_dict
    
    def pop_min_degree(self, degree_pq: Dict) -> int:
        """Pop node with minimum degree from priority queue"""
        min_deg = min(degree_pq.keys())
        node = degree_pq[min_deg].pop(0)
        if len(degree_pq[min_deg]) == 0:
            del degree_pq[min_deg]
        return node
    
    def increment_degree(self, degree_pq: Dict, node: int):
        """Increment degree of node in priority queue"""
        
        current_deg = None
        for deg, nodes in degree_pq.items():
            if node in nodes:
                current_deg = deg
                nodes.remove(node)
                if len(nodes) == 0:
                    del degree_pq[deg]
                break
            
        if current_deg is not None:
            new_deg = current_deg + 1
            if new_deg not in degree_pq:
                degree_pq[new_deg] = []
            degree_pq[new_deg].append(node)
            
    def decrement_degree(self, degree_pq: Dict, node: int):
        """Decrement degree of node in priority queue"""
        
        current_deg = None
        for deg, nodes in degree_pq.items():
            if node in nodes:
                current_deg = deg
                nodes.remove(node)
                if len(nodes) == 0:
                    del degree_pq[deg]
                break
        
        if current_deg is not None and current_deg > 0:
            new_deg = current_deg - 1
            if new_deg not in degree_pq:
                degree_pq[new_deg] = []
            degree_pq[new_deg].append(node)
    
    def sort_neighbours_compressed(
        self,
        G: nx.Graph,
        node: int,
        neighbours: List[int],
        edge_weights: Dict
    ) -> List[int]:
        """
        Sort neighbors with compression (from C++ compressColumn)
        Implements 'asc', 'desc', 'random' sorting from C++
        """
        def get_weight(n):
                edge = (node, n) if (node, n) in edge_weights else (n, node)
                return edge_weights.get(edge, 1.0)
        if self.neighbor_sort_method == 'asc':
            return sorted(neighbours, key=get_weight)
        
        elif self.neighbor_sort_method == 'desc':
            return sorted(neighbours, key=get_weight, reverse=True)
        
        elif self.neighbor_sort_method == 'random':
            shuffled = neighbours.copy()
            np.random.shuffle(shuffled)
            return shuffled
        
        elif self.neighbor_sort_method == 'degree':
                return sorted(neighbours, key = lambda n: G.degree(n))
        
        elif self.neighbor_sort_method == 'weight':
            return sorted(neighbours, key=get_weight)
        
        else:
            return neighbours
    
    def build_probabilistic_clique(
        self,
        G: nx.Graph,
        v_i: int,
        sorted_neighbours: List[int],
        edge_weights: Dict,
        degree_pq: Optional[Dict]
    ) -> Tuple[List[Tuple[int, int]], List[float]]:
        """
        Build clique using probabilistic sampling (from C++ getSchurComplement)
        
        This implements the exact algorithm from the C++ code:
        - Compute cumulative weight distribution
        - Sample edges probabilistically
        - Apply column scaling (colScale)
        """
        clique_edges = []
        clique_weights = []
        
        n_neighbors = len(sorted_neighbours)
        if n_neighbors < 2:
            return clique_edges, clique_weights
        
        # Get weights and compute cumulative sums (csum, cumspace from C++)
        vals = []
        for neighbor in sorted_neighbours:
            edge = (v_i, neighbor) if (v_i, neighbor) in edge_weights else (neighbor, v_i)
            w = edge_weights.get(edge, 1.0)
            vals.append(w)
        
        csum = sum(vals)
        cumspace = np.cumsum(vals)
        wdeg = csum
        colScale = 1.0
        
        # Build clique edges (joffset loop from C++)
        for joffset in range(n_neighbors - 1):
            v_j = sorted_neighbours[joffset]
            w = vals[joffset] * colScale
            
            if wdeg == 0:
                break
            
            f = w / wdeg
            
            # Sample k with probability proportional to weights (koff selection from C++)
            r = np.random.uniform(cumspace[joffset] if joffset > 0 else 0, csum)
            koff = n_neighbors - 1
            for k_i in range(joffset + 1, n_neighbors):
                if cumspace[k_i] > r:
                    koff = k_i
                    break
            
            v_k = sorted_neighbours[koff]
            
            # Compute new edge weight (from C++)
            newEdgeVal = f * (1 - f) * wdeg
            
            # Add edge
            edge_tuple = (v_j, v_k) if v_j < v_k else (v_k, v_j)
            clique_edges.append(edge_tuple)
            clique_weights.append(newEdgeVal)
            
            # Update colScale and wdeg (from C++)
            colScale = colScale * (1 - f)
            wdeg = wdeg * (1 - f) * (1 - f)
        
        return clique_edges, clique_weights

    def verify_no_synthetic_nodes(
        self,
        G_original: nx.Graph,
        G_aug: nx.Graph,
        eliminated_nodes: List[int],
        strategy: str
    ):
        """
        Verify that no new synthetic nodes are added during augmentation.
        Only node elimination is allowed - no node creation.
        
        Raises ValueError if any inconsistencies are found.
        
        Args:
            G_original: Original graph before augmentation
            G_aug: Augmented graph after augmentation
            eliminated_nodes: List of eliminated node IDs
            strategy: Augmentation strategy name
        """
        original_nodes = set(G_original.nodes())
        augmented_nodes = set(G_aug.nodes())
        eliminated_set = set(eliminated_nodes)
        expected_remaining = original_nodes - eliminated_set
        
        # CRITICAL CHECK 1: No new nodes allowed
        new_nodes = augmented_nodes - original_nodes
        if new_nodes:
            logger.error(f"[{strategy}] FATAL: {len(new_nodes)} synthetic nodes created: {list(new_nodes)[:20]}")
            raise ValueError(
                f"Augmentation created {len(new_nodes)} new nodes! "
                f"Only elimination is allowed. New nodes: {list(new_nodes)[:10]}"
            )
        
        # CRITICAL CHECK 2: Eliminated nodes must be removed
        still_present = eliminated_set.intersection(augmented_nodes)
        if still_present:
            logger.error(f"[{strategy}] FATAL: {len(still_present)} eliminated nodes still present: {list(still_present)[:20]}")
            raise ValueError(
                f"Eliminated nodes still in graph: {list(still_present)[:10]}"
            )
        
        # CRITICAL CHECK 3: Node set must match exactly
        if augmented_nodes != expected_remaining:
            missing = expected_remaining - augmented_nodes
            extra = augmented_nodes - expected_remaining
            
            error_msg = []
            if missing:
                error_msg.append(f"{len(missing)} expected nodes missing: {list(missing)[:10]}")
            if extra:
                error_msg.append(f"{len(extra)} unexpected nodes present: {list(extra)[:10]}")
            
            logger.error(f"[{strategy}] FATAL: Node set mismatch - {', '.join(error_msg)}")
            raise ValueError(f"Node set mismatch: {', '.join(error_msg)}")
        
        # CRITICAL CHECK 4: All edges must connect only remaining nodes
        invalid_edges = []
        for u, v in G_aug.edges():
            if u not in expected_remaining or v not in expected_remaining:
                invalid_edges.append((u, v))
        
        if invalid_edges:
            logger.error(f"[{strategy}] FATAL: {len(invalid_edges)} edges connect invalid nodes: {invalid_edges[:20]}")
            raise ValueError(
                f"{len(invalid_edges)} edges connect non-existent or eliminated nodes! "
                f"Sample: {invalid_edges[:10]}"
            )
        
        logger.info(f"[{strategy}] ✓ No synthetic nodes - consistency verified")
    
    def force_cleanup_eliminated_nodes(
        self,
        G_aug: nx.Graph,
        eliminated_nodes: List[int],
        edge_weights: Optional[Dict] = None
    ) -> Tuple[nx.Graph, int]:
        """
        Force removal of any eliminated nodes that are still present in the graph.
        Connects neighbors of removed nodes using Schur complement approach.
        
        This is a safety function to handle cases where node elimination failed.
        
        Args:
            G_aug: Augmented graph that may still contain eliminated nodes
            eliminated_nodes: List of nodes that should have been eliminated
            edge_weights: Dictionary of edge weights
            
        Returns:
            cleaned_graph: Graph with all eliminated nodes forcefully removed
            num_cleaned: Number of nodes that were forcefully removed
        """
        eliminated_set = set(eliminated_nodes)
        still_present = [n for n in eliminated_nodes if G_aug.has_node(n)]
        
        if not still_present:
            logger.info("No cleanup needed - all eliminated nodes already removed")
            return G_aug, 0
        
        logger.warning(f"Force cleaning {len(still_present)} nodes that should have been eliminated: {still_present[:20]}")
        
        cleaned_count = 0
        
        for node in still_present:
            if not G_aug.has_node(node):
                continue
            
            # Get neighbors before removal
            neighbors = list(G_aug.neighbors(node))
            
            if len(neighbors) > 1:
                # Connect neighbors in a clique-like fashion
                for i in range(len(neighbors)):
                    for j in range(i + 1, len(neighbors)):
                        u, v = neighbors[i], neighbors[j]
                        
                        # Skip if nodes don't exist or are also being eliminated
                        if not G_aug.has_node(u) or not G_aug.has_node(v):
                            continue
                        if u in eliminated_set or v in eliminated_set:
                            continue
                        
                        # Get edge weights from graph directly
                        w_u = G_aug[node][u].get('weight', 1.0) if G_aug.has_edge(node, u) else 1.0
                        w_v = G_aug[node][v].get('weight', 1.0) if G_aug.has_edge(node, v) else 1.0
                        
                        # Compute Schur complement weight
                        deg_node = len(neighbors)
                        new_weight = (w_u * w_v) / max(deg_node, 1.0)
                        
                        # Add or update edge
                        if G_aug.has_edge(u, v):
                            G_aug[u][v]['weight'] = G_aug[u][v].get('weight', 0) + new_weight
                        else:
                            G_aug.add_edge(u, v, weight=new_weight)
            
            # Remove the node - this will automatically remove all its edges
            G_aug.remove_node(node)
            cleaned_count += 1
            logger.debug(f"Force removed node {node} with {len(neighbors)} neighbors")
        
        logger.info(f"Force cleanup complete: removed {cleaned_count} nodes")
        return G_aug, cleaned_count

    def get_final_node_mapping(
        self,
        G_original: nx.Graph,
        G_aug: nx.Graph,
        original_nodes: List[int],
        eliminated_nodes: List[int],
        strategy: str
    ) -> Tuple[List[int], Dict[int, int]]:
        """
        Get the final node list that matches the augmented graph exactly.
        
        This ensures node features align with graph structure without
        eliminating additional nodes beyond the intended ratio.
        
        Args:
            G_original: Original graph
            G_aug: Augmented graph after elimination
            original_nodes: Original ordered list of node IDs
            eliminated_nodes: List of eliminated node IDs
            strategy: Augmentation strategy name
            
        Returns:
            final_node_list: Ordered list of node IDs in augmented graph
            node_mapping: Dict mapping node IDs to indices in final list
        """
        aug_nodes = sorted(G_aug.nodes())
        eliminated_set = set(eliminated_nodes)
        
        # Check for consistency
        expected_remaining = set(original_nodes) - eliminated_set
        actual_nodes = set(aug_nodes)
        
        # Identify discrepancies
        extra_nodes = actual_nodes - expected_remaining  # Nodes that shouldn't be there
        missing_nodes = expected_remaining - actual_nodes  # Expected nodes that are missing
        
        if extra_nodes:
            logger.warning(f"[{strategy}] Found {len(extra_nodes)} unexpected nodes in graph: {list(extra_nodes)[:10]}")
        
        if missing_nodes:
            logger.warning(f"[{strategy}] Missing {len(missing_nodes)} expected nodes from graph: {list(missing_nodes)[:10]}")
        
        # Use actual augmented graph nodes as source of truth
        final_node_list = aug_nodes
        node_mapping = {node: idx for idx, node in enumerate(final_node_list)}
        
        logger.info(f"[{strategy}] Final node mapping: {len(final_node_list)} nodes")
        logger.info(f"[{strategy}] Original: {len(original_nodes)}, Eliminated: {len(eliminated_nodes)}, "
                   f"Expected: {len(expected_remaining)}, Actual: {len(actual_nodes)}")

        return final_node_list, node_mapping

    def restore_edges_for_isolated_nodes(
        self,
        G_original: nx.Graph,
        G_aug: nx.Graph,
        final_node_list: List[int],
        strategy: str,
        max_edges_per_node: int = 3,
        min_edges_per_node: int = 1
    ) -> Tuple[nx.Graph, int]:
        """
        Restore edges from original graph for nodes that became isolated in augmentation.
        
        This ensures all nodes in final_node_list have at least one edge by borrowing
        edges from the original graph. Randomly selects a subset of original neighbors
        to avoid connecting all isolated nodes to the same node.
        
        Args:
            G_original: Original graph before augmentation
            G_aug: Augmented graph (may have isolated nodes)
            final_node_list: List of nodes that should be in the augmented graph
            strategy: Augmentation strategy name
            max_edges_per_node: Maximum number of edges to restore per isolated node
            min_edges_per_node: Minimum number of edges to restore per isolated node
            
        Returns:
            G_aug: Augmented graph with restored edges
            num_restored: Number of edges restored
        """
        final_node_set = set(final_node_list)
        isolated_nodes = []
        
        # Find nodes with no edges
        for node in final_node_list:
            if G_aug.has_node(node):
                if G_aug.degree(node) == 0:
                    isolated_nodes.append(node)
            else:
                # Node not in graph at all - add it
                G_aug.add_node(node)
                isolated_nodes.append(node)
        
        if not isolated_nodes:
            logger.info(f"[{strategy}] No isolated nodes - all nodes have edges")
            return G_aug, 0
        
        logger.warning(f"[{strategy}] Found {len(isolated_nodes)} isolated nodes, restoring edges from original graph")
        
        edges_restored = 0
        
        for node in isolated_nodes:
            if not G_original.has_node(node):
                logger.warning(f"[{strategy}] Node {node} not in original graph - cannot restore edges")
                continue
            
            # Get original neighbors
            original_neighbors = list(G_original.neighbors(node))
            
            # Filter to only neighbors that are in final node list (not eliminated)
            valid_neighbors = [n for n in original_neighbors if n in final_node_set]
            
            if not valid_neighbors:
                logger.warning(f"[{strategy}] Node {node} has no valid neighbors in final node list")
                # As a fallback, randomly connect to nodes in final list
                if len(final_node_list) > 1:
                    # Pick random nodes that are not itself
                    other_nodes = [n for n in final_node_list if n != node]
                    if other_nodes:
                        # Randomly select 1-3 nodes to connect to
                        num_connections = min(min_edges_per_node, len(other_nodes))
                        fallback_neighbors = np.random.choice(other_nodes, size=num_connections, replace=False)
                        
                        for fallback_neighbor in fallback_neighbors:
                            G_aug.add_edge(node, fallback_neighbor, weight=1.0)
                            edges_restored += 1
                        
                        logger.warning(f"[{strategy}] Added {num_connections} random fallback edges for node {node}")
                continue
            
            # Randomly select a subset of valid neighbors to restore
            num_edges_to_restore = min(
                max_edges_per_node,
                max(min_edges_per_node, len(valid_neighbors))
            )
            
            # If we have few valid neighbors, use all of them
            if len(valid_neighbors) <= max_edges_per_node:
                selected_neighbors = valid_neighbors
            else:
                # Randomly sample neighbors
                selected_neighbors = np.random.choice(
                    valid_neighbors, 
                    size=num_edges_to_restore, 
                    replace=False
                ).tolist()
            
            # Restore edges to selected neighbors
            for neighbor in selected_neighbors:
                if not G_aug.has_edge(node, neighbor):
                    # Get original edge weight
                    if G_original.has_edge(node, neighbor):
                        weight = G_original[node][neighbor].get('weight', 1.0)
                    else:
                        weight = 1.0
                    
                    G_aug.add_edge(node, neighbor, weight=weight)
                    edges_restored += 1
        
        logger.info(f"[{strategy}] Restored {edges_restored} edges for {len(isolated_nodes)} isolated nodes")
        return G_aug, edges_restored

    def to_pytorch_geometric(
        self,
        G: nx.Graph,
        node_features: Optional[np.ndarray] = None,
        node_mapping: Optional[Dict] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Convert augmented NetworkX graph to PyTorch Geometric format
        
        Args:
            G: NetworkX graph
            node_features: Node feature matrix
            node_mapping: Mapping from original node IDs to indices
        
        Returns:
            edge_index: [2, num_edges] tensor
            edge_weight: [num_edges] tensor
            x: [num_nodes, feature_dim] tensor (if features provided)
        """
        # Create node mapping if not provided
        if node_mapping is None:
            nodes = sorted(G.nodes())
            node_mapping = {node: idx for idx, node in enumerate(nodes)}
        else:
            nodes = sorted([n for n in node_mapping.keys() if G.has_node(n)])
            
        edge_list = []
        edge_weights = []
        
        for u, v, data in G.edges(data=True):
            if u in node_mapping and v in node_mapping:
                u_idx = node_mapping[u]
                v_idx = node_mapping[v]
                weight = data.get('weight', 1.0)
                
                edge_list.append([u_idx, v_idx])
                edge_weights.append(weight)
                edge_list.append([v_idx, u_idx])
                edge_weights.append(weight)
                
        if len(edge_list) == 0:
            logger.warning('No edges found in augmented graph')
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_weights = torch.zeros(0)
        else:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
            edge_weights = torch.tensor(edge_weights, dtype=torch.float)
        
        x = None
        if node_features is not None:
            x = torch.tensor(node_features, dtype=torch.float)
        return edge_index, edge_weights, x

    def create_node_name_mapping(
        self,
        G:nx.Graph,
        node_names: List[str],
        strategy: str
    ) -> Dict[int, str]:
        """
        Create explicit node_id -> node_name mapping before augmentation.
        This ensures we can track which names correspond to which indices.
        
        Args:
            G: Original graph
            node_names: List of node names (must match G.nodes() order)
            strategy: Augmentation strategy name
            
        Returns:
            node_id_to_name: Dict mapping node IDs to names
        """
        
        nodes = list(G.nodes())
        
        if node_names is None:
            logger.warning(f"[{strategy}] No node names provided - using node IDs as names")
            return {node: f"Node_{node}" for node in nodes}
        
        if len(node_names) != len(nodes):
            logger.error(
            f"[{strategy}] Node names length ({len(node_names)}) != "
            f"graph nodes ({len(nodes)})"
            )
            raise ValueError("Node names and graph nodes count mismatch!")
        
        # node_names should already be List[str], but verify
        if not isinstance(node_names, list):
            logger.warning(f"[{strategy}] node_names is {type(node_names)}, expected list")
            if isinstance(node_names, (np.ndarray, torch.Tensor)):
                node_names = list(node_names) if isinstance(node_names, np.ndarray) else node_names.tolist()
            else:
                node_names = list(node_names)
        
        # Create Mapping
        node_id_to_name = {node_id: name for node_id, name  in zip(nodes, node_names)}
        
        known_genes = ['TP53', 'KRAS', 'PIK3CA']
        for gene in known_genes:
            if gene in node_names:
                idx = node_names.index(gene)
                node_id = nodes[idx]
                mapped_name = node_id_to_name[node_id]
                if mapped_name != gene:
                    logger.error(
                        f"[{strategy}] Mapping error: {gene} at index {idx} "
                        f"(node_id={node_id}) maps to '{mapped_name}'"
                    )
                    raise ValueError(f"Node name mapping validation failed for {gene}")
        
        logger.info(f"[{strategy}] Created node name mapping for {len(nodes)} nodes")
        return node_id_to_name
    
    def verify_node_name_mapping(
        self,
        node_id_to_name_original: Dict[int, str],
        G_aug: nx.Graph,
        eliminated_nodes: List[int],
        strategy: str
    ) -> Dict[int, str]:
        """
        Verify node names are still correctly mapped after augmentation.
        
        Args:
            node_id_to_name_original: Original node ID -> name mapping
            G_aug: Augmented graph
            eliminated_nodes: List of eliminated node IDs
            strategy: Augmentation strategy name
            
        Returns:
            node_id_to_name_aug: Validated mapping for augmented graph
        """
        
        aug_nodes = set(G_aug.nodes())
        eliminated_set = set(eliminated_nodes)
        
        #  Create augmented mapping (exclude eliminated nodes)
        
        node_id_to_name_aug = {
            node_id: name
                for node_id, name in node_id_to_name_original.items()
                if node_id in aug_nodes and node_id not in eliminated_set
        }
        
        # Verify no eliminated nodes are in augmented graph
        for node_id in eliminated_set:
            if node_id in aug_nodes:
                logger.error(
                    f"[{strategy}] Eliminated node {node_id} "
                    f"('{node_id_to_name_original.get(node_id, 'UNKNOWN')}') still in graph!"
                )
                raise ValueError(f"Eliminated node {node_id} still present after augmentation")
        
        # Verify all augmented nodes have names
        for node_id in aug_nodes:
            if node_id not in node_id_to_name_aug:
                logger.error(
                    f"[{strategy}] Augmented graph has node {node_id} with no name mapping!"
                )
                raise ValueError(f"Node {node_id} in augmented graph has no name")
        
        logger.info(f"[{strategy}] Verified node name mapping for {len(node_id_to_name_aug)} augmented nodes")
        return node_id_to_name_aug
    
    def align_features_labels_with_names(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        node_names: List[str],
        node_id_to_name_aug: Dict[int, str],
        final_node_list: List[int],
        strategy: str
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Align features and labels using node names as the source of truth.
        This ensures labels match features correctly after augmentation.
        
        Args:
            features: Original feature matrix [num_original, feature_dim]
            labels: Original labels [num_original] - np.ndarray
            node_names: Original node names [num_original] - List[str]
            node_id_to_name_aug: Mapping from augmented node IDs to names
            final_node_list: Ordered list of node IDs in augmented graph
            strategy: Augmentation strategy name
            
        Returns:
            aligned_features: [num_augmented, feature_dim] np.ndarray
            aligned_labels: [num_augmented] np.ndarray
            aligned_names: [num_augmented] List[str]
        """
        
        # Validate inputs
        if not isinstance(features, np.ndarray):
            logger.error(f"[{strategy}] features must be np.ndarray, got {type(features)}")
            raise TypeError(f"features must be np.ndarray")
        
        if not isinstance(labels, np.ndarray):
            logger.error(f"[{strategy}] labels must be np.ndarray, got {type(labels)}")
            raise TypeError(f"labels must be np.ndarray")
        
        if len(features) != len(labels) or len(features) != len(node_names):
            logger.error(
                f"[{strategy}] Input length mismatch: "
                f"features={len(features)}, labels={len(labels)}, names={len(node_names)}"
            )
            raise ValueError("Input arrays must have same length")
        
        logger.info(f"[{strategy}] Aligning {len(final_node_list)} nodes using name mapping...")
        
        # Create name -> (feature, label) mapping from original data
        name_to_data = {}
        for i, name in enumerate(node_names):
            name_to_data[name] = {
                'feature': features[i],
                'label': labels[i],
                'original_index': i
            }
        
        logger.info(f"[{strategy}] Created mapping for {len(name_to_data)} unique names")
        
        # Initialize output arrays
        num_aug = len(final_node_list)
        feature_dim = features.shape[1]
        
        aligned_features = np.zeros((num_aug, feature_dim), dtype=features.dtype)
        aligned_labels = np.zeros(num_aug, dtype=labels.dtype)
        aligned_names = []
        
        missing_names = []
        
        # Align data using node IDs -> names mapping
        for i, node_id in enumerate(final_node_list):
            # Get name for this node ID
            name = node_id_to_name_aug.get(node_id)
            
            if name is None:
                logger.error(f"[{strategy}] Node ID {node_id} has no name mapping!")
                raise ValueError(f"Node {node_id} at position {i} has no name")
            
            # Get data for this name
            if name not in name_to_data:
                logger.warning(f"[{strategy}] Name '{name}' (node {node_id}) not in original data")
                missing_names.append(name)
                # Use zeros for features and -100 for label (ignore index)
                aligned_features[i] = np.zeros(feature_dim, dtype=features.dtype)
                aligned_labels[i] = -100
                aligned_names.append(name)
            else:
                data = name_to_data[name]
                aligned_features[i] = data['feature']
                aligned_labels[i] = data['label']
                aligned_names.append(name)
        
        # Log statistics
        if missing_names:
            logger.warning(
                f"[{strategy}] {len(missing_names)} names not found in original data: "
                f"{missing_names[:10]}"
            )
        
        # Verify output shapes
        assert aligned_features.shape[0] == num_aug, "Features shape mismatch"
        assert aligned_labels.shape[0] == num_aug, "Labels shape mismatch"
        assert len(aligned_names) == num_aug, "Names length mismatch"
        
        logger.info(
            f"[{strategy}] Alignment complete: "
            f"features={aligned_features.shape}, "
            f"labels={aligned_labels.shape}, "
            f"names={len(aligned_names)}"
        )
        
        # CRITICAL VALIDATION: Check known driver genes
        known_drivers = ['TP53', 'KRAS', 'PIK3CA']
        for gene in known_drivers:
            if gene in aligned_names:
                # Find position in aligned_names list
                gene_position = aligned_names.index(gene)
                # Get corresponding label from aligned_labels array
                gene_label = aligned_labels[gene_position]
                
                logger.info(f"[{strategy}] {gene}: position={gene_position}, label={gene_label}")
                
                if gene_label != 1 and gene_label != -100:
                    logger.error(
                        f"[{strategy}] CRITICAL: {gene} has incorrect label {gene_label} after alignment!"
                    )
                    raise ValueError(f"{gene} has wrong label after alignment: {gene_label}")
                elif gene_label == 1:
                    logger.info(f"[{strategy}] ✓ {gene} correctly labeled as driver (label=1)")
        
        logger.info(f"[{strategy}] ✓ Alignment validation passed")

        return aligned_features, aligned_labels, aligned_names