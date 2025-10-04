import torch
import networkx as nx
import numpy as np
from typing import List, Dict, Optional, Tuple
import scipy.sparse as sp
from scipy.linalg import block_diag
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
                edge_weights: Optional[Dict] = None
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
            return self.augment_priority(G, node_features, edge_weights)
        
        elif self.elimination_strategy == 'random':
            return self.augment_random(G, node_features, edge_weights)
        
        elif self.elimination_strategy == 'coarsening':
            return self.augment_coarsening(G, node_features, edge_weights)
        else:
            raise ValueError(f'Unknown elimination strategy: {self.elimination_strategy}')
        
    def augment_priority(
        self,
        G: nx.Graph,
        node_features: Optional[torch.Tensor] = None,
        edge_weights: Optional[Dict] = None
    ) -> Tuple[nx.Graph, Optional[torch.Tensor], Dict]:
        """
        Priority-based elimination (PriorityPreconditioner from C++)
        Eliminates nodes in order of degree (lowest first)
        """
        
        G_aug = G.copy()
        num_nodes = G.number_of_nodes()
        elimination_count = int(self.elimination_ratio * num_nodes)
        node_features = node_features.numpy()
        
        logger.info(f"Priority augmentation: {num_nodes} nodes, eliminating {elimination_count} nodes")
        
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
                if G_aug.has_edge(u, v):
                    G_aug[u][v]['weight'] = G_aug[u][v].get('weight', 0)
                    
                else:
                    G_aug.add_edge(u, v, weight = weight)
                    added_edges.append((u, v))
                    if u in degree_pq:
                        self.increment_degree(degree_pq, u)
                    
                    if v in degree_pq:
                        self.increment_degree(degree_pq, v)
                        
        augmented_features = None
        if node_features is not None:
            nodes = list(G.nodes())
            augmented_features = self.update_node_features(
                node_features, nodes, eliminated_nodes, G_aug
            )
            augmented_features = torch.from_numpy(augmented_features)
            
        metadata = {
            'original_nodes': num_nodes,
            'augmented_nodes': G_aug.number_of_nodes(),
            'eliminated_nodes': len(eliminated_nodes),
            'original_edges': G_aug.number_of_edges(),
            'added_edges': len(added_edges),
            'avg_clique_size': np.mean(clique_sizes) if clique_sizes else 0,
            'eliminated_node_ids': eliminated_nodes,
            'strategy': 'priority'
        }
        
        logger.info(f"Priority augmentation complete: {metadata['original_nodes']} → "
                   f"{metadata['augmented_nodes']} nodes")
        
        return G_aug, augmented_features, metadata
    
    def augment_random(
        self, 
        G: nx.Graph,
        node_features: Optional[torch.Tensor],
        edge_weights: Optional[Dict]
    ) -> Tuple[nx.Graph, Optional[torch.Tensor], Dict]:
        """
        Random elimination (RandomPreconditioner from C++)
        Eliminates nodes in random order
        """
        
        G_aug = G.copy()
        num_nodes = G_aug.number_of_nodes()
        elimination_count = int(self.elimination_ratio * num_nodes)
        node_features = node_features.numpy()
        
        logger.info(f"Random augmentation: {num_nodes} nodes, eliminating {elimination_count} nodes")
        
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
                if G_aug.has_edge(u, v):
                    G_aug[u][v]['weight'] = G_aug[u][v].get('weight', 0)
                else:
                    G_aug.add_edge(u, v, weight = weight)
                    added_edges.append((u, v))
                    
        augmented_features = None
        if node_features is not None:
            nodes = list(G_aug.nodes())
            augmented_features = self.update_node_features(
                node_features, nodes, eliminated_nodes, G_aug
            )
            augmented_features = torch.from_numpy(augmented_features)
        
        metadata = {
            'original_nodes': num_nodes,
            'augmented_nodes': G_aug.number_of_nodes(),
            'eliminated_nodes': len(eliminated_nodes),
            'original_edges': G_aug.number_of_edges(),
            'added_edges': len(added_edges),
            'avg_clique_size': np.mean(clique_sizes) if clique_sizes else 0,
            'eliminated_node_ids': eliminated_nodes,
            'strategy': 'random'
        }
        
        logger.info(f'Random Augmentation Complete')
        return G_aug, augmented_features, metadata
    
    def augment_coarsening(
        self,
        G: nx.Graph,
        node_features: Optional[torch.Tensor] = None,
        edge_weights: Optional[Dict] = None
    ) -> Tuple[nx.Graph, Optional[torch.Tensor], Dict]:
        """
        Coarsening-based elimination (CoarseningPreconditioner from C++)
        Collapses nodes onto neighbors instead of creating full cliques
        """
        
        G_aug = G.copy()
        num_nodes = G_aug.number_of_nodes()
        elimination_count = int(self.elimination_ratio * num_nodes)
        node_features = node_features.numpy()
        
        logger.info(f"Coarsening augmentation: {num_nodes} nodes, eliminating {elimination_count} nodes")
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
        
        augmented_features = None
        if node_features is not None:
            nodes = list(G.nodes())
            augmented_features = self.update_node_features_coarsened(
                node_features, nodes, eliminated_nodes, collapsed_mapping, G_aug
            )
        
        metadata = {
            'original_nodes': num_nodes,
            'augmented_nodes': G_aug.number_of_nodes(),
            'eliminated_nodes': len(eliminated_nodes),
            'original_edges': G.number_of_edges(),
            'augmented_edges': G_aug.number_of_edges(),
            'collapsed_mapping': collapsed_mapping,
            'strategy': 'coarsening'
        }
        
        logger.info(f'Coarsening augmentation complete!')
        return G_aug, augmented_features, metadata
    
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
        
        if current_deg is not None and current_deg > 1:
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
        
        n_neighbours = len(sorted_neighbours)
        
        if n_neighbours < 2:
            return clique_edges, clique_weights
        
        vals = []
        for neighbour in sorted_neighbours:
            edge = (v_i, neighbour) if (v_i, neighbour) in edge_weights else (neighbour, v_i)
            w = edge_weights.get(edge, 1.0)
            vals.append(w)
        
        csum = sum(vals)
        cumspace = np.cumsum(vals)
        wdeg = csum
        colScale = 1.0
        
        for joffset in range(n_neighbours - 1):
            v_j = sorted_neighbours[joffset]
            w = vals[joffset] * colScale
            
            if wdeg == 0:
                break
            
            f = w/wdeg
            
            r = np.random.uniform(cumspace[joffset] if joffset > 0 else 0)
            koff = n_neighbours - 1
            for k_i in range(joffset + 1, n_neighbours):
                if cumspace[k_i] > r:
                    koff = k_i
                    break
            
            
            v_k = sorted_neighbours[koff]
            
            new_edge_val = f * (1-f) * wdeg
            
            edge_tuple = (v_j, v_k) if v_j < v_k else (v_k, v_j)
            clique_edges.append(edge_tuple)
            clique_weights.append(new_edge_val)
            
            colScale = colScale * (1-f)
            wdeg = wdeg * (1-f) * (1-f)
            
        return clique_edges, clique_weights
    
    def update_node_features_coarsened(
        self,
        features: np.ndarray,
        original_nodes: List[int],
        eliminated_nodes: List[int],
        collapsed_mapping: Dict,
        G_aug: nx.Graph
    ) -> np.ndarray:
        """
        Update features for coarsening strategy
        Features of collapsed nodes are aggregated into their targets
        """
        node_to_idx = {node: idx for idx, node in enumerate(original_nodes)}
        remaining_nodes = [n for n in original_nodes if n not in eliminated_nodes]
        
        augmented_features = features.copy()
        
        for v_i, target in collapsed_mapping.items():
            if v_i not in node_to_idx or target not in node_to_idx:
                continue
            
            idx_i = node_to_idx[v_i]
            idx_target = node_to_idx[target]
            
            augmented_features[idx_target] += augmented_features[idx_i]
            
        keep_mask = np.array([node not in eliminated_nodes for node in original_nodes])
        return augmented_features[keep_mask]
    
    def sort_neighbours(
        self,
        G: nx.Graph,
        node: int,
        neighbours: List[int],
        edge_weights: Dict
    ) -> List[int]:
        """
        Sort neighbors according to N_s function
        
        Methods:
        - 'degree': Sort by degree (ascending)
        - 'weight': Sort by edge weight to node (ascending)
        - 'random': Random ordering
        """
        
        if self.neighbor_sort_method == 'degree':
            return sorted(neighbours, key= lambda n: G.degree(n))
        elif self.neighbor_sort_method == 'weight':
            def get_weight(n):
                edge = (node, n) if (node, n) in edge_weights else (n, node)
                return edge_weights.get(edge, 1.0)
            return sorted(neighbours, key=get_weight)
        
        elif self.neighbor_sort_method == 'random':
            shuffled = neighbours.copy()
            np.random.shuffle(shuffled)
            return shuffled
        
        else:
            logger.warning(f"Unknown sort method '{self.neighbor_sort_method}', using degree")
            return sorted(neighbours, key=lambda n: G.degree(n))
    
    def build_approximate_clique(
        self,
        G: nx.Graph,
        v_i: int,
        sorted_neighbours: List[int],
        edge_weights: Dict
    ) -> Tuple[List[Tuple[int, int]], List[float]]:
        """
        Build approximate clique among neighbors of v_i
        
        For each pair of neighbors (v_a, v_b), compute conditional probability
        and add weighted edge to form clique approximation.
        
        Returns:
            clique_edges: List of edge tuples
            clique_weights: List of corresponding weights
        """
        
        clique_edges = []
        clique_weights = []
        
        n_neighbours = len(sorted_neighbours)
        if n_neighbours < 2:
            return clique_edges, clique_weights
        
        for a in range(n_neighbours - 1):
            v_a = sorted_neighbours[a]
            
            edge_ia = (v_i, v_a) if (v_i, v_a) in edge_weights else (v_a, v_i)
            w_ia = edge_weights.get(edge_ia, 1.0)
            
            best_b = a + 1
            best_prob = -float('inf')
            
            for b in range(a+1, n_neighbours):
                v_b = sorted_neighbours[b]
                
                edge_ib = (v_i, v_b) if (v_i, v_b) in edge_weights else (v_b, v_i)
                w_ib = edge_weights.get(edge_ib, 1.0)
                
                # Compute conditional probability P(v_b | v_a)
                # Using weight-based probability: higher weights = higher probability
                conditional_prob = self.compute_conditional_probability(
                    G, v_a, v_b, w_ia, w_ib, edge_weights
                )
                
                if conditional_prob > best_prob:
                    best_prob = conditional_prob
                    best_b = b
            
            v_b = sorted_neighbours[best_b]
            
            edge_ib = (v_i, v_b) if (v_i, v_b) in edge_weights else (v_b, v_i)
            w_ib = edge_weights.get(edge_ib, 1.0)
            
            # Compute new edge weight: w_ab = w_ia * w_ib / degree(v_i)
            # This approximates the Schur complement contribution
            degree_i = G.degree(v_i)
            w_ab = (w_ia * w_ib) / max(degree_i, 1.0)
            
            edge_ab = (v_a, v_b) if v_a < v_b else (v_b, v_a)
            clique_edges.append(edge_ab)
            clique_weights.append(w_ab)
            
        return clique_edges, clique_weights
    
    def calculate_conditional_probability(
        self,
        G: nx.Graph,
        v_a: int,
        v_b: int,
        w_ia: float,
        w_ib: float,
        edge_weights: Dict
    ) -> float:
        """
        Compute conditional probability P(v_b | v_a) for clique approximation
        
        Uses combination of:
        - Weight similarity
        - Degree similarity
        - Existing connection strength
        """
        # Weight based component
        weight_component = w_ia * w_ib
        
        # Degree based component
        deg_a = G.degree(v_a)
        deg_b = G.degree(v_b)
        degree_similarity = 1.0 / (1.0 + abs(deg_a - deg_b))
        
        edge_ab = (v_a, v_b) if (v_a, v_b) in edge_weights else (v_b, v_a)
        existing_weight = edge_weights.get(edge_ab, 1.0)
        
        conditional_prob = (
            0.5 * weight_component +
            0.3 * degree_similarity +
            0.2 * existing_weight
        )
        
        return conditional_prob
    
    def update_node_features(
        self,
        features: np.ndarray,
        original_nodes: List[int],
        eliminated_nodes: List[int],
        G_aug: nx.Graph
    ) -> np.ndarray:
        """
        Update node features after augmentation
        
        Options:
        1. Simply remove eliminated nodes' features
        2. Aggregate eliminated features into neighbors (more sophisticated)
        """
        if not self.preserve_features:
            keep_mask = np.array([node not in eliminated_nodes for node in original_nodes])
            return features[keep_mask]
        else:
            # Aggregate eliminated features into remaining neighbors
            # This preserves information from eliminated nodes
            node_to_idx = {node: idx for idx, node in enumerate(original_nodes)}
            remaining_nodes = [n for n in original_nodes if n not in eliminated_nodes]
            
            augmented_features = features.copy()
            for v_i in eliminated_nodes:
                if v_i not in node_to_idx:
                    continue
                
                idx_i = node_to_idx[v_i]
                feature_i = features[idx_i]
                
                # Find neighbors in augmented graph (nodes in original that remain)
                # Note: neighbors were connected via clique, so they inherit information
                neighbours_in_original = [n for n in original_nodes if G_aug.has_node(n) and n in remaining_nodes]
                
                if len(neighbours_in_original) > 0:
                    # Distribute features proportionally
                    contribution = feature_i / len(neighbours_in_original)
                    for neighbour in neighbours_in_original:
                        neighbour_idx = node_to_idx[neighbour]
                        augmented_features[neighbour_idx] += contribution
                    
            keep_mask = np.array([n not in eliminated_nodes for n in original_nodes])
            return augmented_features[keep_mask]
    
    def generate_multiple_views(
        self,
        G: nx.Graph,
        node_features: Optional[np.ndarray] = None,
        edge_weights: Optional[Dict] = None,
        num_views: int = 2
    ) -> List[Tuple[nx.Graph, Optional[np.ndarray], Dict]]:
        """
        Generate multiple augmented views for contrastive learning
        
        Args:
            G: Original graph
            node_features: Node features
            edge_weights: Edge weights
            num_views: Number of augmented views to generate
        
        Returns:
            List of (augmented_graph, augmented_features, metadata) tuples
        """
        
        views = []
        
        for i in range(num_views):
            logger.info(f"Generating augmented view {i+1}/{num_views}")
            
            if self.random_seed is not None:
                np.random.seed(self.random_seed + i)
            
            aug_graph, aug_features, metadata = self.augment(
                G, node_features, edge_weights
            )
            
            views.append((aug_graph, aug_features, metadata))
        
        return views
    
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

class ContrastiveSchurAugmentation:
    """
    Wrapper class for contrastive learning with Schur complement augmentation
    
    Generates pairs of augmented views for contrastive learning frameworks
    """
    
    def __init__(
        self,
        elimination_ratio: float = 0.2,
        num_views: int = 2,
        neighbor_sort_method: str = 'weight',
        random_seed: Optional[int] = None
    ):
        self.num_views = num_views
        self.augmenters = [
            SchurComplementAugmentation(
                elimination_ratio=elimination_ratio,
                neighbor_sort_method=neighbor_sort_method,
                random_seed=random_seed + i if random_seed else None
            )
            for i in range(self.num_views)
        ]
        
        logger.info(f"Initialized ContrastiveSchurAugmentation with {num_views} views")
    
    def generate_contrastive_pairs(
        self,
        G: nx.Graph,
        node_features: Optional[torch.Tensor] = None, 
        edge_weights: Optional[Dict] = None
    ) -> List[Tuple[nx.Graph, Optional[np.ndarray], Dict]]:
        """
        Generate multiple augmented views for contrastive learning
        
        Returns list of augmented graph tuples for contrastive pairs
        """  
        augmented_views = []
        
        for i, augmenter in enumerate(self.augmenters):
            logger.info(f"Generating contrastive view {i+1}/{self.num_views}")
            aug_graph, aug_features, metadata = augmenter.augment(
                G, node_features, edge_weights
            )
            
            augmented_views.append((aug_graph,aug_features, metadata))
        
        return augmented_views
    