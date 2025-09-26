import networkx as nx
import os
import numpy as np
import argparse
import pickle

class Network:
    def __init__(self, edge_index, nodes, num_nodes):
        self.edge_index = edge_index
        self.nodes = nodes
        self.num_nodes = num_nodes
        self.G = self.build_graph()
    
    def build_graph(self):
        G = nx.Graph()
        for i, node in enumerate(nodes):
            G.add_node(i, name = node)
        edges = self.edge_index.t().tolist()
        G.add_edges_from(edges)
        return G
    
    def save_graph(self, file_name ,path):
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, file_name + '.pkl'), 'wb') as f:
            pickle.dump(self.G, f)
    
    def load_graph(self, file_name, path):
        with open(os.path.join(path, file_name + '.pkl'), 'rb') as f:
            self.G = pickle.load(f)
        return self.G
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build and save network graph from dataset')
    parser.add_argument('dataset_file', type=str, help='Dataset pickle file (e.g., dataset_GGNet.pkl)')
    parser.add_argument('output_path', type=str, help='Path to save the graph')
    parser.add_argument('--graph_name', type=str, default=None, help='Name for the saved graph file')
    
    args = parser.parse_args()
    
    dataset_file = args.dataset_file
    output_path = args.output_path
    graph_name = args.graph_name
    
    # Extract dataset type from filename if graph_name not provided
    if graph_name is None:
        if 'GGNet' in dataset_file:
            graph_name = 'GGNet_graph'
        elif 'PathNet' in dataset_file:
            graph_name = 'PathNet_graph'
        elif 'PPNet' in dataset_file:
            graph_name = 'PPNet_graph'
        else:
            graph_name = 'network_graph'
    
    try:
        with open(dataset_file, 'rb') as f:
            data_dict = pickle.load(f)
            f.close()
        
        edge_index = data_dict['edge_index']
        nodes = data_dict['node_name']
        num_nodes = len(nodes)
        
        net = Network(edge_index, nodes, num_nodes)
        net.build_graph()
        net.save_graph(graph_name, output_path)
        
        print('Successfully created the graph file and saved')
    
    except Exception as e:
        print(f"Error processing dataset: {e}")
        exit(1)
    