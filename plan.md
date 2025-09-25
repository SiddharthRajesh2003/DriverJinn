1. Build a gene-gene interaction network from the STRING database file.
1. Clean the network file and annotate it with the corresponding gene names.
1. Use multi-omics data to create node features for each gene in the network. (Use gene expression data to score genes and create the features. )
1. Single Graph with Differential Features:
    - Nodes: Genes 
    - Node features: [tumor_expression, normal_expression, log_fold_change, p_value]
    - Edge features: [tumor_coexpression, normal_coexpression, differential_coexpression]
    - Target: Driver gene prediction
1. Generate feature subspaces for the network before training the model
1. Design our GNN model to take the network and the feature subspace as input.