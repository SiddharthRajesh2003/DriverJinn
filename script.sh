module load conda
conda activate gnn_env

python -m graph_builder.curvature_pipeline --dataset_file data/dataset_GGNet.pkl  --method both \
        --augment --num_views 2 --elimination_ratio 0.2 --strategy priority

python -m graph_builder.curvature_pipeline --dataset_file data/dataset_PathNet.pkl --method both \
        --augment --num_views 2 --elimination_ratio 0.2 --strategy priority

python -m graph_builder.curvature_pipeline --dataset_file data/dataset_PPNet.pkl --method both \
        --augment --num_views 2 --elimination_ratio 0.2 --strategy priority