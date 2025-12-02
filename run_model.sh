python -m model.train_model --dataset_file curvature_output/GGNet_contrastive_v2_random_r0.2.pkl \
    --num_epochs 200 --hidden_channels 128 --num_folds 5 --model_out_prefix GGNet_random \
    --reduce_model_size

python -m model.train_model --dataset_file curvature_output/GGNet_contrastive_v2_priority_r0.2.pkl \
    --num_epochs 200 --hidden_channels 128 --num_folds 5 --model_out_prefix GGNet_priority \
    --reduce_model_size

python -m model.train_model --dataset_file curvature_output/GGNet_contrastive_v2_coarsening_r0.2.pkl \
    --num_epochs 200 --hidden_channels 128 --num_folds 5 --model_out_prefix GGNet_coarsening \
    --reduce_model_size

python -m model.train_model --dataset_file curvature_output/PPNet_contrastive_v2_random_r0.2.pkl \
    --num_epochs 200 --hidden_channels 128 --num_folds 5 --model_out_prefix PPNet_random \
    --reduce_model_size

python -m model.train_model --dataset_file curvature_output/PPNet_contrastive_v2_priority_r0.2.pkl \
    --num_epochs 200 --hidden_channels 128 --num_folds 5 --model_out_prefix PPNet_priority \
    --reduce_model_size

python -m model.train_model --dataset_file curvature_output/PPNet_contrastive_v2_coarsening_r0.2.pkl \
    --num_epochs 200 --hidden_channels 128 --num_folds 5 --model_out_prefix PPNet_coarsening \
    --reduce_model_size

python -m model.train_model --dataset_file curvature_output/PathNet_contrastive_v2_random_r0.2.pkl \
    --num_epochs 200 --hidden_channels 128 --num_folds 5 --model_out_prefix PathNet_random \
    --reduce_model_size

python -m model.train_model --dataset_file curvature_output/PathNet_contrastive_v2_priority_r0.2.pkl \
    --num_epochs 200 --hidden_channels 128 --num_folds 5 --model_out_prefix PathNet_priority \
    --reduce_model_size

python -m model.train_model --dataset_file curvature_output/PathNet_contrastive_v2_coarsening_r0.2.pkl \
    --num_epochs 200 --hidden_channels 128 --num_folds 5 --model_out_prefix PathNet_coarsening \
    --reduce_model_size