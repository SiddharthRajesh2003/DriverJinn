python -m model.train_model --dataset_file curvature_output/GGNet_contrastive_v2_random_r0.2.pkl \
    --epochs 200 --hidden_channels 128 --num_folds 5 --model_out_prefix GGNet_random \
    --reduce_model_size