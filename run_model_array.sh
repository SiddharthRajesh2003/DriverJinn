#!/bin/bash

#SBATCH -J DriverGene
#SBATCH -p hopper
#SBATCH -o run_model_fold%a_%j.txt
#SBATCH -e run_model_fold%a_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sidrajes@iu.edu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --qos=hopper
#SBATCH --time=48:00:00
#SBATCH --mem=50GB
#SBATCH -A r00750
#SBATCH --array=1-5

base=your/base/dir/DriverJinn/
cd $base

module load conda
conda activate conda_envs/envs/gnn_env
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Running fold ${SLURM_ARRAY_TASK_ID} on $(hostname)"

python -m model.train_model \
    --dataset_file curvature_output/GGNet_contrastive_v2_random_r0.2_stratified5CV.pkl \
    --num_epochs 1000 \
    --hidden_channels 128 \
    --projection_dim 96 \
    --num_layers 2 \
    --num_heads 2 \
    --num_folds 5 \
    --train_metrics_dir model_results_$(date +%F) \
    --specific_folds ${SLURM_ARRAY_TASK_ID} \
    --model_out_prefix GGNet_random_r0.2_fold${SLURM_ARRAY_TASK_ID} \
    --temperature 0.3 \
    --dropout 0.3 \
    --attention_mode hybrid \
    --pathway_aggregator hierarchical \
    --aggregation max \
    --negative_slope 0.25 \
    --attention_chunk_size 1000 \
    --gradient_accumulation_steps 8 \
    --mixed_precision \
    --early_stopping_patience 15 \
    --validation_frequency 20 \
    --scheduler_patience 5 \
    --scheduler_factor 0.5 \
    --concat_heads \
    --ranking_loss_type bpr \
    --ranking_loss_scale 5 \
    --ranking_loss_samples 512 \
    --contrastive_weight 0.2 \
    --learning_rate 5e-4 \
    --weight_decay 5e-4
