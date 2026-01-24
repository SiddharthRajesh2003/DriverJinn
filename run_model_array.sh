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

base=/N/project/Krolab/Siddharth/DriverGenePred
cd $base

module load conda
conda activate conda_envs/envs/gnn_env
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Running fold ${SLURM_ARRAY_TASK_ID} on $(hostname)"

python -m model.train_model \
    --dataset_file curvature_output/GGNet_contrastive_v2_random_r0.1.pkl \
    --num_epochs 800 \
    --hidden_channels 144 \
    --projection_dim 72 \
    --num_layers 3 \
    --num_heads 4 \
    --num_folds 5 \
    --specific_folds ${SLURM_ARRAY_TASK_ID} \
    --model_out_prefix GGNet_random_r0.2_fold${SLURM_ARRAY_TASK_ID} \
    --temperature 0.7 \
    --attention_mode hybrid \
    --pathway_aggregator hierarchical \
    --attention_chunk_size 1000 \
    --gradient_accumulation_steps 36 \
    --mixed_precision \
    --scheduler_patience 60 \
    --early_stopping_patience 100 \
    --concat_heads \
    --ranking_loss_type bpr \
    --focal_gamma 2.0 \
    --use_focal \
    --contrastive_weight 0.3
