#!/bin/bash

#SBATCH -J HyperSearch
#SBATCH -p hopper
#SBATCH -o hyperparam_search_%j.txt
#SBATCH -e hyperparam_search_%j.err
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

base=/N/project/Krolab/Siddharth/DriverGenePred
cd $base

module load conda
conda activate conda_envs/envs/gnn_env
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -m model.hyperparameter_search \
    --dataset curvature_output/GGNet_contrastive_v2_random_r0.1.pkl \
    --output_dir hyperparam_results \
    --n_trials 50 \
    --n_folds 1 \
    --epochs_per_trial 100 \
    --patience 30 \
    --study_name GGNet_hypersearch
