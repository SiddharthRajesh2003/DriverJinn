#!/bin/bash

#SBATCH -J DriverGene
#SBATCH -p hopper
#SBATCH -o run_model_%a_%j.txt
#SBATCH -e run_model_%a_%j.err
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
#SBATCH --array=1-90

# 3 networks x 3 elimination strategies x 2 ratios x 5 folds = 90 tasks, evaluated on HGDC's released splits.
# Submit with one run tag so every task writes to the same folders, however long the queue takes:
#   sbatch --export=ALL,RUN_TAG=2026-10-16 run_model_array.sh
RUN_TAG=${RUN_TAG:?set RUN_TAG at submission, e.g. sbatch --export=ALL,RUN_TAG=2026-10-16 run_model_array.sh}

NETS=(GGNet PathNet PPNet)
STRATS=(random priority coarsening)
RATIOS=(0.1 0.2)

i=$((SLURM_ARRAY_TASK_ID - 1))
FOLD=$((i % 5 + 1))
CFG=$((i / 5))
NET=${NETS[$((CFG / 6))]}
STRAT=${STRATS[$((CFG % 6 / 2))]}
RATIO=${RATIOS[$((CFG % 2))]}
PREFIX=${NET}_${STRAT}_r${RATIO}

base=your/base/dir/DriverJinn/
cd $base

module load conda
conda activate conda_envs/envs/gnn_env
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Task ${SLURM_ARRAY_TASK_ID}: ${PREFIX} fold ${FOLD} on $(hostname)"

python -m model.train_model \
    --dataset_file curvature_output/${NET}_contrastive_v2_${STRAT}_r${RATIO}_stratified5CV.pkl \
    --split_file baseline_comparison/HGDC/data/${NET}/dataset_${NET}_ten_5CV.pkl \
    --split_repeat 0 \
    --inner_val_size 0.15 \
    --num_epochs 1000 \
    --hidden_channels 128 \
    --projection_dim 96 \
    --num_layers 2 \
    --num_heads 4 \
    --num_folds 5 \
    --train_metrics_dir model_results_${RUN_TAG} \
    --model_out_dir trained_models_${RUN_TAG} \
    --specific_folds ${FOLD} \
    --model_out_prefix ${PREFIX}_fold${FOLD} \
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

# After every task has finished, aggregate each configuration (same RUN_TAG):
#   for net in GGNet PathNet PPNet; do for s in random priority coarsening; do for r in 0.1 0.2; do
#     python -m model.aggregate_fold_results --prefix ${net}_${s}_r${r} --results_dir model_results_${RUN_TAG}
#   done; done; done
