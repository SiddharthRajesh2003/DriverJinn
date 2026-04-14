#!/bin/bash
set -e

COMMAND=$1
shift

case "$COMMAND" in
    preprocess)
        if [[ $# -eq 0 || "$1" == "--help" || "$1" == "-h" ]]; then
            echo "Usage: docker run cydarthvader/driverjinn preprocess [options]"
            echo ""
            echo "Options:"
            echo "  --dataset_file         Input dataset pickle file (required)"
            echo "  --output_dir           Output directory [default: ./curvature_output]"
            echo "  --method               Curvature method: ollivier | forman | both"
            echo "  --augment              Enable graph augmentation"
            echo "  --num_views            Number of augmented views [default: 2]"
            echo "  --elimination_ratio    Ratio of edges to eliminate [default: 0.2]"
            echo "  --strategy             Elimination strategy: priority | random | coarsening"
            echo "  --no_aug_curvature     Disable curvature on augmented views"
            echo "  --no_split             Skip train/val/test splitting"
            echo "  --test_size            Test split size [default: 0.2]"
            echo "  --val_size             Validation split size [default: 0.1]"
            echo "  --no_stratify          Disable stratified splitting"
            echo "  --random_seed          Random seed [default: 42]"
            echo "  --use_existing_mask    Use existing train/test mask"
            echo "  --train_ratio_from_mask  Train ratio when using existing mask [default: 0.8]"
            echo "  --use_kfold            Use k-fold cross-validation"
            echo "  --n_folds              Number of folds [default: 5]"
            exit 0
        fi
        python -m graph_builder.curvature_pipeline "$@"
        ;;
    train_model)
        if [[ $# -eq 0 || "$1" == "--help" || "$1" == "-h" ]]; then
            echo "Usage: docker run cydarthvader/driverjinn train_model [options]"
            echo ""
            echo "Options:"
            echo "  --dataset_file              Path to preprocessed pickle file (required)"
            echo "  --train_metrics_dir         Directory to save training metrics"
            echo "  --model_out_dir             Directory to save model checkpoints"
            echo "  --model_out_prefix          Prefix for model output files"
            echo "  --num_folds                 Number of CV folds [default: 5]"
            echo "  --specific_folds            Which folds to run (e.g. 1 2 3)"
            echo "  --num_epochs                Number of training epochs [default: 200]"
            echo "  --hidden_channels           Hidden layer size [default: 256]"
            echo "  --projection_dim            Contrastive projection dim [default: 128]"
            echo "  --num_heads                 Attention heads [default: 4]"
            echo "  --num_layers                GNN layers [default: 3]"
            echo "  --concat_heads              Concatenate attention heads"
            echo "  --attention_mode            Attention mode: hybrid | ... [default: hybrid]"
            echo "  --pathway_aggregator        Aggregator type [default: attention]"
            echo "  --aggregation               Graph aggregation: add | max | mean [default: add]"
            echo "  --negative_slope            LeakyReLU negative slope [default: 0.2]"
            echo "  --attention_chunk_size      Chunk size for attention [default: 1000]"
            echo "  --temperature               Contrastive loss temperature [default: 0.4]"
            echo "  --contrastive_weight        Weight of contrastive loss [default: 0.1]"
            echo "  --dropout                   Dropout rate [default: 0.2]"
            echo "  --learning_rate             Learning rate [default: 1e-3]"
            echo "  --weight_decay              Weight decay [default: 1e-5]"
            echo "  --scheduler_patience        LR scheduler patience [default: 50]"
            echo "  --scheduler_factor          LR reduction factor [default: 0.7]"
            echo "  --early_stopping_patience   Early stopping patience [default: 50]"
            echo "  --ranking_loss_type         Ranking loss: bpr | margin [default: bpr]"
            echo "  --ranking_loss_scale        Ranking loss scale [default: 10.0]"
            echo "  --ranking_loss_samples      Ranking loss samples [default: 256]"
            echo "  --ranking_margin            Margin for ranking loss [default: 0.5]"
            echo "  --focal_gamma               Focal loss gamma [default: 2.0]"
            echo "  --use_focal                 Use focal loss"
            echo "  --gradient_accumulation_steps  Gradient accumulation steps [default: 8]"
            echo "  --mixed_precision           Enable mixed precision training"
            echo "  --max_views_per_step        Max augmented views per step [default: 2]"
            echo "  --max_augmented_views       Max augmented views [default: 3]"
            echo "  --validation_frequency      Validate every N epochs [default: 10]"
            echo "  --decay                     EMA decay rate [default: 0.999]"
            echo "  --emergency_mode            Enable emergency/debug mode"
            echo "  --reduce_model_size         Reduce model size"
            echo "  --return_all_layers         Return features from all GNN layers"
            echo "  --seed                      Random seed [default: 42]"
            exit 0
        fi
        python -m model.train_model "$@"
        ;;
    *)
        echo "Usage: docker run cydarthvader/driverjinn [preprocess|train_model] [options]"
        echo ""
        echo "Commands:"
        echo "  preprocess   Run curvature pipeline (graph_builder.curvature_pipeline)"
        echo "  train_model  Run model training (model.train_model)"
        echo ""
        echo "Run with --help for command-specific options:"
        echo "  docker run cydarthvader/driverjinn preprocess --help"
        echo "  docker run cydarthvader/driverjinn train_model --help"
        exit 1
        ;;
esac
