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
            echo "  --split_file                HGDC dataset with released 10x5CV splits (e.g. baseline_comparison/HGDC/data/GGNet/dataset_GGNet_ten_5CV.pkl)"
            echo "  --split_repeat              Which of the 10 HGDC 5-fold repeats to use [default: 0]"
            echo "  --inner_val_size            Fraction of each training fold held out for early stopping [default: 0.15]"
            echo "  --train_metrics_dir         Directory to save training metrics"
            echo "  --model_out_dir             Directory to save model checkpoints"
            echo "  --model_out_prefix          Prefix for model output files"
            echo "  --num_folds                 Number of CV folds [default: 5]"
            echo "  --specific_folds            Which folds to run (e.g. 1 2 3)"
            echo "  --num_epochs                Number of training epochs [default: 1200]"
            echo "  --hidden_channels           Hidden layer size [default: 128]"
            echo "  --projection_dim            Contrastive projection dim [default: 96]"
            echo "  --num_heads                 Attention heads [default: 4]"
            echo "  --num_layers                GNN layers [default: 2]"
            echo "  --concat_heads              Concatenate attention heads"
            echo "  --attention_mode            Attention mode: standard | edge_feature | bias | gated | hybrid [default: hybrid]"
            echo "  --pathway_aggregator        Aggregator type: attention | concat | hierarchical | mean [default: attention]"
            echo "  --aggregation               Graph aggregation: add | max | mean [default: max]"
            echo "  --negative_slope            LeakyReLU negative slope [default: 0.2]"
            echo "  --attention_chunk_size      Chunk size for attention [default: 1000]"
            echo "  --temperature               Contrastive loss temperature [default: 0.3]"
            echo "  --contrastive_weight        Weight of contrastive loss [default: 0.2]"
            echo "  --dropout                   Dropout rate [default: 0.3]"
            echo "  --learning_rate             Learning rate [default: 5e-4]"
            echo "  --weight_decay              Weight decay [default: 5e-4]"
            echo "  --scheduler_patience        LR scheduler patience [default: 5]"
            echo "  --scheduler_factor          LR reduction factor [default: 0.7]"
            echo "  --early_stopping_patience   Early stopping patience [default: 15]"
            echo "  --ranking_loss_type         Ranking loss: pairwise | sampled_pairwise | bpr | listwise | approxndcg [default: bpr]"
            echo "  --ranking_loss_scale        Ranking loss scale [default: 5]"
            echo "  --ranking_loss_samples      Ranking loss samples [default: 512]"
            echo "  --ranking_margin            Margin for ranking loss [default: 0.5]"
            echo "  --focal_gamma               Focal loss gamma [default: 2.0]"
            echo "  --use_focal                 Use focal loss"
            echo "  --gradient_accumulation_steps  Gradient accumulation steps [default: 8]"
            echo "  --mixed_precision           Enable mixed precision training"
            echo "  --max_views_per_step        Max augmented views per step [default: 2]"
            echo "  --max_augmented_views       Max augmented views [default: 2]"
            echo "  --validation_frequency      Validate every N epochs [default: 10]"
            echo "  --decay                     EMA decay rate [default: 0.999]"
            echo "  --emergency_mode            Enable emergency/debug mode"
            echo "  --reduce_model_size         Reduce model size"
            echo "  --return_all_layers         Return features from all GNN layers"
            echo "  --seed                      Random seed [default: 42]"
            echo ""
            echo "Settings used for the reported results (run_model_array.sh), shown for GGNet, random, r0.1, fold 1:"
            echo "  docker run cydarthvader/driverjinn train_model \\"
            echo "    --dataset_file curvature_output/GGNet_contrastive_v2_random_r0.1_stratified5CV.pkl \\"
            echo "    --split_file baseline_comparison/HGDC/data/GGNet/dataset_GGNet_ten_5CV.pkl \\"
            echo "    --split_repeat 0 --inner_val_size 0.15 --num_epochs 1000 --hidden_channels 128 \\"
            echo "    --projection_dim 96 --num_layers 2 --num_heads 4 --num_folds 5 \\"
            echo "    --train_metrics_dir model_results --model_out_dir trained_models \\"
            echo "    --specific_folds 1 --model_out_prefix GGNet_random_r0.1_fold1 \\"
            echo "    --temperature 0.3 --dropout 0.3 --attention_mode hybrid --pathway_aggregator hierarchical \\"
            echo "    --aggregation max --negative_slope 0.25 --attention_chunk_size 1000 \\"
            echo "    --gradient_accumulation_steps 8 --mixed_precision --early_stopping_patience 15 \\"
            echo "    --validation_frequency 20 --scheduler_patience 5 --scheduler_factor 0.5 --concat_heads \\"
            echo "    --ranking_loss_type bpr --ranking_loss_scale 5 --ranking_loss_samples 512 \\"
            echo "    --contrastive_weight 0.2 --learning_rate 5e-4 --weight_decay 5e-4"
            exit 0
        fi
        python -m model.train_model "$@"
        ;;
    aggregate)
        if [[ $# -eq 0 || "$1" == "--help" || "$1" == "-h" ]]; then
            echo "Usage: docker run cydarthvader/driverjinn aggregate [options]"
            echo ""
            echo "Options:"
            echo "  --prefix               Base prefix for fold directories, e.g. GGNet_random_r0.1 (required)"
            echo "  --num_folds            Number of folds to aggregate [default: 5]"
            echo "  --results_dir          Directory containing fold subdirectories [default: model_results]"
            echo "  --output_dir           Output directory [default: model_results/<prefix>_aggregated]"
            exit 0
        fi
        python -m model.aggregate_fold_results "$@"
        ;;
    hyperparam_search)
        if [[ $# -eq 0 || "$1" == "--help" || "$1" == "-h" ]]; then
            echo "Usage: docker run cydarthvader/driverjinn hyperparam_search [options]"
            echo ""
            echo "Options:"
            echo "  --dataset              Path to preprocessed pickle file (required)"
            echo "  --output_dir           Output directory for results [default: hyperparam_results]"
            echo "  --n_trials             Number of Optuna trials [default: 50]"
            echo "  --n_folds              CV folds per trial [default: 1]"
            echo "  --epochs_per_trial     Max epochs per trial [default: 500]"
            echo "  --patience             Early stopping patience [default: 30]"
            echo "  --study_name           Optuna study name [default: None]"
            echo "  --seed                 Random seed [default: 42]"
            exit 0
        fi
        python -m model.hyperparameter_search "$@"
        ;;
    *)
        echo "Usage: docker run cydarthvader/driverjinn [preprocess|train_model|aggregate|hyperparam_search] [options]"
        echo ""
        echo "Commands:"
        echo "  preprocess        Run curvature pipeline (graph_builder.curvature_pipeline)"
        echo "  train_model       Run model training (model.train_model)"
        echo "  aggregate         Aggregate fold results (model.aggregate_fold_results)"
        echo "  hyperparam_search Run hyperparameter search (model.hyperparameter_search)"
        echo ""
        echo "Run with --help for command-specific options:"
        echo "  docker run cydarthvader/driverjinn preprocess --help"
        echo "  docker run cydarthvader/driverjinn train_model --help"
        echo "  docker run cydarthvader/driverjinn aggregate --help"
        echo "  docker run cydarthvader/driverjinn hyperparam_search --help"
        exit 1
        ;;
esac
