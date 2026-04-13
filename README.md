# DriverJinn

This project is still in development with fixes being continuously generated.

This model is currently being trained on NVIDIA H100 GPU.

A Graph Neural Network framework for cancer driver gene prediction using curvature-enhanced graph representations and contrastive learning.

---

## End-to-End Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 1 — Preprocessing (curvature_pipeline.py)                             │
│                                                                             │
│  data/*.pkl  ──►  Load data (features, edge_index, labels, node_names)      │
│                 ↓                                                           │
│             Build NetworkX graph  (build_network.py)                        │
│                 ↓                                                           │
│             Compute curvatures   (curvature_calculator.py)                  │
│               • Ollivier Ricci (discrete)                                   │
│               • Forman (combinatorial)                                      │
│                 ↓                                                           │
│             Integrate features   (curvature_integration.py)                 │
│               • 58 original dims + 12 curvature dims + 3 summary dims       │
│               • = 73 total feature dimensions per node                      │
│                 ↓                                                           │
│             Create 5-fold stratified splits (seed=42)                       │
│                 ↓                                                           │
│             Generate augmented views  (schur_complement.py)                 │
│               • Eliminates ~20% of nodes, adds fill-in edges                │
│               • Produces 2 augmented graph variants                         │
│                 ↓                                                           │
│  Output: curvature_output/GGNet_contrastive_v2_random_r0.2.pkl              │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 2 — Hyperparameter Search  [optional]  (hyperparameter_search.py)     │
│                                                                             │
│  • Optuna Bayesian optimization (TPE sampler, Hyperband pruning)            │
│  • 50 trials, 1 fold, optimizes val NDCG@50                                 │
│  • Search space: LR, hidden_channels, num_layers, num_heads,                │
│    contrastive_weight, dropout, cosine_T0, weight_decay, focal_gamma        │
│  Output: best_params_*.json                                                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 3 — Model Training  (train_model.py)                                  │
│                                                                             │
│  FOR each fold k = 1 … 5:                                                   │
│    ├── Load preprocessed data + augmented views                             │
│    ├── Fit StandardScaler on fold k's training nodes only (no leakage)      │
│    ├── Instantiate ContrastiveDriverGenePredictor  (DriverGenePredictor.py) │
│    │     • CurvatureAwareGNN encoder (dual-pathway: pos/neg curvature)      │
│    │     • Multi-head attention aggregator                                  │
│    │     • Projection head (contrastive)                                    │
│    │     • Ranking head (scoring)                                           │
│    │                                                                        │
│    └── Training loop (up to num_epochs):                                    │
│          ┌──────────────────────────────────────────────────────────┐       │
│          │  Augmented view 1  ──►  encoder  ──►  projector          │       │
│          │  Augmented view 2  ──►  encoder  ──►  projector   ──►    │       │
│          │                        NT-Xent / InfoNCE loss (L_c)      │       │
│          │                                                          │       │
│          │  Original graph   ──►  encoder  ──►  ranking head ──►    |       │
│          │                        BPR ranking loss  (L_r)           │       │
│          │                          • Curriculum hard negatives     │       │
│          │                          • Focal weighting               │       │
│          │                          • hard_frac: 25%→75% over time  │       │
│          │                                                          │       │
│          │  Total loss = α·L_c + (1-α)·L_r                          │       │
│          └──────────────────────────────────────────────────────────┘       │
│                                                                             │
│          Optimizations:                                                     │
│            • ReduceLROnPlateau (factor, patience configurable)               │
│            • Linear warmup (10 epochs)                                      │
│            • Gradient accumulation (8 steps)                                │
│            • Mixed precision (FP16)                                         │
│            • EMA (decay=0.999)                                              │
│            • Gradient checkpointing                                         │
│                                                                             │
│          Validation every val_freq epochs:                                  │
│            • Metrics: NDCG@50, AUROC, AUPRC, MRR, P@50                      │
│            • Early stopping on smoothed NDCG@50 (EMA α=0.3)                 │
│            • Best checkpoint saved at peak val NDCG@50                      │
│                                                                             │
│  Output per fold:                                                           │
│    • trained_models/.../fold_k_best_model.pt                                │
│    • model_results/.../fold_k_metrics.csv                                   │
│    • model_results/.../fold_k_all_genes_scored.csv                          │
│    • model_results/.../fold_k_training_history.csv                          │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 4 — Aggregation  (aggregate_fold_results.py)                          │
│                                                                             │
│  • Aggregate per-fold gene scores:                                          │
│      - Mean score, std, median score                                        │
│      - Median rank (Borda count)                                            │
│      - Significance count (# folds where gene is significant)               │
│      - consensus_significant: significant in ≥50% of folds                  │
│  • Aggregate metrics: mean ± std NDCG@50, AUROC, AUPRC across folds         │
│  • Novel predictions only: filters to true_labels == 0                      │
│    (excludes known drivers from ranked output)                              │
│                                                                             │
│  Output:                                                                    │
│    • model_results/.../aggregated_all_genes.csv                             │
│    • model_results/.../aggregated_metrics_summary.csv                       │
│    • model_results/.../aggregated_training_curves.png                       │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Table of Contents

- [Installation](#installation)
- [Project Structure](#project-structure)
- [Pipeline Overview](#pipeline-overview)
  - [Step 1: Graph Curvature Enhancement](#step-1-graph-curvature-enhancement)
  - [Step 2: Hyperparameter Search](#step-2-hyperparameter-search)
  - [Step 3: Model Training](#step-3-model-training)
  - [Step 4: Result Aggregation](#step-4-result-aggregation)
- [HPC Usage with SLURM](#hpc-usage-with-slurm)
- [Output Files](#output-files)

---

## Installation

### Conda Environment

```bash
conda env create -p conda_envs/envs/gnn_env -f environment.yaml
conda activate conda_envs/envs/gnn_env
```

### Manual Installation

```bash
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu118
pip install torch-geometric
pip install GraphRicciCurvature
pip install matplotlib seaborn pandas scikit-learn optuna
```

### Docker Installation

```bash
docker pull cydarthvader/driverjinn:latest
```

### Key Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| PyTorch | 2.7.1 | Deep learning framework |
| PyTorch Geometric | 2.6.1 | Graph neural network layers |
| Optuna | 4.6.0 | Hyperparameter optimization |
| scikit-learn | 1.7.2 | Stratified k-fold, metrics |
| GraphRicciCurvature | - | Ollivier/Forman curvature computation |
| networkx | 3.5 | Graph analysis |

---

## Project Structure

```
GNNDriver/
├── data/                       # Input gene interaction network datasets
├── graph_builder/              # Graph curvature enhancement pipeline
│   ├── curvature_pipeline.py   #   Main preprocessing entry point
│   ├── build_network.py        #   Network construction
│   ├── curvature_calculator.py #   Ollivier & Forman curvature
│   ├── curvature_integration.py#   Curvature feature integration
│   └── schur_complement.py     #   Schur complement graph augmentation
├── model/                      # Model training and evaluation
│   ├── train_model.py          #   Main training entry point
│   ├── DriverGenePredictor.py  #   Model architecture
│   ├── curvature_aware_gnn.py  #   Curvature-aware GNN layers
│   ├── message_passing.py      #   Message passing mechanisms
│   ├── multi_layer_attention.py#   Multi-head attention layers
│   ├── support_models.py       #   EMA, RankingLoss, EarlyStopping, WarmupScheduler
│   ├── hyperparameter_search.py#   Optuna-based hyperparameter optimization
│   └── aggregate_fold_results.py#  Cross-fold result aggregation
├── Downstream/                 # Downstream analysis and visualizations
│   ├── plots/                  #   Generated figures
│   └── cosmic/                 #   COSMIC cancer gene census reference data
├── utils/
│   └── logging_manager.py      # Logging configuration
├── curvature_output/           # Preprocessed graph datasets (generated)
├── model_results/              # Training results and plots (generated)
├── trained_models/             # Model checkpoints (generated)
├── hyperparam_results/         # Hyperparameter search outputs (generated)
├── run_model_array.sh          # SLURM script: parallel fold training (array job)
├── run_hyperparam_search.sh    # SLURM script: hyperparameter search
├── preprocess_script.sh        # SLURM script: graph preprocessing
├── environment.yaml            # Conda environment specification
├── requirements.txt            # Pip requirements
└── Dockerfile                  # Docker configuration
```

---

## Pipeline Overview

The full pipeline consists of four sequential steps:

### Step 0: Dataset Format

The graph dataset should be a dictionary object stored within a pickle file:

```bash
feature <class 'torch.Tensor'>
node_name <class 'list'>
edge_index <class 'torch.Tensor'>
feature_name <class 'list'>
label <class 'torch.Tensor'>
```

### Step 1: Graph Curvature Enhancement

Computes Ollivier and Forman curvatures on gene interaction networks, generates augmented graph views via Schur complement elimination, and creates stratified k-fold cross-validation splits.

```bash
python -m graph_builder.curvature_pipeline \
    --dataset_file data/dataset_GGNet.pkl \
    --method both \
    --augment \
    --num_views 2 \
    --elimination_ratio 0.1 \
    --strategy random \
    --use_kfold \
    --n_folds 5
```

#### Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--dataset_file` | str | **Required** | Input dataset pickle file |
| `--output_dir` | str | `./curvature_output` | Output directory |
| `--method` | choice | `both` | Curvature method: `ollivier`, `forman`, `both` |
| `--augment` | flag | - | Generate augmented views via Schur complement |
| `--num_views` | int | 2 | Number of augmented views |
| `--elimination_ratio` | float | 0.2 | Ratio of nodes to eliminate (0.1-0.3) |
| `--strategy` | choice | `priority` | Elimination strategy: `priority`, `random`, `coarsening` |
| `--use_kfold` | flag | - | Use k-fold cross-validation |
| `--n_folds` | int | 5 | Number of CV folds |
| `--random_seed` | int | 42 | Random seed |
| `--no_normalize` | flag | - | Skip curvature normalization |
| `--no_aug_curvature` | flag | - | Skip curvature for augmented graphs |
| `--no_split` | flag | - | Skip train/val split creation |
| `--no_stratify` | flag | - | Disable stratified splitting |
| `--test_size` | float | 0.2 | Test set proportion |
| `--val_size` | float | 0.1 | Validation set proportion |
| `--use_existing_mask` | flag | - | Use existing boolean mask from dataset |
| `--train_ratio_from_mask` | float | 0.8 | Train vs val ratio when using existing mask |

**Output:** A pickle file in `curvature_output/` containing the enhanced graph, augmented views, curvature features, and k-fold split masks.

---

### Step 2: Hyperparameter Search

Runs Optuna-based Bayesian optimization using TPE sampling and Hyperband pruning. Optimizes for NDCG@50 on a single fold for speed.

```bash
python -m model.hyperparameter_search \
    --dataset curvature_output/GGNet_contrastive_v2_random_r0.1.pkl \
    --output_dir hyperparam_results \
    --n_trials 50 \
    --n_folds 1 \
    --epochs_per_trial 500 \
    --patience 30 \
    --study_name GGNet_hypersearch
```

#### Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--dataset` | str | **Required** | Path to curvature-enhanced dataset |
| `--output_dir` | str | `hyperparam_results` | Output directory |
| `--n_trials` | int | 50 | Number of Optuna trials |
| `--n_folds` | int | 1 | Folds per trial (1 recommended for search) |
| `--epochs_per_trial` | int | 500 | Max epochs per trial |
| `--patience` | int | 30 | Early stopping patience |
| `--study_name` | str | auto-generated | Name for the Optuna study |

#### Search Space

| Category | Parameters | Range |
|----------|-----------|-------|
| Architecture | `hidden_channels` | [64, 96, 128, 144] |
| | `projection_dim` | [32, 64, 72, 96] |
| | `num_gnn_layers` | 2-4 |
| | `num_attention_heads` | [1, 2, 4] |
| | `attention_mode` | [standard, edge_feature, bias, gated, hybrid] |
| | `pathway_aggregator` | [attention, concat, mean, hierarchical] |
| | `concat_heads` | [True, False] |
| Regularization | `dropout` | 0.1-0.4 |
| | `negative_slope` | 0.1-0.3 |
| | `temperature` | 0.3-1.0 |
| Optimization | `learning_rate` | 5e-5 to 5e-3 (log) |
| | `weight_decay` | 1e-6 to 1e-4 (log) |
| | `gradient_accumulation_steps` | [8, 16, 24, 36] |
| | `use_ema` | [True, False] |
| | `ema_decay` | 0.99-0.999 |
| Loss | `contrastive_weight` | 0.1-0.4 |
| | `ranking_loss_scale` | 5.0-15.0 |
| | `ranking_loss_type` | [bpr, pairwise] |
| | `ranking_margin` | 0.3-1.0 |
| | `use_focal` | [True, False] |
| | `focal_gamma` | 1.0-3.0 |
| Message Passing | `aggregation` | [add, mean, max] |
| Scheduler | `scheduler_patience` | 20-150 |
| | `scheduler_factor` | 0.5-0.8 |

**Output:** Best parameters JSON, all trial results CSV, and optimization plots in `hyperparam_results/`.

---

## Model Architecture (DriverGenePredictor.py)

```
Input: Node features (73 dims) + edge_index + curvature edge attributes
    │
    ▼
CurvatureAwareGNN  (curvature_aware_gnn.py)
    • Dual-pathway: positive curvature edges / negative curvature edges
    • Multi-hop: 1-hop + 2-hop neighborhoods
    • Graph Attention Networks (GAT) with hybrid attention modes
    ↓ gradient checkpointed
HybridAggregator  (multi_layer_attention.py)
    • Multi-pathway attention-based aggregation
    ↓
  ┌────────────────────────┐
  │                        │
ProjectionHead         RankingHead
(contrastive loss)     (BPR ranking loss)
  │                        │
  ▼                        ▼
NT-Xent / InfoNCE     Scalar gene score
(aligns views)        (driver likelihood)
```

---

## Loss Function

```
Total Loss = α · L_contrastive + (1 - α) · L_ranking

L_contrastive = NT-Xent (InfoNCE) between augmented view pairs
                temperature τ = 0.3–0.4

L_ranking = BPR (Bayesian Personalized Ranking)
            - Curriculum hard negatives: hard_frac ramps 25%→75% over training
            - Global hard mining: torch.topk over all non-driver nodes
            - Focal weighting: (1 - sigmoid(diff))^γ down-weights easy pairs
```

---

### Step 3: Model Training

Trains the contrastive driver gene predictor with ranking-based loss. Supports training all folds sequentially or specific folds individually (for parallel HPC execution).

```bash
python -m model.train_model \
    --dataset_file curvature_output/GGNet_contrastive_v2_random_r0.2.pkl \
    --num_epochs 1000 \
    --hidden_channels 144 \
    --projection_dim 96 \
    --num_layers 3 \
    --num_heads 4 \
    --specific_folds 1 \
    --model_out_prefix GGNet_random_r0.2_fold1 \
    --temperature 0.31 \
    --dropout 0.325 \
    --negative_slope 0.285 \
    --attention_mode standard \
    --pathway_aggregator hierarchical \
    --gradient_accumulation_steps 64 \
    --ranking_margin 0.516 \
    --ranking_loss_scale 6.17 \
    --mixed_precision \
    --scheduler_patience 300 \
    --scheduler_factor 0.786 \
    --early_stopping_patience 300 \
    --concat_heads \
    --ranking_loss_type bpr \
    --focal_gamma 2.7 \
    --use_focal \
    --contrastive_weight 0.35 \
    --weight_decay 1e-5 \
    --learning_rate 5e-4 \
    --seed 42
```

#### Arguments

**Data & I/O:**

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--dataset_file` | str | **Required** | Input curvature-enhanced dataset pickle file |
| `--train_metrics_dir` | str | `model_results` | Output directory for metrics and plots |
| `--model_out_dir` | str | `trained_models` | Directory for model checkpoints |
| `--model_out_prefix` | str | `""` | Prefix for output filenames |

**Training Configuration:**

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--num_folds` | int | 5 | Number of cross-validation folds |
| `--specific_folds` | int list | None | Train only specific folds (space-separated, 1-indexed) |
| `--num_epochs` | int | 200 | Training epochs per fold |
| `--seed` | int | 42 | Random seed for reproducibility |

**Model Architecture:**

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--hidden_channels` | int | 256 | Hidden layer dimensions |
| `--projection_dim` | int | 128 | Contrastive projection dimensions |
| `--num_layers` | int | 3 | Number of GNN layers |
| `--num_heads` | int | 4 | Number of attention heads |
| `--concat_heads` | flag | False | Concatenate attention heads (vs average) |
| `--negative_slope` | float | 0.2 | LeakyReLU negative slope in GNN layers |
| `--attention_mode` | choice | `hybrid` | `standard`, `edge_feature`, `bias`, `gated`, `hybrid` |
| `--pathway_aggregator` | choice | `attention` | `attention`, `concat`, `hierarchical`, `mean` |
| `--aggregation` | str | `add` | Message passing aggregation method |

**Optimization:**

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--learning_rate` | float | 1e-3 | AdamW learning rate |
| `--weight_decay` | float | 1e-5 | AdamW weight decay (L2 regularization) |
| `--dropout` | float | 0.2 | Dropout rate |
| `--temperature` | float | 0.4 | NT-Xent contrastive loss temperature |
| `--gradient_accumulation_steps` | int | 8 | Gradient accumulation steps |
| `--mixed_precision` | flag | - | Enable FP16 mixed precision training |
| `--decay` | float | 0.999 | EMA decay rate for model weights |

**Loss Configuration:**

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--contrastive_weight` | float | 0.1 | Contrastive loss weight; ranking weight = 1 - this |
| `--ranking_loss_type` | choice | `bpr` | `pairwise`, `sampled_pairwise`, `bpr`, `listwise`, `approxndcg` |
| `--ranking_loss_scale` | float | 10.0 | Ranking loss scaling factor |
| `--ranking_margin` | float | 0.5 | Pairwise ranking margin |
| `--focal_gamma` | float | 2.0 | Focal loss gamma (higher = focus on hard examples) |
| `--use_focal` | flag | - | Enable focal weighting in ranking loss |

**Scheduler & Early Stopping:**

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--scheduler_patience` | int | 50 | Epochs to wait before reducing LR |
| `--scheduler_factor` | float | 0.7 | Factor to reduce LR by on plateau |
| `--early_stopping_patience` | int | 50 | Epochs without improvement before stopping |
| `--validation_frequency` | int | 10 | Validate every N epochs |

**Memory Optimization:**

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--attention_chunk_size` | int | 1000 | Chunk size for attention computation |
| `--max_augmented_views` | int | 3 | Max augmented views in memory |
| `--max_views_per_step` | int | 2 | Max views per training step |
| `--emergency_mode` | flag | - | Aggressive memory reduction (auto-sets params) |
| `--reduce_model_size` | flag | - | Reduce to hidden=64, proj=32, layers=2, heads=1 |
| `--return_all_layers` | flag | - | Return all GNN layer outputs |

**Output per fold:** Model checkpoint, training metrics CSV, training history pickle, training curves plot, metrics comparison plot, and scored genes CSV.

---

### Step 4: Result Aggregation

Combines results from parallel fold training runs into cross-validation summary statistics, consensus gene predictions, and combined plots.

```bash
python -m model.aggregate_fold_results \
    --prefix GGNet_random_r0.2 \
    --num_folds 5 \
    --results_dir model_results
```

#### Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--prefix` | str | **Required** | Base prefix for fold directories |
| `--num_folds` | int | 5 | Number of folds to aggregate |
| `--results_dir` | str | `model_results` | Directory containing fold subdirectories |
| `--output_dir` | str | auto | Output directory (default: `<results_dir>/<prefix>_aggregated`) |

**Output:**
- Cross-validation summary statistics (mean/std AUROC, NDCG@50, Precision@50, etc.)
- Aggregated gene scores with consensus significance analysis
- Combined training curves across all folds
- Metrics comparison bar and box plots

---

## HPC Usage with SLURM

### Parallel Training (one fold per GPU)

Use `run_model_array.sh` to train each fold independently via SLURM job arrays:

```bash
sbatch run_model_array.sh
```

This submits 5 jobs (`--array=1-5`), each training a single fold on its own GPU. After all jobs complete, run the aggregation script to combine results.

### Hyperparameter Search

```bash
sbatch run_hyperparam_search.sh
```

### Preprocessing (all network/strategy combinations)

```bash
sbatch preprocess_script.sh
```

---

## Output Files

### Per-Fold Training Output

Located in `model_results/<prefix>/`:

| File | Description |
|------|-------------|
| `<prefix>_fold_<N>_metrics.csv` | Validation metrics for the fold |
| `<prefix>_fold_<N>_history.pkl` | Full training history (loss, LR, all metrics per epoch) |
| `<prefix>_fold_<N>_all_genes_scored.csv` | All genes ranked by predicted driver score |
| `<prefix>_training_curves.png` | Training/validation curves plot |
| `<prefix>_metrics_comparison.png` | Metrics comparison plot |

### Aggregated Output

Located in `model_results/<prefix>_aggregated/`:

| File | Description |
|------|-------------|
| `<prefix>_cv_summary.txt` | Cross-validation mean/std for all metrics |
| `<prefix>_all_genes_aggregate_scores.csv` | All genes with mean score, std, median rank, consensus significance |
| `<prefix>_consensus_significant_genes.csv` | Genes significant in >=50% of folds |
| `<prefix>_aggregate_summary.txt` | Summary statistics for aggregated predictions |
| `<prefix>_combined_training_curves.png` | Training curves overlaid across folds |
| `<prefix>_metrics_comparison.png` | Cross-fold metrics comparison |

### Hyperparameter Search Output

Located in `hyperparam_results/`:

| File | Description |
|------|-------------|
| `<study>_best_params.json` | Best hyperparameter configuration |
| `<study>_results.csv` | All trial results |
| `<study>_all_results.json` | Detailed trial results with fold scores |
| `<study>_plots.png` | Optimization history and parameter importance |
