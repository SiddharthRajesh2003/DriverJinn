# model

GNN-based cancer driver gene ranking using curvature-aware message passing and contrastive learning.

## Architecture

```
node features (27 dims)
        │
        ▼
CurvatureConstrainedMessagePassing   ← curvature-filtered edges, multi-head attention
  (message_passing.py)
        │
        ▼
CurvatureAwareGNN                    ← multi-pathway (pos/neg/all curvature), multi-layer
  (curvature_aware_gnn.py)            output aggregated via MultiLayerAttention
        │
        ▼
ContrastiveDriverGenePredictor       ← projection heads, contrastive + ranking loss,
  (DriverGenePredictor.py)             post-hoc FP filtering via curvature features
        │
        ▼
ranked gene list + NDCG@50 / AUROC
```

## Files

| File | Description |
|------|-------------|
| `message_passing.py` | `CurvatureConstrainedMessagePassing` — extends PyG's `MessagePassing` with curvature-based edge filtering and configurable attention (standard, edge_feature, bias, gated, hybrid). |
| `curvature_aware_gnn.py` | `CurvatureAwareGNN` — stacks message passing layers across positive/negative/all curvature pathways; supports one-hop and two-hop variants. |
| `multi_layer_attention.py` | `MultiLayerAttention` — memory-efficient aggregation of multi-layer GNN outputs using chunked attention and gradient checkpointing. |
| `DriverGenePredictor.py` | `ContrastiveDriverGenePredictor` — wraps the GNN with projection heads, computes contrastive and ranking losses, produces final driver gene rankings. |
| `support_models.py` | Training utilities: `ProjectionHead`, `RankingLoss`, `WarmupScheduler`, `EarlyStopping`, `EMA`. |
| `train_model.py` | Training entry point — k-fold cross-validation loop, loss computation, metric tracking, checkpoint management, and learning curve visualization. |
| `aggregate_fold_results.py` | Post-training utility — aggregates metrics across parallel fold runs, computes mean/std/confidence intervals, generates summary plots. |
| `hyperparameter_search.py` | Optuna-based hyperparameter tuning with TPE sampler and HyperBand pruning. |

## Training Pipeline

1. Load pickle from `graph_builder` (original graph + two augmented views)
2. k-fold cross-validation; per fold: fit a fresh `StandardScaler` on training nodes only
3. Forward pass on both augmented views → contrastive loss (NT-Xent)
4. Forward pass on original graph → ranking loss (ListMLE / approx NDCG)
5. Combined loss with `contrastive_weight` mixing coefficient
6. Gradient accumulation, EMA weight averaging, early stopping on smoothed NDCG@50

## Current Best Hyperparameters

| Parameter | Value |
|-----------|-------|
| `hidden_channels` | 128 |
| `num_heads` | 4 |
| `temperature` | 0.3 |
| `dropout` | 0.3 |
| `contrastive_weight` | 0.2 |
| `lr` | 5e-4 |
| `weight_decay` | 5e-4 |
| `scheduler_factor` | 0.5 |
| `scheduler_patience` | 5 |

Mean val NDCG@50 ~0.46 · Best fold NDCG@50 ~0.52 · Best fold AUROC ~0.81
