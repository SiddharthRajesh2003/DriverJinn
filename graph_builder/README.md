# graph_builder

Constructs the protein-protein interaction (PPI) network, computes Ricci curvature features, and produces augmented graph views for contrastive learning.

## Data Flow

```
raw edges + node info
        │
        ▼
  build_network.py       ← NetworkX graph, saved as pickle
        │
        ▼
curvature_calculator.py  ← Ollivier-Ricci and Forman-Ricci per edge
        │
        ▼
curvature_integration.py ← edge curvatures → node features (mean/std/min/max/median)
        │                   concatenated with original node features (12 + 3×5 = 27 dims)
        ▼
schur_complement.py      ← augmented graph views via random Schur complement elimination
        │
        ▼
   pickle output          ← consumed by model/train_model.py
```

`curvature_pipeline.py` orchestrates all steps above end-to-end and is the entry point for HPC cluster runs.

## Files

| File | Description |
|------|-------------|
| `build_network.py` | `Network` class — converts edge indices and node metadata into a NetworkX graph; handles pickle save/load. |
| `curvature_calculator.py` | `EdgeCurvature` class — wraps GraphRicciCurvature to compute Ollivier-Ricci and Forman-Ricci curvature for every edge. |
| `curvature_integration.py` | `CurvatureFeatureIntegrator` class — aggregates edge curvatures into per-node features and appends them to the original feature matrix. Fits a `StandardScaler` on training nodes. |
| `schur_complement.py` | `SchurComplementAugmentation` class — implements Algorithm 1 (Gaussian elimination with clique approximation) to generate augmented graph views for contrastive learning. Supports priority-based, random, and coarsening elimination strategies. |
| `curvature_pipeline.py` | End-to-end pipeline orchestrator. Patches multiprocessing for SLURM/PBS/LSF environments, then runs network build → curvature → integration → augmentation. |

## Output

The pipeline writes a pickle containing:
- original graph (`Data` object)
- augmented view 1 and view 2 (Schur complement graphs)
- integrated node feature matrix
- curvature edge attributes
