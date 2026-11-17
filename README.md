# DriverJinn

This project is still in development with fixes being continuously generated.

The reported results were trained on NVIDIA H100 GPUs.

A Graph Neural Network framework for cancer driver gene prediction using curvature-enhanced graph representations and contrastive learning.

## Table of Contents

- [Results (HGDC protocol, October 2026)](#results-hgdc-protocol-october-2026)
  - [Baseline comparison](#baseline-comparison-same-hgdc-splits)
- [End-to-End Pipeline](#end-to-end-pipeline)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Pipeline Overview](#pipeline-overview)
  - [Step 1: Graph Curvature Enhancement](#step-1-graph-curvature-enhancement)
  - [Step 2: Hyperparameter Search](#step-2-hyperparameter-search)
  - [Step 3: Model Training](#step-3-model-training)
  - [Step 4: Result Aggregation](#step-4-result-aggregation)
  - [Step 5: Candidate Adjustment and Figures](#step-5-candidate-adjustment-and-figures)
- [Docker Usage](#docker-usage)
- [Apptainer Usage](#apptainer-usage)
- [HPC Usage with SLURM](#hpc-usage-with-slurm)
- [Output Files](#output-files)


## Results (HGDC protocol, October 2026)

> **Note:** These are computational predictions and have not been validated experimentally. Results come from `model_results_2026-10-16/` (18 configurations × 5 folds). Figures and tables are produced by `Downstream/publication_figures.qmd`. The full write-up is in [RESULTS.md](RESULTS.md), and the methods are in [MODEL_MATH.md](MODEL_MATH.md).
>
> **Earlier results are withdrawn.** The March–May 2026 findings that were listed here (NDCG@50 = 0.478, the 21 enriched hallmarks, the significance-based consensus genes and the seven-gene chromatin cluster) came from code with three bugs. Gene scores used live weights while metrics used EMA weights. The significance test indexed genes incorrectly. Fold aggregation stacked folds by row position, so every "aggregate" ranking was really fold 1's ranking. Those claims should not be cited.

---

### Evaluation protocol

- **Splits:** HGDC's released 5-fold splits (repeat 0) from `baseline_comparison/HGDC/data/<net>/dataset_<net>_ten_5CV.pkl`. Train and test genes are drawn only from HGDC's labelled genes (796 drivers, 2,187 non-drivers; 1,809 / 1,301 labelled genes are present in GGNet / PathNet). This makes the numbers directly comparable with HGDC, EMOGI and MTGCN.
- **Model selection:** 15% of each fold's training genes (stratified) form an inner validation set used for early stopping. Each test fold is evaluated once, at the end.
- **Unlabelled genes** are never used as labels. They are only scored.
- **Candidate ranking:** genes are ranked by mean within-fold rank, because raw scores are not calibrated across folds. A *consensus candidate* is an unlabelled gene that is among the top 100 unlabelled genes in ≥ 3 of 5 folds. No significance test is applied.
- **Length and degree adjustment:** before candidates are named, each unlabelled gene is ranked only against unlabelled genes in the same coding-length decile and network-degree quintile, and FLAGS genes (Shyr et al. 2014) are removed. A *robust candidate* is an adjusted consensus candidate in ≥ 8 of the 18 configurations.

---

### Test-set performance

Mean ± SD over the 5 HGDC test folds (about 39% of test genes are drivers). AUPRC is average precision:

| Network | Strategy | ρ | AUROC | AUPRC | NDCG@50 | P@50 |
|---------|----------|---|-------|-------|---------|------|
| PathNet | priority | 0.2 | **0.835 ± 0.021** | **0.804 ± 0.029** | 0.880 ± 0.045 | 0.852 ± 0.054 |
| PathNet | priority | 0.1 | 0.829 ± 0.019 | 0.795 ± 0.028 | 0.874 ± 0.030 | 0.852 ± 0.023 |
| PathNet | coarsening | 0.1 | 0.829 ± 0.018 | 0.795 ± 0.023 | 0.878 ± 0.035 | 0.852 ± 0.036 |
| PathNet | coarsening | 0.2 | 0.828 ± 0.017 | 0.794 ± 0.027 | 0.873 ± 0.034 | 0.844 ± 0.038 |
| PathNet | random | 0.2 | 0.822 ± 0.018 | 0.781 ± 0.037 | 0.857 ± 0.057 | 0.840 ± 0.060 |
| PathNet | random | 0.1 | 0.822 ± 0.021 | 0.779 ± 0.040 | 0.851 ± 0.064 | 0.824 ± 0.065 |
| PPNet | coarsening | 0.2 | 0.831 ± 0.026 | 0.773 ± 0.044 | 0.878 ± 0.072 | 0.872 ± 0.046 |
| PPNet | priority | 0.2 | 0.830 ± 0.024 | 0.773 ± 0.043 | 0.884 ± 0.069 | 0.880 ± 0.040 |
| PPNet | coarsening | 0.1 | 0.828 ± 0.030 | 0.766 ± 0.052 | 0.860 ± 0.078 | 0.848 ± 0.050 |
| PPNet | random | 0.1 | 0.829 ± 0.026 | 0.762 ± 0.043 | 0.856 ± 0.063 | 0.844 ± 0.046 |
| PPNet | random | 0.2 | 0.828 ± 0.031 | 0.762 ± 0.048 | 0.859 ± 0.065 | 0.836 ± 0.065 |
| PPNet | priority | 0.1 | 0.825 ± 0.032 | 0.757 ± 0.055 | 0.849 ± 0.083 | 0.844 ± 0.052 |
| GGNet | random | 0.1 | 0.813 ± 0.020 | 0.752 ± 0.022 | 0.886 ± 0.031 | 0.856 ± 0.033 |
| GGNet | random | 0.2 | 0.813 ± 0.020 | 0.751 ± 0.017 | 0.890 ± 0.037 | 0.864 ± 0.043 |
| GGNet | priority | 0.1 | 0.812 ± 0.022 | 0.749 ± 0.019 | 0.896 ± 0.031 | 0.872 ± 0.039 |
| GGNet | priority | 0.2 | 0.813 ± 0.021 | 0.749 ± 0.020 | 0.881 ± 0.031 | 0.852 ± 0.030 |
| GGNet | coarsening | 0.2 | 0.814 ± 0.021 | 0.748 ± 0.022 | 0.886 ± 0.029 | 0.860 ± 0.028 |
| GGNet | coarsening | 0.1 | 0.806 ± 0.033 | 0.734 ± 0.043 | 0.855 ± 0.044 | 0.820 ± 0.051 |

**Key observations:**
- **Network choice dominates.** Mean AUPRC is about 0.79 for PathNet, 0.77 for PPNet and 0.75 for GGNet.
- **The Schur elimination strategy and ratio have at most a modest effect.** Within a network, AUPRC spans 0.016–0.025 across the six settings, comparable to the fold-to-fold SD (0.017–0.055). The one consistent pattern is on PathNet, where random elimination scores about 0.02 lower than priority and coarsening at both ratios; no paired fold-level test has been run on this. No ablation without contrastive augmentation has been run yet, so the contribution of the augmentation itself is untested.
- NDCG@50 and P@50 are much higher than in earlier runs because the test set now contains only labelled genes. They are not comparable with the withdrawn numbers.
- **Tuned hyperparameters did not help.** All 18 configurations were retrained with Optuna-tuned hyperparameters (`model_results_2026-11-13/`). Mean AUROC moved 0.823 → 0.828, AUPRC 0.768 → 0.772 and NDCG@50 0.872 → 0.870, all within the fold SD (~0.03). The table above therefore keeps the October 2026 run. Consensus candidate lists overlap only ~30–60% between the two runs, so individual novel-gene calls should be treated as unstable.

---

### Baseline comparison (same HGDC splits)

All baselines were retrained by `baseline_comparison/run_comparison.py` on HGDC's repeat-0 5-fold splits, so every method sees exactly the same train/test genes. The DriverJinn rows are the reference configuration (random, ρ = 0.2). AUPRC (AP) is average precision; AUPRC (trapz) is the trapezoidal area that HGDC, EMOGI and MTGCN publish. *Folds won* is the number of the 5 test folds on which DriverJinn has the higher AP than that method. Rows are sorted by AP within each network.

| Network | Method | AUROC | AUPRC (AP) | AUPRC (trapz) | P@50 | DriverJinn folds won |
|---------|--------|-------|------------|---------------|------|----------------------|
| GGNet | SCIS-CDG | **0.851 ± 0.017** | **0.807 ± 0.025** | **0.807 ± 0.025** | **0.912 ± 0.046** | 0/5 |
| GGNet | SVM | 0.841 ± 0.022 | 0.779 ± 0.040 | 0.777 ± 0.042 | 0.880 ± 0.042 | 1/5 |
| GGNet | HGDC | 0.827 ± 0.021 | 0.779 ± 0.023 | 0.778 ± 0.023 | 0.892 ± 0.052 | 0/5 |
| GGNet | *DriverJinn* | 0.813 ± 0.020 | 0.751 ± 0.017 | 0.750 ± 0.017 | 0.864 ± 0.043 | — |
| GGNet | MTGCN | 0.818 ± 0.028 | 0.750 ± 0.046 | 0.749 ± 0.047 | 0.824 ± 0.085 | 2/5 |
| GGNet | ChebNet | 0.788 ± 0.025 | 0.734 ± 0.033 | 0.734 ± 0.033 | 0.852 ± 0.063 | 3/5 |
| GGNet | EMOGI | 0.787 ± 0.018 | 0.731 ± 0.027 | 0.730 ± 0.027 | 0.836 ± 0.057 | 4/5 |
| GGNet | GCN | 0.619 ± 0.031 | 0.515 ± 0.033 | 0.510 ± 0.033 | 0.596 ± 0.070 | 5/5 |
| GGNet | GAT | 0.588 ± 0.031 | 0.462 ± 0.028 | 0.467 ± 0.033 | 0.480 ± 0.032 | 5/5 |
| PPNet | SCIS-CDG | **0.861 ± 0.016** | **0.833 ± 0.021** | **0.833 ± 0.022** | **0.948 ± 0.023** | 0/5 |
| PPNet | HGDC | 0.836 ± 0.014 | 0.795 ± 0.014 | 0.794 ± 0.014 | 0.900 ± 0.024 | 1/5 |
| PPNet | SVM | 0.839 ± 0.014 | 0.788 ± 0.031 | 0.786 ± 0.033 | 0.896 ± 0.017 | 1/5 |
| PPNet | *DriverJinn* | 0.828 ± 0.031 | 0.762 ± 0.048 | 0.760 ± 0.048 | 0.836 ± 0.065 | — |
| PPNet | EMOGI | 0.826 ± 0.010 | 0.757 ± 0.036 | 0.762 ± 0.033 | 0.836 ± 0.078 | 2/5 |
| PPNet | ChebNet | 0.826 ± 0.020 | 0.754 ± 0.026 | 0.760 ± 0.023 | 0.848 ± 0.039 | 3/5 |
| PPNet | MTGCN | 0.821 ± 0.015 | 0.739 ± 0.029 | 0.738 ± 0.029 | 0.800 ± 0.051 | 3/5 |
| PPNet | GCN | 0.803 ± 0.019 | 0.734 ± 0.018 | 0.733 ± 0.018 | 0.832 ± 0.030 | 3/5 |
| PPNet | GAT | 0.769 ± 0.049 | 0.677 ± 0.076 | 0.677 ± 0.070 | 0.776 ± 0.103 | 5/5 |
| PathNet | SCIS-CDG | **0.870 ± 0.018** | **0.864 ± 0.014** | **0.863 ± 0.014** | **0.932 ± 0.018** | 0/5 |
| PathNet | MTGCN | 0.850 ± 0.023 | 0.837 ± 0.018 | 0.836 ± 0.018 | 0.912 ± 0.011 | 1/5 |
| PathNet | HGDC | 0.840 ± 0.025 | 0.829 ± 0.030 | 0.828 ± 0.030 | 0.900 ± 0.037 | 1/5 |
| PathNet | SVM | 0.843 ± 0.019 | 0.823 ± 0.017 | 0.822 ± 0.018 | 0.892 ± 0.018 | 1/5 |
| PathNet | EMOGI | 0.828 ± 0.022 | 0.807 ± 0.015 | 0.814 ± 0.017 | 0.884 ± 0.026 | 1/5 |
| PathNet | ChebNet | 0.825 ± 0.011 | 0.804 ± 0.012 | 0.812 ± 0.009 | 0.884 ± 0.036 | 1/5 |
| PathNet | *DriverJinn* | 0.822 ± 0.018 | 0.781 ± 0.037 | 0.780 ± 0.038 | 0.840 ± 0.060 | — |
| PathNet | GCN | 0.801 ± 0.045 | 0.775 ± 0.050 | 0.773 ± 0.051 | 0.860 ± 0.035 | 3/5 |
| PathNet | GAT | 0.788 ± 0.015 | 0.745 ± 0.018 | 0.745 ± 0.020 | 0.816 ± 0.030 | 4/5 |

**Takeaways:**
- **DriverJinn does not beat the state of the art.** SCIS-CDG is best on every network and every metric, and beats DriverJinn on all 15 test folds (mean AP gap 0.056 / 0.071 / 0.082 on GGNet / PPNet / PathNet). HGDC and the SVM baseline also beat DriverJinn on 13 and 12 of 15 folds respectively.
- **DriverJinn is mid-pack.** It clearly beats plain GCN and GAT, and is roughly tied with EMOGI, ChebNet and MTGCN on GGNet and PPNet (within one fold SD). On PathNet it ranks below every method except GCN and GAT.
- **Even the best DriverJinn configuration falls short.** PathNet priority ρ = 0.2 (AP 0.804 ± 0.029, Table above) still trails SCIS-CDG (0.864), MTGCN (0.837), HGDC (0.829) and SVM (0.823) on PathNet.
- AP and trapezoidal AUPRC differ by ≤ 0.008 for every method, so the choice of AUPRC flavour does not change any ranking.
- Per-fold scores are in `baseline_comparison/results/folds_<method>.csv`, means in `summary.csv`, and paired fold-level differences in `pairwise.csv`. SCIS-CDG runs in a separate Linux environment (DGL + CUDA 11.8).

---

### Recovery of known drivers (GGNet, random, ρ = 0.2)

| Measure | Value | Random |
|---------|-------|--------|
| Precision@10 (test fold) | 0.96 | 0.39 |
| Precision@50 | 0.86 | 0.39 |
| Precision@200 | 0.59 | 0.39 |
| Median out-of-fold percentile, CGC Tier 1 (n = 484) | 0.77 | — |
| Median out-of-fold percentile, non-CGC labelled genes (n = 1,223) | 0.37 | — |

Known drivers rank consistently at the top of every fold. ATM, EP300, ARID1A, CREBBP, KMT2C and KMT2D are in the top 10 in most folds. Adjusted consensus lists overlap strongly across the six augmentation settings (Jaccard 0.55–0.85 on GGNet).

---

### Length and hub bias in raw rankings

Raw rankings favour long genes and network hubs, so the candidates below use the adjusted rankings. Earlier versions of this section listed raw candidates such as OBSCN and SYNE1. These are FLAGS genes, so they are no longer candidates.

| Across the 18 configurations | Raw | Adjusted, FLAGS removed |
|------------------------------|-----|-------------------------|
| Consensus candidates per configuration | 58–96 | 51–84 |
| Share in the longest CDS decile (10% expected) | 41–73% | ≤ 5.2% |
| Median CDS length | 2.8–7.9 kb | 1.1–1.5 kb |

The adjustment changes the lists substantially: only 16–38 genes per configuration are shared between the raw and adjusted lists.

Hub bias also affects performance on labelled genes. Out-of-fold percentile correlates with degree (Spearman r = 0.32 on GGNet), and test AUROC is lowest in the lowest-degree quintile (0.73 vs 0.76–0.83 in the other quintiles).

---

### Unlabelled CGC genes as external validation

HGDC leaves some COSMIC CGC (v103) genes unlabelled, and the model never sees them as positives. Adjusted consensus candidates are enriched for these genes in 15 of 18 configurations (one-sided Fisher, P < 0.05): 5 of 6 on GGNet, 6 of 6 on PathNet and 4 of 6 on PPNet. Raw lists were significant in all 12 GGNet and PathNet configurations but in none on PPNet. Example configurations:

| Configuration | Raw: candidates / CGC / OR / P | Adjusted: candidates / CGC / OR / P |
|---------------|--------------------------------|-------------------------------------|
| GGNet random ρ=0.2 | 96 / 6 / 10.5 / 4.8e-5 | 84 / 3 / 5.5 / 0.020 |
| PathNet priority ρ=0.2 | 67 / 5 / 10.4 / 2.2e-4 | 69 / 3 / 5.6 / 0.020 |
| PathNet random ρ=0.2 | 83 / 4 / 6.4 / 5.0e-3 | 51 / 3 / 7.8 / 8.8e-3 |
| PPNet random ρ=0.2 | 61 / 2 / 5.1 / 0.064 | 63 / 1 / 2.4 / 0.35 |

Recovered CGC genes include ITGB1, CBFB, CTNNA1, POLR2A, RELA, RRAS2 and SOS1. The absolute counts are small (1–4 per configuration after adjustment), because HGDC already labels most CGC genes.

No COSMIC hallmark is enriched among consensus candidates (smallest unadjusted P = 0.30). This test has limited power, because hallmarks are annotated only for CGC genes.

---

### Robust candidates

Twenty-four unlabelled genes are adjusted consensus candidates in ≥ 8 of the 18 configurations. Numbers in brackets give the configurations recovering each gene:

| Group | Genes |
|-------|-------|
| RTK signalling | EPHA2 (18), SH3GL2 (12), SHC1 (10), CRKL (10), BTC (10), FYN (9) |
| Chromatin & transcription factors | HDAC1 (17), HDAC2 (13), CBFB (12), BRD2 (11), PRMT6 (11), MEIS1 (11) |
| Integrin adhesion | ITGB1 (17), ITGB3 (13), ITGA2B (9), FLRT2 (8) |
| RNA polymerase II | POLR2K (12), POLR2C (10), POLR2G (8), MNAT1 (8) |
| Cell cycle & survival | CDKN2B (12), PPP1R13L (11), RFC2 (8), BNIP3L (8) |

ITGB1 (Tier 2) and CBFB (Tier 1) are CGC genes that HGDC does not label. Per-network counts and KEGG enrichment are in [RESULTS.md](RESULTS.md) §3.7.

Candidate lists depend heavily on the network. Under random elimination at ρ = 0.2, 72 adjusted candidates are unique to GGNet, 37 to PathNet and 49 to PPNet, and only EPHA2 and HDAC1 are shared by all three. Only EPHA2, ITGB1 and HDAC1 are recovered in at least five of the six configurations on every network.

---

### HDAC1 and HDAC2

HDAC1 (17 of 18) and HDAC2 (13 of 18) are the most consistently recovered chromatin regulators. The earlier seven-gene "SIN3–HDAC–SWI/SNF axis" is not supported:

| Gene | Consensus configs (of 18): raw → adjusted | Notes |
|------|-------------------------------------------|-------|
| HDAC1 | 12 → 17 | At least 5 of 6 configurations on every network |
| HDAC2 | 8 → 13 | All 12 GGNet and PathNet configurations; 1 on PPNet |
| KMT2E | 8 → 4 | Not a robust candidate |
| SIN3A | 9 → 2 | PathNet only (priority and coarsening, ρ = 0.2) |
| SMARCC1 | 0 → 1 | Not supported |
| SMARCC2 | 4 → 0 | Not supported |

RBBP4 is excluded because it is an HGDC labelled non-driver, so it cannot be a candidate under this protocol.

---

### Known limitations

- **Checkpoints do not reproduce the saved scores.** Reloading an October 2026 checkpoint gives per-gene scores with Spearman 0.84 against the saved `*_all_genes_scored.csv`; the cause is unresolved. All results above come from the saved scores, and per-gene pathway-attention analysis is blocked until this is resolved.
- **Not competitive with current baselines.** On the same splits DriverJinn trails SCIS-CDG, HGDC and an SVM on every network (see [Baseline comparison](#baseline-comparison-same-hgdc-splits)).
- **Analyses not yet run:** the no-augmentation, curvature-feature and attention-mode ablations.

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
│    ├── Load HGDC's fold (--split_file); labelled genes only                 │
│    ├── Hold out 15% of training genes (stratified) for early stopping       │
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
│            • ReduceLROnPlateau (factor, patience configurable)              │
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
│          Test fold evaluated once, after training                           │
│                                                                             │
│  Output per fold:                                                           │
│    • trained_models/.../fold_k_best_model.pt                                │
│    • model_results/.../*_kfold_metrics.csv                                  │
│    • model_results/.../*_all_genes_scored.csv                               │
│    • model_results/.../*_training_history.pkl                               │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 4 — Aggregation  (aggregate_fold_results.py)                          │
│                                                                             │
│  • Aggregate per-fold gene scores (aligned by gene_id):                     │
│      - Mean / median / std of within-fold rank; mean score                  │
│      - folds_top100: # folds where gene is in top 100 unlabelled genes      │
│      - consensus_top100: unlabelled and folds_top100 ≥ 3                    │
│  • Aggregate test metrics: mean ± std AUROC, AUPRC, NDCG@50, P@50, MRR      │
│                                                                             │
│  Output:                                                                    │
│    • model_results/.../aggregated_all_genes.csv                             │
│    • model_results/.../aggregated_metrics_summary.csv                       │
│    • model_results/.../combined_training_curves.png                         │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 5 — Candidate adjustment  (Downstream/length_adjusted_candidates.py)  │
│                                                                             │
│  • Re-rank unlabelled genes within coding-length decile × degree quintile   │
│  • Remove FLAGS genes, recompute consensus (top 100 in ≥ 3 of 5 folds)      │
│                                                                             │
│  Output: Downstream/length_adjusted/                                        │
│  Figures: Downstream/publication_figures.qmd                                │
└─────────────────────────────────────────────────────────────────────────────┘
```

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
├── data/                       # Input networks and HGDC's 10×5CV splits (dataset_<net>_ten_5CV.pkl)
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
│   ├── aggregate_fold_results.py#  Cross-fold result aggregation
│   └── test_ranking_pipeline.py#   Tests for the ranking pipeline
├── Downstream/                 # Downstream analysis and visualizations
│   ├── length_adjusted_candidates.py # Length/degree-adjusted consensus candidates
│   ├── flags_shyr2014.txt      #   FLAGS gene list
│   ├── length_adjusted/        #   Adjusted candidate tables (generated)
│   ├── publication_figures.qmd #   Current figures and tables
│   ├── poster_plots.qmd        #   Poster figures (carries withdrawn numbers)
│   ├── plots/                  #   Generated figures
│   └── cosmic/                 #   COSMIC cancer gene census reference data
├── utils/
│   └── logging_manager.py      # Logging configuration
├── curvature_output/           # Preprocessed graph datasets (generated)
├── model_results_<tag>/        # Training results and plots (generated)
├── trained_models_<tag>/       # Model checkpoints (generated)
├── hyperparam_results/         # Hyperparameter search outputs (generated)
├── RESULTS.md                  # Results write-up
├── MANUSCRIPT.md               # Manuscript draft
├── MODEL_MATH.md               # Methods and model mathematics
├── run_model_array.sh          # SLURM script: 90-task array (18 configurations × 5 folds)
├── run_hyperparam_search.sh    # SLURM script: hyperparameter search
├── preprocess_script.sh        # SLURM script: graph preprocessing
├── environment.yaml            # Conda environment specification
├── requirements.txt            # Pip requirements
├── entrypoint.sh               # Docker entrypoint
└── Dockerfile                  # Docker configuration
```

## Pipeline Overview

The full pipeline consists of five sequential steps:

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
    --dataset curvature_output/GGNet_contrastive_v2_random_r0.2_stratified5CV.pkl \
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
                temperature τ = 0.3

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
    --dataset_file curvature_output/GGNet_contrastive_v2_random_r0.2_stratified5CV.pkl \
    --split_file data/dataset_GGNet_ten_5CV.pkl \
    --split_repeat 0 \
    --inner_val_size 0.15 \
    --num_epochs 1000 \
    --hidden_channels 128 \
    --projection_dim 96 \
    --num_layers 2 \
    --num_heads 4 \
    --num_folds 5 \
    --train_metrics_dir model_results_$(date +%F) \
    --model_out_dir trained_models_$(date +%F) \
    --model_out_prefix GGNet_random_r0.2 \
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
```

#### Arguments

**Data & I/O:**

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--dataset_file` | str | **Required** | Input curvature-enhanced dataset pickle file |
| `--split_file` | str | None | HGDC dataset with released 10×5CV splits. Replaces the dataset's own folds: train/test over HGDC's labelled genes, with a 15% inner validation split for early stopping. Used for all reported results |
| `--split_repeat` | int | 0 | Which of HGDC's 10 five-fold repeats to use |
| `--inner_val_size` | float | 0.15 | Fraction of each training fold held out for early stopping |
| `--train_metrics_dir` | str | `model_results` | Output directory for metrics and plots |
| `--model_out_dir` | str | `trained_models` | Directory for model checkpoints |
| `--model_out_prefix` | str | `""` | Prefix for output filenames |

**Training Configuration:**

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--num_folds` | int | 5 | Number of cross-validation folds |
| `--specific_folds` | int list | None | Train only specific folds (space-separated, 1-indexed) |
| `--num_epochs` | int | 1200 | Training epochs per fold (reported results used 1000) |
| `--seed` | int | 42 | Random seed for reproducibility |

**Model Architecture:**

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--hidden_channels` | int | 128 | Hidden layer dimensions |
| `--projection_dim` | int | 96 | Contrastive projection dimensions |
| `--num_layers` | int | 2 | Number of GNN layers |
| `--num_heads` | int | 4 | Number of attention heads |
| `--concat_heads` | flag | False | Concatenate attention heads (vs average) |
| `--negative_slope` | float | 0.2 | LeakyReLU negative slope in GNN layers |
| `--attention_mode` | choice | `hybrid` | `standard`, `edge_feature`, `bias`, `gated`, `hybrid` |
| `--pathway_aggregator` | choice | `attention` | `attention`, `concat`, `hierarchical`, `mean` |
| `--aggregation` | str | `max` | Message passing aggregation method |

**Optimization:**

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--learning_rate` | float | 5e-4 | AdamW learning rate |
| `--weight_decay` | float | 5e-4 | AdamW weight decay (L2 regularization) |
| `--dropout` | float | 0.3 | Dropout rate |
| `--temperature` | float | 0.3 | NT-Xent contrastive loss temperature |
| `--gradient_accumulation_steps` | int | 8 | Gradient accumulation steps |
| `--mixed_precision` | flag | - | Enable FP16 mixed precision training |
| `--decay` | float | 0.999 | EMA decay rate for model weights |

**Loss Configuration:**

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--contrastive_weight` | float | 0.2 | Contrastive loss weight; ranking weight = 1 - this |
| `--ranking_loss_type` | choice | `bpr` | `pairwise`, `sampled_pairwise`, `bpr`, `listwise`, `approxndcg` |
| `--ranking_loss_scale` | float | 5.0 | Ranking loss scaling factor |
| `--ranking_loss_samples` | int | 512 | Number of pairs sampled for the ranking loss |
| `--ranking_margin` | float | 0.5 | Pairwise ranking margin |
| `--focal_gamma` | float | 2.0 | Focal loss gamma (higher = focus on hard examples) |
| `--use_focal` | flag | - | Enable focal weighting in ranking loss |

**Scheduler & Early Stopping:**

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--scheduler_patience` | int | 5 | Validation checks to wait before reducing LR |
| `--scheduler_factor` | float | 0.7 | Factor to reduce LR by on plateau (reported results used 0.5) |
| `--early_stopping_patience` | int | 15 | Validation checks without improvement before stopping |
| `--validation_frequency` | int | 10 | Validate every N epochs |

**Memory Optimization:**

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--attention_chunk_size` | int | 1000 | Chunk size for attention computation |
| `--max_augmented_views` | int | 2 | Max augmented views in memory |
| `--max_views_per_step` | int | 2 | Max views per training step |
| `--emergency_mode` | flag | - | Aggressive memory reduction (auto-sets params) |
| `--reduce_model_size` | flag | - | Reduce to hidden=64, proj=32, layers=2, heads=1 |
| `--return_all_layers` | flag | - | Return all GNN layer outputs |

**Output per fold:** Model checkpoint, test metrics CSV (including `AUPRC_trapz`, trapezoidal AUPRC for comparison with published baselines), training history pickle, training curves plot, metrics comparison plot, and scored genes CSV.

---

### Step 4: Result Aggregation

Combines results from parallel fold training runs into cross-validation summary statistics, consensus gene predictions, and combined plots.

```bash
python -m model.aggregate_fold_results \
    --prefix GGNet_random_r0.2 \
    --num_folds 5 \
    --results_dir model_results_<tag>
```

The script expects one directory per fold, `<results_dir>/<prefix>_fold<N>/`, which is the layout `run_model_array.sh` writes. Files named `aggregated` inside a single fold's directory cover that fold only.

#### Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--prefix` | str | **Required** | Base prefix for fold directories |
| `--num_folds` | int | 5 | Number of folds to aggregate |
| `--results_dir` | str | `model_results` | Directory containing fold subdirectories |
| `--output_dir` | str | auto | Output directory (default: `<results_dir>/<prefix>_aggregated`) |

**Output:**
- Cross-validation summary statistics (mean/std AUROC, NDCG@50, Precision@50, etc.)
- Aggregated gene ranking (mean within-fold rank) and consensus candidates (unlabelled genes in the top 100 unlabelled genes in ≥ 3 folds)
- Combined training curves across all folds
- Metrics comparison bar and box plots

`AUPRC_trapz` is not carried into `<prefix>_combined_metrics.csv`; read it from each fold's `*_kfold_metrics.csv`.

---

### Step 5: Candidate Adjustment and Figures

Re-ranks unlabelled genes within coding-length decile × degree quintile bins, removes FLAGS genes, and recomputes consensus candidates and CGC enrichment. Run from the repo root:

```bash
python Downstream/length_adjusted_candidates.py
quarto render Downstream/publication_figures.qmd
```

The results directory (`model_results_2026-10-16`) and the HGDC and GENCODE input paths are set at the top of the script. Coding lengths come from `baseline_comparison/HGDC/preprocess_data/gene_mutation/gencode.v28.basic.annotation.gff3.gz`, so this step needs a local copy of the [HGDC repository](https://github.com/NWPU-903PR/HGDC) in `baseline_comparison/HGDC/`.

**Output:** in `Downstream/length_adjusted/`, one `<config>_length_adjusted.csv` per configuration, plus `summary.csv`, `consensus_counts.csv` and `module_genes.csv`.

---

## Docker Usage

1. Download the docker image

```bash
docker pull cydarthvader/driverjinn
```

1. Run the preprocessing script with the following command

```bash
docker run -v "your/working/directory:/app" cydarthvader/driverjinn preprocess --dataset_file data/dataset_PathNet.pkl --method both --augment --num_views 2 --elimination_ratio 0.2 --strategy random --use_kfold
```

1. Run the training script with this command

```bash
docker run --gpus all -v "your/working/directory:/app" cydarthvader/driverjinn \
    train_model \
     --dataset_file curvature_output/GGNet_contrastive_v2_random_r0.2_stratified5CV.pkl  \
     --num_epochs 100 \
     --hidden_channels 32 \
     --projection_dim 16 \
     --num_layers 1 \
     --num_heads 4 \
     --concat_heads \
     --num_folds 5 \
     --model_out_prefix GGNet_random_r0.1 \
     --temperature 0.3 \
     --model_out_dir trained_model_$(date +%F) \
     --train_metrics_dir model_results_$(date +%F) \
     --dropout 0.3 \
     --attention_mode hybrid \
     --pathway_aggregator hierarchical \
     --aggregation max \
     --negative_slope 0.25 \
     --attention_chunk_size 1000 \
     --gradient_accumulation_steps 8 \
     --early_stopping_patience 15 \
     --validation_frequency 20 \
     --scheduler_patience 5 \
     --scheduler_factor 0.5 \
     --ranking_loss_type bpr \
     --ranking_loss_scale 5 \
     --ranking_loss_samples 512 \
     --contrastive_weight 0.2 \
     --learning_rate 5e-4 \
     --weight_decay 5e-4
```

## Apptainer Usage

```bash
apptainer run --nv docker://cydarthvader/driverjinn \
    train_model \
     --dataset_file curvature_output/GGNet_contrastive_v2_random_r0.2_stratified5CV.pkl  \
     --num_epochs 100 \
     --hidden_channels 32 \
     --projection_dim 16 \
     --num_layers 1 \
     --num_heads 4 \
     --concat_heads \
     --num_folds 5 \
     --model_out_prefix GGNet_random_r0.2 \
     --temperature 0.3 \
     --model_out_dir trained_model_$(date +%F) \
     --train_metrics_dir model_results_$(date +%F) \
     --dropout 0.3 \
     --attention_mode hybrid \
     --pathway_aggregator hierarchical \
     --aggregation max \
     --negative_slope 0.25 \
     --attention_chunk_size 1000 \
     --gradient_accumulation_steps 8 \
     --early_stopping_patience 15 \
     --validation_frequency 20 \
     --scheduler_patience 5 \
     --scheduler_factor 0.5 \
     --ranking_loss_type bpr \
     --ranking_loss_scale 5 \
     --ranking_loss_samples 512 \
     --contrastive_weight 0.2 \
     --learning_rate 5e-4 \
     --weight_decay 5e-4
```

---

## HPC Usage with SLURM

### Parallel Training (one fold per GPU)

Use `run_model_array.sh` to train all 18 configurations on HGDC's splits. `RUN_TAG` is required, so that every task writes to the same `model_results_<RUN_TAG>/` and `trained_models_<RUN_TAG>/` folders however long the queue takes:

```bash
mkdir -p slurm_logs
sbatch --export=ALL,RUN_TAG=$(date +%F) run_model_array.sh
```

This submits 90 tasks (`--array=1-90`: 3 networks × 3 strategies × 2 ratios × 5 folds), each training a single fold on its own GPU. SLURM does not create the `slurm_logs/` directory itself. The partition (`h100-single`), account, email address, and `base` path in the script are specific to our cluster, so change them before submitting. The script also reads splits from `baseline_comparison/HGDC/data/<net>/`, which is not tracked in git; the tracked copies in `data/` are byte-identical, so point `--split_file` there if you don't have a local copy of HGDC.

After all tasks complete, aggregate each configuration:

```bash
for net in GGNet PathNet PPNet; do for s in random priority coarsening; do for r in 0.1 0.2; do
    python -m model.aggregate_fold_results --prefix ${net}_${s}_r${r} --results_dir model_results_${RUN_TAG}
done; done; done
```

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

Located in `model_results_<tag>/<run>/`, where `<run>` is the `--model_out_prefix` value (`<prefix>_fold<N>` for array jobs):

| File | Description |
|------|-------------|
| `<run>_kfold_metrics.csv` | Test metrics for the fold, including `AUPRC_trapz` in runs newer than `model_results_2026-10-16` |
| `<run>_training_history.pkl` | Full training history (loss, LR, all metrics per epoch) |
| `<run>_fold_<N>_all_genes_scored.csv` | All genes ranked by predicted driver score |
| `<run>_fold_<N>_known_drivers_scores.csv` | Scores for known drivers |
| `<run>_fold_<N>_scoring_summary.txt` | Summary of the fold's gene scoring |
| `<run>_training_curves.png` | Training/validation curves plot |
| `<run>_metrics_comparison.png` | Metrics comparison plot |

### Aggregated Output

Located in `model_results_<tag>/<prefix>_aggregated/`:

| File | Description |
|------|-------------|
| `<prefix>_combined_metrics.csv` | Per-fold test metrics plus mean/std rows |
| `<prefix>_aggregated_all_genes.csv` | All genes ranked by mean within-fold rank, with `folds_top100` and `consensus_top100` |
| `<prefix>_aggregated_consensus_top100.csv` | Consensus candidates: unlabelled genes in the top 100 unlabelled genes in ≥ 3 folds |
| `<prefix>_aggregated_summary.txt` | Summary of the aggregated ranking |
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