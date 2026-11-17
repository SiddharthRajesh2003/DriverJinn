# Downstream

Post-prediction validation and visualization pipeline for DriverJinn. All analysis lives in `poster_plots.qmd` (Quarto notebook); outputs go to `plots/`.

## Inputs

| Source | Path | Content |
|--------|------|---------|
| Model results | `../model_results_*/` | Per-fold ranked gene lists, metrics, training histories |
| Curvature graph | `../curvature_output/GGNet_graph_with_curvatures.pkl` | PPI network with Ollivier-Ricci edge curvatures |
| COSMIC reference | `cosmic/` | Cancer Gene Census v103 (GRCh38) — tier labels and hallmark annotations |

## Analysis Sections (`poster_plots.qmd`)

### 1. Training Performance
Validation NDCG@50 and AUROC learning curves across 5 folds. Summary table of peak metrics and epoch counts per fold.

### 2. Model Validation
- **ROC curves** — per-fold with mean ± 1 std band; true label = COSMIC Tier 1
- **Enrichment curve** — cumulative precision recovering Tier 1 vs any CGC gene as rank increases
- **Score distributions** — violin plots stratified by CGC tier (T1 / T2 / non-CGC)
- **Precision@K** — bar charts at K = 10, 20, 50, 100, 200 with per-fold error bars

### 3. Stability & Consistency
- **Rank stability heatmap** — top 30 consensus genes × 5 folds
- **Per-fold metrics bar chart** — NDCG@50, AUROC, AUPRC, Precision@50 with mean ± std markers

### 4. Biological Enrichment
- **Hallmark ORA** — Fisher's exact test on top 100 predictions against COSMIC hallmarks; FDR-BH correction
- **KEGG pathway enrichment** — Enrichr-based analysis of top 50 genes (requires `gseapy` and internet)

### 5. Novel Candidate Discovery
Top consensus-significant genes absent from COSMIC CGC, with aggregate rank, mean score, and fold support count.

### 6. Cross-Model Consensus
Merges rankings from 6 Schur augmentation strategies:

| Strategy | Elimination rates |
|----------|------------------|
| random | r = 0.1, 0.2 |
| coarsening | r = 0.1, 0.2 |
| priority | r = 0.1, 0.2 |

Consensus score = mean rank across all 6 models; `models_consensus` counts how many models flag a gene as significant.

### 7. Cross-Architecture Consensus
Compares GGNet, PathNet, and PPNet rankings. Consensus rank = mean rank across 3 architectures. Top 20 table highlights novel candidates (gold) and Tier 1 genes (light red).

### 8. PPI Network Analysis
- **Degree vs score scatter** — Spearman correlation between network centrality and model predictions, colored by CGC tier
- **Degree distribution** — all genes vs CGC Tier 1 vs top-100 predicted
- **PPI subnetwork** — top 50 predicted genes; node size = model score, node color = CGC tier

### 9. Chromatin Remodeling Cluster Deep Dive
Focused analysis on 7 genes (SIN3A, HDAC1, HDAC2, SMARCC1, SMARCC2, RBBP4, KMT2E) spanning the SIN3/HDAC, NuRD/PRC2, SWI/SNF, and KMT complexes. Curvature-annotated subgraph with edges colored by Ollivier-Ricci value (blue = bottleneck, red = positive).

## Output Plots (`plots/`)

| File | Section |
|------|---------|
| `roc_curves.png` | Model Validation |
| `poster_enrichment_curve.png` | Model Validation |
| `poster_score_violin.png` | Model Validation |
| `poster_precision_at_k.png` | Model Validation |
| `poster_rank_stability.png` | Stability |
| `poster_metrics_summary.png` | Stability |
| `cgc_hallmark_enrichment.png` | Biological Enrichment |
| `poster_kegg_enrichment.png` | Biological Enrichment |
| `novel.png` | Cross-Architecture Consensus |
| `consensus_subnetwork.png` | PPI Network Analysis |
| `cluster_curvature_subgraph.png` | Chromatin Cluster |

## Reference Data (`cosmic/`)

COSMIC Cancer Gene Census v103 (GRCh38):

- `Cosmic_CancerGeneCensus_v103_GRCh38.tsv` — master gene list with tier (1/2), role in cancer, mutation types, and tumour type associations; used to label predictions as T1 / T2 / novel
- `Cosmic_CancerGeneCensusHallmarksOfCancer_v103_GRCh38.tsv` — hallmark annotations for Tier 1 genes with impact direction (promotes/suppresses) and PubMed citations; used for ORA
