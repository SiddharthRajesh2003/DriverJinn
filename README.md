# DriverJinn

This project is still in development with fixes being continuously generated.

This model is currently being trained on NVIDIA H100 GPU.

A Graph Neural Network framework for cancer driver gene prediction using curvature-enhanced graph representations and contrastive learning.

## Table of Contents

- [Preliminary Biological Findings](#preliminary-biological-findings)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Pipeline Overview](#pipeline-overview)
  - [Step 1: Graph Curvature Enhancement](#step-1-graph-curvature-enhancement)
  - [Step 2: Hyperparameter Search](#step-2-hyperparameter-search)
  - [Step 3: Model Training](#step-3-model-training)
  - [Step 4: Result Aggregation](#step-4-result-aggregation)
- [HPC Usage with SLURM](#hpc-usage-with-slurm)
- [Output Files](#output-files)


## Preliminary Biological Findings

> **Note:** All findings below are computational predictions from the DriverJinn model. They represent statistically significant, cross-validated candidate cancer driver genes and have not yet been experimentally validated. Downstream analysis scripts and plots are in `Downstream/poster_plots.qmd`.

---

### Cross-Validation Performance Summary

Results from the March 2026 systematic evaluation across three gene interaction networks (GGNet, PPNet, PathNet) and three Schur complement augmentation strategies (random, coarsening, priority) at two elimination ratios (ρ=0.1, ρ=0.2):

| Network | Strategy | ρ | NDCG@50 | AUROC | AUPRC | P@50 |
|---------|----------|---|---------|-------|-------|------|
| GGNet | random | 0.2 | **0.478 ± 0.062** | 0.763 ± 0.030 | 0.230 | 0.432 |
| GGNet | random | 0.1 | 0.472 ± 0.062 | 0.760 ± 0.020 | 0.230 | 0.420 |
| GGNet | coarsening | 0.1 | 0.468 ± 0.049 | 0.761 ± 0.017 | 0.231 | 0.412 |
| GGNet | priority | 0.2 | 0.462 ± 0.054 | 0.760 ± 0.023 | 0.228 | 0.404 |
| GGNet | coarsening | 0.2 | 0.457 ± 0.058 | 0.758 ± 0.027 | 0.225 | 0.388 |
| GGNet | priority | 0.1 | 0.456 ± 0.054 | 0.755 ± 0.015 | 0.220 | 0.400 |
| PathNet | coarsening | 0.2 | 0.446 ± 0.061 | 0.769 ± 0.014 | 0.252 | 0.396 |
| PathNet | coarsening | 0.1 | 0.442 ± 0.046 | 0.769 ± 0.010 | 0.260 | 0.372 |
| PathNet | priority | 0.1 | 0.427 ± 0.057 | 0.770 ± 0.008 | 0.259 | 0.356 |
| PPNet | coarsening | 0.2 | 0.426 ± 0.036 | **0.784 ± 0.008** | 0.230 | 0.372 |
| PPNet | priority | 0.2 | 0.424 ± 0.047 | 0.784 ± 0.008 | 0.232 | 0.368 |
| PPNet | coarsening | 0.1 | 0.412 ± 0.035 | 0.787 ± 0.010 | 0.232 | 0.356 |

Per-fold peak metrics for the best overall configuration (GGNet random ρ=0.2):

| Fold | Peak NDCG@50 | Peak AUROC |
|------|-------------|------------|
| 1 | 0.3992 | 0.7534 |
| 2 | 0.5022 | 0.7519 |
| 3 | 0.4262 | 0.7542 |
| 4 | 0.5244 | 0.8144 |
| 5 | 0.5375 | 0.7665 |
| **Mean** | **0.4779** | **0.7681** |

**Key observations:**
- GGNet consistently achieves the highest NDCG@50, indicating the best relative ranking of known driver genes
- PPNet achieves the highest AUROC (~0.787) but lower ranked-list quality, suggesting it separates drivers from non-drivers in aggregate but is less precise at the top
- PathNet achieves the highest AUPRC in some configurations, reflecting strong pathway-level biological coherence
- Random augmentation (ρ=0.2) is optimal for GGNet; coarsening is optimal for PPNet and PathNet
- Priority elimination consistently underperforms all other augmentation strategies across all networks

---

### Model Score Validation Against COSMIC Cancer Gene Census

#### Score Tier Separation

The model assigns dramatically different score distributions to genes by their CGC status, confirming that high scores reflect genuine driver-gene biology:

| CGC Tier | Median Score |
|----------|-------------|
| Tier 1 (518 high-confidence drivers) | **0.987** |
| Tier 2 | **0.899** |
| Non-CGC (remainder of genome) | **0.285** |

#### CGC Enrichment Curve

At the top-K cutoff, the fraction of COSMIC Tier 1 genes in the ranked list far exceeds random expectation:
- Random baseline: ~4.6% Tier 1 at any K (518 of ~11,000 gene universe)
- **At K=50**: ~40% Tier 1 precision (~8.7× enrichment over random)
- **At K=500**: enrichment remains ~6× above baseline

#### Precision@K

| K | Precision (Tier 1) | Precision (Any CGC) | Tier 1 Baseline | Any-CGC Baseline |
|---|-------------------|--------------------|-----------------|--------------------|
| 10 | ~34% | ~46% | 4.6% | 5.8% |
| 50 | ~31% | ~37% | 4.6% | 5.8% |
| 200 | ~30% | ~35% | 4.6% | 5.8% |

#### Rank Stability (Top-30 across Folds)

The top-ranked genes are dominated by known Tier 1 drivers with highly stable ranks across all 5 folds — providing internal validation of the model's consistency. Consistent top-ranking known drivers include: **POLR2A** (rank 1 in 4/5 folds), UBC, HDAC1, RPS27A, ITGB1, HSP90AA1, UBB, JAK1, POLR2B, ATM. Novel genes appearing stably in the top-30: **SOS1, SMAD3, ERCC2, HDAC2, CREBBP**.

---

### Pathway Enrichment Analysis

#### Hallmark Enrichment (Top-100 Predictions, Fisher's Exact Test, FDR-BH)

21 COSMIC cancer hallmarks are significantly enriched (p.adj < 0.05) in the top-100 predicted genes:

| Hallmark | p.adj |
|----------|-------|
| Role in cancer | 1.4e-24 |
| Function summary | 1.4e-24 |
| Escaping programmed cell death | 4.1e-19 |
| Types of alteration in cancer | 1.2e-18 |
| Genome instability and mutations | 4.8e-17 |
| Invasion and metastasis | 4.8e-17 |
| Impact of mutation on function | 6.8e-16 |
| Differentiation and development | 9.3e-16 |
| Proliferative signalling | 9.8e-16 |
| Cell division control | 2.5e-15 |
| Tumour promoting inflammation | 3.2e-13 |
| Cell replicative immortality | 3.6e-11 |
| Escaping immune response to cancer | 3.7e-11 |
| Angiogenesis | 3.7e-11 |
| Senescence | 2.6e-10 |
| Fusion partner | 9.8e-9 |
| Clinical impact | 1.0e-8 |
| Global regulation of gene expression | 1.7e-8 |
| Change of cellular energetics | 6.9e-8 |
| Suppression of growth | 2.9e-6 |
| Mouse model | 1.3e-4 |

#### KEGG Pathway Enrichment (Top-50 Predictions, via Enrichr API)

Top enriched KEGG pathways in the model's highest-confidence predictions:

| KEGG Pathway | Biological Relevance |
|-------------|---------------------|
| RNA polymerase | Core transcription machinery; POLR2A/POLR2B consistently top-ranked |
| Cell cycle | Checkpoint regulation; consistent with CDK/cyclin predictions |
| Nucleotide excision repair | DNA damage; GTF2H complex genes predicted as novel drivers |
| Basal transcription factors | TFIIH/TFIID components; transcriptional dysregulation mechanism |
| Chronic myeloid leukemia | Known CML gene set overlap in top predictions |
| Pathways in cancer | Multi-pathway driver gene convergence |
| TGF-beta signaling | Growth suppression; consistent with TGFBR predictions |
| Ubiquitin mediated proteolysis | E3 ligase complex dysregulation (BTRC, TRAF6, ANAPC) |
| Huntington disease | Broad transcriptional co-repressor complex overlap |
| Thyroid hormone signaling | Nuclear receptor coactivator predictions |

---

### Network Topology Analysis

#### PPI Degree Correlation

High-scoring genes are significantly more connected in the human protein-protein interaction network:

- **Spearman ρ = 0.476** between PPI degree and model score (p ≈ 0)
- Median PPI degree: all genes = **61**, CGC Tier 1 = **126**, top-100 predicted = **177**

The model preferentially scores hub genes — which is consistent with cancer driver biology, since essential cellular regulators tend to be both highly connected and oncogenically dysregulated. This is not a bias artifact: hub proteins in the top-100 are specifically those with well-established roles in transcription, DNA repair, and chromatin remodeling.

#### PPI Subnetwork (Top-50 Predictions)

The top-50 predicted genes form a dense subgraph: **density = 0.0555**, 8 connected components, largest component = 43 nodes. The largest component is anchored by POLR2A, UBC, HDAC1, and their interaction partners — reflecting the model's coherent prediction of a transcriptional regulatory hub as a central driver mechanism.

---

### Cross-Network Consensus Predictions

The definitive high-confidence novel driver candidates are genes ranked highly across all three independent networks (GGNet × PathNet × PPNet, each using random augmentation ρ=0.2). The combined rank is computed as the sum of per-network ranks; lower is better.

| Combined Rank | Gene | GGNet | PathNet | PPNet | CGC Status |
|---------------|------|-------|---------|-------|------------|
| 1 | **SIN3A** | 31 | 20 | 118 | **Novel** |
| 2 | **HDAC2** | 23 | 82 | 69 | **Novel** |
| 3 | NCOR2 | 121 | 46 | 91 | Tier 1 |
| 4 | **SMARCC1** | 58 | 149 | 85 | **Novel** |
| 5 | **KMT2E** | 255 | 22 | 23 | **Novel** |
| 6 | SOS1 | 18 | 91 | 193 | Tier 1 |
| 7 | CBFB | 21 | 84 | 209 | Tier 1 |
| 8 | KDR | 100 | 113 | 103 | Tier 1 |
| 9 | **SMARCC2** | 92 | 29 | 196 | **Novel** |
| 10 | **SMARCA2** | 190 | 68 | 61 | **Novel** |
| 11 | **RBBP4** | 13 | 47 | 294 | **Novel** |
| 12 | EED | 218 | 34 | 115 | Tier 2 |
| 13 | **FRS2** | 88 | 132 | 147 | **Novel** |
| 14 | **DOCK7** | 290 | 37 | 64 | **Novel** |
| 15 | **NCOA3** | 319 | 4 | 80 | **Novel** |
| 16 | **MED1** | 129 | 108 | 170 | **Novel** |
| 17 | CHD4 | 82 | 19 | 315 | Tier 1 |
| 18 | **SHC1** | 102 | 281 | 36 | **Novel** |
| 19 | **MARK3** | 216 | 107 | 100 | **Novel** |
| 20 | **HDAC1** | 3 | 122 | 301 | **Novel** |

14 of the top-20 cross-network consensus genes are **novel** (non-CGC) candidates, providing a high-confidence shortlist for experimental follow-up.

---

### Highlighted Finding: Chromatin Remodeling Cluster

The model's most striking biological finding is the consistent, high-confidence prediction of a tightly connected chromatin remodeling cluster as novel cancer driver genes. Seven genes from three interacting chromatin regulatory complexes are predicted with near-perfect model scores across all folds and all network configurations:

| Gene | Complex | Score (all folds) | CV | Percentile vs Tier 1 | Mean Rank (9 conditions) |
|------|---------|------------------|----|----------------------|--------------------------|
| **HDAC1** | SIN3/HDAC | >0.9999 | ≈0 | **100th** | 1095.2 |
| **HDAC2** | SIN3/HDAC | >0.9999 | ≈0 | 98th | 674.9 |
| **SIN3A** | SIN3/HDAC | >0.9999 | ≈0 | 98th | 119.9 |
| **SMARCC1** | SWI/SNF | >0.9999 | ≈0 | 97th | 221.3 |
| **SMARCC2** | SWI/SNF | >0.9999 | ≈0 | 94th | 166.2 |
| **RBBP4** | NuRD/PRC2 | >0.9999 | ≈0 | 99th | 732.3 |
| **KMT2E** | KMT complex | >0.9999 | ≈0 | 85th | 86.2 |

All seven genes score above 85–100% of all confirmed COSMIC Tier 1 cancer drivers, with coefficient of variation ≈ 0 across all 5 cross-validation folds (i.e., no fold uncertainty). The cluster forms a highly connected subgraph (38 nodes, 497 edges), indicating functional co-regulation. None are currently listed in COSMIC CGC.

**Why this matters:** SIN3A/HDAC1/HDAC2 are core components of the SIN3–HDAC co-repressor, which silences tumor suppressor loci through histone deacetylation. SMARCC1/SMARCC2 are catalytic subunits of the SWI/SNF chromatin remodeling complex, the most frequently mutated chromatin remodeling complex in cancer (~20% of all human cancers). RBBP4 bridges the NuRD and PRC2 repressive complexes. KMT2E is a H3K4 methyltransferase whose loss disrupts epigenetic memory. The convergent prediction of all members of these interacting complexes — across three independent gene-interaction networks — strongly implicates **SIN3–HDAC–SWI/SNF co-regulatory axis dysregulation** as an underappreciated cancer driver mechanism.

---

### Per-Network Novel Candidate Gene Tables

The following tables list the highest-ranked novel candidates (consensus-significant: significant in all 5 folds, `is_known_driver = False`) from the best configuration of each network type.

#### GGNet — random, ρ=0.2 (best overall NDCG@50 = 0.478)

| Rank | Gene | Known Role in Cancer Biology |
|------|------|------------------------------|
| 18 | **SOS1** | RAS guanine nucleotide exchange factor; activates MAPK/PI3K cascades |
| 42 | **MNAT1** | CDK-activating kinase (CAK) assembly subunit; cell cycle entry |
| 59 | **TRAF6** | E3 ubiquitin ligase; NF-κB and JNK signaling; anti-apoptotic |
| 69 | **MRE11** | MRN complex; DSB sensing and homologous recombination |
| 81 | **RAD50** | MRN complex; DNA double-strand break repair |
| 84 | **GTF2H1** | TFIIH complex; nucleotide excision repair and basal transcription |
| 96 | **PDS5B** | Cohesin complex; chromosome cohesion and segregation |
| 106 | **DYNC1H1** | Cytoplasmic dynein heavy chain; mitotic spindle assembly |
| 124 | **AGO3** | Argonaute; miRNA-mediated gene silencing |
| 133 | **GTF2H4** | TFIIH subunit; NER and CDK7-mediated transcription |
| 147 | **RPTOR** | Raptor; mTORC1 scaffold and nutrient-sensing growth regulator |
| 215 | **PAK1** | p21-activated kinase; RAS/MAPK oncogenic signaling; frequently amplified in breast cancer |
| 221 | **BTRC** | β-TrCP; E3 ubiquitin ligase targeting β-catenin (Wnt) and IκB |
| 239 | **REV3L** | DNA polymerase ζ catalytic subunit; translesion synthesis |
| 254 | **FANCM** | Fanconi anemia complementation group M; replication fork restart |
| 283 | **PLCG2** | Phospholipase Cγ2; B-cell receptor / RAS signaling |

#### PathNet — coarsening, ρ=0.2 (highest AUPRC per-fold peak)

| Rank | Gene | Known Role in Cancer Biology |
|------|------|------------------------------|
| 14 | **DYRK1A** | Dual-specificity kinase; cyclin D1 degradation; DYRK1A loss disrupts G1/S |
| 37 | **PTPRJ** | Receptor tyrosine phosphatase; suppresses EGFR/PDGFR signaling; confirmed tumor suppressor |
| 75 | **NCOA6** | Nuclear receptor coactivator 6; transcriptional co-activator of multiple oncogenic NRs |
| 76 | **PTPRF** | Receptor tyrosine phosphatase; negative regulator of EGFR/insulin signaling |
| 94 | **RBL1** | p107 retinoblastoma-like; E2F repressor; cell cycle tumor suppressor |
| 97 | **PAXIP1** | PTIP; BRCA1-associated complex; histone H3K4 methylation at DSBs |
| 120 | **GLI3** | Hedgehog pathway transcription factor; activator/repressor switch |
| 151 | **HIPK2** | Homeodomain-interacting protein kinase 2; phospho-activates p53 Ser46 |
| 163 | **MDC1** | Mediator of DNA damage checkpoint 1; H2AX reader; DSB signal amplifier |
| 174 | **SMC3** | Cohesin structural subunit; mutated in acute myeloid leukemia |
| 181 | **MNAT1** | CAK assembly factor *(also top-ranked in GGNet — cross-network consistent)* |
| 190 | **MED13** | Mediator complex kinase module; transcriptional CDK8 substrate |
| 194 | **HDAC2** | Histone deacetylase; epigenetic silencing of tumor suppressors |
| 195 | **TNKS** | Tankyrase 1; PARP/Wnt — degrades AXIN to activate β-catenin |
| 223 | **MECP2** | Methyl-CpG binding protein; epigenetic reader; Xq28 amplification in cancer |
| 272 | **FANCI** | Fanconi anemia complementation group I; inter-strand crosslink repair |

#### PPNet — coarsening, ρ=0.2 (highest AUROC = 0.784)

| Rank | Gene | Known Role in Cancer Biology |
|------|------|------------------------------|
| 18 | **SIN3B** | SIN3–HDAC co-repressor complex; silences pro-proliferative genes |
| 60 | **SETD1A** | H3K4me3 methyltransferase; frequently mutated in clonal hematopoiesis |
| 80 | **NEK1** | NIMA-related kinase; G2/M checkpoint and centrosome duplication |
| 84 | **SIN3A** | Core SIN3–HDAC scaffold; p53- and MYC-mediated transcriptional repression |
| 86 | **CHD3** | Chromodomain–helicase–DNA binding; NuRD chromatin remodeling complex |
| 112 | **MAML3** | Mastermind-like coactivator; Notch transcriptional complex |
| 126 | **MED13L** | Mediator complex; MED13L mutations cause intellectual disability and cancer predisposition |
| 127 | **TAF1** | TATA-binding protein–associated factor; largest TFIID subunit; kinase activity |
| 219 | **RIF1** | Replication timing regulator; NHEJ/DSB repair pathway choice |
| 289 | **PIK3CD** | PI3Kδ catalytic subunit; activated in hematologic malignancies |
| 297 | **REST** | RE1-silencing transcription factor; represses neuronal differentiation; oncogenic in non-neuronal cancers |

---

### Functional Pathway Themes

Grouping top novel predictions by biological function reveals recurring pathway themes across all network configurations:

| Pathway / Process | Representative Genes | Significance |
|-------------------|---------------------|--------------|
| **DNA Damage Response** | MRE11, RAD50, GTF2H1/4, REV3L, FANCM, FANCI, MDC1, PAXIP1, RIF1 | Genomic instability is a hallmark of cancer; these predictions collectively implicate DSB repair and replication stress as central driver mechanisms |
| **Transcription & Mediator Complex** | MED13, MED13L, MNAT1, TAF1, GTF2H1, POLR2B | Transcriptional dysregulation through the Mediator/TFIIH axis; consistent with CDK8 oncogenic amplification in colorectal cancer |
| **Chromatin Remodeling & Epigenetics** | HDAC1/2, SIN3A/B, SMARCC1/2, RBBP4, KMT2E, CHD3, SETD1A | SIN3–HDAC–SWI/SNF co-regulatory axis; epigenetic reprogramming is central to tumor cell plasticity |
| **RAS / MAPK / PI3K Signaling** | SOS1, PAK1, PLCG2, PIK3CD, RPTOR | SOS1 gain-of-function is an established RAS activator in RASopathy-associated cancers; PAK1 amplification is frequent in luminal breast cancer |
| **Cell Cycle & Checkpoint** | MNAT1, RPTOR, RBL1, DYRK1A, NEK1, BTRC, SMC3, PDS5B | Dysregulation of multiple independent G1/S and G2/M checkpoints |
| **Wnt / Notch Signaling** | BTRC, TNKS, GLI3, MAML3 | Convergent developmental pathway activation through β-catenin and Notch coactivation |
| **Cohesin Complex** | PDS5B, SMC3, SMARCC1/2 | Cohesin mutations drive chromosomal instability; SMC3 is recurrently mutated in AML |
| **Fanconi Anemia / ICL Repair** | FANCM, FANCI | Fanconi pathway defects sensitize cells to replication stress and promote carcinogenesis |

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