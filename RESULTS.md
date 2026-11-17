# 3. Results

---

## 3.1 Evaluation setting

We trained DriverJinn on the three gene interaction networks released with HGDC [4]: GGNet, PathNet and PPNet. For each network we varied the Schur elimination strategy (random, priority-ordered or coarsening) and the elimination ratio ρ (0.1 or 0.2), giving 18 configurations in total. Every configuration was evaluated on HGDC's released 5-fold cross-validation splits (repeat 0), so that our results can be compared directly with methods benchmarked on the same partitions.

Training and test genes were drawn only from HGDC's labelled set, which contains 796 drivers and 2,187 non-drivers overall; of these, 1,809 genes are present in GGNet, 1,301 in PathNet and `[TBD]` in PPNet. Within each training fold, a stratified 15% of genes was held out for early stopping, and each test fold was scored exactly once. Drivers make up about 39% of the test genes in each network, so a random ranking has an expected AUROC of 0.50 and an expected AUPRC and Precision@K of 0.39.

We report test-fold AUROC, AUPRC (computed as average precision), NDCG@50 and Precision@50 as mean ± SD over the five folds. Because the test sets contain only labelled genes, top-of-list metrics are high in absolute terms and should not be compared with values computed over the full gene universe. Unless stated otherwise, figures show GGNet with random elimination at ρ = 0.2 as the reference configuration. The best configuration by mean test AUPRC, PathNet with priority elimination at ρ = 0.2, supplies the headline numbers reported in the Abstract.

## 3.2 Predictive performance on held-out test folds

In the reference configuration, DriverJinn achieves a test AUROC of 0.813 ± 0.020 and an AUPRC of 0.751 ± 0.017, compared with 0.50 and 0.39 expected for a random ranking (Fig 1a–c). Performance at the top of the list is stronger still: NDCG@50 is 0.890 ± 0.037 and Precision@50 is 0.864 ± 0.043. Training is stable across folds. Inner-validation NDCG@50 plateaus after roughly 250–400 epochs in every fold, and early stopping terminates training between about 600 and 1,000 epochs (Fig 1d; Supplementary Fig S1).

Table 1 summarises all 18 configurations. AUROC ranges from 0.806 to 0.835 and AUPRC from 0.734 to 0.804. The best configuration, PathNet with priority elimination at ρ = 0.2, reaches an AUROC of 0.835 ± 0.021 and an AUPRC of 0.804 ± 0.029.

**Table 1. Test-fold performance of DriverJinn across networks and augmentation settings.** Values are mean ± SD over the five HGDC test folds. Rows are grouped by network and sorted by AUPRC. The best value in each column is shown in bold.

| Network | Strategy | ρ | AUROC | AUPRC | NDCG@50 | P@50 |
|---------|----------|---|-------|-------|---------|------|
| PathNet | priority | 0.2 | **0.835 ± 0.021** | **0.804 ± 0.029** | 0.880 ± 0.045 | 0.852 ± 0.054 |
| PathNet | priority | 0.1 | 0.829 ± 0.019 | 0.795 ± 0.028 | 0.874 ± 0.030 | 0.852 ± 0.023 |
| PathNet | coarsening | 0.1 | 0.829 ± 0.018 | 0.795 ± 0.023 | 0.878 ± 0.035 | 0.852 ± 0.036 |
| PathNet | coarsening | 0.2 | 0.828 ± 0.017 | 0.794 ± 0.027 | 0.873 ± 0.034 | 0.844 ± 0.038 |
| PathNet | random | 0.2 | 0.822 ± 0.018 | 0.781 ± 0.037 | 0.857 ± 0.057 | 0.840 ± 0.060 |
| PathNet | random | 0.1 | 0.822 ± 0.021 | 0.779 ± 0.040 | 0.851 ± 0.064 | 0.824 ± 0.065 |
| PPNet | coarsening | 0.2 | 0.831 ± 0.026 | 0.773 ± 0.044 | 0.878 ± 0.072 | 0.872 ± 0.046 |
| PPNet | priority | 0.2 | 0.830 ± 0.024 | 0.773 ± 0.043 | 0.884 ± 0.069 | **0.880 ± 0.040** |
| PPNet | coarsening | 0.1 | 0.828 ± 0.030 | 0.766 ± 0.052 | 0.860 ± 0.078 | 0.848 ± 0.050 |
| PPNet | random | 0.1 | 0.829 ± 0.026 | 0.762 ± 0.043 | 0.856 ± 0.063 | 0.844 ± 0.046 |
| PPNet | random | 0.2 | 0.828 ± 0.031 | 0.762 ± 0.048 | 0.859 ± 0.065 | 0.836 ± 0.065 |
| PPNet | priority | 0.1 | 0.825 ± 0.032 | 0.757 ± 0.055 | 0.849 ± 0.083 | 0.844 ± 0.052 |
| GGNet | random | 0.1 | 0.813 ± 0.020 | 0.752 ± 0.022 | 0.886 ± 0.031 | 0.856 ± 0.033 |
| GGNet | random | 0.2 | 0.813 ± 0.020 | 0.751 ± 0.017 | 0.890 ± 0.037 | 0.864 ± 0.043 |
| GGNet | priority | 0.1 | 0.812 ± 0.022 | 0.749 ± 0.019 | **0.896 ± 0.031** | 0.872 ± 0.039 |
| GGNet | priority | 0.2 | 0.813 ± 0.021 | 0.749 ± 0.020 | 0.881 ± 0.031 | 0.852 ± 0.030 |
| GGNet | coarsening | 0.2 | 0.814 ± 0.021 | 0.748 ± 0.022 | 0.886 ± 0.029 | 0.860 ± 0.028 |
| GGNet | coarsening | 0.1 | 0.806 ± 0.033 | 0.734 ± 0.043 | 0.855 ± 0.044 | 0.820 ± 0.051 |

`[TBD]` Comparison with HGDC, EMOGI, MTGCN and SCIS-CDG [4,26,27,20] on the same splits, using trapezoidal AUPRC to match the published values.

## 3.3 The choice of network matters more than the augmentation setting

Network choice was the largest source of variation in performance. Averaged over the six augmentation settings, test AUPRC was 0.791 on PathNet, 0.766 on PPNet and 0.747 on GGNet (Fig 3d).

By contrast, the elimination strategy and ratio had only small effects. Within a network, AUPRC varied by 0.016 (PPNet) to 0.025 (PathNet) across the six settings, which is comparable to the fold-to-fold SD of individual configurations (0.017–0.055). On GGNet, five of the six settings fell within 0.004 of one another, and only coarsening at ρ = 0.1 was noticeably lower. The one consistent pattern was on PathNet, where random elimination scored about 0.02 lower in AUPRC than priority and coarsening at both ratios. Taken together, these results show at most a modest effect of Schur elimination strategy on predictive performance, and we do not interpret the PathNet difference as established without a paired fold-level test (`[TBD]`).

The gene rankings themselves were also robust to the augmentation setting. On GGNet, the Jaccard overlap of adjusted consensus candidates (§3.6) between pairs of augmentation settings ranged from 0.55 to 0.85 (Fig 3b).

These comparisons address how the augmentation is configured, not whether it helps. We have not yet compared the full model with one trained on the ranking loss alone (`[TBD]`: no-augmentation ablation), nor run the curvature-feature and attention-mode ablations described in §1.4 (`[TBD]`).

## 3.4 Recovery of known drivers

DriverJinn places known drivers at the top of held-out rankings (Fig 2). In the reference configuration, Precision@10 was 0.96, a 2.5-fold enrichment over random; Precision@50 was 0.86 (2.2-fold) and Precision@200 was 0.59 (1.5-fold) (Fig 2a). Screening the top 10% of each test ranking recovered 23% of that fold's drivers (Fig 2b).

To assess how labelled genes rank when they are not seen during training, we computed each gene's out-of-fold percentile, defined as its percentile position in the fold in which it was held out. This score clearly separated COSMIC Cancer Gene Census (CGC) v103 genes from other labelled genes. The median out-of-fold percentile was 0.77 for CGC Tier 1 genes (n = 484) and 0.37 for non-CGC labelled genes (n = 1,223), most of which are HGDC non-drivers (one-sided Mann–Whitney U test, P = 3.5 × 10⁻⁹¹). CGC Tier 2 genes (n = 102) were also ranked above non-CGC genes (P = 7.2 × 10⁻²¹) (Fig 2c).

The top of the ranking was stable across folds. ATM, EP300, ARID1A, CREBBP, KMT2C and KMT2D ranked within the top 10 of the full 11,183-gene ranking in most folds, including the fold in which each was a held-out test gene (Fig 3a). The same canonical drivers led the rankings on every network: ATM, KMT2C, CREBBP, KMT2D, KDM6A, EP300 and SMARCA4 had mean ranks between 2.7 and 23 across GGNet, PathNet and PPNet.

## 3.5 Raw rankings are biased toward network hubs and long genes

Two systematic biases were evident in the raw rankings. The first was a bias toward highly connected genes. The out-of-fold percentile of labelled genes increased with network degree (Spearman r_s = 0.32 on GGNet; Fig 4a), and discrimination was weakest among poorly connected genes: test AUROC computed within the lowest-degree quintile (degree 1–17) was about 0.73, compared with 0.76–0.83 in the other quintiles (Fig 4b). Among unlabelled genes, the ubiquitin hubs UBC, UBB and RPS27A were among the highest ranked on GGNet.

The second was a bias toward long genes. Across configurations, 41–73% of raw consensus candidates fell in the longest decile of coding sequence (CDS) length, where 10% would be expected by chance, and their median CDS length was 2.8–7.9 kb (Fig 4c). These candidates included TTN, OBSCN, SYNE1, DST, MACF1 and RYR2. Long genes accumulate passenger mutations in proportion to their length and are a well-known source of false positives in driver prediction.

To correct for both biases, we ranked each unlabelled gene only against unlabelled genes of similar CDS length and network degree, and we removed FLAGS genes, which are frequently mutated by chance [Shyr 2014] (Methods §2.7). After adjustment, no more than 0–4 consensus candidates per configuration (≤ 5%) fell in the longest length decile, and their median CDS length dropped to 1.1–1.5 kb (Fig 4c). The adjustment changed the candidate lists substantially: only 16–38 genes per configuration were shared between the raw and adjusted lists. All candidate analyses below use the adjusted lists unless stated otherwise.

## 3.6 Consensus candidates are enriched for unlabelled cancer genes

A small number of CGC v103 genes are not labelled in HGDC, so DriverJinn never sees them as positives during training. These genes provide an external check on how the model ranks unlabelled genes. We defined a consensus candidate as an unlabelled gene that ranked among the top 100 unlabelled genes in at least three of the five folds.

In 15 of the 18 configurations, adjusted consensus candidates (51–84 genes per configuration) were enriched for unlabelled CGC genes relative to all other unlabelled genes (one-sided Fisher's exact test, P < 0.05; odds ratio 5.5–9.6; Fig 4d). The three exceptions were GGNet coarsening at ρ = 0.1 (P = 0.10), PPNet random at ρ = 0.1 (P = 0.07) and PPNet random at ρ = 0.2 (P = 0.35). The absolute numbers are small, with 1–4 CGC genes per configuration, because HGDC already labels most CGC genes. The recovered CGC genes include ITGB1, CBFB, CTNNA1 and POLR2A.

Length and degree adjustment lowered the odds ratios but made the enrichment more consistent across networks. With the raw lists, odds ratios were 6.4–16.9 and enrichment was significant in all 12 GGNet and PathNet configurations, but in none of the six PPNet configurations. After adjustment, four of the six PPNet configurations were significant (Fig 4d). This pattern suggests that part of the raw enrichment on GGNet and PathNet came from long, highly connected CGC genes, and that the same bias masked the signal on PPNet.

No COSMIC hallmark of cancer was enriched among consensus candidates (smallest unadjusted P = 0.30). This test has limited power, because hallmarks are annotated only for CGC genes and HGDC already labels about 94% of them.

## 3.7 Robust candidates and the pathways they share

Candidate lists depended strongly on the network. Among adjusted consensus candidates under random elimination at ρ = 0.2, 72 were found only on GGNet, 49 only on PPNet and 37 only on PathNet, and just 2 were found on all three networks (Fig 3c). Because the three networks are built from different types of evidence, we treated recurrence across configurations, and especially across networks, as the main indicator of a robust prediction. We therefore ranked candidates by the number of the 18 configurations in which they were recovered.

Twenty-four unlabelled genes were adjusted consensus candidates in at least 8 of the 18 configurations (Fig 5a; Table 2). The most robust were EPHA2 (18 of 18), ITGB1 (17; a CGC Tier 2 gene not labelled by HGDC), HDAC1 (17), ITGB3 (13) and HDAC2 (13). Adjustment generally strengthened these genes: 21 of the 24 were recovered in more configurations after adjustment than before. Four of them, SH3GL2, CRKL, MNAT1 and RFC2, were not consensus candidates in any raw configuration, because longer or more highly connected genes had pushed them down the raw rankings.

Based on UniProt functional annotation, the 24 genes fall into five groups: integrin adhesion (ITGB1, ITGB3, ITGA2B, FLRT2); receptor tyrosine kinase (RTK) signalling (EPHA2, SH3GL2, SHC1, CRKL, BTC, FYN); chromatin regulation and transcription factors (HDAC1, HDAC2, CBFB, BRD2, PRMT6, MEIS1); RNA polymerase II and its associated kinase (POLR2K, POLR2C, POLR2G, MNAT1); and cell cycle and survival (CDKN2B, PPP1R13L, RFC2, BNIP3L). On GGNet, the robust candidates are connected to one another and to a shared set of CGC Tier 1 neighbours, including HSP90AB1, PBX1, HMGA1, ZFHX3, NSD2 and LRP1B (Fig 5b).

These groups differed in which networks recovered them. The four RNA polymerase II genes were consensus candidates in all six GGNet configurations but rarely on PPNet. Conversely, CBFB, BRD2, PRMT6, MEIS1, CDKN2B and PPP1R13L were recovered on PathNet and PPNet but never on GGNet. Only EPHA2, ITGB1 and HDAC1 were recovered in at least five of the six configurations on every network.

**Table 2. Robust candidates: unlabelled genes that are adjusted consensus candidates in at least 8 of 18 configurations.** Network columns give the number of configurations (out of 6 per network) in which the gene is a consensus candidate. "Raw" gives the total across all 18 configurations for the unadjusted ranking. FLRT2 is not present in PathNet.

| Gene | Group | GGNet | PathNet | PPNet | Total | Raw | CGC |
|------|-------|-------|---------|-------|-------|-----|-----|
| EPHA2 | RTK signalling | 6 | 6 | 6 | 18 | 13 | — |
| ITGB1 | Integrin adhesion | 6 | 6 | 5 | 17 | 15 | Tier 2 |
| HDAC1 | Chromatin & TFs | 6 | 5 | 6 | 17 | 12 | — |
| ITGB3 | Integrin adhesion | 2 | 5 | 6 | 13 | 12 | — |
| HDAC2 | Chromatin & TFs | 6 | 6 | 1 | 13 | 8 | — |
| SH3GL2 | RTK signalling | 6 | 1 | 5 | 12 | 0 | — |
| CBFB | Chromatin & TFs | 0 | 6 | 6 | 12 | 6 | Tier 1 |
| POLR2K | RNA Pol II | 6 | 3 | 3 | 12 | 4 | — |
| CDKN2B | Cell cycle/survival | 0 | 6 | 6 | 12 | 2 | — |
| BRD2 | Chromatin & TFs | 0 | 5 | 6 | 11 | 11 | — |
| PRMT6 | Chromatin & TFs | 0 | 5 | 6 | 11 | 4 | — |
| MEIS1 | Chromatin & TFs | 0 | 5 | 6 | 11 | 11 | — |
| PPP1R13L | Cell cycle/survival | 0 | 5 | 6 | 11 | 12 | — |
| SHC1 | RTK signalling | 1 | 3 | 6 | 10 | 3 | — |
| CRKL | RTK signalling | 0 | 4 | 6 | 10 | 0 | — |
| BTC | RTK signalling | 0 | 6 | 4 | 10 | 2 | — |
| POLR2C | RNA Pol II | 6 | 4 | 0 | 10 | 6 | — |
| ITGA2B | Integrin adhesion | 5 | 1 | 3 | 9 | 5 | — |
| FYN | RTK signalling | 6 | 1 | 2 | 9 | 3 | — |
| FLRT2 | Integrin adhesion | 4 | — | 4 | 8 | 5 | — |
| POLR2G | RNA Pol II | 6 | 2 | 0 | 8 | 6 | — |
| MNAT1 | RNA Pol II | 6 | 0 | 2 | 8 | 0 | — |
| RFC2 | Cell cycle/survival | 6 | 2 | 0 | 8 | 0 | — |
| BNIP3L | Cell cycle/survival | 0 | 6 | 2 | 8 | 2 | — |

To examine a broader set of predictions, we took the 190 non-CGC genes that were adjusted consensus candidates in at least three configurations and tested them for KEGG pathway enrichment against all unlabelled genes (one-sided Fisher's exact test with Benjamini–Hochberg correction; Fig 5c). These genes were enriched for cancer and growth-signalling pathways, including Pathways in cancer (28 genes, 4.4-fold, FDR = 1.2 × 10⁻⁸), Rap1 signalling (17 genes, 6.2-fold, FDR = 2.4 × 10⁻⁷), Ras signalling (15 genes, 5.1-fold), MAPK signalling (17 genes, 4.2-fold), Cell cycle (11 genes, 7.0-fold) and Melanoma (7 genes, 12.3-fold). They were also enriched for RNA polymerase (6 genes, 13.8-fold) and Basal transcription factors (7 genes, 10.6-fold), reflecting the RNA polymerase II group recovered on GGNet.

## 3.8 HDAC1 and HDAC2

The class I histone deacetylases HDAC1 and HDAC2 were the most consistently recovered chromatin regulators. HDAC1 was a consensus candidate in 17 of 18 configurations and HDAC2 in 13. Most of HDAC2's support came from GGNet and PathNet: across the 12 configurations on these two networks, HDAC1 was a consensus candidate in 11 and HDAC2 in all 12. On GGNet, HDAC1 ranked 8th to 11th among unlabelled genes in five of the six configurations. In contrast, SIN3A, the scaffold of the SIN3–HDAC co-repressor complex, was a consensus candidate only on PathNet (priority at ρ = 0.2 in 4 of 5 folds; coarsening at ρ = 0.2 in 3 of 5). We therefore report HDAC1 and HDAC2 as robust candidates and SIN3A as a network-specific one.

## 3.9 Multi-omic profile of the robust candidates

To understand which input features distinguish the robust candidates, we compared their pan-cancer feature percentiles with those of labelled drivers, labelled non-drivers and other unlabelled genes (Fig 6). Robust candidates had higher mean mutation-frequency percentiles across the 16 TCGA cancer types than other unlabelled genes (two-sided Mann–Whitney U test, P = 0.022), with a median that lay between those of drivers and non-drivers. They did not differ from other unlabelled genes in differential methylation (P = 0.47) or differential expression (P = 0.55) (Fig 6b).

The robust candidates therefore resemble drivers mainly in their mutation frequency. Beyond that, their prioritisation appears to depend on network context rather than on a distinctive methylation or expression signature. Individual genes vary (Fig 6a): SH3GL2 and FLRT2 have consistently high differential expression and methylation percentiles, and CDKN2B has high differential expression.

## 3.10 Summary

On HGDC's published splits, DriverJinn ranks held-out drivers well, with AUROC of 0.81–0.84 and AUPRC of 0.73–0.80 across 18 configurations, and it places canonical drivers at the top of every fold. Performance depends mainly on the choice of network, whereas the Schur elimination strategy and ratio have at most small effects. Raw candidate lists are biased toward long genes and network hubs. After each gene is ranked only against genes of similar length and degree, consensus candidates remain enriched for unlabelled CGC genes in 15 of 18 configurations and concentrate in integrin adhesion, RTK signalling, chromatin regulation, RNA polymerase II and cell-cycle functions. EPHA2, ITGB1 and HDAC1 are recovered in at least five configurations on every network. These are computational predictions and have not yet been tested experimentally.

---

## Drafting notes (delete before submission)

**Carried over from the previous draft**

- **Missing analyses** marked `[TBD]`: (1) baseline table (HGDC, EMOGI, MTGCN, SCIS-CDG) on the same splits; (2) no-augmentation ablation; (3) curvature-feature and attention-mode ablations, which the last paragraph of §1.4 promises. Either run them or remove the promise from §1.4.
- **AUPRC flavour:** Table 1 reports average precision. Published baselines use trapezoidal AUPRC; recompute (`auprc_trapz`) before building the baseline table.
- **MRR** is computed (`*_combined_metrics.csv`) but omitted; it is near-degenerate on a 39%-positive test set. Either add a column or drop MRR from §1.4 item 5.
- **Fig 5a vs §3.7 counts:** totals come from `Downstream/plots/publication/tables/robust_candidates.csv` (adjusted, FLAGS removed). The 190-gene KEGG query comes from `length_adjusted/consensus_counts.csv` (`configs_flags ≥ 3`, non-CGC); check against the notebook's printed `KEGG query:` line.
- **Fig 5 caption** says "24 unlabelled genes"; these include CBFB (CGC Tier 1, unlabelled by HGDC) and ITGB1 (Tier 2). Text and caption agree.
- **§3.8** uses adjusted values only; the README's SIN3A ranks ("PathNet 6–47") are raw and should not be cited here.
- **Figure/panel numbering** follows `publication_figures.qmd` as of 2026-10-28.
- **Pathway-attention analysis** (contribution 3) cannot be shown yet: checkpoints do not reproduce the saved scores (Spearman 0.84). Resolve, or retrain with attention export, before writing an interpretability subsection.
- **[Shyr 2014]** needs a reference number in the MODEL_MATH.md list.

**New in this revision**

- **§3.3 claim corrected.** The old text said strategies differ by "at most 0.02 AUPRC" and that this is "smaller than the fold-to-fold SD of every configuration". PathNet spans 0.779–0.804 (0.025), which exceeds several PathNet SDs (e.g. 0.023). Random elimination is also lower on PathNet at both ratios. Rewritten as "at most modest"; a paired fold-level test (e.g. Wilcoxon or a mixed model on the 5 shared folds) would let you make a firmer statement.
- **§3.10 claim corrected.** "EPHA2, ITGB1 and HDAC1 are recovered on all three networks" was too broad: many Table 2 genes appear on all three. Now says "at least five configurations on every network", matching §3.7.
- **Spearman symbol** changed from ρ to r_s in §3.5 to avoid clashing with the elimination ratio ρ.
- **§3.5 AUROC by degree:** the old wording ("AUROC for drivers in the lowest-degree quintile") implied a single-class AUROC. Now "computed within the lowest-degree quintile"; confirm that is what the code does.
- **§3.1 counts:** PPNet labelled-gene count is missing. Also, 796/2,983 ≈ 27% drivers overall but ~39% within the test sets; a reviewer will notice, so state per-network driver/non-driver counts (ties to the open "gene counts per filtering step" item).
- **§3.6 causal language softened** ("therefore partly driven by" → "suggests").
- **§3.9 multiple testing:** P = 0.022 is one of three tests and would not survive Bonferroni (0.066). Either report corrected P-values or frame the mutation-frequency result as suggestive.