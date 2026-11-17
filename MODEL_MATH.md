# DriverJinn — Methods (cited draft)

Rewritten from the working mathematical formulation, with primary-source citations
throughout. Numbered references `[n]` resolve against the list at the end.
Code anchors are preserved so the manuscript and the implementation stay checkable
against each other.

---

## Editorial notes — read before using this draft

Six substantive changes, plus three things to verify against the code.

**1. The effective-resistance claim has been split in two (M3).** The old draft
stated exact resistance preservation next to the exact Schur complement, then
disclosed in a later paragraph that the complement is actually *sampled*. A
reviewer will read that as the exact guarantee applying to what you ran, which it
does not. The rewrite states the exact property as a property of Kron reduction
[21,22], then states what the sampled construction actually gives you (unbiased in
expectation, spectral approximation with high probability) [23], in the same
paragraph as the equation. This is honest and still a strong argument.

**2. M4's novelty claim is recalibrated, and it becomes stronger, not weaker.**
The six-pathway design — curvature regime × hop depth — and the curvature-homophily
descriptor are Attali *et al.*'s framework [12], not yours; claiming them invites a
reviewer who knows that paper to distrust the rest. But Attali *et al.* close their
paper by naming curvature-conditioned attention as future work: they state a plan to
build an attention mechanism using not only node features but also edge curvature.
DriverJinn is that mechanism. Framing M4 as "we implement what [12] proposed and
did not build, and apply it to driver-gene ranking" is a cleaner and more defensible
claim than "this is our contribution."

**3. M1 now cites prior curvature work on cancer networks.** Sandhu *et al.* [1]
already showed Ollivier–Ricci curvature separates tumour from normal gene
co-expression networks. Not citing this looks like you don't know the field; citing
it establishes that curvature is a validated signal in this domain and that you are
extending it from a global network descriptor to a per-gene predictive feature.

**4. M6's argument against AUROC is now sourced.** Saito & Rehmsmeier [43] is the
canonical bioinformatics citation for AUROC being deceptive under class imbalance,
and it was published in a venue this readership reads. The argument reads as a
methodological position rather than a preference once it is attached.

**5. M5 is reframed as positive-unlabelled learning explicitly.** "Unlabelled, not
negative" is the PU-learning setting [32,33,34,35], and there is a Briefings review of PU
learning in bioinformatics [35] you can cite in one clause. This converts a design
choice that looks idiosyncratic into a recognised problem formulation.

**6. Terminology.** The draft used "Forman–Ricci" for what `GraphRicciCurvature`
computes by default, which is *augmented* Forman–Ricci [46,13]. S1.2 now says so. The
distinction matters because Topping *et al.*'s Balanced Forman [10] is a third,
different variant, and an ablation reviewer will ask which one you used.

### Verified against the code — all three confirmed

- **Per-fold scaler (2.2) ✓.** `normalize_features_per_fold`
  ([train_model.py:438](model/train_model.py#L438)) constructs a fresh
  `StandardScaler()` at [:464](model/train_model.py#L464) and fits it on
  `raw_features_np[train_mask_np]`; it is called from inside the fold loop at
  [:637](model/train_model.py#L637) and reads `original.get('x_raw', features)`
  to avoid double-normalising. The description is accurate.
- **NDCG gain convention (2.6) ✓.** [DriverGenePredictor.py:1139](model/DriverGenePredictor.py#L1139)
  sets `gains = sorted_labels` — linear gain, no exponentiation.
- **Ollivier idleness (S1.1) ✓.** `alpha=0.5` is passed explicitly at
  [curvature_calculator.py:35](graph_builder/curvature_calculator.py#L35). The
  claim that it is also the library default has been dropped from S1.1 as an
  unnecessary hostage.

**7. Kothapalli [19] is the missing link in M3, and it must be cited.** The draft
credited Schur-complement augmentation to SCIS-CDG [20] and the sampling routine to
Kyng & Sachdeva [23]. But Kothapalli introduced the randomized Schur complement
*as a graph-contrastive augmentor* at ICML 2023 — generating unbiased approximations
of Schur complements and treating the resulting graphs as augmented views, with
theoretical justification and an explicit connection to graph diffusion. That is
your M3, one year before SCIS-CDG. Omitting it and citing only the cancer paper
would look like citing the application while skipping the method, which reviewers in
this area do notice. Corrected lineage: Kyng & Sachdeva [23] → Kothapalli [19] →
SCIS-CDG [20] → DriverJinn.

**8. HGDC gives you a quantitative, published justification for the curvature
premise on your exact networks.** Zhang *et al.* measured the homophily ratio of
driver genes — the fraction of a driver's neighbours that are themselves drivers —
and found 0.0951 on GGNet, 0.1853 on PathNet and 0.2085 on PPNet, concluding all
three are heterophilic. Heterophily and negative Ricci curvature are the same
phenomenon viewed from two angles: a gene whose neighbours are unlike it and unlike
each other sits on edges whose endpoint neighbourhoods barely overlap, which is what
negative $\kappa$ measures. So HGDC has already shown, on these networks, that
drivers occupy the positions curvature is built to detect. M1 now states this. It is
the strongest single sentence available for motivating the method, and it costs you
nothing because HGDC is a baseline you were going to cite anyway.

### Two corrections HGDC forces

**A. Label policy — RESOLVED against the code; 2.5 rewritten.** The draft claimed
ranking pairs are drawn from "the labelled positives and the curated negatives."
The code does not do this. Only one binary `label` vector is loaded — the dataset
keys read are `edge_index`, `feature`, `x`, `label`, `kfold_splits`,
`ollivier_curvature`, with no negative mask
([train_model.py:400](model/train_model.py#L400) initialises all-zero and copies
driver labels in; [:1392](model/train_model.py#L1392) comments `true_labels == 0`
as "Only unknowns"). `RankingLoss.forward` then takes
`non_driver_mask = (labels == 0)`, i.e. every non-positive gene in the training
mask. The scoring summaries confirm the partition: 11,395 genes = 716 known
drivers + 10,679 "unknown".

What is not used is a different key. `dataset_*.pkl` ships **`mask`**, HGDC's
flag for which genes carry an annotation at all (positives + EMOGI curated
negatives). `curvature_pipeline.py:195` can read it, but only under
`--use_existing_mask`; [preprocess_script.sh](preprocess_script.sh) passes
`--use_kfold`, which routes to `create_kfold_splits` and builds StratifiedKFold
splits over `np.arange(num_nodes)` — all 11,395 genes. `train_model.py` never
reads `mask`. The `_ten_5CV.pkl` variants carry HGDC's own `split_set` (their
10×5CV protocol); that key is never read either.

Consequence (runs up to May 2026): label 0 pooled curated non-drivers with
unannotated genes, and that pooled class was the negative class in the BPR loss
*and* in AUROC/AUPRC/NDCG, so the reported metrics were not comparable with
published baselines, which evaluate on the annotated subset.

**RESOLVED (Oct 2026): HGDC's protocol adopted.** `train_model.py --split_file`
now reads HGDC's released `split_set` (repeat 0 of 10) through
`load_hgdc_splits` ([train_model.py:440](model/train_model.py#L440)): train and
test folds cover HGDC's annotated genes only, unannotated genes are never used as
labels, and an inner validation split handles model selection so each test fold is
evaluated once. Sections 2.0, 2.5, 2.6, S2, S6.3 and S7.2–S7.3 are rewritten to
match. All results produced before this change must be regenerated before any number
from them enters the manuscript.

**B. GGNet is not a protein interaction network.** The three networks are different
in kind: PathNet is built from KEGG and Reactome pathways, GGNet is a gene–gene
interaction network extracted from the Encyclopedia of RNA Interactomes, and PPNet
is the STRING PPI network. M1's biological reading — "protein complexes," "shared
complex" — is correct for PPNet, roughly correct for PathNet, and wrong for GGNet,
whose edges are RNA-level interactions. M1 now uses network-neutral language
("functional module," "co-regulated set") with the complex-specific reading marked
as the PPNet case.

### Data provenance — largely resolved

Since the source data is the HGDC release, most of the `[DATA-REF]` gaps close by
citing [4] and its repository (`github.com/NWPU-903PR/HGDC`). Specifically: the
gene feature matrix is $X \in \mathbb{R}^{N \times 58}$, so $d_0 = 58$ and
$\tilde{x}_v \in \mathbb{R}^{73}$; the three networks are as described in B above;
positives are CDGs and negatives follow EMOGI's compilation.

Still to supply yourself: the CGC/COSMIC version behind the positive set (cite
Tate *et al.* [29] for the database and Sondka *et al.* [28] for the Census gene
list, and give the version number); how your 716 positives relate to HGDC's CDG list
(identical? filtered? intersected with your networks?); your gene counts before and
after filtering, since $n = 11{,}395$ differs slightly from the published PPNet node
count and you should say why. For the primary sources behind the three networks
themselves, lift references 33, 37 and 38 from HGDC's own reference list rather
than reconstructing them — that guarantees you cite the same releases.

---

# PART I — MAIN TEXT

---

## 2.0 Dataset

DriverJinn is evaluated on three gene interaction networks of different
construction, all taken from the HGDC release [4] so that features, labels and
networks match published baselines: **GGNet**, a gene–gene interaction network
derived from RNA-level interactomes; **PathNet**, built from KEGG and Reactome
pathways; and **PPNet**, the STRING protein interaction network. The three
networks are built over different gene sets, not a shared one — $n$, and the
label split within it, is per-network:

| Network | Genes ($n$) | Drivers | Curated non-drivers | Unlabelled |
|---|---|---|---|---|
| GGNet   | 11,183 | 708 | 1,101 | 9,374 |
| PathNet | 7,695  | 598 | 701   | 6,396 |
| PPNet   | 11,395 | 716 | 1,026 | 9,653 |

(counted from HGDC's `dataset_*_ten_5CV.pkl`; drivers and curated non-drivers
together form HGDC's annotated set). Every network shares the same
58-dimensional multi-omics feature schema $X \in \mathbb{R}^{n \times 58}$,
later extended to 73 dimensions with the fifteen curvature descriptors of 2.2;
only $n$ and the specific genes present differ. Elsewhere in this document,
"716 known drivers" (2.6, S8) refers specifically to PPNet and should not be
read as a figure shared across all three networks — state the per-network
count wherever a specific run is being described.

Labels follow HGDC [4]: known drivers are positives, EMOGI-compiled likely
non-drivers [26] are negatives, and every remaining gene is unlabelled. Training
and evaluation use only the annotated genes, through HGDC's released fold
assignment (2.5); unlabelled genes are scored but never used as labels. The
COSMIC/CGC version and tier filter underlying HGDC's own driver list is not
independently verified here.

This is distinct from the downstream validation and enrichment analysis
(2.7; `Downstream/publication_figures.qmd`,
`Downstream/length_adjusted_candidates.py`), which never participates in training and
instead labels predictions against a separately sourced reference: COSMIC
Cancer Gene Census v103, GRCh38 [28,29] (`Downstream/cosmic/`), split into
Tier 1 and Tier 2. The two uses of "CGC" in this paper are not guaranteed to
be version-matched, and this difference should be stated plainly rather than
implied to be one dataset.

Because training and evaluation use HGDC's own annotated genes and fold
assignment, the metrics in 2.6 are computed over the same genes, negative
population and test folds as the published baselines on these networks [4].

---

## 2.1 Discrete Ricci curvature on the interaction network

Gene interaction networks are not geometrically uniform. Some edges sit inside
densely interconnected functional modules — co-regulated sets, signalling cassettes,
and in a protein interaction network such as PPNet, physical complexes — where any
two partners of a gene are themselves likely to interact. Other edges are bridges:
the sole route between two otherwise separate modules, carrying all information flow
between them. Discrete Ricci curvature distinguishes these two regimes with a single
scalar per edge, and it has already been shown to carry cancer-relevant signal:
Sandhu *et al.* found that Ollivier–Ricci curvature reliably separates tumour from
normal gene co-expression networks, and related it quantitatively to network entropy
as a measure of robustness [1]. Subsequent work extended the same geometry to drug
response [2] and to prognosis in ovarian carcinoma [3]. DriverJinn takes this
signal from a global network descriptor to a per-gene predictive feature and an
inductive bias inside message passing.

There is direct evidence that drivers occupy the low-curvature regions of the
specific networks used here. Zhang *et al.* measured the homophily ratio of known
driver genes — the fraction of a driver's neighbours that are themselves drivers —
and obtained 0.0951 on GGNet, 0.1853 on PathNet and 0.2085 on PPNet, well below the
0.5 threshold at which a network is considered heterophilic [4]. A driver whose
neighbours are largely non-drivers, and whose neighbours are largely unlike one
another, sits on edges whose endpoint neighbourhoods barely overlap — precisely the
configuration that yields $\kappa(u,v) < 0$. Heterophily as measured by [4] and
negative Ricci curvature are therefore two views of the same structural fact, and
GGNet, the most heterophilic of the three networks, is also the one on which
homophily-assuming GCN architectures degrade most [4]. Curvature offers a way to
condition on that structure rather than average over it.

For an edge $(u,v)$, let $\mu_u$ be the lazy random-walk measure at $u$ — mass
$\alpha = 0.5$ retained at $u$, the remainder distributed over neighbours
proportionally to interaction confidence weights. Following Ollivier [5,6], the
Ollivier–Ricci curvature is the normalised optimal-transport deficit between
endpoint measures:

$$
\kappa(u,v) \;=\; 1 \;-\; \frac{W_1(\mu_u,\, \mu_v)}{d(u,v)}
$$

where $W_1$ is the 1-Wasserstein (earth-mover's) distance [7,8] and $d(u,v)$ the
weighted shortest-path distance. Curvature is computed with the
`GraphRicciCurvature` library [9]
([graph_builder/curvature_calculator.py:35](graph_builder/curvature_calculator.py#L35)).

**Biological reading.** $\kappa(u,v) > 0$ means the two genes' neighbourhoods
overlap substantially, so probability mass transports cheaply between them — the
signature of a shared complex or co-regulated module, where function is distributed
redundantly across partners. $\kappa(u,v) < 0$ means the neighbourhoods are
near-disjoint and all mass must funnel across the single edge — a bottleneck.
Perturbing a gene at a bottleneck propagates across module boundaries, whereas
perturbing a gene inside a redundant module is buffered by its partners. The same
quantity has an independent justification from the learning side: Topping *et al.*
proved that negatively curved edges are precisely the edges responsible for
over-squashing in message-passing networks, the failure mode in which information
from distant nodes is compressed to the point of loss [10,11]. Curvature therefore
identifies both the topological positions we expect drivers to occupy and the
positions where a naive GNN would fail to propagate their signal — which is why it
is used both as a node feature (2.2) and inside message passing (2.4).

> Supplementary: exact transport LP and coupling formulation (S1.1); augmented
> Forman–Ricci curvature, used for node features only (S1.2).

---

## 2.2 Curvature feature construction

Per-edge curvatures are aggregated onto genes. For gene $v$ with incident edges
$\mathcal{E}_v$ and curvature multiset
$\mathcal{K}_v = \{\kappa(u,v) : (u,v) \in \mathcal{E}_v\}$, fifteen descriptors are
appended to the raw omics features:

$$
\tilde{x}_v \;=\; \bigl[\, x_v \,\big\|\, \phi_v \,\bigr] \;\in\; \mathbb{R}^{d_0 + 15}
$$

where $x_v \in \mathbb{R}^{58}$ is the multi-omics feature vector of the HGDC
release [4], giving $\tilde{x}_v \in \mathbb{R}^{73}$.

**Table 1 — curvature descriptors ($\phi_v$).**

| # | Descriptor | Definition | Biological reading |
|---|---|---|---|
| 1–6 | Ollivier block | mean, sd, min, max, median, count over $\mathcal{K}_v$ | distribution of transport geometry across the gene's interactions |
| 7–12 | Augmented Forman block | same six statistics over $\mathcal{K}^{AF}_v$ | combinatorial corroboration, robust where transport is degenerate |
| 13 | $\deg^+_v$ | $\sum_{(u,v)} \mathbb{1}[\kappa(u,v) > 0]$ | number of redundant, in-module interactions |
| 14 | $\deg^-_v$ | $\sum_{(u,v)} \mathbb{1}[\kappa(u,v) < 0]$ | number of bottleneck interactions |
| 15 | $\mathrm{hom}_v$ | $\frac{1}{\lvert\mathcal{E}_v\rvert}\sum_{u \sim v} \bigl(1 + \lvert\bar{\kappa}_v - \bar{\kappa}_u\rvert\bigr)^{-1}$ | curvature homophily, after [12] |

Including both discretisations is deliberate rather than redundant. Samal *et al.*
showed the two are strongly but not perfectly correlated across model and real
networks, with agreement highest for the augmented Forman variant [13]; where
transport is degenerate the combinatorial measure remains informative, and the two
blocks together are cheaper than either alone would be reliable. The homophily
descriptor $\mathrm{hom}_v$ follows the curvature-constrained homophily metric of
Attali *et al.* [12]: it approaches 1 when a gene sits in a curvature-homogeneous
neighbourhood and drops at the interface between a dense module and a tree-like
bridge — the position many known drivers occupy
([graph_builder/curvature_integration.py:92-175](graph_builder/curvature_integration.py#L92-L175)).

**Normalisation.** Features are stored unnormalised; within each cross-validation
fold a `StandardScaler` [14] is fit on that fold's **training nodes only** and
applied to all nodes and every augmented view
([model/train_model.py:435](model/train_model.py#L435), applied per fold at
[model/train_model.py:636](model/train_model.py#L636)). Scaler parameters are
therefore estimated without reference to held-out genes, so no distributional
information from the validation or test split enters training.

> Supplementary: full per-fold standardisation equations and zero-variance
> handling (S2).

---

## 2.3 Schur-complement graph augmentation

Contrastive learning requires two stochastic views of the network [15,16,17]. The
standard graph augmentations — random edge dropping and node dropping [18] — are
poorly suited to interaction networks: deleting a bridge edge disconnects a
functional module, so the two views disagree about pathway connectivity and the
model is trained toward an invariance that is biologically false. This concern is
not unique to us; Zhu *et al.* argue on general grounds that graph augmentations
should preserve intrinsic structure rather than perturb uniformly [17].

The alternative we adopt originates with Kothapalli [19], who introduced a
randomized topological augmentor that generates unbiased approximations of a
Laplacian's Schur complements and treats the resulting graphs as contrastive views,
with theoretical justification and an explicit connection to graph diffusion. That
connection is worth noting here: HGDC independently found that a diffusion-derived
auxiliary network raises the driver homophily ratio on all three networks — from
0.0951 to 0.1338 on GGNet, 0.1853 to 0.2685 on PathNet, 0.2085 to 0.2163 on PPNet —
and attributed its performance gain to that effect [4]. A Schur-complement view is
reached by a different route but shares the diffusion semantics. SCIS-CDG
subsequently applied the same class of augmentor to cancer driver networks [20];
DriverJinn follows that line.

DriverJinn instead **marginalises genes out**. Partition
$V = A \,\dot\cup\, B$ with $B$ a randomly chosen eliminated set,
$|B| = \lfloor \rho n \rfloor$, and block the graph Laplacian $L = D - W$
accordingly. The augmented view is the Schur complement onto the retained set:

$$
L^{(S)} \;=\; L_{AA} \;-\; L_{AB}\, L_{BB}^{-1}\, L_{BA}
$$

This operation is Kron reduction [21]. $L^{(S)}$ is again a valid graph Laplacian on
$A$, and it preserves effective resistance [22] exactly:
$R^{L^{(S)}}_{\text{eff}}(u,v) = R^{L}_{\text{eff}}(u,v)$ for every $u,v \in A$ [21].
Removing a gene therefore preserves how strongly every remaining pair of genes is
connected through the network as a whole, while changing the local wiring. The two
views differ in structure but agree in global flow geometry — which is what makes
them a valid positive pair for contrastive learning.

**What is actually computed.** The complement is *sampled*, not formed exactly:
materialising $L_{AB}L_{BB}^{-1}L_{BA}$ creates a $\deg(v)^2$ clique per eliminated
gene, which is intractable at $|E| \sim 6\times10^5$. DriverJinn substitutes the
randomised elimination step of Kyng & Sachdeva [23], which contributes
$\deg(v) - 1$ edges per eliminated gene in place of $\binom{\deg(v)}{2}$, and whose
expectation is the exact complement (S3.1). Resistance preservation therefore holds
in expectation and, by the matrix-martingale analysis of [23], as a spectral
approximation with high probability, rather than exactly.

Two views are drawn independently at $\rho = 0.2$ under one of three elimination
orders (`random`, `priority`, `coarsening`; `random` performs best).

> Supplementary: the sampling recursion (S3.1); elimination-order comparison (S3.2);
> cross-view gene alignment (S3.3).

---

## 2.4 Curvature-aware attention

Graph attention [24,25] scores an interaction from the two genes' features alone.
Attali *et al.* [12] showed that constraining message passing by edge curvature —
propagating separately along positively and negatively curved edges, at one and two
hops — mitigates over-squashing and outperforms curvature-based rewiring [10] on
node classification. They also identify, as future work, an attention mechanism that
conditions on edge curvature rather than only on node features. DriverJinn
implements that mechanism and applies it to driver-gene ranking: how much a gene
attends to a partner depends on the *topological role* of the interaction connecting
them, not only on what the partner looks like.

Let $h^k_v$ be gene $v$'s representation at head $k$ and $\kappa_{vu}$ the curvature
of edge $(v,u)$. The default (`hybrid`) logit is a learned mixture of two ways of
injecting curvature — as an additional evidence channel inside the score, and as a
prior added on top of it:

$$
e^k_{vu} \;=\;
\omega_1 \underbrace{\sigma_{\text{LR}}\!\left(\mathbf{a}_{\text{f}}^{k\top}\bigl[h^k_v \,\|\, h^k_u \,\|\, \beta\kappa_{vu}\bigr]\right)}_{\text{curvature as feature}}
\;+\;
\omega_2 \underbrace{\left[\sigma_{\text{LR}}\!\left(\mathbf{a}_{\text{b}}^{k\top}\bigl[h^k_v \,\|\, h^k_u\bigr]\right) + b_k \kappa_{vu}\right]}_{\text{curvature as bias}}
$$

with $\boldsymbol{\omega} = \operatorname{softmax}(\boldsymbol{\omega}_{\text{raw}})$
learned per layer, $\beta$ and $b_k$ learned scales, and $\sigma_{\text{LR}}$ a
LeakyReLU. Logits are normalised over each gene's neighbourhood by a
numerically-stabilised softmax to give attention weights $\alpha^k_{vu}$
([model/message_passing.py](model/message_passing.py)). The bias term is the graph
analogue of an additive attention prior; the feature term treats curvature as an
edge attribute in the sense of [25].

**Multi-pathway encoding.** Following the curvature-regime × hop-depth factorisation
of [12], the encoder runs six independent stacks, one for each combination of
curvature regime $\{$positive, negative, all$\}$ and propagation depth
$\{$1-hop, 2-hop$\}$, and combines them with a learned per-gene attention over
pathways ([model/curvature_aware_gnn.py](model/curvature_aware_gnn.py),
[model/multi_layer_attention.py](model/multi_layer_attention.py)). The resulting
pathway weights $a^{(c,\eta)}_v$ are directly interpretable: they report whether a
given gene's score is driven by its bottleneck interactions or its in-module ones,
and are reported for top-ranked predictions. Interpretability of this kind is a
standing expectation for network-based driver methods [26,27].

> Supplementary: the four alternative attention modes and their ablation (S4.1);
> stabilised softmax (S4.2); two-hop propagation and layer composition (S4.3); the
> six-pathway encoder and two-stage aggregation in full (S5).

---

## 2.5 Training objective

Labels are taken from the HGDC release [4]. Known drivers $\mathcal{P}$ are
positives, and the likely non-drivers compiled by the procedure of EMOGI [26] are
negatives $\mathcal{N}$; together they form the annotated set
$\mathcal{A} = \mathcal{P} \cup \mathcal{N}$. Every remaining gene $\mathcal{U}$
is treated as **unlabelled, never negative**: a gene absent from the COSMIC Cancer
Gene Census [28,29] may simply be undiscovered, and driver catalogues have grown
substantially with each pan-cancer release [30,31]. This is the
positive-unlabelled (PU) setting [32,33], well established in bioinformatics
[34,35]. Rather than assign these genes a label, DriverJinn leaves them out of
every supervised term and scores them only at inference, so each fold's model
ranks them as genes it has never seen labelled.

**Data partitioning and model selection.** Folds are HGDC's released assignment
(`split_set`, repeat 0 of the 10 provided), which partitions $\mathcal{A}$ into
five disjoint test folds $\mathcal{A}^{(f)}_{\text{te}}$
([train_model.py:440](model/train_model.py#L440)). For fold $f$, the remaining
annotated genes are split once more, stratified by label (85/15, seed 42), into
an inner training set and an inner validation set:

$$
\mathcal{A} \setminus \mathcal{A}^{(f)}_{\text{te}}
\;=\; \mathcal{A}^{(f)}_{\text{tr}} \,\sqcup\, \mathcal{A}^{(f)}_{\text{val}} .
$$

The three sets have strictly separate roles:

| Set | Role | GGNet / PathNet / PPNet (genes per fold) |
|---|---|---|
| $\mathcal{A}^{(f)}_{\text{tr}}$ | supervised ranking loss | ≈1,230 / 885 / 1,185 |
| $\mathcal{A}^{(f)}_{\text{val}}$ | learning-rate schedule, early stopping, checkpoint selection | ≈217 / 156 / 209 |
| $\mathcal{A}^{(f)}_{\text{te}}$ | evaluated **once**, after training; the only numbers reported | 362 / 260 / 349 |

No test-fold gene contributes a label to training, selection or early stopping.
All genes, including $\mathcal{U}$ and the test fold, still take part *unlabelled*
in message passing and in the label-free contrastive term — the standard
transductive setting for node ranking on a fixed graph, shared by HGDC and the GNN
baselines it reports [4,26,27].

The model is trained as a ranker, never as a binary classifier, and the objective
combines a self-supervised and a supervised term:

$$
\mathcal{L} \;=\;
\lambda\, \mathcal{L}_{\text{con}}\bigl(\tilde{G}_1, \tilde{G}_2\bigr)
\;+\;
(1 - \lambda)\, \varsigma\, \mathcal{L}_{\text{rank}}(G)
$$

**Training against augmented positive pairs.** Each gene is scored once in each
of the two Schur-complement views $\tilde{G}_1, \tilde{G}_2$ (2.3), and the two
resulting embeddings of the *same* gene form a positive pair; every other gene
in the batch is a negative. $\mathcal{L}_{\text{con}}$ is NT-Xent [15,16,36] at
temperature $\tau = 0.3$:

$$
\mathcal{L}_{\text{con}} = -\log \frac{\exp(\operatorname{sim}(z_v^{(1)}, z_v^{(2)}) / \tau)}{\sum_{u \neq v} \exp(\operatorname{sim}(z_v^{(1)}, z_u^{(2)}) / \tau)}
$$

with $z_v^{(1)}, z_v^{(2)}$ gene $v$'s embeddings from the two views and
$\operatorname{sim}(\cdot,\cdot)$ cosine similarity. This term requires no
labels, so it learns from the whole network rather than only the 716 annotated
drivers — the same motivation that led MTGCN to add a link-prediction auxiliary
task [27] and later methods to adopt graph contrastive objectives outright
[20,37]. Because the two views differ only in which genes Schur elimination
marginalised out, never in driver labels, minimising $\mathcal{L}_{\text{con}}$
pulls a gene's two view-embeddings together regardless of which neighbours
survived elimination in either view — the encoder is pushed toward
representations that depend on a gene's overall position in the network rather
than the specific local wiring of any one sampled view. This is the sense in
which the term makes the model topology-*invariant*, not topology-independent:
it remains a function of the graph, just not of which perturbed copy of the
graph it was shown.

**Ranking objective.** $\mathcal{L}_{\text{rank}}$ is a Bayesian Personalised
Ranking loss [38] over (driver, curated non-driver) pairs drawn from
$\mathcal{A}^{(f)}_{\text{tr}}$, focal-weighted [39] and trained with
hard-negative mining on a curriculum (S6.3). BPR is the natural choice here: it
optimises the order of pairs directly, which is what a ranked shortlist needs, and
it avoids the hard threshold a binary classifier would need to separate driver
from non-driver.

$\lambda = 0.2$, and the ranking term is rescaled by $\varsigma = 5$ to match
the dynamic range of the contrastive term; the effective weighting between the
two objectives is set by $\lambda$ and $\varsigma$ jointly. Contrastive
learning runs on the augmented views; ranking runs on the original graph.
Augmentation shapes the representation; it never defines the ranking target, so
no prediction is ever made on a perturbed network, and every gene in the
network remains scoreable at inference.

> Supplementary: NT-Xent and BPR in full, the hard-negative curriculum, and
> alternative ranking losses (S6).

---

## 2.6 Evaluation

All reported metrics are computed on HGDC's five test folds
$\mathcal{A}^{(f)}_{\text{te}}$ (2.5), each evaluated once, with the
exponential-moving-average weights [40,41] of the checkpoint selected on the inner
validation set (S7.2); figures are mean ± SD over the five folds. The primary metric is NDCG@50
[42]: with $v_{(1)},\dots,v_{(K)}$ the top-$K$ genes by score and $y_v \in \{0,1\}$,

$$
\mathrm{NDCG@}K \;=\; \frac{1}{\mathrm{IDCG@}K} \sum_{k=1}^{K} \frac{y_{v_{(k)}}}{\log_2(k+1)},
\qquad
\mathrm{IDCG@}K = \sum_{k=1}^{\min(K,\,|\mathcal{P}|)} \frac{1}{\log_2(k+1)}
$$

using linear gain; under binary relevance this is identical to the exponential gain
$2^{y}-1$ convention [42]
([model/DriverGenePredictor.py:1122](model/DriverGenePredictor.py#L1122)).

**Why NDCG@50 and Precision@50 rather than AUROC.** The ranking is deployed over
the whole network, where known drivers are rare (716 of 11,395 genes, ≈6%, on
PPNet), so AUROC would be dominated by the order of the vast unlabelled majority.
Saito & Rehmsmeier showed that ROC analysis is visually and
inferentially misleading under exactly this kind of imbalance, and that
precision-oriented measures give a more faithful picture of what a classifier will
do in deployment [43]. More importantly, AUROC does not match the use case: the
output of this method is a shortlist a lab will follow up experimentally, so what
matters is how many true drivers appear in the top tens of genes. NDCG@50 measures
that, with a position discount. AUROC, AUPRC, Precision@K, Recall@K
($K \in \{10,20,50,100\}$) and MRR are all reported for comparability with prior
work [26,27,37], since the published baselines on these three networks report AUC
and AUPR under 10 repetitions of 5-fold cross-validation [4] and a comparison is
otherwise impossible.

Because DriverJinn is trained and tested on HGDC's own annotated genes and fold
assignment (2.5), its AUROC and AUPRC are computed over the same genes, negative
population and test folds as the published HGDC, EMOGI, MTGCN, GCN, GAT, Chebnet
and SVM figures [4], and can be tabulated against them directly. Two caveats
remain. First, those figures average 10 repetitions of 5-fold cross-validation,
whereas DriverJinn reports repetition 0; this is stated alongside any comparison
table. Second, on HGDC's test folds drivers make up ≈39–46% of genes, so a random
ranking already scores NDCG@50 and Precision@50 near that share; these values are
not comparable with any computed over the full network. No prior method on these
networks reports NDCG@K, so where a ranking comparison is required, baselines are
rerun from their released implementations on the same folds and scored with the
same NDCG@K code as DriverJinn.

**Two AUPRC conventions.** DriverJinn's AUPRC is average precision [14], the
step-wise sum over the PR curve with no interpolation (S8). HGDC and SCIS-CDG
instead report the trapezoidal area under the PR curve, which interpolates
linearly between operating points and so overstates precision where recall
jumps [66]. Both are computed
([model/DriverGenePredictor.py:752-756](model/DriverGenePredictor.py#L752-L756));
the trapezoidal value (`auprc_trapz`) is used only when tabulating against those
baselines, and the convention is stated in every table.

> Supplementary: full metric definitions (S8).

---

## 2.7 Candidate prioritisation and enrichment

Unlabelled genes have no label to evaluate against, so the candidate list is
built from their scores under the five test-fold models. Raw scores are not
calibrated across folds, so within fold $f$ each unlabelled gene $u$ gets only a
rank $r^{(f)}(u)$ among the unlabelled genes. A gene is a *consensus candidate* if
$r^{(f)}(u) \le 100$ in at least three of the five folds
([model/aggregate_fold_results.py:125](model/aggregate_fold_results.py#L125)), and
candidates are ordered by mean within-fold rank.

**Length and degree correction.** Raw rankings over-represent very long genes and
network hubs. In the raw consensus lists, 27–69 of the 58–96 candidates per
configuration fall in the longest coding-length decile, and among labelled genes
the out-of-fold percentile rises with degree (Spearman $\rho = 0.32$). Both
patterns are known confounders: long genes accumulate passenger mutations at the
background rate [61,67], and hubs aggregate more messages and have more labelled
neighbours. Before any gene is named, its within-fold score is therefore replaced
by its percentile within a stratum of genes of similar size and connectivity,

$$
\pi^{(f)}(u) \;=\; \frac{\bigl|\{w \in B(u) : s^{(f)}_w \le s^{(f)}_u\}\bigr|}{|B(u)|},
\qquad
B(u) = \{\,w : \mathrm{dec}_L(w) = \mathrm{dec}_L(u),\ \mathrm{quin}_D(w) = \mathrm{quin}_D(u)\,\}
$$

where $\mathrm{dec}_L$ is the coding-length decile and $\mathrm{quin}_D$ the degree
quintile, both taken over the unlabelled genes, giving 50 strata. Coding length is
the union of CDS intervals across GENCODE v28 basic transcripts [68], the
annotation shipped with HGDC's preprocessing [4]. Degree is the undirected degree
in the network the model was trained on. Genes are re-ranked by $\pi^{(f)}$, with
ties broken by raw score. Strata are used rather than a regression on length
because the effect is not linear: it sits almost entirely in the longest decile.
The script asserts that $|\rho(\pi, \text{length})| < 0.05$ after adjustment.
Finally, the FLAGS genes — the 100 genes most frequently mutated in public exomes
of healthy individuals [67] — are removed from the unlabelled pool, because the
longest-length decile is broad enough that genes such as TTN still lead it.
Consensus is then recomputed exactly as above, so raw and adjusted lists are
directly comparable.

**Enrichment.** Each consensus list is tested for Cancer Gene Census v103
membership [28,29] against the remaining unlabelled genes with a one-sided
Fisher's exact test, and results are reported for both raw and adjusted lists
([Downstream/length_adjusted_candidates.py](Downstream/length_adjusted_candidates.py)).
Because the unlabelled pool excludes HGDC's annotated genes, a CGC gene here is
one that HGDC's training labels missed. This makes the test a check against an
independent reference, not a recovery of training positives.

---

# PART II — SUPPLEMENTARY METHODS

---

## S1. Curvature definitions in full

### S1.1 Ollivier–Ricci

Definition after Ollivier [5,6]. Lazy random-walk measure at $u$ with idleness
$\alpha$:

$$
\mu_u^\alpha(z) \;=\;
\begin{cases}
\alpha, & z = u,\\[4pt]
(1-\alpha)\,\dfrac{w_{uz}}{\sum_{z' \in \mathcal{N}(u)} w_{uz'}}, & z \in \mathcal{N}(u),\\[8pt]
0, & \text{otherwise,}
\end{cases}
$$

and the transport distance

$$
W_1(\mu_u, \mu_v) \;=\; \inf_{\pi \in \Pi(\mu_u,\mu_v)} \sum_{z, z'} d(z,z')\, \pi(z,z'),
$$

with $\Pi(\mu_u,\mu_v)$ the couplings having marginals $\mu_u, \mu_v$ [7,8]. Solved
per edge as a linear program by `GraphRicciCurvature`'s `OllivierRicci` [9],
$\alpha = 0.5$, weighted, 4 processes
([graph_builder/curvature_calculator.py:35](graph_builder/curvature_calculator.py#L35)).
The lazy-walk parameterisation with $\alpha \in (0,1)$ follows [44]; $\alpha = 0.5$
is passed explicitly and used throughout.

### S1.2 Augmented Forman–Ricci

The `FormanRicci` estimator in [9] implements the **augmented** Forman–Ricci
curvature [13], which accounts for the two-dimensional simplices (triangles) present
in the graph, rather than the un-augmented edge-only form of [45,46]. This is the
variant used for node features (Table 1, rows 7–12). It is distinct from the
Balanced Forman curvature of Topping *et al.* [10], which we do not compute; a
comparison of the three is a natural ablation.

$$
\kappa^F(u,v) \;=\; w_{uv}\!\left(
\frac{w_u}{w_{uv}} + \frac{w_v}{w_{uv}}
\;-\; \sum_{\substack{z \sim u,\; z' \sim v \\ z \neq v,\; z' \neq u}}
\left[\frac{w_u}{\sqrt{w_{uv} w_{uz}}} + \frac{w_v}{\sqrt{w_{uv} w_{vz'}}}\right]
\right)
$$

$O(1)$ per edge, versus a transport LP per edge for Ollivier — the reason both are
computed as features but only Ollivier enters message passing
([graph_builder/curvature_calculator.py:22-53](graph_builder/curvature_calculator.py#L22-L53)).

## S2. Per-fold standardisation

For fold $f$, the scaler is fitted on every gene outside the test fold,
$V^{(f)}_{\text{fit}} = V \setminus \mathcal{A}^{(f)}_{\text{te}}$ (2.5):

$$
\hat{x}_{v,j}^{(f)} = \frac{\tilde{x}_{v,j} - \mu_j^{(f)}}{\sigma_j^{(f)}},
\qquad
\mu_j^{(f)} = \frac{1}{|V^{(f)}_{\text{fit}}|}\!\!\sum_{v \in V^{(f)}_{\text{fit}}}\!\! \tilde{x}_{v,j},
$$

with $\sigma_j^{(f)} \leftarrow 1$ wherever $\sigma_j^{(f)} = 0$ [14]
([train_model.py:672](model/train_model.py#L672)). Standardisation uses no labels,
so fitting on unlabelled genes leaks nothing; fitting on the ≈1,200 annotated
training genes alone would instead take statistics from a small, driver-enriched
sample and mis-scale the unlabelled majority the model is meant to rank. Test-fold
genes are excluded, so the scaler carries no distributional information from
them. The same fold-specific scaler is applied to the original graph and to both
augmented views.

## S3. Schur complement: sampling, orders, alignment

### S3.1 Sampled elimination

The single-vertex elimination step below is the randomised Cholesky /
approximate-Gaussian-elimination routine of Kyng & Sachdeva [23]; see also [20] for
its use as a contrastive augmentation on gene networks.

Eliminating $v$ with neighbours $u_1,\dots,u_r$ in sort order, weights
$w_j = w_{vu_j}$, $W_0 = \sum_j w_j$, $\gamma_0 = 1$. For $j = 1,\dots,r-1$:

$$
f_j = \frac{\gamma_{j-1} w_j}{W_{j-1}},
\qquad
u_{k(j)} \sim \Pr[k] \propto w_k \ \ (k > j),
$$

$$
w^{\text{new}}_{u_j u_{k(j)}} \mathrel{+}= f_j\,(1 - f_j)\, W_{j-1},
\qquad
\gamma_j = \gamma_{j-1}(1 - f_j),
\qquad
W_j = W_{j-1}(1 - f_j)^2 .
$$

$r-1$ new edges instead of $\binom{r}{2}$, with
$\mathbb{E}\bigl[L^{(S)}_{\text{sampled}}\bigr] \approx L^{(S)}$; the spectral
approximation guarantee follows from the matrix-martingale argument of [23]
([graph_builder/schur_complement.py:641-706](graph_builder/schur_complement.py#L641-L706)).
The fallback cleanup path uses the deterministic weight
$w^{\text{new}}_{uu'} \mathrel{+}= w_{vu}w_{vu'} / \max(\deg v, 1)$
([graph_builder/schur_complement.py:835-841](graph_builder/schur_complement.py#L835-L841)).

### S3.2 Elimination orders

| Strategy | Order of $B$ | Effect |
|---|---|---|
| `priority` | ascending degree (min-degree heap) | minimises fill-in; conservative views |
| `random` | uniform permutation | maximum view diversity; best empirical NDCG |
| `coarsening` | collapse onto a neighbour | contracts rather than marginalises |

Minimum-degree ordering is the classical fill-reducing heuristic for sparse Cholesky
[47]; the coarsening variant is closest in spirit to spectrally-guaranteed graph
reduction [48].

### S3.3 Cross-view alignment

Since $B_1 \neq B_2$, the two views have different node sets and different
indexings. The contrastive loss is restricted to $A_1 \cap A_2$, matched **by gene
symbol**, not by node index
([model/train_model.py:911-925](model/train_model.py#L911-L925)).

## S4. Attention in full

Node features are projected into $K$ heads,
$h_v = W^{(\ell)}\hat{x}_v \in \mathbb{R}^{KF}$, $F = \texttt{hidden\_channels}$.

### S4.1 Modes

**(a) `standard`** — plain GAT [24], curvature entering only via hard edge filtering
(S4.4), i.e. the curvature-constrained propagation of [12]:

$$
e^k_{vu} = \sigma_{\text{LR}}\!\left(\mathbf{a}^{k\top}\bigl[h^k_v \,\|\, h^k_u\bigr]\right)
$$

**(b) `edge_feature`** — curvature as a concatenated channel in the sense of edge-
featured attention [25], learnable scale $\beta$:

$$
e^k_{vu} = \sigma_{\text{LR}}\!\left(\mathbf{a}^{k\top}\bigl[h^k_v \,\|\, h^k_u \,\|\, \beta\,\kappa_{vu}\bigr]\right),
\qquad \mathbf{a}^k \in \mathbb{R}^{2F+1}
$$

**(c) `bias`** — curvature as a per-head additive shift:

$$
e^k_{vu} = \sigma_{\text{LR}}\!\left(\mathbf{a}^{k\top}\bigl[h^k_v \,\|\, h^k_u\bigr]\right) + b_k\,\kappa_{vu}
$$

**(d) `gated`** — learned convex mix, gate conditioned on both:

$$
g^k_{vu} = \operatorname{sigmoid}\!\left(\mathbf{W}_g \bigl[\bar{h}_v \,\|\, \bar{h}_u \,\|\, \kappa_{vu}\bigr]\right)_k,
\qquad \bar{h}_v = \tfrac{1}{K}\textstyle\sum_k h^k_v
$$

$$
e^k_{vu} = g^k_{vu}\,\sigma_{\text{LR}}\!\left(\mathbf{a}_{\text{node}}^{k\top}\bigl[h^k_v \| h^k_u\bigr]\right)
+ \bigl(1 - g^k_{vu}\bigr)\,\sigma_{\text{LR}}\!\left(a^k_{\text{curv}}\,\kappa_{vu}\right)
$$

**(e) `hybrid`** — default; see §2.4.

### S4.2 Stabilised softmax

$$
\alpha^k_{vu} \;=\;
\frac{\exp\!\bigl(e^k_{vu} - \max_{w \in \mathcal{N}(v)} e^k_{vw}\bigr)}
{\sum_{w \in \mathcal{N}(v)} \exp\!\bigl(e^k_{vw} - \max_{w'} e^k_{vw'}\bigr) + \varepsilon},
\qquad \varepsilon = 10^{-16}
$$

([model/message_passing.py:349-378](model/message_passing.py#L349-L378)).

### S4.3 Aggregation and hop depth

$$
m_v^{k} = \bigoplus_{u \in \mathcal{N}_c(v)} \alpha^k_{vu} \, h^k_u,
\qquad
m_v = \big\|_{k=1}^{K} m_v^{k} \ \ (\texttt{concat})
\quad\text{or}\quad
\tfrac{1}{K}\textstyle\sum_k m_v^k
$$

with $\bigoplus \in \{$add, mean, max$\}$. Two-hop pathways, following [12], apply
the operator twice with a residual:

$$
h_v^{(2\text{-hop})} = h_v + \tfrac{1}{2}\,\mathcal{A}(h)_v + \tfrac{1}{2}\,\mathcal{A}^2(h)_v
$$

([model/message_passing.py:283-306](model/message_passing.py#L283-L306)). Each layer
closes with batch normalisation [49], ReLU and dropout [50]:

$$
h^{(\ell)}_v = \operatorname{Dropout}_p\!\left(\operatorname{ReLU}\!\left(\operatorname{BN}^{(\ell)}\!\left(m^{(\ell)}_v\right)\right)\right).
$$

### S4.4 Curvature filtering (`standard` mode only)

$$
E^{+} = \{(u,v) : \kappa_{uv} \ge \kappa_{\text{med}}\},
\quad
E^{-} = \{(u,v) : \kappa_{uv} \le \kappa_{\text{med}}\},
\quad
E^{\ast} = E,
$$

with a floor $|E^c| \ge \lceil \varrho_{\min}|E| \rceil$, $\varrho_{\min}=0.15$,
enforced by top-$k$ fallback. This is the hard curvature partition of [12]. It is
bypassed under the default `hybrid` mode, where the pathway distinction is carried
by learned parameters instead.

## S5. Encoder and aggregation

### S5.1 Six-pathway encoder

The pathway factorisation follows [12].

$$
\mathcal{C} \times \mathcal{H} = \{+,-,\ast\} \times \{1\text{-hop}, 2\text{-hop}\}
\;\Longrightarrow\; 6\ \text{pathways}
$$

After shared projection
$h^{(0)}_v = \operatorname{Dropout}\bigl(\operatorname{ReLU}(W_{\text{in}}\hat{x}_v)\bigr)$,
each pathway evolves independently for $L$ layers:

$$
h^{(c,\eta,\ell)}_v = \Psi^{(c,\eta,\ell)}\!\left(h^{(c,\eta,\ell-1)},\, E_c,\, \kappa\right).
$$

Batch-norm statistics are shared across pathways within a layer. Each pathway is
wrapped in non-reentrant gradient checkpointing [51].

### S5.2 Stage 1 — depth (self-attention over layers)

Scaled dot-product self-attention [25] applied along the layer axis. Stack layer
outputs as $H_v \in \mathbb{R}^{L \times D}$:

$$
A = \operatorname{softmax}\!\left(\frac{(H_vW_Q)(H_vW_K)^\top}{\sqrt{D/K}}\right),
\qquad
\bar{h}_v = \operatorname{LayerNorm}\!\left(\frac{1}{L}\sum_{\ell=1}^{L}\bigl(W_O A (H_vW_V)\bigr)_\ell + h^{(\cdot,L)}_v\right)
$$

with layer normalisation as in [52]. Mean-over-layers plus a residual on the final
layer lets the model select its effective depth per gene. Chunked at 1000 nodes;
skipped entirely when $L=1$.

### S5.3 Stage 2 — breadth (attention over pathways)

`attention` (default), with bottleneck MLP $\psi$:

$$
a^{(c,\eta)}_v = \operatorname{softmax}_{(c,\eta)}\bigl(\psi(\bar{h}^{(c,\eta)}_v)\bigr),
\qquad
h_v = \sum_{(c,\eta)} a^{(c,\eta)}_v\, \bar{h}^{(c,\eta)}_v
$$

`hierarchical`, hop then curvature:

$$
\bar{h}^{(c)}_v = \sum_{\eta} \operatorname{softmax}_\eta\bigl(\psi_\eta(\bar{h}^{(c,\eta)}_v)\bigr) \bar{h}^{(c,\eta)}_v,
\qquad
h_v = \sum_{c} \operatorname{softmax}_c\bigl(\psi_c(\bar{h}^{(c)}_v)\bigr) \bar{h}^{(c)}_v
$$

Both are $O(|\mathcal{C}||\mathcal{H}|)$ per node; no cross-pathway attention matrix
is formed. Final projection:

$$
h_v \leftarrow \operatorname{Dropout}\!\left(\operatorname{ReLU}\!\left(\operatorname{LayerNorm}(W_{\text{out}} h_v)\right)\right) \in \mathbb{R}^{F}.
$$

## S6. Losses in full

### S6.1 Heads

The nonlinear projection head follows SimCLR [15], which showed that contrasting in
a projected space rather than the encoder output improves the learned
representation:

$$
z_v = \frac{g(h_v)}{\lVert g(h_v)\rVert_2},
\quad g = W_2 \circ \operatorname{ReLU}\circ \operatorname{BN}\circ W_1;
\qquad
s_v = \operatorname{sigmoid}\!\left(W_2\operatorname{Dropout}\bigl(\operatorname{ReLU}(W_1 h_v)\bigr)\right)
$$

$L_2$-normalisation makes $z_u^\top z_v$ exactly the cosine similarity.

### S6.2 NT-Xent

NT-Xent [15,36], equivalently InfoNCE [16], applied node-wise as in GRACE [17]. On
the $N$ genes shared by both views, stack into $Z \in \mathbb{R}^{2N\times d_z}$,

$$
S_{ij} = \frac{z_i^\top z_j}{\tau} \ \text{clipped to } [-50,50],
\qquad S_{ii} \leftarrow -\infty,
\qquad \pi(i) = (i+N) \bmod 2N,
$$

$$
\mathcal{L}_{\text{con}} = -\frac{1}{2N}\sum_{i=1}^{2N} \log \frac{\exp\bigl(S_{i\,\pi(i)}\bigr)}{\sum_{j\neq i}\exp\bigl(S_{ij}\bigr)}
$$

Both directions are included; every other gene in both views is a negative.

### S6.3 Focal BPR with hard-negative curriculum

BPR [38] with the focal modulating factor of [39]. Of $P$ sampled
(driver, curated non-driver) pairs — both members drawn from the inner training
set $\mathcal{A}^{(f)}_{\text{tr}}$ (2.5) — a fraction $\varphi(t) = 0.25 + 0.5\,t$
at training progress $t$ use the top-scoring curated non-drivers (hard negatives);
the rest are uniform. With
$\Delta_{pq} = s_p - s_q$:

$$
\mathcal{L}_{\text{rank}} = \frac{1}{P}\sum_{(p,q)}
\bigl(1 - \operatorname{sigmoid}(\Delta_{pq})\bigr)^{\gamma}\,
\bigl(-\log\operatorname{sigmoid}(\Delta_{pq})\bigr)
$$

Early training sees easy negatives (stable gradients); late training concentrates on
curated non-drivers being confused with drivers. Unlabelled genes never enter a
pair, so the model is not pushed to demote plausible undiscovered drivers [32]; the
curriculum ceiling of 0.75 keeps a quarter of each batch uniform, so the loss is not
dominated by a few borderline curated negatives.

### S6.4 Alternative ranking losses (implemented, not default)

| `loss_type` | Form | Ref |
|---|---|---|
| `pairwise` | $\frac{1}{\lvert D\rvert\lvert\bar D\rvert}\sum_{p,q}(1-\sigma(\Delta_{pq}))^\gamma[m-\Delta_{pq}]_+$ | [39] |
| `sampled_pairwise` | $\frac{1}{P}\sum (1-\sigma(\Delta))^\gamma\operatorname{softplus}(-\Delta)$ | [38,39] |
| `listwise` | $-\sum_v \tilde{y}_v\log\operatorname{softmax}(s)_v$, $\tilde{y}=y/\lVert y\rVert_1$ | [53] |
| `approxndcg` | soft ranks $\hat{r}_i = 1+\sum_{j\neq i}\sigma\bigl((s_j-s_i)/\tau_r\bigr)$, $\tau_r=0.1$, into DCG | [54] |

## S7. Optimisation, complexity, runtime

### S7.1 Training loop

Per epoch, over $T = 8$ gradient-accumulation steps
([model/train_model.py:809-1090](model/train_model.py#L809-L1090)):

1. Deterministic view pairing
   $(i_1,i_2) = \bigl((eT+t)\bmod n_V,\ (eT+t+1)\bmod n_V\bigr)$.
2. Input noise $x \leftarrow x + \epsilon$, $\epsilon \sim \mathcal{N}(0, 0.01^2 I)$,
   on both views and the original graph.
3. Each step contributes $\lambda\mathcal{L}_{\text{con}}/T$; the ranking term is
   backpropagated once per epoch on the full graph.
4. Gradient clipping to unit norm after unscaling AMP gradients [55].
5. AdamW [56], $\eta_0 = 5\times10^{-4}$, weight decay $5\times10^{-4}$.

Implemented in PyTorch [57] and PyTorch Geometric [58].

### S7.2 EMA and scheduling

Polyak-style parameter averaging [40], in the exponential-moving-average form used
for semi-supervised training in [41]:

$$
\theta^{\text{EMA}} \leftarrow \gamma\,\theta^{\text{EMA}} + (1-\gamma)\,\theta,
\qquad \gamma = 0.999
$$

EMA weights are used for **all** evaluation and for the final ranked gene list.
Model selection uses only the inner validation set $\mathcal{A}^{(f)}_{\text{val}}$
(2.5), evaluated every 20 epochs. The monitored signal is a smoothed inner-validation
NDCG@50 over successive checks $c$,

$$
\widetilde{\mathrm{NDCG}}_c = \alpha\,\mathrm{NDCG}^{(c)}_{@50} + (1-\alpha)\,\widetilde{\mathrm{NDCG}}_{c-1},
\qquad \alpha = 0.3,
$$

fed to `ReduceLROnPlateau` ($\times 0.5$, patience 5 checks) after a 10-epoch
linear warm-up, and to early stopping (patience 15 checks, i.e. 300 epochs).
Smoothing prevents spurious LR cuts from check-to-check variance on the ≈85 drivers
in a GGNet inner validation set. The retained checkpoint is the one with the highest
unsmoothed inner-validation NDCG@50
([train_model.py:1231](model/train_model.py#L1231)). After training, that
checkpoint's EMA weights are restored and the test fold is evaluated exactly once
([train_model.py:1258-1272](model/train_model.py#L1258-L1272)); the same weights
score every gene for the ranked list. Inner-validation values are used only for
selection and are never reported. With drivers forming ≈40% of the annotated genes,
a random ranking scores about 0.4 on this signal, which is why it starts well above
the values seen under the earlier all-gene protocol.

### S7.3 Hyperparameters

| Parameter | Value | Parameter | Value |
|---|---|---|---|
| $F$ (`hidden_channels`) | 128 | $L$ (`num_layers`) | 2 |
| $d_z$ (`projection_dim`) | 96 | $K$ (`num_heads`) | 4 |
| $\tau$ | 0.3 | $\lambda$ | 0.2 |
| dropout $p$ | 0.3 | $\varsigma$ (ranking scale) | 5.0 |
| $\rho$ | 0.2 | views $n_V$ | 2 |
| attention mode | `hybrid` | pathway aggregation | `hierarchical` |
| aggregation $\bigoplus$ | `max` | $\gamma$ (EMA) | 0.999 |
| lr / weight decay | 5e-4 / 5e-4 | accumulation $T$ | 8 |
| $\gamma$ (focal) | 2.0 | pairs $P$ | 512 |
| $\varrho_{\min}$ | 0.15 | LeakyReLU slope | 0.25 |
| epochs (max) | 1000 | folds | 5 (HGDC `split_set`, repeat 0) |
| validation interval | 20 epochs | inner validation fraction | 0.15 |
| LR factor / patience | 0.5 / 5 checks | early-stopping patience | 15 checks |

Values match the cluster run script, [run_model_array.sh](run_model_array.sh).

### S7.4 Complexity

Preprocessing, once per network:

$$
\text{Ollivier–Ricci:}\ O\bigl(|E| \cdot \mathrm{LP}(\bar{d})\bigr),
\qquad
\text{augmented Forman:}\ O(|E|\bar{d}),
\qquad
\text{augmentation:}\ O(\rho n \bar{d})
$$

for mean degree $\bar{d}$; the curvature LP dominates and is computed once and
cached. Per training epoch:

$$
O\bigl(\underbrace{|\mathcal{C}||\mathcal{H}|}_{6}\, L\, K\, (|E|F + nF^2)\bigr)
\;+\;
O\bigl(n L^2 F\bigr)_{\text{layer attn}}
\;+\;
O(N^2 d_z)_{\text{NT-Xent}}
\;+\;
O(P)_{\text{BPR}}
$$

The NT-Xent term is the practical bottleneck: it is quadratic in the number of genes
shared by the two views ($N \approx (1-\rho)n$), which is why the similarity matrix
is built once per accumulation step rather than per pathway.

**Scale actually run:** 3 networks (GGNet, PathNet, PPNet; sizes in 2.0) × 3
elimination strategies × 2 elimination ratios × 5 HGDC test folds = 90 training
runs, one GPU each ([run_model_array.sh](run_model_array.sh)). The largest network
is PPNet, $n = 11{,}395$ genes with $|\mathcal{P}| = 716$ known drivers, from the
HGDC data release [4] `[state filtering relative to the published node count]`.

> **TODO before submission — partly measured.** From the Oct 2026 run log, GGNet
> takes ≈49 s per epoch on one NVIDIA H100 (16.3 min per 20-epoch validation
> interval), so a fold that reaches the 1,000-epoch cap takes ≈13.6 h. Still to
> record: total time per fold after early stopping, peak GPU memory, and
> preprocessing time for the curvature LP. Reviewers at this venue routinely ask;
> a one-line "trains in X h on one GPU" is enough.

## S8. Metric definitions

$$
P@K = \frac{1}{K}\sum_{k \le K} y_{v_{(k)}},
\qquad
R@K = \frac{1}{|\mathcal{P}|}\sum_{k \le K} y_{v_{(k)}},
\qquad
\mathrm{MRR} = \frac{1}{|\mathcal{P}|}\sum_{v \in \mathcal{P}} \frac{1}{r(v)}
$$

$$
\mathrm{AUROC} = \Pr\bigl[s_p > s_q \;\big|\; p \in \mathcal{P},\, q \notin \mathcal{P}\bigr]
$$

$$
\mathrm{AP} = \sum_{k} (R_k - R_{k-1})\,P_k,
\qquad
\mathrm{AUPRC}_{\text{trapz}} = \sum_{k} (R_k - R_{k-1})\,\frac{P_k + P_{k-1}}{2}
$$

with $(R_k, P_k)$ the recall and precision at the $k$-th score threshold. AP is
the reported AUPRC; the trapezoidal form matches HGDC and SCIS-CDG and is used
only for comparison with them (2.6).

with $r(v)$ the 1-indexed rank of driver $v$, $K \in \{10,20,50,100\}$. NDCG as in
[42]; see [43] on the interpretation of AUROC and AUPRC under the present class
imbalance. Also reported: $F_1$ at the optimal PR-curve threshold and at $0.5$, and
the median/mean rank of known drivers
([model/DriverGenePredictor.py:770-840](model/DriverGenePredictor.py#L770-L840)).
Metrics computed with scikit-learn [14].

---

# REFERENCES

Verify volume and page numbers against the publisher record before submission;
these were assembled from secondary listings and a small number may need
correcting. Numbered in order of first citation in the Methods.

1. Sandhu R, Georgiou T, Reznik E, *et al.* Graph curvature for differentiating cancer networks. *Sci Rep* 2015;5:12323.
2. Pouryahya M, Oh JH, Mathews JC, Deasy JO, Tannenbaum AR. Characterizing cancer drug response and biological correlates: a geometric network approach. *Sci Rep* 2018;8:6402.
3. Elkin R, Oh JH, Liu YL, *et al.* Geometric network analysis provides prognostic information in patients with high grade serous carcinoma of the ovary treated with immune checkpoint inhibitors. *NPJ Genom Med* 2021;6:99.
4. Zhang T, Zhang S-W, Xie M-Y, Li Y. A novel heterophilic graph diffusion convolutional network for identifying cancer driver genes. *Brief Bioinform* 2023;24:bbad137. [HGDC — source of the GGNet/PathNet/PPNet networks, the 58-dimensional feature matrix, the driver/non-driver labels, and the driver homophily ratios; data at `github.com/NWPU-903PR/HGDC`]
5. Ollivier Y. Ricci curvature of Markov chains on metric spaces. *J Funct Anal* 2009;256:810–864.
6. Ollivier Y. Ricci curvature of metric spaces. *C R Math Acad Sci Paris* 2007;345:643–646.
7. Villani C. *Optimal Transport: Old and New*. Berlin: Springer, 2009.
8. Rubner Y, Tomasi C, Guibas LJ. The earth mover's distance as a metric for image retrieval. *Int J Comput Vis* 2000;40:99–121.
9. Ni C-C, Lin Y-Y, Luo F, Gao J. Community detection on networks with Ricci flow. *Sci Rep* 2019;9:9984. [`GraphRicciCurvature`]
10. Topping J, Di Giovanni F, Chamberlain BP, Dong X, Bronstein MM. Understanding over-squashing and bottlenecks on graphs via curvature. *ICLR* 2022.
11. Alon U, Yahav E. On the bottleneck of graph neural networks and its practical implications. *ICLR* 2021.
12. Attali H, Buscaldi D, Pernelle N. Curvature constrained MPNNs: improving message passing with local structural properties. *Data Knowl Eng* 2025;156:102382.
13. Samal A, Sreejith RP, Gu J, *et al.* Comparative analysis of two discretizations of Ricci curvature for complex networks. *Sci Rep* 2018;8:8650.
14. Pedregosa F, Varoquaux G, Gramfort A, *et al.* Scikit-learn: machine learning in Python. *J Mach Learn Res* 2011;12:2825–2830.
15. Chen T, Kornblith S, Norouzi M, Hinton G. A simple framework for contrastive learning of visual representations. *ICML* 2020;119:1597–1607. [SimCLR]
16. van den Oord A, Li Y, Vinyals O. Representation learning with contrastive predictive coding. arXiv:1807.03748, 2018. [InfoNCE]
17. Zhu Y, Xu Y, Yu F, *et al.* Deep graph contrastive representation learning. *ICML Workshop on Graph Representation Learning*, 2020. [GRACE]
18. You Y, Chen T, Sui Y, *et al.* Graph contrastive learning with augmentations. *NeurIPS* 2020;33:5812–5823. [GraphCL]
19. Kothapalli V. Randomized Schur complement views for graph contrastive learning. *Proc 40th Int Conf Machine Learning (ICML)*, PMLR 2023;202:17580–17614.
20. Ma X, Li Z, Du Z, *et al.* Advancing cancer driver gene detection via Schur complement graph augmentation and independent subspace feature extraction. *Comput Biol Med* 2024;174:108484. [SCIS-CDG]
21. Dörfler F, Bullo F. Kron reduction of graphs with applications to electrical networks. *IEEE Trans Circuits Syst I* 2013;60:150–163.
22. Klein DJ, Randić M. Resistance distance. *J Math Chem* 1993;12:81–95.
23. Kyng R, Sachdeva S. Approximate Gaussian elimination for Laplacians — fast, sparse, and simple. *FOCS* 2016:573–582.
24. Veličković P, Cucurull G, Casanova A, *et al.* Graph attention networks. *ICLR* 2018.
25. Vaswani A, Shazeer N, Parmar N, *et al.* Attention is all you need. *NeurIPS* 2017;30:5998–6008.
26. Schulte-Sasse R, Budach S, Hnisz D, Marsico A. Integration of multiomics data with graph convolutional networks to identify new cancer genes and their associated molecular mechanisms. *Nat Mach Intell* 2021;3:513–526. [EMOGI]
27. Peng W, Tang Q, Dai W, Chen T. Improving cancer driver gene identification using multi-task learning on graph convolutional network. *Brief Bioinform* 2022;23:bbab432. [MTGCN]
28. Sondka Z, Bamford S, Cole CG, *et al.* The COSMIC Cancer Gene Census: describing genetic dysfunction across all human cancers. *Nat Rev Cancer* 2018;18:696–705.
29. Tate JG, Bamford S, Jubb HC, *et al.* COSMIC: the Catalogue Of Somatic Mutations In Cancer. *Nucleic Acids Res* 2019;47:D941–D947.
30. Repana D, Nulsen J, Dressler L, *et al.* The Network of Cancer Genes (NCG): a comprehensive catalogue of known and candidate cancer genes from cancer sequencing screens. *Genome Biol* 2019;20:1.
31. Bailey MH, Tokheim C, Porta-Pardo E, *et al.* Comprehensive characterization of cancer driver genes and mutations. *Cell* 2018;173:371–385.
32. Bekker J, Davis J. Learning from positive and unlabeled data: a survey. *Mach Learn* 2020;109:719–760.
33. Elkan C, Noto K. Learning classifiers from only positive and unlabeled data. *KDD* 2008:213–220.
34. Yang P, Li X-L, Mei J-P, Kwoh C-K, Ng S-K. Positive-unlabeled learning for disease gene identification. *Bioinformatics* 2012;28:2640–2647.
35. Li F, Dong S, Leier A, *et al.* Positive-unlabeled learning in bioinformatics and computational biology: a brief review. *Brief Bioinform* 2022;23:bbab461.
36. Sohn K. Improved deep metric learning with multi-class N-pair loss objective. *NeurIPS* 2016;29:1857–1865.
37. Peng W, Zhou Z, Dai W, *et al.* Multi-network graph contrastive learning for cancer driver gene identification. *IEEE Trans Netw Sci Eng* 2024;11:3430–3440.
38. Rendle S, Freudenthaler C, Gantner Z, Schmidt-Thieme L. BPR: Bayesian personalized ranking from implicit feedback. *UAI* 2009:452–461.
39. Lin T-Y, Goyal P, Girshick R, He K, Dollár P. Focal loss for dense object detection. *ICCV* 2017:2980–2988.
40. Polyak BT, Juditsky AB. Acceleration of stochastic approximation by averaging. *SIAM J Control Optim* 1992;30:838–855.
41. Tarvainen A, Valpola H. Mean teachers are better role models. *NeurIPS* 2017;30:1195–1204.
42. Järvelin K, Kekäläinen J. Cumulated gain-based evaluation of IR techniques. *ACM Trans Inf Syst* 2002;20:422–446.
43. Saito T, Rehmsmeier M. The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. *PLoS ONE* 2015;10:e0118432.
44. Lin Y, Lu L, Yau S-T. Ricci curvature of graphs. *Tohoku Math J* 2011;63:605–627.
45. Forman R. Bochner's method for cell complexes and combinatorial Ricci curvature. *Discrete Comput Geom* 2003;29:323–374.
46. Sreejith RP, Mohanraj K, Jost J, *et al.* Forman curvature for complex networks. *J Stat Mech* 2016;2016:063206.
47. George A, Liu JWH. The evolution of the minimum degree ordering algorithm. *SIAM Rev* 1989;31:1–19.
48. Loukas A. Graph reduction with spectral and cut guarantees. *J Mach Learn Res* 2019;20:1–42.
49. Ioffe S, Szegedy C. Batch normalization: accelerating deep network training by reducing internal covariate shift. *ICML* 2015:448–456.
50. Srivastava N, Hinton G, Krizhevsky A, Sutskever I, Salakhutdinov R. Dropout: a simple way to prevent neural networks from overfitting. *J Mach Learn Res* 2014;15:1929–1958.
51. Chen T, Xu B, Zhang C, Guestrin C. Training deep nets with sublinear memory cost. arXiv:1604.06174, 2016.
52. Ba JL, Kiros JR, Hinton GE. Layer normalization. arXiv:1607.06450, 2016.
53. Cao Z, Qin T, Liu T-Y, Tsai M-F, Li H. Learning to rank: from pairwise approach to listwise approach. *ICML* 2007:129–136.
54. Qin T, Liu T-Y, Li H. A general approximation framework for direct optimization of information retrieval measures. *Inf Retr* 2010;13:375–397.
55. Micikevicius P, Narang S, Alben J, *et al.* Mixed precision training. *ICLR* 2018.
56. Loshchilov I, Hutter F. Decoupled weight decay regularization. *ICLR* 2019.
57. Paszke A, Gross S, Massa F, *et al.* PyTorch: an imperative style, high-performance deep learning library. *NeurIPS* 2019;32:8024–8035.
58. Fey M, Lenssen JE. Fast graph representation learning with PyTorch Geometric. *ICLR Workshop on Representation Learning on Graphs and Manifolds*, 2019.

*Cited in Results/Discussion rather than Methods (baselines and context).*

59. Zhao W, Gu X, Chen S, *et al.* MODIG: integrating multi-omics and multi-dimensional gene network for cancer driver gene identification based on graph attention network model. *Bioinformatics* 2022;38:4901–4907.
60. Vogelstein B, Papadopoulos N, Velculescu VE, *et al.* Cancer genome landscapes. *Science* 2013;339:1546–1558.
61. Lawrence MS, Stojanov P, Polak P, *et al.* Mutational heterogeneity in cancer and the search for new cancer-associated genes. *Nature* 2013;499:214–218. [MutSigCV]
62. Mularoni L, Sabarinathan R, Deu-Pons J, Gonzalez-Perez A, López-Bigas N. OncodriveFML: a general framework to identify coding and non-coding regions with cancer driver mutations. *Genome Biol* 2016;17:128.
63. Tokheim CJ, Papadopoulos N, Kinzler KW, Vogelstein B, Karchin R. Evaluating the evaluation of cancer driver genes. *Proc Natl Acad Sci USA* 2016;113:14330–14335. [20/20+]
64. Han Y, Yang J, Qian X, *et al.* DriverML: a machine learning algorithm for identifying driver genes in cancer sequencing studies. *Nucleic Acids Res* 2019;47:e45.
65. Cowen L, Ideker T, Raphael BJ, Sharan R. Network propagation: a universal amplifier of genetic associations. *Nat Rev Genet* 2017;18:551–562.

*Added Oct 2026 for 2.6–2.7; renumber into citation order before submission.*

66. Davis J, Goadrich M. The relationship between precision-recall and ROC curves. *ICML* 2006:233–240.
67. Shyr C, Tarailo-Graovac M, Gottlieb M, Lee JJM, van Karnebeek C, Wasserman WW. FLAGS, frequently mutated genes in public exomes. *BMC Med Genomics* 2014;7:64.
68. Frankish A, Diekhans M, Ferreira AM, *et al.* GENCODE reference annotation for the human and mouse genomes. *Nucleic Acids Res* 2019;47:D766–D773.

---

# PART III — INTERNAL

*Delete before submission.*

## P1. Manuscript gaps not covered by this document

The math is now sourced; these determine acceptance and are not derivable from code.

- [ ] **Baseline comparison.** HGDC [4] already publishes AUC/AUPR for HGDC, MTGCN,
      EMOGI, GCN, GAT, Chebnet and SVM on exactly GGNet/PathNet/PPNet under 10×
      5-CV. The protocol now matches (HGDC's own folds, repeat 0), so those numbers
      are quotable with a note that DriverJinn reports one of the ten repeats; run
      `--split_repeat 1..9` if compute allows. You still need:
      SCIS-CDG [20] — omitting it will be read as avoidance, since DriverJinn shares
      its augmentation class; and NDCG@K for the GNN baselines, which nobody
      reports, so you must rerun from released code. Add MutSigCV [61],
      OncodriveFML [62], 20/20+ [63], DriverML [64] for coverage of the
      non-network literature.
- [ ] **Data provenance.** Mostly resolved via [4]; still needed: COSMIC/CGC
      version and tier filter [28,29]; the relationship between your 716 positives
      and HGDC's CDG list; gene counts before and after filtering, with the
      discrepancy against the published PPNet node count explained. Lift HGDC's
      references 33, 37, 38 verbatim for the three network sources.
- [x] **Label policy.** Resolved (editorial note A, 2.5): curated non-drivers are the
      negatives, unlabelled genes are never labels, HGDC's folds are used, and model
      selection runs on an inner validation split.
- [x] **Regenerate every number.** Done from `model_results_2026-10-16` (18 configs
      × 5 folds); MANUSCRIPT.md's abstract and drafting notes carry the new values.
      The pre-Oct-2026 figures ("8.7-fold enrichment", hallmark counts, CGC tier
      medians, the chromatin cluster) must not reappear; the poster and
      `Downstream/poster_plots.qmd` still carry them.
- [x] **Length / degree bias in candidates.** Corrected (2.7); results in
      `Downstream/length_adjusted/`. Remaining: Fig 5 of `publication_figures.qmd`
      still plots raw ranks and should switch to the adjusted ones.
- [ ] **Trapezoidal AUPRC values.** `auprc_trapz` (2.6, S8) was added after the
      2026-10-16 runs, so no values exist yet. Rerun, or recompute from saved
      per-gene scores and HGDC's `split_set`, before the baseline table is built.
- [ ] **Biological validation of novel predictions.** Material exists in
      [Downstream/](Downstream/). Needs literature support for top-ranked unlabelled
      genes. This is the section that gets the paper accepted here.
- [ ] **Ablation.** Curvature features on/off; attention mode (S4.1); elimination
      strategy (S3.2); $\lambda$ sensitivity. Add Ollivier vs augmented Forman vs
      Balanced Forman [10] as an ablation now that S1.2 distinguishes them.
- [ ] **Availability.** Repository, trained models, ranked gene list as supplementary
      table.
- [ ] **Runtime** (S7.4 TODO).

## P2. Corrections to the poster figure

[curvature_gnn_math.html](curvature_gnn_math.html) contains six statements that
would be wrong if carried into the manuscript.

| Poster | Correct |
|---|---|
| Encoder is a standard GAT | Curvature-conditioned attention, `hybrid` mode (§2.4). Curvature sits *inside* the attention; the poster equation omits it. |
| Single encoder, $z_v \in \mathbb{R}^{128}$ | Six pathways over $\{+,-,\ast\}\times\{1,2\text{-hop}\}$ after [12], collapsed by layer-then-pathway attention (S5). |
| $\tilde{x}_v = [x_v \| \bar\kappa_v, \kappa^{\min}_v, \kappa^{\max}_v] \in \mathbb{R}^{d+15}$ | Dimension right, contents wrong: 6 Ollivier + 6 augmented Forman + $(\deg^+,\deg^-,\mathrm{hom})$ (Table 1). |
| Views formed by exact Schur complement | Sampled randomised-Cholesky sparsifier [23] (S3.1). |
| $\mathcal{L}_{\text{rank}} = -\sum\log\sigma(s_p-s_q)$ | Focal-weighted BPR [38,39] with hard-negative curriculum $0.25 \to 0.75$ (S6.3). |
| $\mathcal{L} = (1-\lambda)\mathcal{L}_{\text{rank}} + \lambda\mathcal{L}_{\text{con}}$; NDCG gain $2^{\mathbb{1}[\cdot]}-1$ | Missing $\varsigma = 5$, so the effective split is not 80/20 (§2.5). NDCG uses linear gain — equivalent here, but state the convention. |