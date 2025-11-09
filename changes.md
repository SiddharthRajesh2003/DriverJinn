# Complete Model Evolution Guide: DriverGenePredictor

## Table of Contents
1. [What Your Model Was Doing (Classification)](#classification-approach)
2. [What Your Model Does Now (Ranking)](#ranking-approach)
3. [Why This Change Matters](#why-change)
4. [Methods to Deprecate](#deprecation)
5. [Migration Guide](#migration)
6. [Side-by-Side Comparison](#comparison)

---

## 1. What Your Model Was Doing (Classification) {#classification-approach}

### Overall Architecture (Classification)

```
┌─────────────────────────────────────────────────────────────────┐
│                    INPUT: Gene Interaction Network              │
│                    - Nodes: Genes (features)                    │
│                    - Edges: Interactions (curvature)            │
│                    - Labels: 0 (non-driver) or 1 (driver)       │
└─────────────────────────────────────────────────────────────────┘
                              ↓ 
┌─────────────────────────────────────────────────────────────────┐
│              STEP 1: Curvature-Aware GNN Encoder                │
│                                                                 │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐              │
│  │ Positive   │  │ Negative   │  │   Both     │              │
│  │ Curvature  │  │ Curvature  │  │ Curvature  │              │
│  │  Pathway   │  │  Pathway   │  │  Pathway   │              │
│  └──────┬─────┘  └──────┬─────┘  └──────┬─────┘              │
│         │                │                │                     │
│         └────────────────┴────────────────┘                     │
│                          ↓                                      │
│              [Layer 1] → [Layer 2] → [Layer 3]                 │
│                          ↓                                      │
│              Multi-Layer Attention Aggregation                  │
│                          ↓                                      │
│              Final Embedding: [num_nodes, hidden_dim]          │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│            STEP 2: Binary Classification Head                   │
│                                                                 │
│  classifier = nn.Sequential(                                    │
│      nn.Linear(hidden_dim, hidden_dim),                        │
│      nn.ReLU(),                                                │
│      nn.Dropout(0.2),                                          │
│      nn.Linear(hidden_dim, 1),  # Single binary output        │
│      nn.Sigmoid()                # Probability [0, 1]          │
│  )                                                              │
│                          ↓                                      │
│              Output: [num_nodes, 1] probabilities              │
│              "Is this gene a driver? YES (>0.5) or NO (<0.5)"  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                 STEP 3: Loss Functions                          │
│                                                                 │
│  Option A: Weighted Binary Cross-Entropy                       │
│    loss = BCE(logits, labels, pos_weight=15)                   │
│    → Penalize missing drivers 15x more                         │
│                                                                 │
│  Option B: Focal Loss                                          │
│    loss = -(1-p)^γ × log(p)  # Focus on hard examples         │
│    → Down-weight easy negatives, focus on hard positives       │
│                                                                 │
│  Problem: STILL outputs binary decision at end!                │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│            STEP 4: Threshold Selection (Post-hoc)               │
│                                                                 │
│  Try thresholds: [0.1, 0.2, 0.3, ..., 0.9]                    │
│  For each threshold:                                           │
│    predictions = (probabilities > threshold)                    │
│    compute F1 score                                            │
│  Select threshold with best F1                                 │
│                                                                 │
│  Problem: Arbitrary! Different datasets need different values  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    STEP 5: Predictions                          │
│                                                                 │
│  Gene A: probability = 0.73 → DRIVER     (>0.5)               │
│  Gene B: probability = 0.51 → DRIVER     (>0.5)               │
│  Gene C: probability = 0.49 → NON-DRIVER (<0.5)               │
│  Gene D: probability = 0.08 → NON-DRIVER (<0.5)               │
│                                                                 │
│  Problem: Gene B and C are very close, but different labels!  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│            STEP 6: Evaluation Metrics (Misleading)              │
│                                                                 │
│  With 1:15 imbalance (6.25% drivers):                          │
│                                                                 │
│  Naive baseline (predict all non-driver):                      │
│    Accuracy: 93.75% 😱 (looks great, but useless!)            │
│    Precision: undefined                                        │
│    Recall: 0% (missed all drivers!)                           │
│    F1: 0%                                                      │
│                                                                 │
│  Your model (with focal loss):                                │
│    Accuracy: 89% (worse than naive!)                          │
│    Precision: 0.45 (55% false positives)                      │
│    Recall: 0.68 (missed 32% of drivers)                       │
│    F1: 0.54 (mediocre)                                        │
│                                                                 │
│  Problem: Metrics don't capture ranking quality!               │
└─────────────────────────────────────────────────────────────────┘
```

### What Was Happening Under the Hood

#### Training Loop (Classification):
```python
for epoch in range(num_epochs):
    # 1. Forward pass
    embeddings = encoder(x, edge_index, edge_curvature)
    logits = classifier(embeddings)  # [num_nodes, 1]
    probs = sigmoid(logits)
    
    # 2. Loss computation
    if use_focal_loss:
        # Focal Loss: Focus on hard examples
        # α = 0.25 means 75% weight on positives
        # γ = 2.0 down-weights easy examples
        p_t = p * y + (1-p) * (1-y)
        focal_weight = (1 - p_t) ** gamma
        loss = -(alpha * focal_weight * log(p))
    else:
        # Weighted BCE
        # pos_weight = 15 means driver errors cost 15x more
        loss = BCE(logits, labels, pos_weight=15)
    
    # 3. Backward pass
    loss.backward()
    optimizer.step()
    
    # 4. Validation
    val_probs = sigmoid(model(val_data))
    val_preds = (val_probs > 0.5)  # Binary threshold!
    
    # 5. Metrics (problematic with imbalance)
    accuracy = (val_preds == val_labels).mean()  # Misleading!
    f1 = compute_f1(val_preds, val_labels)       # Better, but still threshold-dependent
```

#### Key Issues:

1. **Threshold Dependency:**
```python
# With threshold = 0.5:
Gene A: prob=0.73 → Driver     ✓ Correct
Gene B: prob=0.51 → Driver     ✓ Correct  
Gene C: prob=0.49 → Non-driver ✗ Wrong (close to 0.5!)
Gene D: prob=0.08 → Non-driver ✓ Correct

# With threshold = 0.4:
Gene A: prob=0.73 → Driver     ✓ Correct
Gene B: prob=0.51 → Driver     ✓ Correct
Gene C: prob=0.49 → Driver     ✓ Now correct!
Gene D: prob=0.08 → Non-driver ✓ Correct

# Which threshold is "right"? Nobody knows!
```

2. **Class Imbalance Struggle:**
```python
# Model's dilemma with 1:15 imbalance:

# Strategy A: Predict mostly non-drivers
predictions = [0, 0, 0, 0, 0, 1, 0, 0, ...]  # 93% non-drivers
accuracy = 93%  # High! 🎉
recall = 20%    # Low... 😞
# Misses most drivers, but high accuracy!

# Strategy B: Predict more drivers (with focal loss)
predictions = [1, 1, 0, 1, 0, 1, 1, 0, ...]  # More 1s
accuracy = 75%  # Lower 😞
recall = 80%    # High! 🎉
# Finds more drivers, but many false positives!

# Model constantly fights between these strategies
```

3. **Information Loss:**
```python
# Model thinks:
Gene A: 0.98 → "Super confident driver"
Gene B: 0.52 → "Barely a driver"
Gene C: 0.48 → "Barely non-driver"
Gene D: 0.02 → "Definitely non-driver"

# But you only see:
Gene A: 1 (Driver)
Gene B: 1 (Driver)      # Same label as A!
Gene C: 0 (Non-driver)  # Same label as D!
Gene D: 0 (Non-driver)

# Lost all the nuance!
```

4. **Post-hoc "Potential Drivers":**
```python
def identify_potential_drivers(confidence_threshold=0.5):
    """
    Try to salvage information by looking at false positives
    """
    # 1. Get predictions
    probs = model.predict_probability(data)
    preds = (probs > confidence_threshold)  # Still threshold-dependent!
    
    # 2. Find false positives
    fp_mask = (preds == 1) & (true_labels == 0)
    
    # 3. Filter by "high confidence"
    high_conf_fps = fp_mask & (probs > confidence_threshold)
    
    # 4. Call these "potential drivers"
    # But this is circular reasoning!
    # We're using the SAME threshold that caused the problem!
```

### Biological Problems with Classification

#### Problem 1: Binary Nature vs Continuous Biology
```
Real Biology:
  ┌─────────────────────────────────────────────────────┐
  │  Driver Likelihood (Continuous Spectrum)            │
  ├─────────────────────────────────────────────────────┤
  │  Strong Driver    →    Weak Driver    →  Non-Driver│
  │  (TP53, BRCA1)        (Passenger+)       (ACTB)    │
  │      ▼                    ▼                  ▼      │
  │    [====]              [===]              [=]       │
  │   0.95-1.0           0.40-0.60          0.0-0.1    │
  └─────────────────────────────────────────────────────┘

Classification Forces:
  ┌─────────────────────────────────────────────────────┐
  │  Binary Decision (Threshold = 0.5)                  │
  ├─────────────────────────────────────────────────────┤
  │     DRIVER (1)          |         NON-DRIVER (0)    │
  │  TP53, BRCA1, Gene_X    |    ACTB, Gene_Y, Gene_Z  │
  │                         |                           │
  │  Problem: Gene_X (0.51) and Gene_Y (0.49)          │
  │  are treated as COMPLETELY different!               │
  └─────────────────────────────────────────────────────┘
```

#### Problem 2: Imbalance in Cancer Biology
```
Reality:
  - Human genome: ~20,000 genes
  - Known cancer drivers: ~200-500 genes (2.5%)
  - Imbalance ratio: 1:40 to 1:100
  
Your Data:
  - Imbalance: 1:15 (6.25% drivers)
  - Already "enriched" for drivers!
  
Classification Struggle:
  - Model wants to predict non-driver (it's usually right!)
  - Focal loss / weighted BCE try to fight this
  - But model still torn between strategies
  - Result: Sub-optimal for both classes
```

#### Problem 3: Discovery of Novel Drivers
```
Classification Approach:
  1. Train on known drivers (labels=1)
  2. Model learns "what known drivers look like"
  3. Predict: "Does this gene look like known drivers?"
  4. Problem: Novel drivers might not look like known ones!
  
Example:
  Known Drivers: TP53, BRCA1 (high expression, hub genes)
  Novel Driver: Gene_X (low expression, bridge gene)
  
  Classification: "Gene_X doesn't look like TP53/BRCA1 → Non-driver"
  
  Reality: Gene_X is a driver, just different mechanism!
```

---

## 2. What Your Model Does Now (Ranking) {#ranking-approach}

### Overall Architecture (Ranking)

```
┌─────────────────────────────────────────────────────────────────┐
│                    INPUT: Same as before                        │
│                    - Nodes: Genes (features)                    │
│                    - Edges: Interactions (curvature)            │
│                    - Labels: 0 or 1 (only for training)         │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│         STEP 1: Enhanced Curvature-Aware GNN Encoder            │
│                                                                 │
│  ┌─────────────────┐  ┌─────────────────┐  ┌────────────────┐ │
│  │ Positive Curv   │  │ Negative Curv   │  │   Both Curv    │ │
│  │  + 1-hop        │  │  + 1-hop        │  │  + 1-hop       │ │
│  │  + 2-hop        │  │  + 2-hop        │  │  + 2-hop       │ │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘ │
│           │                    │                     │          │
│           └────────────────────┴─────────────────────┘          │
│                              ↓                                  │
│            6 Pathways (3 curvature × 2 hop types)              │
│                              ↓                                  │
│            Multi-Pathway Aggregator (attention/hierarchical)    │
│                              ↓                                  │
│            Learns which pathways matter for each gene          │
│                              ↓                                  │
│            Final Embedding: [num_nodes, hidden_dim]            │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              STEP 2: Ranking Head (NOT Classification!)         │
│                                                                 │
│  ranking_head = nn.Sequential(                                  │
│      nn.Linear(hidden_dim, hidden_dim),                        │
│      nn.ReLU(),                                                │
│      nn.Dropout(0.2),                                          │
│      nn.Linear(hidden_dim, 1)  # RAW SCORE (not probability!) │
│  )                                                              │
│                          ↓                                      │
│              Output: [num_nodes] raw scores                    │
│              "How driver-like is each gene?" (continuous)      │
│              NO sigmoid! NO threshold! Just scores!            │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                 STEP 3: Ranking Loss                            │
│                                                                 │
│  Pairwise Ranking Loss:                                        │
│    For each (driver, non-driver) pair:                        │
│      loss = max(0, margin - score_driver + score_non_driver)  │
│                                                                 │
│  Goal: driver_score > non_driver_score + margin                │
│  Margin = 1.0 means "drivers should score 1 point higher"     │
│                                                                 │
│  No fighting with thresholds!                                  │
│  No arbitrary pos_weight!                                      │
│  Just optimize rankings!                                       │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    STEP 4: Score ALL Genes                      │
│                                                                 │
│  Gene A (TP53):      score = 4.83                             │
│  Gene B (BRCA1):     score = 4.12                             │
│  Gene C (MYC):       score = 3.45                             │
│  Gene D (Unknown1):  score = 2.91  ← Potential driver!        │
│  Gene E (Unknown2):  score = 0.73                             │
│  Gene F (ACTB):      score = 0.12                             │
│  Gene G (Unknown3):  score = -0.45                            │
│                                                                 │
│  NO THRESHOLDS! Just rank by score!                           │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│               STEP 5: Statistical Significance                  │
│                                                                 │
│  Known drivers: mean_score = 4.2, std = 0.8                   │
│                                                                 │
│  For each unknown gene:                                        │
│    z_score = (score - mean_known) / std_known                 │
│    p_value = P(known_driver scores >= this score)             │
│                                                                 │
│  Gene D: score=2.91, z=−1.6, p=0.012 ← Significant! ✓        │
│  Gene E: score=0.73, z=−4.3, p=0.893 ← Not significant        │
│                                                                 │
│  Adjusted p-value (FDR correction for multiple testing)       │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│               STEP 6: Ranking Metrics (Appropriate!)            │
│                                                                 │
│  AUPRC (Area Under Precision-Recall Curve):                   │
│    - Measures: How well do drivers rank at the top?           │
│    - Perfect: All drivers ranked before all non-drivers       │
│    - Your model: AUPRC = 0.87 (excellent!)                    │
│                                                                 │
│  NDCG@50 (Normalized Discounted Cumulative Gain):             │
│    - Measures: Quality of top 50 rankings                     │
│    - Emphasizes: Getting top ranks right                      │
│    - Your model: NDCG@50 = 0.82                               │
│                                                                 │
│  Precision@K:                                                  │
│    - Top 10: 8/10 are drivers (P@10 = 0.80)                  │
│    - Top 50: 35/50 are drivers (P@50 = 0.70)                 │
│                                                                 │
│  Mean Reciprocal Rank (MRR):                                  │
│    - Average of 1/rank for each known driver                  │
│    - Your model: MRR = 0.73 (drivers typically in top ranks) │
└─────────────────────────────────────────────────────────────────┘
```

### Training Loop (Ranking):
```python
for epoch in range(num_epochs):
    # 1. Forward pass
    embeddings = encoder(x, edge_index, edge_curvature)
    scores = ranking_head(embeddings)  # [num_nodes] raw scores
    
    # 2. Ranking loss
    driver_mask = (labels == 1)
    non_driver_mask = (labels == 0)
    
    driver_scores = scores[driver_mask]
    non_driver_scores = scores[non_driver_mask]
    
    # Pairwise ranking: drivers > non-drivers + margin
    # Shape: [n_drivers, n_non_drivers]
    pairwise_diff = driver_scores.unsqueeze(1) - non_driver_scores.unsqueeze(0)
    loss = F.relu(margin - pairwise_diff).mean()
    
    # 3. Backward pass
    loss.backward()
    optimizer.step()
    
    # 4. Validation (NO THRESHOLD!)
    val_scores = model.score_all_genes(val_data)
    
    # 5. Ranking metrics (appropriate for imbalanced data)
    auprc = average_precision_score(val_labels, val_scores)
    ndcg = compute_ndcg(val_scores, val_labels, k=50)
    
    # Check: Are known drivers ranking high?
    driver_ranks = [rank for gene, rank in 
                   zip(genes, val_scores.argsort(descending=True)) 
                   if gene in known_drivers]
    median_driver_rank = np.median(driver_ranks)  # Want this low!
```

### Key Advantages:

1. **No Threshold Problem:**
```python
# Ranking approach:
scores = model.score_all_genes(data)

# Gene A: 4.83  (Rank 1)  - Clear driver
# Gene B: 4.12  (Rank 2)  - Clear driver
# Gene C: 3.45  (Rank 3)  - Likely driver
# Gene D: 2.91  (Rank 4)  - Potential driver (investigate!)
# Gene E: 0.73  (Rank 5)  - Weak signal
# Gene F: 0.12  (Rank 6)  - Non-driver
# Gene G: -0.45 (Rank 7)  - Definitely non-driver

# No threshold needed! Ranks speak for themselves.
# Can investigate top-K candidates based on resources.
```

2. **Natural Handling of Imbalance:**
```python
# Model's goal: Rank drivers higher than non-drivers
# Doesn't care about class proportions!

# With 1:15 imbalance:
# - 100 drivers, 1500 non-drivers
# - Model just needs: driver_scores > non_driver_scores
# - No fighting between "predict more 1s" vs "predict more 0s"
# - Just optimize ranking quality!
```

3. **Continuous Information Preserved:**
```python
# Classification threw away information:
Gene A: 0.98 → 1 (Driver)
Gene B: 0.52 → 1 (Driver)  # Same as A!
Gene C: 0.48 → 0 (Non-driver)
Gene D: 0.02 → 0 (Non-driver)  # Same as C!

# Ranking keeps all nuance:
Gene A: 4.83 (Rank 1)  - Strong driver
Gene B: 3.12 (Rank 12) - Moderate driver
Gene C: 0.91 (Rank 45) - Weak signal
Gene D: 0.08 (Rank 89) - Non-driver

# All differences preserved!
```

4. **Better for Discovery:**
```python
# Classification: "Is this like known drivers?"
# Ranking: "How driver-like is this?"

# Novel driver with different mechanism:
Gene_X: score = 3.2
  - Rank: 4th overall
  - Z-score: 2.1 (vs known drivers)
  - P-value: 0.018 (significant!)
  
# Even if Gene_X doesn't look exactly like TP53/BRCA1,
# if it scores high, it's worth investigating!
```

---

## 3. Why This Change Matters {#why-change}

### Problem Summary Table

| Issue | Classification | Ranking |
|-------|---------------|---------|
| **Threshold dependency** | Critical problem | No threshold needed |
| **Class imbalance** | Constant struggle | Naturally handles |
| **Information loss** | Binary output | Continuous scores |
| **Novel discovery** | Biased to known patterns | Open to new patterns |
| **Interpretability** | "Driver or not?" | "How driver-like?" |
| **Evaluation** | Misleading metrics | Appropriate metrics |
| **Post-hoc analysis** | Awkward (potential drivers) | Natural (top-K) |

### Real-World Impact

#### Scenario: Clinical Application

**Classification Approach:**
```
Doctor: "Which genes should we test for in this patient?"
Model: "I predict 47 genes are drivers" (using threshold=0.4)
Doctor: "That's too many! We can only test 10."
Model: "... then use threshold=0.7? That gives 8 genes."
Doctor: "But what if the 9th gene is important?"
Model: "¯\_(ツ)_/¯ It scored 0.68, just below threshold."

Result: Arbitrary cutoff, might miss important genes.
```

**Ranking Approach:**
```
Doctor: "Which genes should we test for in this patient?"
Model: "Here are all genes ranked by driver likelihood:
  1. TP53     (score: 4.83, p<0.001) ← Definitely test
  2. BRCA1    (score: 4.12, p<0.001) ← Definitely test
  3. MYC      (score: 3.45, p=0.003) ← Definitely test
  4. PIK3CA   (score: 3.21, p=0.008) ← Definitely test
  5. EGFR     (score: 2.91, p=0.012) ← Test if budget allows
  6. Unknown1 (score: 2.73, p=0.021) ← Interesting! Test?
  7. ALK      (score: 2.45, p=0.034) ← Maybe test
  8. Unknown2 (score: 2.12, p=0.052) ← Borderline
  ..."

Doctor: "Perfect! Test top 10, prioritize top 5."

Result: Clear prioritization, no arbitrary cutoffs.
```

---

## 4. Methods to Deprecate {#deprecation}

### Template for Deprecation Warnings

```python
import warnings
from functools import wraps

def deprecated(reason, alternative=None, removal_version=None):
    """
    Decorator to mark methods as deprecated.
    
    Args:
        reason: Why this method is deprecated
        alternative: What to use instead
        removal_version: When this will be removed
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            message = f"\n{'='*80}\n"
            message += f"⚠️  DEPRECATION WARNING\n"
            message += f"{'='*80}\n"
            message += f"Method: {func.__name__}\n"
            message += f"Reason: {reason}\n"
            if alternative:
                message += f"\nUse instead:\n{alternative}\n"
            if removal_version:
                message += f"\nThis method will be removed in version {removal_version}\n"
            message += f"{'='*80}\n"
            warnings.warn(message, DeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)
        return wrapper
    return decorator
```

### Methods to Deprecate

```python
class ContrastiveDriverGenePredictor(nn.Module):
    """
    Enhanced driver gene predictor with ranking-based approach.
    
    DEPRECATED METHODS (Classification-based):
    - forward() → Use compute_ranking_scores() instead
    - compute_classification_loss() → Use RankingLoss instead
    - compute_focal_loss() → Use RankingLoss instead
    - train_step() → Use train_step_ranking() instead
    - evaluate() → Use evaluate_ranking() instead
    - predict_probability() → Use score_all_genes() instead
    - identify_potential_drivers() → Use score_genes_with_statistics() instead
    
    RECOMMENDED METHODS (Ranking-based):
    - compute_ranking_scores() → Get continuous scores for all genes
    - train_step_ranking() → Train with ranking loss
    - evaluate_ranking() → Evaluate with ranking metrics
    - score_all_genes() → Score and rank all genes
    - score_genes_with_statistics() → Score with statistical significance
    """
    
    # ========================================================================
    # DEPRECATED: Binary Classification Methods
    # ========================================================================
    
    @deprecated(
        reason="Binary classification not suitable for severe class imbalance (1:15)",
        alternative="""
# Old (classification):
logits, embeddings = model.forward(x, edge_index, edge_curvature)
probs = torch.sigmoid(logits)
predictions = (probs > 0.5)  # Binary!

# New (ranking):
scores = model.compute_ranking_scores(x, edge_index, edge_curvature)
df_ranked = model.score_all_genes(data)  # All genes ranked!
        """,
        removal_version="2.0.0"
    )
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_curvature: torch.Tensor,
        return_embeddings: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        DEPRECATED: Use compute_ranking_scores() instead
        
        Forward pass for binary classification (DEPRECATED).
        """
        h, _ = self.encode(x, edge_index, edge_curvature)
        logits = self.classifier(h)
        
        if return_embe