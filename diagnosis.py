"""
Comprehensive Diagnostic Script for Debugging Poor Model Performance

Run this to identify the root causes of low NDCG/AUROC scores.
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy import stats

def diagnose_data_quality(data, labels, train_mask, val_mask):
    """Comprehensive data quality checks"""
    
    print("=" * 80)
    print("DATA QUALITY DIAGNOSTICS")
    print("=" * 80)
    
    features = data.get('x', data.get('feature'))
    edge_index = data['edge_index']
    edge_curvature = data.get('ollivier_curvature', data.get('forman_curvature'))
    
    issues = []
    
    # 1. Check for NaN/Inf
    print("\n1. NaN/Inf Check:")
    print("-" * 60)
    
    nan_features = torch.isnan(features).any(dim=1).sum().item()
    inf_features = torch.isinf(features).any(dim=1).sum().item()
    nan_curvature = torch.isnan(edge_curvature).sum().item()
    
    print(f"  Nodes with NaN features: {nan_features}")
    print(f"  Nodes with Inf features: {inf_features}")
    print(f"  Edges with NaN curvature: {nan_curvature}")
    
    if nan_features > 0 or inf_features > 0 or nan_curvature > 0:
        issues.append("❌ NaN/Inf values detected in data")
    else:
        print("  ✓ No NaN/Inf values")
    
    # 2. Class Balance
    print("\n2. Class Balance:")
    print("-" * 60)
    
    total_nodes = len(labels)
    num_drivers = labels.sum().item()
    num_non_drivers = total_nodes - num_drivers
    driver_ratio = num_drivers / total_nodes
    
    print(f"  Total nodes: {total_nodes}")
    print(f"  Driver genes: {num_drivers} ({driver_ratio*100:.2f}%)")
    print(f"  Non-driver genes: {num_non_drivers} ({(1-driver_ratio)*100:.2f}%)")
    print(f"  Imbalance ratio: 1:{num_non_drivers/num_drivers:.1f}")
    
    if driver_ratio < 0.01:
        issues.append("⚠️  Extreme class imbalance (<1% drivers)")
    elif driver_ratio < 0.05:
        print("  ⚠️  High class imbalance (1-5% drivers)")
    else:
        print("  ✓ Reasonable class balance")
    
    # 3. Train/Val Split Quality
    print("\n3. Train/Val Split Quality:")
    print("-" * 60)
    
    train_drivers = labels[train_mask].sum().item()
    val_drivers = labels[val_mask].sum().item()
    train_total = train_mask.sum().item()
    val_total = val_mask.sum().item()
    
    print(f"  Train: {train_total} nodes, {train_drivers} drivers ({train_drivers/train_total*100:.2f}%)")
    print(f"  Val: {val_total} nodes, {val_drivers} drivers ({val_drivers/val_total*100:.2f}%)")
    
    if train_drivers < 10:
        issues.append("❌ Too few drivers in training set (<10)")
    
    if val_drivers < 5:
        issues.append("❌ Too few drivers in validation set (<5)")
    
    # 4. Feature Quality
    print("\n4. Feature Quality:")
    print("-" * 60)
    
    # Check feature variance
    feature_std = features.std(dim=0)
    zero_variance_features = (feature_std < 1e-6).sum().item()
    
    print(f"  Feature dimension: {features.shape[1]}")
    print(f"  Zero-variance features: {zero_variance_features}")
    print(f"  Mean feature std: {feature_std.mean().item():.4f}")
    print(f"  Feature value range: [{features.min().item():.4f}, {features.max().item():.4f}]")
    
    if zero_variance_features > features.shape[1] * 0.1:
        issues.append("⚠️  >10% features have zero variance")
    
    # Check if features are all zeros
    all_zero_nodes = (features.abs().sum(dim=1) == 0).sum().item()
    if all_zero_nodes > 0:
        issues.append(f"❌ {all_zero_nodes} nodes have all-zero features")
    
    # 5. Curvature Distribution
    print("\n5. Curvature Distribution:")
    print("-" * 60)
    
    curv_mean = edge_curvature.mean().item()
    curv_std = edge_curvature.std().item()
    curv_min = edge_curvature.min().item()
    curv_max = edge_curvature.max().item()
    
    print(f"  Mean: {curv_mean:.4f}")
    print(f"  Std: {curv_std:.4f}")
    print(f"  Range: [{curv_min:.4f}, {curv_max:.4f}]")
    
    positive_curv = (edge_curvature > 0).sum().item()
    negative_curv = (edge_curvature < 0).sum().item()
    zero_curv = (edge_curvature == 0).sum().item()
    
    print(f"  Positive curvature: {positive_curv} ({positive_curv/len(edge_curvature)*100:.1f}%)")
    print(f"  Negative curvature: {negative_curv} ({negative_curv/len(edge_curvature)*100:.1f}%)")
    print(f"  Zero curvature: {zero_curv} ({zero_curv/len(edge_curvature)*100:.1f}%)")
    
    if zero_curv > len(edge_curvature) * 0.5:
        issues.append("⚠️  >50% edges have zero curvature")
    
    # 6. Graph Connectivity
    print("\n6. Graph Connectivity:")
    print("-" * 60)
    
    num_nodes = features.shape[0]
    num_edges = edge_index.shape[1]
    avg_degree = num_edges / num_nodes
    
    print(f"  Nodes: {num_nodes}")
    print(f"  Edges: {num_edges}")
    print(f"  Average degree: {avg_degree:.2f}")
    
    # Check for isolated nodes
    node_degrees = torch.zeros(num_nodes, dtype=torch.long)
    node_degrees.scatter_add_(0, edge_index[0], torch.ones_like(edge_index[0]))
    isolated_nodes = (node_degrees == 0).sum().item()
    
    print(f"  Isolated nodes: {isolated_nodes}")
    
    if isolated_nodes > 0:
        issues.append(f"⚠️  {isolated_nodes} isolated nodes (no edges)")
    
    if avg_degree < 2:
        issues.append("⚠️  Very sparse graph (avg degree < 2)")
    
    # 7. Label Distribution in Graph
    print("\n7. Label Distribution in Graph:")
    print("-" * 60)
    
    # Check if drivers are clustered or scattered
    driver_indices = torch.where(labels == 1)[0]
    
    if len(driver_indices) > 0:
        # Get driver neighbors
        driver_edges = []
        for driver_idx in driver_indices:
            neighbors = edge_index[1][edge_index[0] == driver_idx]
            driver_edges.extend(neighbors.tolist())
        
        if len(driver_edges) > 0:
            driver_neighbor_labels = labels[torch.tensor(driver_edges)]
            driver_neighbor_ratio = driver_neighbor_labels.float().mean().item()
            print(f"  Drivers' neighbors that are drivers: {driver_neighbor_ratio*100:.2f}%")
            
            if driver_neighbor_ratio < 0.01:
                issues.append("⚠️  Drivers are isolated (few driver neighbors)")
    
    # Summary
    print("\n" + "=" * 80)
    print("DIAGNOSIS SUMMARY")
    print("=" * 80)
    
    if len(issues) == 0:
        print("✓ No critical data quality issues detected")
    else:
        print(f"Found {len(issues)} issues:")
        for i, issue in enumerate(issues, 1):
            print(f"{i}. {issue}")
    
    return issues


def test_baseline_models(data, labels, train_mask, val_mask):
    """Test simple baseline models to see if data is learnable"""
    
    print("\n" + "=" * 80)
    print("BASELINE MODEL COMPARISON")
    print("=" * 80)
    
    features = data.get('x', data.get('feature')).cpu().numpy()
    labels_np = labels.cpu().numpy()
    train_mask_np = train_mask.cpu().numpy()
    val_mask_np = val_mask.cpu().numpy()
    
    X_train = features[train_mask_np]
    y_train = labels_np[train_mask_np]
    X_val = features[val_mask_np]
    y_val = labels_np[val_mask_np]
    
    results = {}
    
    # 1. Random Baseline
    print("\n1. Random Baseline:")
    print("-" * 60)
    random_scores = np.random.rand(len(y_val))
    random_auroc = roc_auc_score(y_val, random_scores)
    random_auprc = average_precision_score(y_val, random_scores)
    print(f"  AUROC: {random_auroc:.4f}")
    print(f"  AUPRC: {random_auprc:.4f}")
    results['Random'] = {'auroc': random_auroc, 'auprc': random_auprc}
    
    # 2. Logistic Regression
    print("\n2. Logistic Regression (features only):")
    print("-" * 60)
    try:
        lr = LogisticRegression(max_iter=1000, class_weight='balanced')
        lr.fit(X_train, y_train)
        lr_scores = lr.predict_proba(X_val)[:, 1]
        lr_auroc = roc_auc_score(y_val, lr_scores)
        lr_auprc = average_precision_score(y_val, lr_scores)
        print(f"  AUROC: {lr_auroc:.4f}")
        print(f"  AUPRC: {lr_auprc:.4f}")
        results['LogisticRegression'] = {'auroc': lr_auroc, 'auprc': lr_auprc}
    except Exception as e:
        print(f"  Error: {e}")
        results['LogisticRegression'] = {'auroc': 0.0, 'auprc': 0.0}
    
    # 3. Random Forest
    print("\n3. Random Forest (features only):")
    print("-" * 60)
    try:
        rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
        rf.fit(X_train, y_train)
        rf_scores = rf.predict_proba(X_val)[:, 1]
        rf_auroc = roc_auc_score(y_val, rf_scores)
        rf_auprc = average_precision_score(y_val, rf_scores)
        print(f"  AUROC: {rf_auroc:.4f}")
        print(f"  AUPRC: {rf_auprc:.4f}")
        
        # Feature importance
        importances = rf.feature_importances_
        top_features = np.argsort(importances)[-10:][::-1]
        print(f"  Top 10 important features: {top_features.tolist()}")
        
        results['RandomForest'] = {'auroc': rf_auroc, 'auprc': rf_auprc}
    except Exception as e:
        print(f"  Error: {e}")
        results['RandomForest'] = {'auroc': 0.0, 'auprc': 0.0}
    
    # 4. Your GNN Model (from results)
    print("\n4. Your GNN Model:")
    print("-" * 60)
    print(f"  AUROC: 0.6916")
    print(f"  AUPRC: 0.2095")
    results['YourGNN'] = {'auroc': 0.6916, 'auprc': 0.2095}
    
    # Analysis
    print("\n" + "=" * 80)
    print("BASELINE ANALYSIS")
    print("=" * 80)
    
    print("\nModel Performance Summary:")
    print(f"{'Model':<20} {'AUROC':<10} {'AUPRC':<10}")
    print("-" * 40)
    for model, scores in results.items():
        print(f"{model:<20} {scores['auroc']:<10.4f} {scores['auprc']:<10.4f}")
    
    # Diagnosis
    print("\nDiagnosis:")
    
    rf_auroc = results.get('RandomForest', {}).get('auroc', 0)
    lr_auroc = results.get('LogisticRegression', {}).get('auroc', 0)
    gnn_auroc = results['YourGNN']['auroc']
    
    if rf_auroc < 0.65:
        print("❌ CRITICAL: Even Random Forest performs poorly (<0.65 AUROC)")
        print("   → Problem is with the DATA/FEATURES, not the model architecture")
        print("   → Action: Improve feature engineering or get better features")
    elif lr_auroc < 0.70 and rf_auroc > 0.75:
        print("⚠️  Non-linear relationships exist (RF >> LR)")
        print("   → GNN should help, but needs tuning")
    
    if gnn_auroc < lr_auroc:
        print("❌ CRITICAL: GNN worse than simple Logistic Regression!")
        print("   → Model is not learning from graph structure")
        print("   → Action: Check edge_index, curvature computation, or simplify model")
    elif gnn_auroc < rf_auroc - 0.05:
        print("⚠️  GNN underperforming Random Forest")
        print("   → Graph structure not helping, or model undertrained")
        print("   → Action: Tune hyperparameters or train longer")
    else:
        print("✓ GNN competitive with baselines")
        print("  → Model architecture OK, needs hyperparameter tuning")
    
    return results


def analyze_predictions(model, data, labels, val_mask, device='cuda'):
    """Analyze model predictions to understand failures"""
    
    print("\n" + "=" * 80)
    print("PREDICTION ANALYSIS")
    print("=" * 80)
    
    model.eval()
    with torch.no_grad():
        scores, embeddings = model.forward(
            data['x'].to(device),
            data['edge_index'].to(device),
            data.get('ollivier_curvature', data.get('forman_curvature')).to(device),
            return_embeddings=True
        )
    
    scores = scores.cpu().numpy()
    labels_np = labels.cpu().numpy()
    val_mask_np = val_mask.cpu().numpy()
    
    val_scores = scores[val_mask_np]
    val_labels = labels_np[val_mask_np]
    
    # 1. Score distribution
    print("\n1. Score Distribution:")
    print("-" * 60)
    
    driver_scores = val_scores[val_labels == 1]
    nondriver_scores = val_scores[val_labels == 0]
    
    print(f"  Drivers:")
    print(f"    Mean: {driver_scores.mean():.4f}")
    print(f"    Std: {driver_scores.std():.4f}")
    print(f"    Range: [{driver_scores.min():.4f}, {driver_scores.max():.4f}]")
    
    print(f"  Non-drivers:")
    print(f"    Mean: {nondriver_scores.mean():.4f}")
    print(f"    Std: {nondriver_scores.std():.4f}")
    print(f"    Range: [{nondriver_scores.min():.4f}, {nondriver_scores.max():.4f}]")
    
    # Statistical test
    t_stat, p_value = stats.ttest_ind(driver_scores, nondriver_scores)
    print(f"\n  T-test: t={t_stat:.4f}, p={p_value:.6f}")
    
    if p_value > 0.05:
        print("  ❌ Distributions NOT significantly different (p > 0.05)")
        print("     → Model cannot discriminate drivers from non-drivers")
    else:
        print("  ✓ Distributions significantly different")
    
    # 2. Top predictions
    print("\n2. Top 50 Predictions:")
    print("-" * 60)
    
    top_50_indices = np.argsort(-val_scores)[:50]
    top_50_labels = val_labels[top_50_indices]
    
    num_drivers_in_top50 = top_50_labels.sum()
    print(f"  Drivers in top 50: {num_drivers_in_top50} ({num_drivers_in_top50/50*100:.1f}%)")
    
    if num_drivers_in_top50 < 5:
        print("  ❌ Almost no drivers in top 50 (<10%)")
    elif num_drivers_in_top50 < 20:
        print("  ⚠️  Few drivers in top 50 (<40%)")
    else:
        print("  ✓ Good driver enrichment in top 50")
    
    # 3. Embedding quality
    print("\n3. Embedding Quality:")
    print("-" * 60)
    
    if embeddings is not None:
        embeddings_np = embeddings.cpu().numpy()[val_mask_np]
        
        # Check for NaN
        nan_embeddings = np.isnan(embeddings_np).any(axis=1).sum()
        print(f"  Embeddings with NaN: {nan_embeddings}")
        
        # Check for collapse
        embedding_std = embeddings_np.std(axis=0).mean()
        print(f"  Mean dimension std: {embedding_std:.4f}")
        
        if embedding_std < 0.1:
            print("  ❌ Embedding collapse detected (low variance)")
            print("     → Model is not learning diverse representations")
        
        # Check separation
        driver_emb = embeddings_np[val_labels == 1]
        nondriver_emb = embeddings_np[val_labels == 0]
        
        if len(driver_emb) > 0 and len(nondriver_emb) > 0:
            driver_centroid = driver_emb.mean(axis=0)
            nondriver_centroid = nondriver_emb.mean(axis=0)
            centroid_distance = np.linalg.norm(driver_centroid - nondriver_centroid)
            
            avg_within_std = (driver_emb.std() + nondriver_emb.std()) / 2
            separation_ratio = centroid_distance / avg_within_std
            
            print(f"  Centroid distance: {centroid_distance:.4f}")
            print(f"  Average within-class std: {avg_within_std:.4f}")
            print(f"  Separation ratio: {separation_ratio:.4f}")
            
            if separation_ratio < 0.5:
                print("  ❌ Poor class separation in embedding space")


def plot_diagnostics(data, labels, train_mask, val_mask, save_path='diagnostics.png'):
    """Create diagnostic plots"""
    
    features = data.get('x', data.get('feature')).cpu().numpy()
    edge_curvature = data.get('ollivier_curvature', data.get('forman_curvature')).cpu().numpy()
    labels_np = labels.cpu().numpy()
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Class distribution
    ax = axes[0, 0]
    class_counts = [len(labels_np) - labels_np.sum(), labels_np.sum()]
    ax.bar(['Non-driver', 'Driver'], class_counts)
    ax.set_ylabel('Count')
    ax.set_title('Class Distribution')
    ax.set_yscale('log')
    
    # 2. Feature statistics
    ax = axes[0, 1]
    feature_means = features.mean(axis=0)
    ax.hist(feature_means, bins=50)
    ax.set_xlabel('Feature Mean')
    ax.set_ylabel('Count')
    ax.set_title('Feature Mean Distribution')
    
    # 3. Feature variance
    ax = axes[0, 2]
    feature_stds = features.std(axis=0)
    ax.hist(feature_stds, bins=50)
    ax.set_xlabel('Feature Std')
    ax.set_ylabel('Count')
    ax.set_title('Feature Variance Distribution')
    
    # 4. Curvature distribution
    ax = axes[1, 0]
    ax.hist(edge_curvature, bins=100, alpha=0.7)
    ax.axvline(0, color='red', linestyle='--', label='Zero')
    ax.set_xlabel('Edge Curvature')
    ax.set_ylabel('Count')
    ax.set_title('Curvature Distribution')
    ax.legend()
    
    # 5. Train/Val split
    ax = axes[1, 1]
    split_data = {
        'Train': [train_mask.sum().item() - labels[train_mask].sum().item(), 
                  labels[train_mask].sum().item()],
        'Val': [val_mask.sum().item() - labels[val_mask].sum().item(), 
                labels[val_mask].sum().item()]
    }
    x = np.arange(2)
    width = 0.35
    ax.bar(x - width/2, [split_data['Train'][0], split_data['Val'][0]], width, label='Non-driver')
    ax.bar(x + width/2, [split_data['Train'][1], split_data['Val'][1]], width, label='Driver')
    ax.set_xticks(x)
    ax.set_xticklabels(['Train', 'Val'])
    ax.set_ylabel('Count')
    ax.set_title('Train/Val Split')
    ax.legend()
    ax.set_yscale('log')
    
    # 6. Feature correlation with labels
    ax = axes[1, 2]
    correlations = []
    for i in range(features.shape[1]):
        corr = np.corrcoef(features[:, i], labels_np)[0, 1]
        if not np.isnan(corr):
            correlations.append(abs(corr))
    
    if len(correlations) > 0:
        ax.hist(correlations, bins=50)
        ax.set_xlabel('|Correlation with Label|')
        ax.set_ylabel('Count')
        ax.set_title('Feature-Label Correlation')
        ax.axvline(np.mean(correlations), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(correlations):.4f}')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Diagnostic plots saved to {save_path}")
    plt.close()


# Main diagnostic function
def run_full_diagnostics(model, data, labels, train_mask, val_mask, device='cuda'):
    """Run all diagnostics"""
    
    print("\n" + "="*80)
    print("COMPREHENSIVE MODEL DIAGNOSTICS")
    print("Current Performance: NDCG@50=0.469, AUROC=0.692, AUPRC=0.210")
    print("="*80)
    
    # 1. Data quality
    data_issues = diagnose_data_quality(data, labels, train_mask, val_mask)
    
    # 2. Baseline comparison
    baseline_results = test_baseline_models(data, labels, train_mask, val_mask)
    
    # 3. Prediction analysis
    if model is not None:
        analyze_predictions(model, data, labels, val_mask, device)
    
    # 4. Generate plots
    plot_diagnostics(data, labels, train_mask, val_mask)
    
    # Final recommendations
    print("\n" + "="*80)
    print("RECOMMENDED ACTIONS (Priority Order)")
    print("="*80)
    
    rf_auroc = baseline_results.get('RandomForest', {}).get('auroc', 0)
    
    if rf_auroc < 0.70:
        print("\n🔴 PRIORITY 1: DATA/FEATURE PROBLEMS")
        print("   Your features are not predictive enough.")
        print("   Actions:")
        print("   1. Add more informative features (biological pathways, expression, etc.)")
        print("   2. Improve feature normalization/scaling")
        print("   3. Remove noisy/irrelevant features")
        print("   4. Check if you have the right labels")
    else:
        print("\n🟢 Features are OK (RF AUROC > 0.70)")
    
    print("\n🔴 PRIORITY 2: MODEL TRAINING ISSUES")
    print("   Actions:")
    print("   1. Increase training epochs (try 200-300)")
    print("   2. Lower learning rate (try 5e-4 or 2e-4)")
    print("   3. Increase contrastive_weight to 0.5-0.7")
    print("   4. Reduce dropout to 0.1")
    print("   5. Simplify model (use 2 layers, hidden_dim=64)")
    
    print("\n🟡 PRIORITY 3: GRAPH STRUCTURE")
    print("   Actions:")
    print("   1. Verify edge_index is correct")
    print("   2. Check curvature computation")
    print("   3. Try attention_mode='bias' (simpler)")
    print("   4. Use only ['both'] curvature type")
    
    return data_issues, baseline_results


if __name__ == "__main__":
    import pickle
    with open('curvature_output/GGNet_contrastive_v2_random_r0.2.pkl', 'rb') as f:
        data_dict = pickle.load(f)
    
    feature = data_dict.get('feature')
    labels = data_dict.get('label')
    run_full_diagnostics(model = None, data = feature, labels=labels, train_mask, val_mask, device='cuda')
