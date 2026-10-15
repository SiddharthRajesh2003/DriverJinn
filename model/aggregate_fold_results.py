"""
Aggregate results from parallel fold training runs.

Usage:
    python -m model.aggregate_fold_results --prefix GGNet_random_r0.1 --num_folds 5
"""

import argparse
import pickle
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_fold_metrics(results_dir: Path, prefix: str, num_folds: int) -> pd.DataFrame:
    """Load metrics from each fold directory and combine."""
    all_metrics = []

    for fold in range(1, num_folds + 1):
        fold_dir = results_dir / f"{prefix}_fold{fold}"
        metrics_file = fold_dir / f"{prefix}_fold{fold}_kfold_metrics.csv"

        if not metrics_file.exists():
            print(f"Warning: {metrics_file} not found, skipping fold {fold}")
            continue

        df = pd.read_csv(metrics_file)
        # Filter out mean/std rows if present (from single-fold runs)
        df = df[~df['Fold'].isin(['Mean', 'Std'])]
        df['Fold'] = fold  # Ensure correct fold number
        all_metrics.append(df)
        print(f"Loaded fold {fold}: {len(df)} rows")

    if not all_metrics:
        raise ValueError(f"No metrics files found for prefix '{prefix}'")

    return pd.concat(all_metrics, ignore_index=True)


def load_fold_histories(results_dir: Path, prefix: str, num_folds: int) -> Dict:
    """Load training histories from each fold."""
    all_histories = {}

    for fold in range(1, num_folds + 1):
        fold_dir = results_dir / f"{prefix}_fold{fold}"
        history_file = fold_dir / f"{prefix}_fold{fold}_training_history.pkl"

        if not history_file.exists():
            print(f"Warning: {history_file} not found, skipping fold {fold}")
            continue

        try:
            with open(history_file, 'rb') as f:
                history = pickle.load(f)
            all_histories[fold] = history
            print(f"Loaded training history for fold {fold}")
        except (EOFError, pickle.UnpicklingError) as e:
            print(f"Warning: training history for fold {fold} is corrupted ({e}), skipping")

    return all_histories


def load_fold_gene_scores(results_dir: Path, prefix: str, num_folds: int) -> List[pd.DataFrame]:
    """Load gene scores from each fold."""
    all_scores = []

    for fold in range(1, num_folds + 1):
        fold_dir = results_dir / f"{prefix}_fold{fold}"

        # Try multiple naming patterns
        candidate_patterns = [
            f"{prefix}_fold{fold}_fold_1_all_genes_scored.csv",  # Array job pattern
            f"{prefix}_fold{fold}_all_genes_scored.csv",
            f"{prefix}_fold{fold}_gene_scores.csv",
            f"{prefix}_fold{fold}_fold_{fold}_gene_scores.csv",
        ]

        scores_file = None
        for pattern in candidate_patterns:
            candidate = fold_dir / pattern
            if candidate.exists():
                scores_file = candidate
                break

        if scores_file is None:
            # Try glob for any gene score file
            matches = list(fold_dir.glob("*genes_scored*.csv")) + list(fold_dir.glob("*gene_scores*.csv"))
            if matches:
                scores_file = matches[0]

        if scores_file is None:
            print(f"Warning: gene scores not found for fold {fold}, skipping")
            continue

        df = pd.read_csv(scores_file)
        df['source_fold'] = fold
        all_scores.append(df)
        print(f"Loaded gene scores for fold {fold}: {len(df)} genes from {scores_file.name}")

    return all_scores


def compute_summary_statistics(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean and std across folds."""
    numeric_cols = metrics_df.select_dtypes(include=[np.number]).columns
    numeric_cols = [c for c in numeric_cols if c != 'Fold']

    mean_values = metrics_df[numeric_cols].mean()
    std_values = metrics_df[numeric_cols].std()

    mean_row = {'Fold': 'Mean', **mean_values.to_dict()}
    std_row = {'Fold': 'Std', **std_values.to_dict()}

    summary_df = pd.concat([
        metrics_df,
        pd.DataFrame([mean_row, std_row])
    ], ignore_index=True)

    return summary_df


TOP_K = 100       # Candidate cut-off: rank among unlabelled genes within a fold
MIN_FOLDS = 3     # Folds in which a gene must make the cut-off to be a consensus candidate


def aggregate_gene_scores(score_dfs: List[pd.DataFrame], output_dir: Path, prefix: str):
    """
    Aggregate per-fold gene scores into one ranking.

    - Genes are aligned by gene_id: each fold's file is sorted by that fold's own scores.
    - Genes are ranked by mean within-fold rank, because raw scores are not calibrated across folds.
    - Candidates are unlabelled genes (is_labeled == False; true_label == 0 for older files without that
      column) that rank in the top TOP_K unlabelled genes in at least MIN_FOLDS folds.
    """
    if not score_dfs:
        print("No gene scores to aggregate")
        return None

    print(f"\nAggregating scores from {len(score_dfs)} folds...")

    dfs = [df.set_index('gene_id').sort_index() for df in score_dfs]
    ref = dfs[0]
    for df in dfs[1:]:
        if not df.index.equals(ref.index) or not (df['gene_name'] == ref['gene_name']).all():
            raise ValueError("Folds contain different genes; cannot aggregate")

    is_driver = ref['true_label'].values == 1
    labeled = ref['is_labeled'].values.astype(bool) if 'is_labeled' in ref else is_driver
    unlabeled = ~labeled

    scores = np.column_stack([df['driver_score'].values for df in dfs])     # [num_genes, num_folds], gene-aligned
    ranks = np.column_stack([df['rank'].values for df in dfs])

    unlabeled_rank = np.full(ranks.shape, np.inf)
    for j in range(ranks.shape[1]):
        order = np.argsort(ranks[unlabeled, j])
        r = np.empty(len(order))
        r[order] = np.arange(1, len(order) + 1)
        unlabeled_rank[unlabeled, j] = r                                    # Rank among unlabelled genes only
    folds_top = (unlabeled_rank <= TOP_K).sum(axis=1)

    aggregate_df = pd.DataFrame({
        'gene_id': ref.index.values,
        'gene_name': ref['gene_name'].values,
        'is_known_driver': is_driver,
        'is_labeled': labeled,
        'mean_rank': ranks.mean(axis=1),
        'median_rank': np.median(ranks, axis=1),
        'std_rank': ranks.std(axis=1),
        'mean_score': scores.mean(axis=1),
        'std_score': scores.std(axis=1),
        f'folds_top{TOP_K}': folds_top,
        'total_folds': len(dfs),
        f'consensus_top{TOP_K}': unlabeled & (folds_top >= MIN_FOLDS),
    })
    aggregate_df = aggregate_df.sort_values(['mean_rank', 'median_rank'], kind='stable').reset_index(drop=True)
    aggregate_df.insert(0, 'aggregate_rank', np.arange(1, len(aggregate_df) + 1))

    consensus_genes = aggregate_df[aggregate_df[f'consensus_top{TOP_K}']]
    display_cols = ['aggregate_rank', 'gene_name', 'is_known_driver', 'mean_rank', 'std_rank',
                    f'folds_top{TOP_K}', 'total_folds']

    print(f"\n{'='*80}")
    print("AGGREGATE RESULTS")
    print(f"{'='*80}")
    print(f"\nTotal genes: {len(aggregate_df)}")
    print(f"Known drivers: {aggregate_df['is_known_driver'].sum()} | Labelled genes: {aggregate_df['is_labeled'].sum()}")
    print(f"Consensus candidates (top {TOP_K} unlabelled in >= {MIN_FOLDS} folds): {len(consensus_genes)}")
    print(f"\nTop 20 genes (by mean within-fold rank):")
    print("-" * 80)
    print(aggregate_df[display_cols].head(20).to_string(index=False))
    if len(consensus_genes) > 0:
        print(f"\nTop 20 consensus candidates:")
        print("-" * 80)
        print(consensus_genes[display_cols].head(20).to_string(index=False))

    output_prefix = f"{prefix}_" if prefix else ""

    all_file = output_dir / f"{output_prefix}aggregated_all_genes.csv"
    aggregate_df.to_csv(all_file, index=False)
    print(f"\nSaved aggregated scores to: {all_file}")

    consensus_file = output_dir / f"{output_prefix}aggregated_consensus_top{TOP_K}.csv"
    consensus_genes.to_csv(consensus_file, index=False)
    print(f"Saved {len(consensus_genes)} consensus candidates to: {consensus_file}")

    summary_file = output_dir / f"{output_prefix}aggregated_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("AGGREGATED GENE SCORING SUMMARY (ACROSS ALL FOLDS)\n")
        f.write("="*80 + "\n\n")
        f.write(f"Number of folds: {len(dfs)}\n")
        f.write(f"Total genes: {len(aggregate_df)}\n")
        f.write(f"Known driver genes: {aggregate_df['is_known_driver'].sum()}\n")
        f.write(f"Labelled genes: {aggregate_df['is_labeled'].sum()}\n")
        f.write("Ranking: mean within-fold rank (raw scores are not calibrated across folds)\n\n")
        f.write(f"Consensus candidates (top {TOP_K} unlabelled genes in >= {MIN_FOLDS} folds): {len(consensus_genes)}\n\n")
        f.write("Top 20 genes:\n")
        f.write("-"*80 + "\n")
        f.write(aggregate_df[display_cols].head(20).to_string(index=False))
        f.write("\n\n")
        if len(consensus_genes) > 0:
            f.write("Top 20 consensus candidates:\n")
            f.write("-"*80 + "\n")
            f.write(consensus_genes[display_cols].head(20).to_string(index=False))

    print(f"Saved aggregate summary to: {summary_file}")
    print(f"\n{'='*80}\n")

    return aggregate_df


def plot_combined_training_curves(histories: Dict, output_dir: Path, prefix: str):
    """Plot training curves from all folds - matches train_model.py layout."""
    if not histories:
        print("No training histories to plot")
        return

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Training Curves Across Folds', fontsize=16)

    metrics_to_plot = [
        ('train_loss', 'Training Loss', 'lower'),
        ('train_ndcg', 'Training NDCG', 'upper'),
        ('val_auroc', 'Validation AUROC', 'upper'),
        ('val_ndcg@50', 'Validation NDCG@50', 'upper'),
        ('val_precision@50', 'Validation Precision@50', 'upper'),
        ('learning_rate', 'Learning Rate', 'log')
    ]

    for idx, (metric_key, title, scale) in enumerate(metrics_to_plot):
        ax = axes[idx // 3, idx % 3]

        for fold, history_list in histories.items():
            # history_list might be a list with one element (from single-fold run)
            history = history_list[0] if isinstance(history_list, list) else history_list

            if metric_key in history and len(history[metric_key]) > 0:
                epochs = range(1, len(history[metric_key]) + 1)
                ax.plot(epochs, history[metric_key], label=f'Fold {fold}', alpha=0.7)

        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric_key)
        ax.set_title(title)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        if scale == 'log':
            ax.set_yscale('log')

    plt.tight_layout()
    output_file = output_dir / f"{prefix}_combined_training_curves.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved combined training curves to: {output_file}")


def plot_metrics_comparison(metrics_df: pd.DataFrame, output_dir: Path, prefix: str):
    """Plot comparison of metrics across folds - matches train_model.py layout."""
    # Filter to numeric fold rows only
    plot_df = metrics_df[metrics_df['Fold'].apply(lambda x: str(x).isdigit())].copy()
    plot_df['Fold'] = plot_df['Fold'].astype(int)

    metrics_cols = ['NDCG@50', 'AUROC', 'AUPRC', 'Precision@50', 'MRR']
    available_metrics = [m for m in metrics_cols if m in plot_df.columns]

    if not available_metrics:
        print("No metrics available for comparison plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Performance Metrics Across Folds', fontsize=16)

    # Bar plot - metrics by fold
    x = np.arange(len(available_metrics))
    width = 0.15
    num_folds = len(plot_df)

    for idx, row in plot_df.iterrows():
        fold_idx = plot_df.index.get_loc(idx)
        offset = (fold_idx - num_folds / 2) * width
        values = [row[col] for col in available_metrics]
        axes[0].bar(x + offset, values, width, label=f'Fold {int(row["Fold"])}', alpha=0.8)

    axes[0].set_xlabel('Metrics')
    axes[0].set_ylabel('Score')
    axes[0].set_title('Metrics by Fold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(available_metrics, rotation=45, ha='right')
    axes[0].legend(loc='best')
    axes[0].grid(True, alpha=0.3, axis='y')
    axes[0].set_ylim([0, 1])

    # Box plot - metrics distribution
    box_data = [plot_df[col].values for col in available_metrics]
    bp = axes[1].boxplot(box_data, tick_labels=available_metrics, patch_artist=True)

    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_alpha(0.7)

    axes[1].set_xlabel('Metrics')
    axes[1].set_ylabel('Score')
    axes[1].set_title('Metrics Distribution')
    axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].set_xticklabels(available_metrics, rotation=45, ha='right')
    axes[1].set_ylim([0, 1])

    plt.tight_layout()
    output_file = output_dir / f"{prefix}_metrics_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved metrics comparison to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Aggregate results from parallel fold training')
    parser.add_argument('--prefix', type=str, required=True,
                        help='Base prefix for fold directories (e.g., GGNet_random_r0.1)')
    parser.add_argument('--num_folds', type=int, default=5,
                        help='Number of folds to aggregate')
    parser.add_argument('--results_dir', type=str, default='model_results',
                        help='Results directory containing fold subdirectories')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for aggregated results (default: model_results/<prefix>_aggregated)')

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir) if args.output_dir else results_dir / f"{args.prefix}_aggregated"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'='*60}")
    print(f"AGGREGATING FOLD RESULTS")
    print(f"{'='*60}")
    print(f"Prefix: {args.prefix}")
    print(f"Folds: 1-{args.num_folds}")
    print(f"Results dir: {results_dir}")
    print(f"Output dir: {output_dir}")
    print(f"{'='*60}\n")

    # Load and aggregate metrics
    print("Loading fold metrics...")
    metrics_df = load_fold_metrics(results_dir, args.prefix, args.num_folds)
    summary_df = compute_summary_statistics(metrics_df)

    # Save combined metrics
    metrics_file = output_dir / f"{args.prefix}_combined_metrics.csv"
    summary_df.to_csv(metrics_file, index=False)
    print(f"\nSaved combined metrics to: {metrics_file}")

    # Print summary
    print(f"\n{'='*60}")
    print("CROSS-VALIDATION SUMMARY")
    print(f"{'='*60}")
    print(summary_df.to_string(index=False))

    # Load and plot training histories
    print(f"\n{'='*60}")
    print("Loading training histories...")
    histories = load_fold_histories(results_dir, args.prefix, args.num_folds)
    if histories:
        plot_combined_training_curves(histories, output_dir, args.prefix)

    # Create metrics comparison plot
    plot_metrics_comparison(metrics_df, output_dir, args.prefix)

    # Load and aggregate gene scores
    print(f"\n{'='*60}")
    print("Aggregating gene scores...")
    score_dfs = load_fold_gene_scores(results_dir, args.prefix, args.num_folds)
    if score_dfs:
        aggregate_gene_scores(score_dfs, output_dir, args.prefix)

    print(f"\n{'='*60}")
    print("AGGREGATION COMPLETE")
    print(f"{'='*60}")
    print(f"Results saved to: {output_dir}")


if __name__ == '__main__':
    main()
