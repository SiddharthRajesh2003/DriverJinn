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
import seaborn as sns


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

        with open(history_file, 'rb') as f:
            history = pickle.load(f)
        all_histories[fold] = history
        print(f"Loaded training history for fold {fold}")

    return all_histories


def load_fold_gene_scores(results_dir: Path, prefix: str, num_folds: int) -> List[pd.DataFrame]:
    """Load gene scores from each fold."""
    all_scores = []

    for fold in range(1, num_folds + 1):
        fold_dir = results_dir / f"{prefix}_fold{fold}"
        scores_file = fold_dir / f"{prefix}_fold{fold}_gene_scores.csv"

        if not scores_file.exists():
            # Try alternative naming
            scores_file = fold_dir / f"{prefix}_fold{fold}_fold_{fold}_gene_scores.csv"

        if not scores_file.exists():
            print(f"Warning: gene scores not found for fold {fold}, skipping")
            continue

        df = pd.read_csv(scores_file)
        df['source_fold'] = fold
        all_scores.append(df)
        print(f"Loaded gene scores for fold {fold}: {len(df)} genes")

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


def aggregate_gene_scores(score_dfs: List[pd.DataFrame], output_dir: Path, prefix: str):
    """Aggregate gene scores across folds using rank aggregation."""
    if not score_dfs:
        print("No gene scores to aggregate")
        return

    # Combine all scores
    combined = pd.concat(score_dfs, ignore_index=True)

    # Get gene identifier column
    gene_col = 'gene' if 'gene' in combined.columns else combined.columns[0]
    score_col = 'score' if 'score' in combined.columns else 'mean_score'

    if score_col not in combined.columns:
        # Try to find a score column
        score_candidates = [c for c in combined.columns if 'score' in c.lower()]
        if score_candidates:
            score_col = score_candidates[0]
        else:
            print(f"Could not find score column. Available: {combined.columns.tolist()}")
            return

    # Aggregate by gene
    aggregated = combined.groupby(gene_col).agg({
        score_col: ['mean', 'std', 'count'],
    }).reset_index()

    aggregated.columns = [gene_col, 'mean_score', 'std_score', 'fold_count']
    aggregated = aggregated.sort_values('mean_score', ascending=False)

    # Add rank
    aggregated['rank'] = range(1, len(aggregated) + 1)

    # Save
    output_file = output_dir / f"{prefix}_aggregated_gene_scores.csv"
    aggregated.to_csv(output_file, index=False)
    print(f"Saved aggregated gene scores to: {output_file}")

    # Print top genes
    print(f"\nTop 20 predicted driver genes (aggregated across folds):")
    print(aggregated.head(20).to_string(index=False))

    return aggregated


def plot_combined_training_curves(histories: Dict, output_dir: Path, prefix: str):
    """Plot training curves from all folds."""
    if not histories:
        print("No training histories to plot")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    metrics_to_plot = [
        ('train_loss', 'val_loss', 'Loss'),
        ('train_auroc', 'val_auroc', 'AUROC'),
        ('train_ndcg', 'val_ndcg', 'NDCG@50'),
        ('train_auprc', 'val_auprc', 'AUPRC')
    ]

    colors = plt.cm.tab10(np.linspace(0, 1, len(histories)))

    for ax, (train_key, val_key, title) in zip(axes.flat, metrics_to_plot):
        for (fold, history_list), color in zip(histories.items(), colors):
            # history_list might be a list with one element (from single-fold run)
            history = history_list[0] if isinstance(history_list, list) else history_list

            if train_key in history:
                epochs = range(1, len(history[train_key]) + 1)
                ax.plot(epochs, history[train_key], '--', color=color, alpha=0.5, label=f'Fold {fold} Train')
            if val_key in history:
                epochs = range(1, len(history[val_key]) + 1)
                ax.plot(epochs, history[val_key], '-', color=color, label=f'Fold {fold} Val')

        ax.set_xlabel('Epoch')
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = output_dir / f"{prefix}_combined_training_curves.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved combined training curves to: {output_file}")


def plot_metrics_boxplot(metrics_df: pd.DataFrame, output_dir: Path, prefix: str):
    """Create boxplot of metrics across folds."""
    # Filter to numeric fold rows only
    plot_df = metrics_df[metrics_df['Fold'].apply(lambda x: str(x).isdigit())]

    metrics = ['AUROC', 'AUPRC', 'NDCG@50', 'Precision@50']
    available_metrics = [m for m in metrics if m in plot_df.columns]

    if not available_metrics:
        print("No metrics available for boxplot")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    plot_data = plot_df[available_metrics].melt(var_name='Metric', value_name='Value')
    sns.boxplot(data=plot_data, x='Metric', y='Value', ax=ax)
    sns.stripplot(data=plot_data, x='Metric', y='Value', ax=ax, color='black', alpha=0.5)

    ax.set_title(f'Cross-Validation Metrics Distribution ({len(plot_df)} folds)')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    output_file = output_dir / f"{prefix}_metrics_boxplot.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved metrics boxplot to: {output_file}")


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

    # Create metrics boxplot
    plot_metrics_boxplot(metrics_df, output_dir, args.prefix)

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
