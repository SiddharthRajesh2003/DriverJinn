"""
Checks for gene scoring, cross-fold aggregation and HGDC split loading.

    python -m model.test_ranking_pipeline
"""
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from model.DriverGenePredictor import ContrastiveDriverGenePredictor
from model.aggregate_fold_results import aggregate_gene_scores
from model.train_model import load_hgdc_splits

HGDC_GGNET = Path("baseline_comparison/HGDC/data/GGNet/dataset_GGNet_ten_5CV.pkl")


def scored_fold(scores, labels, names):
    """Mimic score_all_genes: one row per gene, sorted by score (row position = rank - 1)."""
    df = pd.DataFrame({'original_index': np.arange(len(scores)), 'gene_id': np.arange(len(scores)),
                       'gene_name': names, 'driver_score': scores, 'true_label': labels})
    df = df.sort_values('driver_score', ascending=False).reset_index(drop=True)
    df['rank'] = df.index + 1
    df['percentile'] = df['rank'] / len(df) * 100
    return df


def test_driver_likeness_is_aligned_to_genes():
    scores = np.array([0.1, 0.9, 0.5, 0.8, 0.3, 0.7])
    labels = np.array([1, 0, 1, 0, 0, 1], dtype=float)     # drivers score 0.1, 0.5, 0.7
    names = np.array(list("ABCDEF"))
    labeled = np.array([1, 0, 1, 1, 0, 1], dtype=bool)
    fake = SimpleNamespace(device=torch.device('cpu'), score_all_genes=lambda **kw: scored_fold(scores, labels, names))
    df = ContrastiveDriverGenePredictor.score_genes_with_statistics(
        fake, data=None, labels=torch.tensor(labels), node_names=names, labeled_mask=torch.tensor(labeled))
    by_gene = df.set_index('gene_name')
    # Share of the 3 driver scores >= each gene's score, (+1)/(n+1)
    assert np.isclose(by_gene.at['B', 'driver_likeness_p'], (0 + 1) / 4)   # 0.9 beats every driver
    assert np.isclose(by_gene.at['E', 'driver_likeness_p'], (2 + 1) / 4)   # 0.3: drivers 0.5, 0.7 score higher
    assert by_gene.loc[['A', 'C', 'F'], 'driver_likeness_p'].isna().all()   # drivers are not tested
    assert by_gene['is_labeled'].to_dict() == dict(zip(names, labeled))    # looked up by gene, not row position
    assert 'adjusted_pvalue' not in df and 'is_significant' not in df


def test_aggregation_aligns_folds_by_gene():
    rng = np.random.default_rng(0)
    n = 300
    names = np.array([f"G{i}" for i in range(n)])
    labels = np.zeros(n)
    labels[:20] = 1
    base = rng.normal(size=n)
    folds = [scored_fold(base + rng.normal(scale=0.3, size=n), labels, names) for _ in range(5)]
    folds[1]['driver_score'] *= 1000                                       # uncalibrated fold must not dominate
    for df in folds:
        df['is_labeled'] = df['true_label'] == 1
    with tempfile.TemporaryDirectory() as tmp:
        agg = aggregate_gene_scores(folds, Path(tmp), "t").set_index('gene_name')
        assert (Path(tmp) / "t_aggregated_consensus_top100.csv").exists()
    ranks = pd.concat([df.set_index('gene_name')['rank'] for df in folds], axis=1)
    assert np.allclose(agg.loc[names, 'mean_rank'], ranks.loc[names].mean(axis=1))
    # folds_top100: rank among unlabelled genes only
    unl = [df[~df['is_labeled']].reset_index(drop=True) for df in folds]
    top = pd.concat([pd.Series(d.index < 100, index=d['gene_name']) for d in unl], axis=1).sum(axis=1)
    assert (agg.loc[top.index, 'folds_top100'] == top).all()
    assert not agg.loc[agg['is_labeled'], 'consensus_top100'].any()
    assert (agg['consensus_top100'] == (~agg['is_labeled'] & (agg['folds_top100'] >= 3))).all()
    assert agg.sort_values('aggregate_rank')['mean_rank'].is_monotonic_increasing


def test_hgdc_splits():
    if not HGDC_GGNET.exists():
        print(f"  skipped: {HGDC_GGNET} not found")
        return
    hgdc = pd.read_pickle(HGDC_GGNET)
    labels = np.asarray(hgdc['label']).ravel()
    folds = load_hgdc_splits(HGDC_GGNET, 0, hgdc['node_name'], labels)
    assert len(folds) == 5
    labelled = folds[0]['train_mask'] | folds[0]['val_mask'] | folds[0]['test_mask']
    tests = np.zeros(len(labels), dtype=int)
    for f in folds:
        tr, va, te = f['train_mask'], f['val_mask'], f['test_mask']
        assert not (tr & va).any() and not (tr & te).any() and not (va & te).any()
        assert ((tr | va | te) == labelled).all()                           # unlabelled genes are in no mask
        assert abs(va.sum() / (tr | va).sum() - 0.15) < 0.01
        assert abs(labels[va].mean() - labels[tr].mean()) < 0.02           # stratified
        tests += te
    assert (tests[labelled] == 1).all() and (tests[~labelled] == 0).all()  # each labelled gene tested exactly once
    try:
        load_hgdc_splits(HGDC_GGNET, 0, np.asarray(hgdc['node_name'])[::-1], labels)
        raise AssertionError("gene-order mismatch was not caught")
    except ValueError:
        pass


if __name__ == '__main__':
    for test in (test_driver_likeness_is_aligned_to_genes, test_aggregation_aligns_folds_by_gene, test_hgdc_splits):
        test()
        print(f"PASS {test.__name__}")
