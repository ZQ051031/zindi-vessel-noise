"""Controlled-addition metrics with IID and cluster bootstrap uncertainty."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def rmse(y, pred):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(pred)) ** 2)))


def leaderboard_score(y, pred):
    return 1 - rmse(y, pred)


def bootstrap_ci(y, pred, groups=None, n_boot=300, seed=42):
    rng = np.random.default_rng(seed)
    y, pred = np.asarray(y), np.asarray(pred)
    if y.ndim != 1 or y.shape != pred.shape or not len(y) or not np.isfinite(y).all() or not np.isfinite(pred).all():
        raise ValueError('Bootstrap requires nonempty, matching finite 1D arrays')
    groups = np.arange(len(y)) if groups is None else np.asarray(groups)
    if len(groups) != len(y) or pd.isna(groups).any() or n_boot < 1:
        raise ValueError('Bootstrap requires nonmissing groups aligned with observations and n_boot > 0')
    unique = np.unique(groups)
    blocks = [np.flatnonzero(groups == g) for g in unique]
    scores = []
    for _ in range(n_boot):
        ix = np.concatenate([blocks[i] for i in rng.integers(len(blocks), size=len(blocks))])
        scores.append(leaderboard_score(y[ix], pred[ix]))
    return np.quantile(scores, [.025, .975]).tolist()


def evaluate_frames(sub, labels, seed=42):
    if sub.ID.duplicated().any() or labels.ID.duplicated().any() or set(sub.ID) != set(labels.ID):
        raise ValueError('Submission and labels require matching unique IDs')
    merged = labels.merge(sub, on='ID', suffixes=('_true', '_pred'), validate='one_to_one')
    y, p = merged.share_added_vessel_true.to_numpy(), merged.share_added_vessel_pred.to_numpy()
    presence = merged['presence'] if 'presence' in merged else y > 0
    result = {'rmse': rmse(y, p), 'leaderboard_score': leaderboard_score(y, p),
              'auc': float(roc_auc_score(presence, merged.score_added_vessel)) if len(np.unique(presence)) > 1 else None,
              'iid_ci': bootstrap_ci(y, p, seed=seed), 'by_share_bin': {}}
    for col in ['site', 'day', 'background_window_id']:
        if col in merged:
            result[f'{col}_cluster_ci'] = bootstrap_ci(y, p, merged[col], seed=seed)
    for lo, hi in zip([0, .1, .3, .6], [.1, .3, .6, 1.01]):
        ix = (y >= lo) & (y < hi)
        result['by_share_bin'][f'{lo}-{hi}'] = {'n': int(ix.sum()), 'rmse': rmse(y[ix], p[ix]) if ix.any() else None}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--submission', required=True)
    parser.add_argument('--labels', required=True)
    parser.add_argument('--out')
    args = parser.parse_args()
    result = evaluate_frames(pd.read_csv(args.submission), pd.read_csv(args.labels))
    text = json.dumps(result, indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text)


if __name__ == '__main__':
    main()
