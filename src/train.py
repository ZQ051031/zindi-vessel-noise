"""Train on release backgrounds plus explicitly SYNTHETIC controlled additions. Never reads Test.csv."""
import argparse
import json
from pathlib import Path
import time
import joblib
import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import GroupKFold
from .data import load_release, make_windows
from .synthetic import write_synthetic_release, add_controlled_passages
from .features import window_features
from .models import estimators, MODEL_NAMES
from .evaluate import rmse


def train(release=None, out_dir='models', augment_k=3, seed=42, group_by='site',
          folds=4, n_estimators=180, synthetic_days=.5, zero_gate=False, **loader_options):
    start = time.perf_counter()
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    if augment_k < 1:
        raise ValueError('augment_k must be positive')
    if release is None:
        release = write_synthetic_release(root / 'SYNTHETIC_training_release', days=synthetic_days, seed=seed)
    backgrounds, meta = make_windows(load_release(release, **loader_options))
    if not len(backgrounds):
        raise ValueError('No usable 15-minute windows')
    rng = np.random.default_rng(seed)
    xs, ys, ps, metas = [], [], [], []
    for _ in range(augment_k):
        mix, y, presence, kinds = add_controlled_passages(backgrounds, rng)
        xs.append(window_features(mix))
        ys.append(y)
        ps.append(presence)
        augmented_meta = meta.copy()
        augmented_meta['SYNTHETIC_addition'] = True
        augmented_meta['kind'] = kinds
        metas.append(augmented_meta)
    x, y, presence = pd.concat(xs, ignore_index=True), np.concatenate(ys), np.concatenate(ps)
    metadata = pd.concat(metas, ignore_index=True)
    groups = metadata.site if group_by == 'site' else metadata.t.dt.floor('D').astype(str)
    if groups.nunique() < 2:
        raise ValueError(f'Need at least two distinct {group_by} groups; use more sites/days')
    splits = list(GroupKFold(min(folds, groups.nunique())).split(x, y, groups))
    oof = np.zeros((len(y), 4))
    raw_prob = np.zeros(len(y))
    fold_ids = np.zeros(len(y), dtype=int)
    fold_metrics = []
    for fold, (tr, va) in enumerate(splits):
        ridge, reg, clf = estimators(seed, n_estimators)
        iso = IsotonicRegression(out_of_bounds='clip').fit(x.excess.iloc[tr], y[tr])
        ridge.fit(x.iloc[tr], y[tr])
        reg.fit(x.iloc[tr], y[tr])
        if len(np.unique(presence[tr])) > 1:
            clf.fit(x.iloc[tr], presence[tr])
            raw_prob[va] = clf.predict_proba(x.iloc[va])[:, 1]
        else:
            raw_prob[va] = presence[tr].mean()
        oof[va] = np.column_stack([x.excess.iloc[va], iso.predict(x.excess.iloc[va]),
                                   np.clip(ridge.predict(x.iloc[va]), 0, 1), np.clip(reg.predict(x.iloc[va]), 0, 1)])
        fold_ids[va] = fold
        fold_metrics.append({name: rmse(y[va], oof[va, i]) for i, name in enumerate(MODEL_NAMES)})
    weights = nnls(oof, y)[0]
    weights = weights / weights.sum() if weights.sum() else np.array([0, 0, 0, 1.])
    # Cross-fitted calibration keeps each row out of its calibration fit.
    calibrated = np.empty(len(y))
    for tr, va in splits:
        cal = IsotonicRegression(out_of_bounds='clip').fit(raw_prob[tr], presence[tr])
        calibrated[va] = cal.predict(raw_prob[va])
    ensemble = oof @ weights
    use_gate = bool(zero_gate and rmse(y, ensemble * calibrated) < rmse(y, ensemble))
    if use_gate:
        ensemble *= calibrated
    ridge, reg, clf = estimators(seed, n_estimators)
    ridge.fit(x, y)
    reg.fit(x, y)
    if len(np.unique(presence)) > 1:
        clf.fit(x, presence)
    else:
        clf = None
    bundle = {'ridge': ridge, 'lightgbm': reg, 'classifier': clf,
              'isotonic': IsotonicRegression(out_of_bounds='clip').fit(x.excess, y),
              'calibrator': IsotonicRegression(out_of_bounds='clip').fit(raw_prob, presence),
              'weights': weights, 'zero_gate': use_gate, 'features': list(x), 'seed': seed,
              'fallback_mean': float(x.excess.mean()), 'fallback_std': max(float(x.excess.std()), .01),
              'training_labels': 'SYNTHETIC controlled additions; not natural vessel labels'}
    joblib.dump(bundle, root / 'bundle.joblib')
    metrics = {'oof': {name: {'rmse': rmse(y, oof[:, i]), 'score': 1 - rmse(y, oof[:, i])} for i, name in enumerate(MODEL_NAMES)},
               'fold_rmse': fold_metrics, 'seed': seed, 'features': list(x), 'n_background_windows': len(backgrounds),
               'n_training_windows': len(y), 'group_by': group_by, 'weights': weights.tolist(), 'zero_gate': use_gate,
               'training_seconds': time.perf_counter() - start, 'model_bytes': (root / 'bundle.joblib').stat().st_size,
               'validation_note': 'OOF base models; ensemble weights selected on OOF (selection estimate, not nested validation)',
               'SYNTHETIC_labels': True}
    metrics['oof']['ensemble'] = {'rmse': rmse(y, ensemble), 'score': 1 - rmse(y, ensemble)}
    for f, (_, va) in enumerate(splits):
        fold_metrics[f]['ensemble'] = rmse(y[va], ensemble[va])
    result = metadata.assign(share_added_vessel=y, presence=presence, fold=fold_ids, ensemble=ensemble, score_added_vessel=calibrated)
    for i, name in enumerate(MODEL_NAMES):
        result[name] = oof[:, i]
    result.to_parquet(root / 'SYNTHETIC_addition_oof.parquet', index=False)
    (root / 'metrics.json').write_text(json.dumps(metrics, indent=2))
    print(json.dumps({k: v for k, v in metrics.items() if k not in ['features']}, indent=2))
    return bundle, metrics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--release')
    p.add_argument('--out-dir', default='models')
    p.add_argument('--augment-k', type=int, default=3)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--folds', type=int, default=4)
    p.add_argument('--n-estimators', type=int, default=180)
    p.add_argument('--synthetic-days', type=float, default=.5)
    p.add_argument('--group-by', choices=['site', 'period'], default='site')
    p.add_argument('--zero-gate', action='store_true')
    for col in ['site', 'time', 'dur']:
        p.add_argument(f'--{col}-col')
    p.add_argument('--drop-bad-flags', action='store_true', help='Inspect DATA_DICTIONARY.md first: 0/false/good/ok/pass are treated as good')
    p.add_argument('--flag-cols', nargs='+')
    p.add_argument('--duration-tolerance', type=float, default=5)
    train(**vars(p.parse_args()))


if __name__ == '__main__':
    main()
