"""Reproduce historical grouped CV and run true LOSO on the official-release feature cache."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut

from .evaluate import rmse
from .models import MODEL_NAMES
from .train import calibrate_oof, ensemble_weights
from .loso_folds import CANDIDATE_NAMES, baseline_fold, candidate_fold


def score_predictions(labels, prediction, presence=None, probability=None):
    error = rmse(labels, prediction)
    result = {'rmse': error, 'score': 1 - error}
    if probability is not None:
        result['auc'] = float(roc_auc_score(presence, probability))
    return result


def run_baselines(cache, out_dir, jobs):
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata = pd.read_parquet(cache / 'SYNTHETIC_addition_oof.parquet')
    labels = metadata.share_added_vessel.to_numpy()
    presence = metadata.presence.to_numpy()
    sites = metadata.site.to_numpy()
    grouped = list(GroupKFold(4).split(labels, labels, sites))
    loso = list(LeaveOneGroupOut().split(labels, labels, sites))
    for fold_id, (_, valid) in enumerate(grouped):
        if not metadata.fold.iloc[valid].eq(fold_id).all():
            raise ValueError('The cached historical fold assignments do not match GroupKFold(4)')

    def held_out(valid):
        return tuple(sorted(set(sites[valid])))

    unique_folds = {held_out(valid): (train, valid) for train, valid in grouped + loso}
    feature_path = cache / 'SYNTHETIC_addition_features.parquet'
    print(f'Fitting {len(unique_folds)} unique folds for grouped CV and LOSO with {jobs} workers.', flush=True)
    predictions = Parallel(n_jobs=jobs, return_as='generator', max_nbytes=None)(
        delayed(baseline_fold)(feature_path, labels, presence, train, valid)
        for train, valid in unique_folds.values()
    )
    folds = {}
    for key, prediction in zip(unique_folds, predictions):
        folds[key] = prediction
        print('Finished held-out stations', ', '.join(key), flush=True)

    results = {}
    for name, split in [('grouped4', grouped), ('loso', loso)]:
        share = np.empty((len(labels), len(MODEL_NAMES)))
        raw_prob = np.empty(len(labels))
        fold_ids = np.empty(len(labels), dtype=int)
        for fold_id, (_, valid) in enumerate(split):
            share[valid], raw_prob[valid] = folds[held_out(valid)]
            fold_ids[valid] = fold_id
        prob = calibrate_oof(raw_prob, presence, split)
        weights = ensemble_weights(share, labels)
        ensemble = share @ weights
        results[name] = {
            'folds': [held_out(valid) for _, valid in split],
            'weights': weights.tolist(),
            'models': {model: score_predictions(labels, share[:, i]) for i, model in enumerate(MODEL_NAMES)},
        }
        results[name]['models']['ensemble'] = score_predictions(labels, ensemble, presence, prob)
        oof = metadata[['site', 't', 'background_window_id', 'kind', 'share_added_vessel', 'presence']].copy()
        oof['fold'] = fold_ids
        for model, pred in zip(MODEL_NAMES, share.T):
            oof[model] = pred
        oof['ensemble'] = ensemble
        oof['raw_prob'] = raw_prob
        oof['score_added_vessel'] = prob
        oof.to_parquet(out_dir / f'{name}_baseline_oof.parquet', index=False)
    results['n_background_windows'] = int(metadata.background_window_id.nunique())
    results['n_training_examples'] = len(metadata)
    (out_dir / 'baseline_metrics.json').write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2), flush=True)


def run_candidates(out_dir, jobs):
    baseline = pd.read_parquet(out_dir / 'loso_baseline_oof.parquet')
    labels = baseline.share_added_vessel.to_numpy()
    presence = baseline.presence.to_numpy()
    sites = baseline.site.to_numpy()
    loso = list(LeaveOneGroupOut().split(labels, labels, sites))
    for fold_id, (_, valid) in enumerate(loso):
        if not baseline.fold.iloc[valid].eq(fold_id).all():
            raise ValueError('Candidate folds must match the baseline LOSO split')
    checkpoint_dir = out_dir / 'candidate_folds'
    checkpoint_dir.mkdir(exist_ok=True)
    saved = {i: checkpoint_dir / f'fold_{i}.npz' for i in range(len(loso))}
    pending = [i for i, path in saved.items() if not path.exists()]
    print(f'Running two candidates over the same {len(loso)} splits ({len(pending)} pending).', flush=True)
    fold_results = Parallel(n_jobs=jobs, return_as='generator_unordered', max_nbytes=None)(
        delayed(candidate_fold)(out_dir, sites, labels, presence, fold_id, train, valid)
        for fold_id, (train, valid) in enumerate(loso)
        if fold_id in pending
    )
    for fold_id, models in fold_results:
        _, valid = loso[fold_id]
        tmp = saved[fold_id].with_suffix('.tmp.npz')
        np.savez(tmp, indices=valid, **{name: models[name] for name in CANDIDATE_NAMES})
        tmp.replace(saved[fold_id])
        print(f'Finished {sites[valid[0]]} with {len(valid)} examples.', flush=True)

    share = {name: np.empty(len(labels)) for name in CANDIDATE_NAMES}
    raw_prob = {name: np.empty(len(labels)) for name in CANDIDATE_NAMES}
    for fold_id, path in saved.items():
        _, valid = loso[fold_id]
        with np.load(path) as models:
            np.testing.assert_array_equal(models['indices'], valid)
            for name in CANDIDATE_NAMES:
                share[name][valid], raw_prob[name][valid] = models[name]

    results = {}
    for name in CANDIDATE_NAMES:
        probability = calibrate_oof(raw_prob[name], presence, loso)
        results[name] = score_predictions(labels, share[name], presence, probability)
        results[name]['uncalibrated_auc'] = float(roc_auc_score(presence, raw_prob[name]))
        baseline[name] = share[name]
        baseline[f'{name}_raw_prob'] = raw_prob[name]
        baseline[f'{name}_calibrated_prob'] = probability
    baseline.to_parquet(out_dir / 'candidate_oof.parquet', index=False)
    (out_dir / 'candidate_metrics.json').write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2), flush=True)


def blend_weight(labels, first, second):
    delta = first - second
    norm = delta @ delta
    return float(np.clip(delta @ (labels - second) / norm, 0, 1)) if norm else .5


def compare_candidates(out_dir):
    oof = pd.read_parquet(out_dir / 'candidate_oof.parquet')
    labels = oof.share_added_vessel.to_numpy()
    presence = oof.presence.to_numpy()
    predictions = {'current_ensemble': (oof.ensemble.to_numpy(), oof.score_added_vessel.to_numpy())}
    for name in CANDIDATE_NAMES:
        predictions[name] = (oof[name].to_numpy(), oof[f'{name}_calibrated_prob'].to_numpy())

    metrics = {name: score_predictions(labels, share, presence, probability)
               for name, (share, probability) in predictions.items()}
    first, second = sorted(metrics, key=lambda name: metrics[name]['rmse'])[:2]
    y_first, p_first = predictions[first]
    y_second, p_second = predictions[second]
    stations = oof.groupby('site').indices
    row_weights = np.empty(len(oof))
    fold_weights = {}
    for site, valid in stations.items():
        train = oof.site.ne(site).to_numpy()
        fold_weights[site] = blend_weight(labels[train], y_first[train], y_second[train])
        row_weights[valid] = fold_weights[site]
    weight = blend_weight(labels, y_first, y_second)
    predictions['best_two_ensemble'] = (
        row_weights * y_first + (1 - row_weights) * y_second,
        row_weights * p_first + (1 - row_weights) * p_second,
    )
    blended_share, blended_probability = predictions['best_two_ensemble']
    metrics['best_two_ensemble'] = score_predictions(labels, blended_share, presence, blended_probability)
    eligible = [name for name in [*CANDIDATE_NAMES, 'best_two_ensemble']
                if metrics[name]['score'] >= .8987 and metrics[name]['auc'] >= .87964]
    winner = max(eligible, key=lambda name: metrics[name]['score'], default=None)
    model_rows = []
    for name, (share, probability) in predictions.items():
        model_rows.append({'model': name, 'held_out_site': 'ALL', 'examples': len(oof), **metrics[name]})
        for site, indices in stations.items():
            row = score_predictions(labels[indices], share[indices], presence[indices], probability[indices])
            model_rows.append({'model': name, 'held_out_site': site, 'examples': len(indices), **row})
    pd.DataFrame(model_rows).to_csv(out_dir / 'model_holdout_scores.csv', index=False)
    result = {'models': metrics, 'ensemble_members': {first: weight, second: 1 - weight},
              'held_out_site_weight_for_first_member': fold_weights, 'winner': winner}
    (out_dir / 'decision.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--jobs', type=int, default=1)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--candidates', action='store_true')
    mode.add_argument('--compare', action='store_true')
    args = parser.parse_args()
    if args.candidates:
        run_candidates(args.out_dir, args.jobs)
    elif args.compare:
        compare_candidates(args.out_dir)
    elif args.cache is None:
        parser.error('--cache is required to run the baselines')
    else:
        run_baselines(args.cache, args.out_dir, args.jobs)


if __name__ == '__main__':
    main()
