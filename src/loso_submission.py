"""Refit a qualified LOSO pipeline on every official-release training example."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from .features import window_features
from .loso_features import context_features, fit_station_profiles
from .loso_folds import build_feature_rows, candidate_predict, fit_candidate
from .models import predict_features

PIPELINE = 'enriched_loso'


def fit_winner(validation, baseline_model, out):
    decision = json.loads((validation / 'decision.json').read_text())
    winner = decision['winner']
    if winner is None:
        raise ValueError('No pipeline qualified in LOSO validation')
    weights = decision['ensemble_members'] if winner == 'best_two_ensemble' else {winner: 1.}
    weights = {name: weight for name, weight in weights.items() if weight > 0}
    oof = pd.read_parquet(validation / 'candidate_oof.parquet')
    bundle = {'pipeline': PIPELINE, 'weights': weights, 'models': {}, 'calibrators': {}}
    candidates = [name for name in weights if name != 'current_ensemble']
    if candidates:
        background_profiles = np.load(validation / 'background_profiles.npy', mmap_mode='r')
        basis = fit_station_profiles(background_profiles, oof.site.to_numpy()[:len(background_profiles)])
        enriched = np.load(validation / 'enriched_features.npy', mmap_mode='r')
        profiles = np.load(validation / 'mixture_profiles.npy', mmap_mode='r')
        features = build_feature_rows(enriched, profiles, basis, oof.site.to_numpy())
        bundle['station_basis'] = basis
        for name in candidates:
            print(f'Fitting {name} on {len(oof)} training examples.', flush=True)
            bundle['models'][name] = fit_candidate(name, features, oof.share_added_vessel, oof.presence)
            bundle['calibrators'][name] = IsotonicRegression(out_of_bounds='clip').fit(
                oof[f'{name}_raw_prob'], oof.presence)
    if 'current_ensemble' in weights:
        baseline = joblib.load(baseline_model)
        metrics = json.loads((validation / 'baseline_metrics.json').read_text())
        baseline['weights'] = np.array(metrics['loso']['weights'])
        baseline['calibrator'] = IsotonicRegression(out_of_bounds='clip').fit(oof.raw_prob, oof.presence)
        bundle['models']['current_ensemble'] = baseline
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out)


def predict_enriched(bundle, cube):
    legacy = window_features(cube)
    features = None
    if 'station_basis' in bundle:
        extra, profiles = context_features(cube)
        features = build_feature_rows(
            np.column_stack((legacy.to_numpy(dtype=np.float32), extra)), profiles, bundle['station_basis'])
    share = np.zeros(len(cube))
    probability = np.zeros(len(cube))
    for name, weight in bundle['weights'].items():
        estimator = bundle['models'][name]
        if name == 'current_ensemble':
            y, p = predict_features(estimator, legacy[estimator['features']])
        else:
            y, raw = candidate_predict(name, estimator, features)
            p = bundle['calibrators'][name].predict(raw)
        share += weight * y
        probability += weight * p
    return np.clip(share, 0, 1), np.clip(probability, 0, 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validation', type=Path, required=True)
    parser.add_argument('--baseline-model', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    fit_winner(**vars(parser.parse_args()))


if __name__ == '__main__':
    main()
