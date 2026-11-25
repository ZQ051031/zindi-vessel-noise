"""Independent workers for station-held-out fitting."""
from functools import lru_cache
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .loso_features import fit_station_profiles, station_embedding
from .train import fit_validation_fold

CANDIDATE_NAMES = ('enriched_boosting', 'enriched_mlp')
PRESENCE_TARGET_SCALE = 0.35


@lru_cache(maxsize=1)
def load_features(path):
    return pd.read_parquet(path, use_threads=False)


def baseline_fold(path, labels, presence, train_idx, val_idx):
    features = load_features(path)
    return fit_validation_fold(
        features.iloc[train_idx], labels[train_idx], presence[train_idx], features.iloc[val_idx], 42, 180)


def build_feature_rows(features, profiles, basis, row_sites=None):
    return np.column_stack((features, station_embedding(profiles, basis, row_sites)))


def fold_feature_rows(path, sites, train_idx, val_idx):
    features = np.load(path / 'enriched_features.npy', mmap_mode='r')
    profiles = np.load(path / 'mixture_profiles.npy', mmap_mode='r')
    background_profiles = np.load(path / 'background_profiles.npy', mmap_mode='r')
    bg_sites = sites[:len(background_profiles)]
    training_backgrounds = np.isin(bg_sites, np.unique(sites[train_idx]))
    basis = fit_station_profiles(background_profiles[training_backgrounds], bg_sites[training_backgrounds])
    train = build_feature_rows(features[train_idx], profiles[train_idx], basis, sites[train_idx])
    valid = build_feature_rows(features[val_idx], profiles[val_idx], basis)
    return train, valid


def fit_boosting(features, labels, presence):
    params = {
        'num_threads': 1, 'num_leaves': 63, 'learning_rate': .05, 'max_bin': 63,
        'min_data_in_leaf': 28, 'feature_fraction': .68, 'bagging_fraction': .86,
        'bagging_freq': 1, 'lambda_l2': 4., 'verbosity': -1, 'seed': 42,
        'deterministic': True, 'force_col_wise': True,
    }
    data = lgb.Dataset(features, label=labels)
    regressor = lgb.train(dict(params, objective='regression'), data, num_boost_round=300)
    data.set_label(presence)
    classifier = lgb.train(dict(params, objective='binary'), data, num_boost_round=300)
    return regressor, classifier


def fit_mlp(features, labels, presence):
    scale = make_pipeline(SimpleImputer(keep_empty_features=True), StandardScaler(copy=False))
    standardized = scale.fit_transform(features)
    np.clip(standardized, -8, 8, out=standardized)
    model = MLPRegressor(hidden_layer_sizes=(96, 48), alpha=.003, learning_rate_init=.0018,
                         batch_size=1024, max_iter=36, tol=1e-5, n_iter_no_change=7, random_state=42)
    targets = np.column_stack((labels, presence * PRESENCE_TARGET_SCALE)).astype(np.float32)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=ConvergenceWarning, message='Stochastic Optimizer:')
        model.fit(standardized, targets)
    return scale, model


def fit_candidate(name, features, labels, presence):
    if name == 'enriched_boosting':
        return fit_boosting(features, labels, presence)
    if name == 'enriched_mlp':
        return fit_mlp(features, labels, presence)
    raise ValueError(f'Unknown candidate model {name}')


def candidate_predict(name, estimator, features):
    if name == 'enriched_boosting':
        regressor, classifier = estimator
        share = regressor.predict(features, num_threads=1)
        probability = classifier.predict(features, num_threads=1)
    elif name == 'enriched_mlp':
        scale, model = estimator
        prediction = model.predict(np.clip(scale.transform(features), -8, 8))
        share, probability = prediction[:, 0], prediction[:, 1] / PRESENCE_TARGET_SCALE
    else:
        raise ValueError(f'Unknown candidate model {name}')
    return np.clip(share, 0, 1), np.clip(probability, 0, 1)


def candidate_fold(path, sites, labels, presence, fold_id, train_idx, val_idx):
    x_train, x_valid = fold_feature_rows(path, sites, train_idx, val_idx)
    result = {}
    for name in CANDIDATE_NAMES:
        estimator = fit_candidate(name, x_train, labels[train_idx], presence[train_idx])
        result[name] = candidate_predict(name, estimator, x_valid)
    return fold_id, result
