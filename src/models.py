"""Deterministic estimators and inference-only bundle operations."""
import numpy as np
from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from scipy.special import expit
from .features import window_features

MODEL_NAMES = ['baseline', 'isotonic', 'ridge', 'lightgbm']


def estimators(seed=42, n_estimators=180):
    params = dict(n_estimators=n_estimators, learning_rate=.04, num_leaves=15,
                  min_child_samples=12, reg_lambda=2., verbosity=-1, n_jobs=1,
                  deterministic=True, force_row_wise=True, random_state=seed,
                  seed=seed, bagging_seed=seed, feature_fraction_seed=seed)
    return (make_pipeline(SimpleImputer(strategy='median', add_indicator=True, keep_empty_features=True),
                          StandardScaler(), Ridge(alpha=100)),
            LGBMRegressor(objective='regression', **params),
            LGBMClassifier(objective='binary', **params))


def predict_features(bundle, x):
    base = x.excess.to_numpy()
    parts = np.column_stack([base, bundle['isotonic'].predict(base),
                             np.clip(bundle['ridge'].predict(x), 0, 1),
                             np.clip(bundle['lightgbm'].predict(x), 0, 1)])
    if bundle['classifier'] is None:
        score = expit((base - bundle['fallback_mean']) / bundle['fallback_std'])
    else:
        score = bundle['calibrator'].predict(bundle['classifier'].predict_proba(x)[:, 1])
    share = parts @ bundle['weights']
    if bundle['zero_gate']:
        share *= score
    return np.clip(share, 0, 1), np.clip(score, 0, 1)


def predict_cube(bundle, cube):
    return predict_features(bundle, window_features(cube)[bundle['features']])
