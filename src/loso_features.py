"""Fixed window features and fold-fitted acoustic station profiles."""
import argparse
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from .features import GROUPS


def ratio(numerator, denominator):
    a, b = np.broadcast_arrays(numerator, denominator)
    return np.divide(a, b, out=np.full(a.shape, np.nan), where=b > 0)


def median(values, axis):
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='All-NaN slice encountered', category=RuntimeWarning)
        return np.nanmedian(values, axis=axis)


def log_power(power):
    return 10 * np.log10(np.where(power > 0, power, np.nan))


def power_and_floor(db):
    observed = np.isfinite(db)
    peak = np.max(np.where(observed, db, -np.inf), axis=(1, 2))
    peak = np.where(np.isfinite(peak), peak, 0)
    power = np.where(observed, 10 ** ((db - peak[:, None, None]) / 10), 0)
    quiet = np.sort(np.where(observed, power, np.inf), axis=1)[:, :3]
    valid_quiet = np.isfinite(quiet)
    floor = ratio(np.where(valid_quiet, quiet, 0).sum(axis=1), valid_quiet.sum(axis=1))
    return observed, peak, power, floor


def quiet_profile(floor, peak):
    quiet_db = log_power(floor) + peak[:, None]
    anchor = median(quiet_db, axis=1)
    return np.column_stack((anchor, quiet_db - anchor[:, None])).astype(np.float32)


def temporal_context(power, excess, observed):
    energy = power.sum(axis=1)
    transient = excess.sum(axis=1)
    weights = np.nan_to_num(ratio(excess, transient[:, None]), nan=0)
    minutes = np.arange(15)
    centre = weights @ minutes
    spread = np.sqrt(np.maximum(0, weights @ (minutes ** 2) - centre ** 2))
    peak = np.argmax(excess, axis=1)
    at_peak = (np.abs(minutes - peak[:, None]) <= 2)
    before_peak = minutes < peak[:, None]
    both_observed = observed[:, :-1] & observed[:, 1:]
    step_energy = np.where(both_observed, np.abs(np.diff(power, axis=1)), 0).sum(axis=1)

    segments = [slice(0, 5), slice(5, 10), slice(10, 15)]
    return np.column_stack((
        ratio(transient, energy),
        ratio(excess.max(axis=1), transient),
        ratio(np.sum(excess * at_peak, axis=1), transient),
        ratio(np.sum(excess * before_peak, axis=1), transient),
        np.where(transient > 0, centre / 14, np.nan),
        np.where(transient > 0, spread / 14, np.nan),
        ratio(step_energy, energy),
        *(ratio(power[:, s].sum(axis=1), energy) for s in segments),
        *(ratio(excess[:, s].sum(axis=1), transient) for s in segments),
    ))


def context_features(db):
    """Band ratios and temporal energy patterns computed independently per window."""
    observed, peak, power, floor = power_and_floor(np.asarray(db, dtype=np.float64))
    excess = np.where(observed, np.maximum(power - floor[:, None, :], 0), 0)
    band_energy = power.sum(axis=1)
    band_excess = excess.sum(axis=1)
    total = band_energy.sum(axis=1)[:, None]
    band_avg_db = log_power(ratio(band_energy, observed.sum(axis=1)))
    floor_db = log_power(floor)
    adjacent = np.diff(band_avg_db, axis=1)
    curvature = np.diff(floor_db, n=2, axis=1)
    lag_four = band_avg_db[:, 4:] - band_avg_db[:, :-4]
    band_variation = band_avg_db - floor_db
    band_excess_frac = ratio(band_excess, total)
    relative_peaks = ratio(excess.max(axis=1), band_excess)
    spectrum = [adjacent, curvature, lag_four, band_variation, band_excess_frac, relative_peaks]
    temporal = [
        temporal_context(power[:, :, g].sum(axis=2),
                         excess[:, :, g].sum(axis=2), observed[:, :, g].any(axis=2))
        for g in [*GROUPS, slice(None)]
    ]
    return np.column_stack([*spectrum, *temporal]).astype(np.float32), quiet_profile(floor, peak)


def fit_station_profiles(profiles, sites):
    """Fit a low-dimensional basis using only background windows from the training sites."""
    names = np.unique(sites)
    if len(names) < 2:
        raise ValueError('Station profiles require at least two training sites')
    prototypes = np.vstack([median(profiles[sites == site], axis=0) for site in names])
    centre = np.nan_to_num(median(prototypes, axis=0), nan=0)
    prototypes = np.where(np.isfinite(prototypes), prototypes, centre)
    scale = np.maximum(prototypes.std(axis=0), 3.)
    normalized = (prototypes - centre) / scale
    _, _, directions = np.linalg.svd(normalized, full_matrices=False)
    return names, centre, scale, normalized, directions[:4].T


def station_embedding(profiles, basis, sites=None):
    """Soft acoustic embedding. Training rows exclude their own station prototype."""
    names, centre, scale, prototypes, components = basis
    observed = np.isfinite(profiles)
    z = np.where(observed, (profiles - centre) / scale, 0)
    distances = np.column_stack([
        np.sum(np.where(observed, (z - prototype) ** 2, 0), axis=1)
        / np.maximum(1, observed.sum(axis=1))
        for prototype in prototypes
    ])
    if sites is not None:
        distances[np.asarray(sites)[:, None] == names] = np.inf
    nearest = distances.min(axis=1, keepdims=True)
    weights = np.exp(-(distances - nearest) / .6)
    weights /= weights.sum(axis=1, keepdims=True)
    embedding = weights @ prototypes @ components
    return np.column_stack((embedding, z @ components - embedding,
                            np.sqrt(nearest), weights.max(axis=1))).astype(np.float32)


def prepare_features(legacy_path, out_dir, chunk_size=4096):
    """Materialize fold-independent features once while keeping the released inputs local."""
    out_dir = Path(out_dir)
    legacy = pd.read_parquet(legacy_path).to_numpy(dtype=np.float32)
    mixtures = np.load(out_dir / 'mixtures.npy', mmap_mode='r')
    backgrounds = np.load(out_dir / 'backgrounds.npy', mmap_mode='r')
    if len(mixtures) != len(legacy):
        raise ValueError('Replayed mixtures do not align with legacy features')
    example, profile = context_features(mixtures[:1])
    features = np.lib.format.open_memmap(
        out_dir / 'enriched_features.npy', mode='w+', dtype='float32',
        shape=(len(mixtures), legacy.shape[1] + example.shape[1]))
    profiles = np.lib.format.open_memmap(
        out_dir / 'mixture_profiles.npy', mode='w+', dtype='float32', shape=(len(mixtures), profile.shape[1]))
    for start in range(0, len(mixtures), chunk_size):
        rows = slice(start, start + chunk_size)
        extra, profiles[rows] = context_features(mixtures[rows])
        features[rows, :legacy.shape[1]] = legacy[rows]
        features[rows, legacy.shape[1]:] = extra
    features.flush()
    profiles.flush()
    background_profiles = []
    for start in range(0, len(backgrounds), chunk_size):
        rows = slice(start, start + chunk_size)
        _, peak, _, floor = power_and_floor(backgrounds[rows].astype(np.float64))
        background_profiles.append(quiet_profile(floor, peak))
    np.save(out_dir / 'background_profiles.npy', np.concatenate(background_profiles))
    return features.shape


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--legacy-path', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args()
    print(prepare_features(args.legacy_path, args.out_dir))


if __name__ == '__main__':
    main()
