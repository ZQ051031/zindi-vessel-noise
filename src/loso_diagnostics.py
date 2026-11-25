"""Replay the official-release controlled examples and summarize held-out residuals."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .data import load_release, make_windows
from .schema import BAND_CENTRES
from .synthetic import add_controlled_passages


def replay(release, cache, out_dir):
    backgrounds, windows = make_windows(load_release(release))
    metadata = pd.read_parquet(cache / 'SYNTHETIC_addition_oof.parquet')
    n = len(windows)
    draws, remainder = divmod(len(metadata), n)
    if remainder or not draws:
        raise ValueError('Cached labels do not align with the released background windows')
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / 'backgrounds.npy', backgrounds)
    mixtures = np.lib.format.open_memmap(
        out_dir / 'mixtures.npy', mode='w+', dtype='float64', shape=(len(metadata), 15, 25))
    band_weights = np.empty((len(metadata), 25), dtype=np.float32)
    rng = np.random.default_rng(42)
    background_energy = np.power(10., backgrounds.astype(np.float64) / 10)
    for draw in range(draws):
        rows = slice(draw * n, (draw + 1) * n)
        reference = metadata.iloc[rows].reset_index(drop=True)
        if not (np.array_equal(reference.background_window_id, np.arange(n))
                and reference.site.eq(windows.site).all() and reference.t.eq(windows.t).all()):
            raise ValueError('Recreated background windows differ from the original run')
        mixed, shares, presence, kinds = add_controlled_passages(backgrounds, rng)
        np.testing.assert_array_equal(reference.share_added_vessel, shares)
        np.testing.assert_array_equal(reference.presence, presence)
        np.testing.assert_array_equal(reference.kind, kinds)
        mixtures[rows] = mixed
        energy_added = np.maximum(np.power(10., mixed / 10) - background_energy, 0)
        by_band = np.nansum(energy_added, axis=1)
        by_band[presence == 0] = 0
        total = by_band.sum(axis=1, keepdims=True)
        band_weights[rows] = np.divide(by_band, total, out=np.zeros_like(by_band), where=total > 0)
        print(f'Verified controlled-addition draw {draw + 1} of {draws}.', flush=True)
    mixtures.flush()
    np.save(out_dir / 'added_band_weights.npy', band_weights)


def residuals(oof_path, out_dir):
    oof = pd.read_parquet(oof_path)
    weights = np.load(out_dir / 'added_band_weights.npy')
    if weights.shape != (len(oof), len(BAND_CENTRES)):
        raise ValueError('Band-energy weights do not align with out-of-fold predictions')
    oof['error'] = oof.ensemble - oof.share_added_vessel
    oof['sq_error'] = oof.error ** 2
    station_rows = []
    for station, data in oof.groupby('site'):
        station_rows.append({
            'site': station,
            'background_windows': data.background_window_id.nunique(),
            'examples': len(data),
            'rmse': np.sqrt(data.sq_error.mean()),
            'mean_error': data.error.mean(),
            'auc': roc_auc_score(data.presence, data.score_added_vessel),
        })
    stations = pd.DataFrame(station_rows).sort_values('rmse', ascending=False)
    totals = weights.sum(axis=0, dtype=np.float64)
    bands = pd.DataFrame({
        'centre_hz': BAND_CENTRES,
        'weight': totals,
        'positive_windows': (weights > 0).sum(axis=0),
        'weighted_rmse': np.sqrt(oof.sq_error.to_numpy() @ weights / totals),
        'weighted_mean_error': oof.error.to_numpy() @ weights / totals,
    }).sort_values('weighted_rmse', ascending=False)
    stations.to_csv(out_dir / 'station_residuals.csv', index=False)
    bands.to_csv(out_dir / 'band_residuals.csv', index=False)
    print(stations.to_string(index=False))
    print(bands.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    replay_parser = sub.add_parser('replay')
    replay_parser.add_argument('--release', type=Path, required=True)
    replay_parser.add_argument('--cache', type=Path, required=True)
    replay_parser.add_argument('--out-dir', type=Path, required=True)
    residuals_parser = sub.add_parser('residuals')
    residuals_parser.add_argument('--oof', type=Path, required=True)
    residuals_parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'replay':
        replay(args.release, args.cache, args.out_dir)
    else:
        residuals(args.oof, args.out_dir)


if __name__ == '__main__':
    main()
