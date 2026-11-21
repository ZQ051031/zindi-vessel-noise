"""Fixed per-window acoustic transforms; no population/test statistics."""
import numpy as np
import pandas as pd
from .schema import BAND_CENTRES

GROUPS = [slice(0, 4), slice(4, 8), slice(8, 12), slice(12, 16), slice(16, 20), slice(20, 25)]


def dbsum(a, axis=None):
    count = np.isfinite(a).sum(axis=axis)
    s = np.nansum(a, axis=axis)
    return 10 * np.log10(np.where((count > 0) & (s > 0), s, np.nan))


def stats(a):
    a = a[np.isfinite(a)]
    if not len(a):
        return [np.nan] * 7
    return [a.mean(), np.median(a), a.std(), a.min(), a.max(), *np.percentile(a, [10, 90])]


def window_features(cube_db):
    rows = []
    logf = np.log10(BAND_CENTRES)
    for db in np.asarray(cube_db, dtype=float):
        e = 10 ** (db / 10)
        row = {}
        mean_db, floors = [], []
        for j in range(25):
            observed = e[:, j][np.isfinite(e[:, j])]
            floor = np.sort(observed)[:3].mean() if len(observed) else np.nan
            floors.append(floor)
            values = stats(db[:, j])
            mean_db.append(10 * np.log10(observed.mean()) if len(observed) else np.nan)
            for name, value in zip(['mean', 'median', 'std', 'min', 'max', 'p10', 'p90'], values):
                row[f'b{j}_{name}'] = mean_db[-1] if name == 'mean' else value
        mean_db = np.asarray(mean_db)
        valid = np.isfinite(mean_db)
        median = np.median(db[np.isfinite(db)]) if np.isfinite(db).any() else np.nan
        for j in range(25):
            row[f'b{j}_contrast'] = mean_db[j] - median
        excess = np.maximum(e - np.asarray(floors), 0)
        total = np.nansum(e)
        row['excess'] = np.nansum(excess) / total if total > 0 else 0.
        trajectories, levels = [], []
        for i, group in enumerate(GROUPS):
            trajectory = dbsum(e[:, group], axis=1)
            trajectories.append(trajectory)
            levels.append(dbsum(e[:, group]) - 10 * np.log10(max(1, np.isfinite(e[:, group]).any(axis=1).sum())))
            row[f'g{i}_level'] = levels[-1]
            denom = np.nansum(e[:, group])
            row[f'g{i}_excess'] = np.nansum(excess[:, group]) / denom if denom > 0 else np.nan
            row[f'g{i}_missing'] = 1 - np.isfinite(e[:, group]).mean()
        for i in range(6):
            for j in range(i + 1, 6):
                row[f'g{i}_minus_g{j}'] = levels[i] - levels[j]
        row['low_high_ratio_db'] = dbsum(e[:, :12]) - dbsum(e[:, 16:])
        row['spectral_slope'] = np.polyfit(logf[valid], mean_db[valid], 1)[0] if valid.sum() > 1 else np.nan
        weights = 10 ** (mean_db[valid] / 10)
        row['centroid_log_hz'] = np.average(logf[valid], weights=weights) if len(weights) else np.nan
        row['flatness'] = np.exp(np.log(weights).mean()) / weights.mean() if len(weights) else np.nan
        trajectory = dbsum(e, axis=1)
        tv = np.isfinite(trajectory)
        values = trajectory[tv]
        for name, value in zip(['mean', 'median', 'std', 'min', 'max', 'p10', 'p90'], stats(trajectory)):
            row[f'total_pressure_{name}'] = value
        row['range'] = np.ptp(values) if len(values) else np.nan
        row['max_minus_median'] = values.max() - np.median(values) if len(values) else np.nan
        peak = int(np.nanargmax(trajectory)) if len(values) else 0
        row['peak_minute'] = peak / 14 if len(values) else np.nan
        row['above_floor_3db'] = (values > np.sort(values)[:3].mean() + 3).sum() if len(values) else 0
        row['curvature'] = np.polyfit(np.arange(15)[tv], values, 2)[0] if tv.sum() >= 3 else np.nan
        for name, indices in [('rise', np.arange(peak + 1)), ('fall', np.arange(peak, 15))]:
            ix = indices[tv[indices]]
            row[name] = np.polyfit(ix, trajectory[ix], 1)[0] if len(ix) > 1 else np.nan
        def corr(a, b):
            v = np.isfinite(a) & np.isfinite(b)
            return np.corrcoef(a[v], b[v])[0, 1] if v.sum() > 2 and np.std(a[v]) > 1e-10 and np.std(b[v]) > 1e-10 else np.nan
        row['lag1'] = corr(trajectory[:-1], trajectory[1:])
        row['low_high_covariation'] = corr(trajectories[1], trajectories[4])
        row['observed_cells'] = np.isfinite(db).sum()
        row['missing_fraction'] = 1 - np.isfinite(db).mean()
        row['observed_minutes'] = np.isfinite(db).any(axis=1).sum()
        rows.append(row)
    return pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan)
