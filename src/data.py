"""Tolerant release readers. Inspect DATA_DICTIONARY.md before quality screening."""
from contextlib import contextmanager
from pathlib import Path
import re
import tempfile
import zipfile
import numpy as np
import pandas as pd
from .schema import BAND_NAMES, band_index


@contextmanager
def release_root(path):
    path = Path(path)
    if path.is_dir():
        yield path
    else:
        with tempfile.TemporaryDirectory() as tmp:
            with zipfile.ZipFile(path) as z:
                for name in z.namelist():
                    if not (Path(tmp) / name).resolve().is_relative_to(Path(tmp).resolve()):
                        raise ValueError('Unsafe archive path')
                z.extractall(tmp)
            roots = list(Path(tmp).rglob('acoustics'))
            if len(roots) != 1:
                raise ValueError('Expected one acoustics directory')
            yield roots[0].parent


def detect_column(df, override, candidates, required=True):
    if override:
        if override not in df:
            raise ValueError(f'Missing configured column: {override}')
        return override
    lower = {c.lower(): c for c in df}
    for c in candidates:
        if c in lower:
            return lower[c]
    if required:
        raise ValueError(f'Cannot detect column among {candidates}; supply override')


def load_release(path, site_col=None, time_col=None, dur_col=None,
                 drop_bad_flags=False, flag_cols=None, duration_tolerance=5):
    frames = []
    with release_root(path) as root:
        files = sorted((root / 'acoustics').glob('*.parquet'))
        if not files:
            raise ValueError('No acoustics/*.parquet found')
        for file in files:
            raw = pd.read_parquet(file)
            sc = detect_column(raw, site_col, ['site', 'site_id', 'station', 'station_id', 'station_code'], False)
            tc = detect_column(raw, time_col, ['t', 'timestamp', 'time', 'datetime'])
            dc = detect_column(raw, dur_col, ['dur', 'duration', 'duration_seconds', 'duration_s'])
            keep = (pd.to_numeric(raw[dc], errors='coerce') - 60).abs() <= duration_tolerance
            if drop_bad_flags:
                flags = flag_cols if flag_cols is not None else [c for c in raw if re.search(r'flag|qc|quality', c, re.I)]
                for c in flags:
                    if c not in raw:
                        raise ValueError(f'Missing quality flag: {c}')
                    good = raw[c].astype(str).str.lower().isin(['0', '0.0', 'false', 'good', 'ok', 'pass', 'nan', 'none'])
                    keep &= good
            out = pd.DataFrame({'site': raw[sc].astype(str) if sc else file.stem,
                                't': pd.to_datetime(raw[tc], utc=True, errors='raise')}, index=raw.index)
            for b in BAND_NAMES:
                out[b] = np.nan
            seen = set()
            for c in raw:
                m = re.fullmatch(r'TOL_([0-9.]+)', c, re.I)
                if m:
                    j = band_index(m[1])
                    if j in seen:
                        raise ValueError(f'Duplicate frequency in {file}: {c}')
                    seen.add(j)
                    out[BAND_NAMES[j]] = pd.to_numeric(raw[c], errors='raise')
            frames.append(out.loc[keep])
    return pd.concat(frames, ignore_index=True)


def load_stations(path, lon_col=None, lat_col=None, site_col=None):
    with release_root(path) as root:
        df = pd.read_csv(root / 'stations.csv')
    lon = detect_column(df, lon_col, ['lon', 'longitude', 'long', 'longitude_deg'])
    lat = detect_column(df, lat_col, ['lat', 'latitude', 'latitude_deg'])
    site = detect_column(df, site_col, ['site', 'site_id', 'station', 'station_id', 'station_code'])
    return df.rename(columns={lon: 'lon', lat: 'lat', site: 'site'})


def make_windows(df, min_observed_minutes=10, stride=15):
    if stride < 1:
        raise ValueError('stride must be positive')
    cubes, meta = [], []
    for site, part in df.groupby('site', sort=True):
        grid = part.set_index('t')[BAND_NAMES].resample('1min').mean()
        for i in range(0, len(grid) - 14, stride):
            a = grid.iloc[i:i + 15].to_numpy(dtype=np.float32)
            if np.isfinite(a).any(axis=1).sum() >= min_observed_minutes:
                cubes.append(a)
                meta.append({'site': site, 't': grid.index[i]})
    return np.asarray(cubes, dtype=np.float32).reshape(-1, 15, 25), pd.DataFrame(meta, columns=['site', 't'])
