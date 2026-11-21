"""The fixed minute × deci-decade-band interface."""
import re
import numpy as np
import pandas as pd

BAND_CENTRES = 10 ** (1.5 + np.arange(25) / 10)
N_MIN = 15
BAND_NAMES = [f'TOL_{x:.0f}' if x >= 100 else f'TOL_{x:.1f}' for x in BAND_CENTRES]


def band_index(value):
    value = float(value)
    j = int(np.argmin(abs(BAND_CENTRES - value)))
    if abs(BAND_CENTRES[j] - value) / BAND_CENTRES[j] > .03:
        raise ValueError(f'Unrecognised frequency {value}; expected one of 25 centres (3% tolerance)')
    return j


def read_test_csv(path):
    df = pd.read_csv(path, dtype={'ID': str})
    if 'ID' not in df or df.ID.isna().any() or df.ID.duplicated().any():
        raise ValueError('Test requires unique, nonmissing ID')
    cube = np.full((len(df), 15, 25), np.nan, dtype=np.float32)
    seen = set()
    for col in df.columns.drop('ID'):
        m = re.fullmatch(r'm(\d+)_TOL_([0-9.]+)', col)
        if not m:
            raise ValueError(f'Invalid minute-band column: {col}')
        i, j = int(m[1]) - 1, band_index(m[2])
        if not 0 <= i < 15 or (i, j) in seen:
            raise ValueError(f'Duplicate or invalid minute-band column: {col}')
        seen.add((i, j))
        cube[:, i, j] = pd.to_numeric(df[col], errors='raise')
    if len(seen) != 375:
        raise ValueError(f'Expected 375 minute-band columns, found {len(seen)}')
    if np.isinf(cube).any():
        raise ValueError('Infinite acoustic values are not valid observations')
    return df.ID.to_numpy(), cube


def cube_to_wide(ids, cube):
    cube = np.asarray(cube)
    if cube.shape != (len(ids), 15, 25):
        raise ValueError('Expected cube [len(ids),15,25]')
    cols = [f'm{i:02d}_{b}' for i in range(1, 16) for b in BAND_NAMES]
    df = pd.DataFrame(cube.reshape(len(ids), 375), columns=cols)
    df.insert(0, 'ID', ids)
    return df
