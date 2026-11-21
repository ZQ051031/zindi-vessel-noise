"""Entirely SYNTHETIC acoustic examples, not measured vessel acoustics."""
from pathlib import Path
import numpy as np
import pandas as pd
from .schema import BAND_CENTRES as F, BAND_NAMES, cube_to_wide


def synthetic_background(n_windows, rng=None, site_offset=0):
    """SYNTHETIC ambient spectra with biology, coverage gaps and missing cells."""
    rng = np.random.default_rng(42) if rng is None else rng
    n = n_windows * 15
    t = np.arange(n)
    wind = 66 - 5 * np.log2(np.maximum(F, 250) / 250)
    shipping = 14 * np.exp(-.5 * (np.log2(F / 90) / 1.2) ** 2)
    slow = np.convolve(rng.normal(0, 1, n + 30), np.ones(31) / 31, mode='valid') * 4
    db = wind + shipping + site_offset + 2 * np.sin(t[:, None] * 2 * np.pi / 1440)
    db = db + slow[:, None] + rng.normal(0, .6, (n, 25))
    for _ in range(max(1, n // 100)):
        i = rng.integers(n)
        length = min(int(rng.integers(1, 8)), n - i)
        kind = rng.integers(3)
        shape = (np.exp(-.5 * (np.log2(F / 600) / .65) ** 2) if kind == 0
                 else (F > 2000).astype(float) if kind == 1 else np.ones(25))
        db[i:i + length] += rng.uniform(2, 9) * shape
    db += np.repeat(rng.uniform(-3, 3, n_windows), 15)[:, None]
    for w in range(n_windows):
        if rng.random() < .4:
            kind = rng.integers(3)
            time = np.arange(15)
            envelope = np.exp(-.5 * ((time - rng.uniform(-2, 17)) / rng.uniform(2, 7)) ** 2)
            if kind == 0:
                shape = np.exp(-.5 * (np.log2(F / rng.uniform(60, 180)) / 1.4) ** 2)
            elif kind == 1:
                shape = np.maximum(0, np.log10(F / 100)) / 2
                envelope = np.linspace(0, 1, 15) if rng.random() < .5 else np.linspace(1, 0, 15)
            else:
                shape = np.exp(-.5 * (np.log2(F / 600) / .7) ** 2)
                envelope = 1 / (1 + np.exp(-(time - rng.uniform(3, 12))))
            db[w * 15:(w + 1) * 15] += rng.uniform(3, 16) * envelope[:, None] * shape
    if rng.random() < .6:
        db[:, int(rng.integers(21, 25)):] = np.nan
    db[rng.random(db.shape) < .025] = np.nan
    return db.reshape(n_windows, 15, 25).astype(np.float32)


def write_synthetic_release(out_dir, n_sites=4, days=.5, seed=42):
    """Write a small explicitly SYNTHETIC release with fake station coordinates."""
    root = Path(out_dir)
    (root / 'acoustics').mkdir(parents=True, exist_ok=True)
    (root / 'SYNTHETIC').write_text('SYNTHETIC EXAMPLE — NOT REAL RESULTS\n')
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_sites):
        site = f'SYN_SITE_{chr(65 + i)}'
        cube = synthetic_background(max(2, int(days * 96)), rng, rng.normal(0, 3))
        df = pd.DataFrame(cube.reshape(-1, 25), columns=BAND_NAMES)
        df.insert(0, 't', pd.date_range('2024-01-01', periods=len(df), freq='min', tz='UTC'))
        df['dur'], df['site'], df['qc_flag'] = 60, site, 0
        df['SYNTHETIC'] = True
        df.to_parquet(root / 'acoustics' / f'{site}.parquet', index=False)
        rows.append({'site': site, 'lon': 18 + i * .4, 'lat': -34 + .2 * i,
                     'uncertainty_notes': 'SYNTHETIC invented coordinates', 'SYNTHETIC': True})
    pd.DataFrame(rows).to_csv(root / 'stations.csv', index=False)
    pd.DataFrame({'site': [r['site'] for r in rows], 'SYNTHETIC': True}).to_csv(root / 'recorders.csv', index=False)
    (root / 'DATA_DICTIONARY.md').write_text('SYNTHETIC ONLY. t: UTC minute; dur: observed seconds; TOL_*: invented export dB; qc_flag: 0 good, nonzero bad. Blank means missing, never silence. No verified physical calibration. For real releases inspect their dictionary before enabling quality filtering.\n')
    return root


def vessel_passage_reference(rng, kind):
    """SYNTHETIC linear-energy CPA template including a small ambient floor."""
    if kind not in {'small_craft', 'large_ship'}:
        raise ValueError('kind must be small_craft or large_ship')
    large = kind == 'large_ship'
    peak = rng.uniform(50, 150) if large else rng.uniform(500, 4000)
    spectrum = np.exp(-.5 * (np.log2(F / peak) / (1.5 if large else 1.1)) ** 2) + .025
    spectrum[rng.choice(25, 3, replace=False)] *= rng.uniform(1.5, 3)
    d, v, t0 = rng.uniform(50, 500), rng.uniform(100, 250) if large else rng.uniform(250, 600), rng.uniform(-4, 19)
    r = np.sqrt(d ** 2 + (v * (np.arange(15) - t0)) ** 2)
    energy = spectrum[None, :] * (d / r[:, None]) ** 2
    energy *= np.exp(-r[:, None] * (F[None, :] / 1000) ** 1.4 * 1e-5)
    if rng.random() < .5:
        energy *= 1 + .2 * np.sin(np.log(F)[None, :] * 8 + r[:, None] / 100)
    ambient = rng.uniform(.001, .15) * spectrum.mean() * (F / 250) ** rng.uniform(-.6, .2)
    return energy + ambient[None, :]


def add_controlled_passages(background_cube_db, rng, p_none=.3):
    """SYNTHETIC additions scaled using ONLY observed cells; labels are added/mixture energy."""
    bg = np.asarray(background_cube_db, dtype=float)
    one = bg.ndim == 2
    bg = bg[None] if one else bg
    energy = 10 ** (bg / 10)
    mixture, shares, presence, kinds = [], [], [], []
    for db, e in zip(bg, energy):
        mask = np.isfinite(db)
        target = 0 if rng.random() < p_none or not mask.any() else float(.95 * (rng.beta(.8, 3) if rng.random() < .75 else rng.beta(3, 1.5)))
        added = np.zeros((15, 25))
        kind = 'none'
        if target > 0:
            kind = rng.choice(['small_craft', 'large_ship'])
            added = vessel_passage_reference(rng, kind)
            if rng.random() < .25:
                added += vessel_passage_reference(rng, rng.choice(['small_craft', 'large_ship']))
                kind = 'mixed'
            added *= target / (1 - target) * e[mask].sum() / added[mask].sum()
        mix = e + added
        mixture.append(np.where(mask, 10 * np.log10(mix), np.nan))
        shares.append(added[mask].sum() / mix[mask].sum() if mask.any() else 0.)
        presence.append(int(target > 0))
        kinds.append(kind)
    result = (np.asarray(mixture), np.asarray(shares), np.asarray(presence), np.asarray(kinds))
    return tuple(x[0] for x in result) if one else result


def make_synthetic_competition(out_dir, n_test=1500, seed=4242):
    """SYNTHETIC competition with held-out synthetic sites, hidden labels separate from inputs."""
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    (root / 'SYNTHETIC').write_text('SYNTHETIC EXAMPLE — NOT REAL RESULTS\n')
    rng = np.random.default_rng(seed)
    counts = [n_test // 3 + (i < n_test % 3) for i in range(3)]
    bg = np.concatenate([synthetic_background(n, rng, rng.normal(0, 3)) for n in counts])
    mix, y, p, kinds = add_controlled_passages(bg, rng)
    ids = [f'SYNTHETIC_{i:06d}' for i in range(n_test)]
    cube_to_wide(ids, mix).to_csv(root / 'Test.csv', index=False)
    pd.DataFrame({'ID': ids, 'share_added_vessel': 0., 'score_added_vessel': 0.}).to_csv(root / 'sample_submission.csv', index=False)
    pd.DataFrame({'ID': ids, 'share_added_vessel': y, 'presence': p, 'kind': kinds,
                  'site': np.repeat(['SYN_HELDOUT_A', 'SYN_HELDOUT_B', 'SYN_HELDOUT_C'], counts),
                  'SYNTHETIC': True}).to_csv(root / 'SYNTHETIC_test_labels.csv', index=False)
    return root
