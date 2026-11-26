"""A compact dual-view CNN on the released minute-by-band power observations."""
import argparse
from pathlib import Path
import warnings

import numpy as np
from .schema import BAND_CENTRES, N_MIN


def mel_scale(hz):
    return 2595 * np.log10(1 + hz / 700)


def mel_projection():
    """Integrate mel filters assuming uniform power density within each release band."""
    edges = 10 ** (np.r_[np.log10(BAND_CENTRES) - .05, np.log10(BAND_CENTRES[-1]) + .05])
    points = edges[:-1, None] + np.diff(edges)[:, None] * ((np.arange(64) + .5) / 64)
    centres = np.linspace(*mel_scale(edges[[0, -1]]), len(BAND_CENTRES) + 2)
    distance = np.abs(mel_scale(points)[:, :, None] - centres[None, None, 1:-1])
    area = np.maximum(1 - distance / (centres[1] - centres[0]), 0).mean(axis=1)
    return (area / area.sum(axis=0)).astype(np.float32)


MEL_PROJECTION = mel_projection()


def view_channels(db, power, valid):
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', 'All-NaN slice encountered', RuntimeWarning)
        median_db = np.nanmedian(np.where(valid, db, np.nan), axis=1, keepdims=True)
    energy = power.sum(axis=(1, 2), keepdims=True)
    energy = np.divide(power, energy, out=np.zeros_like(power), where=energy > 0)
    return np.stack((
        np.where(valid, np.clip((db - 70) / 18, -6., 6.), 0),
        np.where(valid, np.clip((db - median_db) / 9, -6., 6.), 0),
        np.sqrt(energy) * 5,
        valid.astype(np.float32),
    ), axis=-1)


def encode_spectrogram(cube):
    """Return aligned-size native and mel views, with a missing-cell mask for each."""
    db = np.asarray(cube, dtype=np.float32)
    valid = np.isfinite(db)
    peak = np.max(np.where(valid, db, -np.inf), axis=(1, 2), keepdims=True)
    peak = np.where(np.isfinite(peak), peak, 0)
    power = np.where(valid, np.power(10., np.maximum((db - peak) / 10, -30.)), 0)
    covered = valid.astype(np.float32) @ MEL_PROJECTION
    mel_power = np.divide(power @ MEL_PROJECTION, covered, out=np.zeros_like(power), where=covered > 0)
    mel_db = (10 * np.log10(np.maximum(mel_power, 1e-30))) + peak
    return np.concatenate((
        view_channels(db, power, valid),
        view_channels(mel_db, mel_power, covered > 0),
    ), axis=-1).astype(np.float32, copy=False)


def prepare_spectrograms(mixtures_path, out, chunk_size=2048):
    mixtures = np.load(mixtures_path, mmap_mode='r')
    if mixtures.ndim != 3 or mixtures.shape[1:] != (N_MIN, len(BAND_CENTRES)):
        raise ValueError('Expected one 15-minute, 25-band grid per mixture')
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_suffix('.part.npy')
    encoded = np.lib.format.open_memmap(
        temporary, mode='w+', dtype=np.float32, shape=(*mixtures.shape, 8))
    for start in range(0, len(mixtures), chunk_size):
        rows = slice(start, start + chunk_size)
        encoded[rows] = encode_spectrogram(mixtures[rows])
    encoded.flush()
    temporary.replace(out)
    return encoded.shape


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mixtures', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    print(prepare_spectrograms(args.mixtures, args.out))


if __name__ == '__main__':
    main()
