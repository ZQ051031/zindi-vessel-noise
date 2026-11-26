"""Fit a qualified CNN checkpoint on the entire official-release training set."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.isotonic import IsotonicRegression
import torch

from .round3_cnn import fit_cnn, infer, new_model, validation_data
from .spectrogram_cnn import encode_spectrogram

PIPELINE = 'spectrogram_cnn'


def refit(cache, validation, out, threads):
    decision = json.loads((validation / 'decision.json').read_text())
    winner = decision['winner']
    if winner is None:
        raise ValueError('No CNN checkpoint met the adoption rule')
    metrics = decision['models'][winner]
    if metrics['score'] < .9250 or metrics['auc'] < .96662:
        raise ValueError('CNN checkpoint does not meet the adoption rule')
    checkpoints = decision['cnn_checkpoints']
    selected = int(winner.removeprefix('cnn_epoch_'))
    index = checkpoints.index(selected)
    epochs = max(checkpoints)
    _, _, share, presence, splits = validation_data(cache)

    raw = np.empty(len(share))
    for i, (_, valid) in enumerate(splits):
        path = validation / f'fixed_{epochs}_epochs' / f'fold_{i}.npz'
        with np.load(path) as data:
            np.testing.assert_array_equal(data['indices'], valid)
            raw[valid] = data['predictions'][index, 1]
    calibrator = IsotonicRegression(out_of_bounds='clip').fit(raw, presence)

    features = np.load(validation / 'spectrograms.npy', mmap_mode='r')
    rows = np.arange(len(share))
    for epoch, model in fit_cnn(features, share.astype(np.float32), presence.astype(np.float32),
                                rows, epochs, range(1, selected + 1), threads=threads):
        print(f'Full release fit ended epoch {epoch}/{selected}', flush=True)
        if epoch == selected:
            break
    state = {name: weight.numpy() for name, weight in model.state_dict().items()}
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({'pipeline': PIPELINE, 'state': state, 'calibrator': calibrator}, out)


def predict_cnn(bundle, cube):
    model = new_model(42)
    model.load_state_dict({name: torch.from_numpy(value) for name, value in bundle['state'].items()})
    encoded = encode_spectrogram(cube)
    share, raw = infer(model, encoded, np.arange(len(encoded)))
    return share, bundle['calibrator'].predict(raw)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--validation', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=1)
    refit(**vars(parser.parse_args()))


if __name__ == '__main__':
    main()
