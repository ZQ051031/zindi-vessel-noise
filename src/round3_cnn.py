"""Train and score a spectrogram CNN using the official-release LOSO folds."""
import argparse
import json
from pathlib import Path

from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut
import torch
from torch import nn
from torch.nn import functional as F

from .loso_experiment import score_predictions
from .schema import BAND_CENTRES, N_MIN
from .train import calibrate_oof

BATCH_SIZE = 384
LEARNING_RATE = .0025


class SpectrogramCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Grouped layers keep native and mel frequency coordinates separate.
        self.encoder = nn.Sequential(
            nn.Conv2d(8, 24, 3, stride=(1, 2), padding=1, groups=2, bias=False),
            nn.BatchNorm2d(24), nn.SiLU(),
            nn.Conv2d(24, 32, 3, stride=2, padding=1, groups=2, bias=False),
            nn.BatchNorm2d(32), nn.SiLU(),
            nn.Conv2d(32, 48, 3, padding=1, groups=2, bias=False),
            nn.BatchNorm2d(48), nn.SiLU(),
        )
        self.readout = nn.Sequential(nn.Linear(48 * 7 * 2, 96), nn.LayerNorm(96),
                                     nn.SiLU(), nn.Dropout(.12), nn.Linear(96, 2))

    def forward(self, x):
        z = self.encoder(x)
        pooled = torch.cat((z.mean(dim=2), z.amax(dim=2)), dim=1).flatten(1)
        return self.readout(pooled)


def new_model(seed, threads=1):
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    return SpectrogramCNN().to(memory_format=torch.channels_last)


def train_batch(model, optimizer, features, share, presence, random):
    gain = torch.from_numpy(random.normal(0, .075, (len(features), 1, 1)).astype(np.float32))
    features[..., 0] += gain * features[..., 3]
    features[..., 4] += gain * features[..., 7]
    prediction = model(features.permute(0, 3, 1, 2))
    error = F.mse_loss(prediction[:, 0].sigmoid(), share)
    loss = error + .045 * F.binary_cross_entropy_with_logits(prediction[:, 1], presence)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


@torch.inference_mode()
def infer(model, features, rows):
    model.eval()
    predictions = []
    for start in range(0, len(rows), BATCH_SIZE):
        batch = features[rows[start:start + BATCH_SIZE]]
        forward = model(torch.from_numpy(batch).permute(0, 3, 1, 2)).sigmoid()
        reverse = model(torch.from_numpy(batch[:, ::-1].copy()).permute(0, 3, 1, 2)).sigmoid()
        predictions.append(((forward + reverse) / 2).numpy())
    return np.concatenate(predictions).T


def fit_cnn(features, share, presence, train, epochs, checkpoints, seed=42, threads=1):
    model = new_model(seed, threads)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=.006)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=.00025)
    random = np.random.default_rng(seed)
    for epoch in range(1, epochs + 1):
        model.train()
        order = random.permutation(train)
        for start in range(0, len(order), BATCH_SIZE):
            rows = order[start:start + BATCH_SIZE]
            minutes = slice(None, None, -1) if random.random() < .5 else slice(None)
            x = torch.from_numpy(features[rows, minutes].copy())
            train_batch(model, optimizer, x, torch.from_numpy(share[rows]),
                        torch.from_numpy(presence[rows]), random)
        scheduler.step()
        if epoch in checkpoints:
            yield epoch, model


def run_fold(encoded, share, presence, train, valid, site, path, epochs):
    features = np.load(encoded, mmap_mode='r')
    results = []
    for epoch, model in fit_cnn(features, share, presence, train, epochs, range(1, epochs + 1)):
        print(f'{site} ended epoch {epoch}/{epochs}', flush=True)
        if epoch in (epochs // 2, epochs):
            results.append(infer(model, features, valid))
    temporary = path.with_suffix('.tmp.npz')
    np.savez(temporary, indices=valid, predictions=results)
    temporary.replace(path)
    return site


def validation_data(cache):
    reference = pd.read_parquet(cache / 'candidate_oof.parquet')
    sites = reference.site.to_numpy()
    share = reference.share_added_vessel.to_numpy()
    presence = reference.presence.to_numpy()
    splits = list(LeaveOneGroupOut().split(share, share, sites))
    if (len(splits) != 10 or len(reference) != 318507
            or reference.background_window_id.nunique() != 106169):
        raise ValueError('Expected the official-release validation cohort')
    for i, (_, valid) in enumerate(splits):
        if not reference.fold.iloc[valid].eq(i).all():
            raise ValueError('Fold assignments differ from R1')
    return reference, sites, share, presence, splits


def run_validation(cache, out, jobs, epochs):
    reference, sites, share, presence, splits = validation_data(cache)
    encoded = out / 'spectrograms.npy'
    if np.load(encoded, mmap_mode='r').shape != (len(share), N_MIN, len(BAND_CENTRES), 8):
        raise ValueError('Spectrogram cache does not match the validation cohort')
    out_dir = out / f'fixed_{epochs}_epochs'
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [out_dir / f'fold_{i}.npz' for i in range(len(splits))]
    done = Parallel(n_jobs=jobs, return_as='generator_unordered', max_nbytes=None)(
        delayed(run_fold)(encoded, share.astype(np.float32), presence.astype(np.float32),
                          train, valid, sites[valid[0]], path, epochs)
        for (train, valid), path in zip(splits, paths) if not path.exists())
    for site in done:
        print(f'Finished LOSO predictions for {site}', flush=True)

    predictions = np.empty((2, 2, len(share)))
    for path, (_, valid) in zip(paths, splits):
        with np.load(path) as data:
            np.testing.assert_array_equal(data['indices'], valid)
            predictions[..., valid] = data['predictions']
    scores = {'R1': (reference.enriched_boosting.to_numpy(),
                     reference.enriched_boosting_calibrated_prob.to_numpy())}
    for epoch, (estimate, raw_probability) in zip((epochs // 2, epochs), predictions):
        scores[f'cnn_epoch_{epoch}'] = (estimate, calibrate_oof(raw_probability, presence, splits))
    metrics = {name: score_predictions(share, estimate, presence, probability)
               for name, (estimate, probability) in scores.items()}
    for epoch, (_, raw_probability) in zip((epochs // 2, epochs), predictions):
        metrics[f'cnn_epoch_{epoch}']['uncalibrated_auc'] = float(
            roc_auc_score(presence, raw_probability))
    eligible = [name for name in scores if name != 'R1'
                and metrics[name]['score'] >= .9250 and metrics[name]['auc'] >= .96662]
    result = {'models': metrics,
              'winner': max(eligible, key=lambda name: metrics[name]['score'], default=None),
              'cnn_checkpoints': [epochs // 2, epochs]}
    write_residuals(cache, out, sites, share, presence, splits, scores)
    (out / 'decision.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


def write_residuals(cache, out, sites, share, presence, splits, scores):
    station_rows, band_rows = [], []
    weights = np.load(cache / 'added_band_weights.npy', mmap_mode='r')
    if weights.shape != (len(share), len(BAND_CENTRES)):
        raise ValueError('Band-energy weights do not align with the validation cohort')
    totals = weights.sum(axis=0, dtype=np.float64)
    for name, (estimate, probability) in scores.items():
        error = estimate - share
        for _, valid in splits:
            station_rows.append({
                'model': name, 'station': sites[valid[0]], 'examples': len(valid),
                'mean_error': error[valid].mean(),
                **score_predictions(share[valid], estimate[valid], presence[valid], probability[valid]),
            })
        rmses = np.sqrt((error ** 2) @ weights / totals)
        biases = (error @ weights) / totals
        band_rows.extend({'model': name, 'centre_hz': hz, 'weighted_rmse': rmse, 'weighted_mean_error': bias}
                         for hz, rmse, bias in zip(BAND_CENTRES, rmses, biases))
    pd.DataFrame(station_rows).to_csv(out / 'station_scores.csv', index=False)
    pd.DataFrame(band_rows).to_csv(out / 'band_scores.csv', index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--jobs', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=12)
    args = parser.parse_args()
    if args.epochs < 2:
        parser.error('--epochs must be at least two')
    run_validation(**vars(args))


if __name__ == '__main__':
    main()
