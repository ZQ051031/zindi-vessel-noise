"""Inference only: fixed per-window features and frozen training transformations."""
import argparse
from pathlib import Path
import resource
import time
import joblib
import numpy as np
import pandas as pd
from .schema import read_test_csv
from .models import predict_cube
from .loso_submission import PIPELINE, predict_enriched

COLUMNS = ['ID', 'share_added_vessel', 'score_added_vessel']


def predict(test, model, sample, out):
    start = time.perf_counter()
    ids, cube = read_test_csv(test)
    template = pd.read_csv(sample, dtype={'ID': str})
    if list(template) != COLUMNS or template.ID.isna().any() or template.ID.duplicated().any() or set(template.ID) != set(ids):
        raise ValueError('Sample must have exact submission columns and matching unique IDs')
    bundle = joblib.load(model) if isinstance(model, (str, Path)) else model
    pipeline = bundle.get('pipeline')
    if pipeline == 'spectrogram_cnn':
        from .round3_submission import predict_cnn

        share, score = predict_cnn(bundle, cube)
    else:
        predictor = predict_enriched if pipeline == PIPELINE else predict_cube
        share, score = predictor(bundle, cube)
    df = pd.DataFrame({'ID': ids, 'share_added_vessel': share, 'score_added_vessel': score})
    df = df.set_index('ID').loc[template.ID].reset_index()
    assert list(df) == COLUMNS and len(df) == len(template)
    assert np.isfinite(df.iloc[:, 1:]).all().all()
    assert ((df.iloc[:, 1:] >= 0) & (df.iloc[:, 1:] <= 1)).all().all()
    df.to_csv(out, index=False)
    print(f'Inference seconds: {time.perf_counter() - start:.3f}; peak process RSS MiB: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.1f}')
    return df


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ['test', 'model', 'sample', 'out']:
        p.add_argument('--' + key, required=True)
    predict(**vars(p.parse_args()))


if __name__ == '__main__':
    main()
