"""One-command SYNTHETIC-only demo; no competition data access or submissions."""
import argparse
import json
from pathlib import Path
import time
import pandas as pd
from .synthetic import write_synthetic_release, make_synthetic_competition
from .train import train
from .predict import predict
from .evaluate import evaluate_frames, leaderboard_score
from .schema import read_test_csv
from .features import window_features
from .pressure_map import pressure_map


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out-dir', default='outputs/synthetic_demo')
    p.add_argument('--days', type=float, default=.5)
    p.add_argument('--n-test', type=int, default=150)
    p.add_argument('--augment-k', type=int, default=3)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    start = time.perf_counter()
    out = Path(args.out_dir)
    release = write_synthetic_release(out / 'SYNTHETIC_release', days=args.days, seed=args.seed)
    comp = make_synthetic_competition(out / 'SYNTHETIC_competition', n_test=args.n_test, seed=args.seed + 4200)
    bundle, metrics = train(release, out / 'models', augment_k=args.augment_k, seed=args.seed)
    sub = predict(comp / 'Test.csv', bundle, comp / 'sample_submission.csv', out / 'SYNTHETIC_submission.csv')
    labels = pd.read_csv(comp / 'SYNTHETIC_test_labels.csv')
    result = evaluate_frames(sub, labels, seed=args.seed)
    _, cube = read_test_csv(comp / 'Test.csv')
    result['baseline_score'] = leaderboard_score(labels.share_added_vessel, window_features(cube).excess)
    result['SYNTHETIC'] = True
    pressure_map(release, out / 'maps', out / 'models' / 'bundle.joblib')
    result['demo_seconds'] = time.perf_counter() - start
    (out / 'SYNTHETIC_evaluation.json').write_text(json.dumps(result, indent=2))
    print('SYNTHETIC HELD-OUT SITES:', json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
