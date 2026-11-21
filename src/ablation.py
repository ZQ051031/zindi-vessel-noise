"""Site-held-out controls using identical folds; no test adaptation."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import GroupKFold
from .models import estimators
from .explain import feature_family
from .evaluate import rmse, bootstrap_ci


def ablate(features, oof, out_dir, seed=42, n_estimators=180):
    if len(features) != len(oof) or oof.site.nunique() < 2:
        raise ValueError('Aligned features/OOF with at least two sites required')
    y = oof.share_added_vessel.to_numpy()
    splits = list(GroupKFold(min(4, oof.site.nunique())).split(features, y, oof.site))
    spectral_families = {'per-band levels', 'band contrasts', 'band-group levels/ratios'}
    spectral = [c for c in features if feature_family(c) in spectral_families and not c.endswith(('_std', '_min', '_max', '_p10', '_p90'))]
    temporal = [c for c in features if feature_family(c) == 'temporal/total_pressure features']
    rows = []
    for name, cols in [('excess_only', ['excess']), ('received_energy', ['total_pressure_mean']),
                       ('spectral_only', spectral), ('temporal_only', temporal), ('all_features', list(features))]:
        pred = np.zeros(len(y))
        for tr, va in splits:
            if name in ['excess_only', 'received_energy']:
                iso = IsotonicRegression(out_of_bounds='clip').fit(features[cols[0]].iloc[tr], y[tr])
                pred[va] = iso.predict(features[cols[0]].iloc[va])
            else:
                _, reg, _ = estimators(seed, n_estimators)
                reg.fit(features[cols].iloc[tr], y[tr])
                pred[va] = np.clip(reg.predict(features[cols].iloc[va]), 0, 1)
        ci = bootstrap_ci(y, pred, oof.background_window_id, seed=seed)
        rows.append({'control': name, 'n_features': len(cols), 'rmse': rmse(y, pred), 'score': 1 - rmse(y, pred),
                     'cluster_ci_low': ci[0], 'cluster_ci_high': ci[1]})
    result = pd.DataFrame(rows)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    result.to_csv(out / 'ablation.csv', index=False)
    (out / 'ablation.json').write_text(json.dumps(rows, indent=2))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--features', required=True)
    p.add_argument('--oof', required=True)
    p.add_argument('--out-dir', required=True)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    print(ablate(pd.read_parquet(args.features), pd.read_parquet(args.oof), args.out_dir, args.seed).to_string(index=False))


if __name__ == '__main__':
    main()
