"""Frozen-model importance and TreeSHAP contributions, not causal attribution."""
import argparse
from pathlib import Path
import re
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from .schema import BAND_CENTRES

SYNTHETIC_TITLE = 'SYNTHETIC EXAMPLE — NOT REAL RESULTS\n'


def feature_family(name):
    if 'missing' in name or name.startswith('observed_'):
        return 'missingness'
    if 'excess' in name:
        return 'excess features'
    if name.startswith('b') and re.match(r'b\d+_', name):
        return 'band contrasts' if 'contrast' in name else 'per-band levels'
    if name.startswith('g') or name in ['low_high_ratio_db', 'spectral_slope', 'centroid_log_hz', 'flatness']:
        return 'band-group levels/ratios'
    return 'temporal/total_pressure features'


def explain(bundle, features, out_dir, synthetic=False):
    bundle = joblib.load(bundle) if isinstance(bundle, (str, Path)) else bundle
    x = features[bundle['features']]
    if x.empty:
        raise ValueError('Explanation requires at least one feature row')
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    names = bundle['features']
    gain = pd.DataFrame({'feature': names, 'gain': bundle['lightgbm'].booster_.feature_importance(importance_type='gain')}).sort_values('gain', ascending=False).head(20)
    ridge = bundle['ridge']
    ridge_names = ridge[:-1].get_feature_names_out(names)
    coefficients = pd.DataFrame({'feature': ridge_names, 'standardised_coefficient': ridge[-1].coef_})
    coefficients = coefficients.iloc[np.argsort(-abs(coefficients.standardised_coefficient.to_numpy()))[:20]]
    contributions = bundle['lightgbm'].predict(x, pred_contrib=True)
    values = np.abs(contributions[:, :-1]).mean(axis=0)
    detail = pd.DataFrame({'feature': names, 'mean_absolute_contribution': values,
                           'family': [feature_family(n) for n in names]})
    family = detail.groupby('family', as_index=False).mean_absolute_contribution.sum()
    bands = []
    for j, hz in enumerate(BAND_CENTRES):
        mask = detail.feature.str.match(fr'b{j}_')
        bands.append({'band': j, 'frequency_hz': hz, 'mean_absolute_contribution': detail.loc[mask, 'mean_absolute_contribution'].sum()})
    band = pd.DataFrame(bands)
    tables = {'lightgbm_gain_top20': gain, 'ridge_coefficients_top20': coefficients,
              'contributions_by_feature': detail, 'contributions_by_family': family,
              'contributions_by_band': band}
    for name, table in tables.items():
        table.to_csv(out / f'{name}.csv', index=False)
        if name == 'contributions_by_feature':
            continue
        value = 'gain' if name.startswith('lightgbm') else 'standardised_coefficient' if name.startswith('ridge') else 'mean_absolute_contribution'
        labels = table['feature'] if 'feature' in table else table['family'] if 'family' in table else table.frequency_hz.map(lambda f: f'{f:.0f} Hz')
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(np.arange(len(table)), table[value])
        ax.set_yticks(np.arange(len(table)), labels, fontsize=8)
        ax.invert_yaxis()
        ax.set(xlabel=value.replace('_', ' '), title=(SYNTHETIC_TITLE if synthetic else '') + name.replace('_', ' '))
        fig.tight_layout()
        fig.savefig(out / f'{name}.png', dpi=150)
        plt.close(fig)
    return tables


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--features', required=True, help='Training feature parquet or CSV; never fit on test')
    p.add_argument('--out-dir', required=True)
    p.add_argument('--synthetic', action='store_true')
    args = p.parse_args()
    x = pd.read_parquet(args.features) if args.features.endswith('.parquet') else pd.read_csv(args.features)
    explain(args.model, x, args.out_dir, synthetic=args.synthetic)


if __name__ == '__main__':
    main()
