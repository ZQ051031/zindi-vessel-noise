"""Relative soundscape proxies, not calibrated pressure or measured vessel shares."""
import argparse
from pathlib import Path
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from .data import load_release, load_stations, make_windows, release_root
from .schema import BAND_NAMES
from .features import GROUPS, dbsum
from .models import predict_cube


def pressure_map(release, out_dir, model=None, lon_col=None, lat_col=None, station_site_col=None, **loader_options):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = load_release(release, **loader_options)
    stations = load_stations(release, lon_col=lon_col, lat_col=lat_col, site_col=station_site_col)
    with release_root(release) as root:
        synthetic = (root / 'SYNTHETIC').exists()
    title = 'SYNTHETIC EXAMPLE — NOT REAL RESULTS\n' if synthetic else ''
    common = [b for b in BAND_NAMES if df.groupby('site')[b].apply(lambda x: x.notna().any()).all()]
    if not common:
        raise ValueError('No common observed bands across stations')
    # Missing common bands invalidate a minute's broadband proxy, rather than acting as silence.
    energy = 10 ** (df[common].to_numpy() / 10)
    level = dbsum(energy, axis=1)
    level[~np.isfinite(energy).all(axis=1)] = np.nan
    df['uncalibrated_export_db'] = level
    df['relative_db_own_median'] = df.uncalibrated_export_db - df.groupby('site').uncalibrated_export_db.transform('median')
    df['day'] = df.t.dt.floor('D')
    summaries = []
    for site, part in df.groupby('site'):
        expected = max(1, int((part.t.max() - part.t.min()).total_seconds() / 60) + 1)
        row = {'site': site, 'median_uncalibrated_export_db': part.uncalibrated_export_db.median(),
               'median_relative_db_own_median': part.relative_db_own_median.median(),
               'coverage': part.uncalibrated_export_db.notna().sum() / expected,
               'common_bands': ','.join(common), 'SYNTHETIC': synthetic}
        for i, group in enumerate(GROUPS):
            cols = BAND_NAMES[group]
            row[f'group_{i}_coverage_percent'] = 100 * part[cols].notna().sum().sum() / (expected * len(cols))
        summaries.append(row)
    summary = pd.DataFrame(summaries).merge(stations[['site', 'lon', 'lat']], on='site', how='left', validate='one_to_one')
    if model:
        cube, meta = make_windows(df, stride=5)
        bundle = joblib.load(model)
        share, _ = predict_cube(bundle, cube)
        inferred = meta.assign(inferred_added_vessel_like_share=share).groupby('site').inferred_added_vessel_like_share.mean()
        summary = summary.merge(inferred, on='site', how='left')
    def scatter(column, label, filename):
        fig, ax = plt.subplots(figsize=(9, 6))
        plot = ax.scatter(summary.lon, summary.lat, c=summary[column], s=30 + 300 * summary.coverage, cmap='viridis')
        for _, r in summary.iterrows():
            ax.annotate(r.site, (r.lon, r.lat), xytext=(5, 5), textcoords='offset points')
        ax.set(xlabel='Longitude (degrees)', ylabel='Latitude (degrees)', title=title + label)
        ax.set_aspect(1 / max(.1, np.cos(np.deg2rad(summary.lat.mean()))))
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        km = max(1, round((xmax - xmin) * 111 * np.cos(np.deg2rad(summary.lat.mean())) / 4))
        x, y = xmin + .1 * (xmax - xmin), ymin + .08 * (ymax - ymin)
        ax.plot([x, x + km / (111 * np.cos(np.deg2rad(summary.lat.mean())))], [y, y], color='black', lw=3)
        ax.text(x, y, f'{km} km', va='bottom')
        fig.colorbar(plot, ax=ax, label=label)
        fig.tight_layout()
        fig.savefig(out / filename, dpi=120)
        plt.close(fig)
    scatter('median_uncalibrated_export_db', 'Total sound: uncalibrated export dB (common bands)', 'station_map.png')
    if model:
        scatter('inferred_added_vessel_like_share', 'Model-inferred added-vessel-like share — NOT measured', 'inferred_share_map.png')
    for value, filename, label in [('uncalibrated_export_db', 'daily_levels.png', 'uncalibrated export dB'),
                                     ('relative_db_own_median', 'daily_relative.png', "dB re station's own median")]:
        daily = df.pivot_table(index='site', columns='day', values=value, aggfunc='median').reindex(summary.site)
        fig, ax = plt.subplots(figsize=(10, 5))
        im = ax.imshow(daily.to_numpy(), aspect='auto', cmap='viridis')
        ax.set_yticks(range(len(daily)), daily.index)
        ax.set_xticks(range(len(daily.columns)), [str(x.date()) for x in daily.columns], rotation=30)
        ax.set_title(title + 'Daily median total sound: ' + label)
        fig.colorbar(im, ax=ax, label=label)
        fig.tight_layout()
        fig.savefig(out / filename, dpi=120)
        plt.close(fig)
    summary.to_csv(out / 'station_summary.csv', index=False)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--release', required=True)
    p.add_argument('--out-dir', required=True)
    p.add_argument('--model')
    p.add_argument('--lon-col')
    p.add_argument('--lat-col')
    p.add_argument('--station-site-col')
    for col in ['site', 'time', 'dur']:
        p.add_argument(f'--{col}-col')
    p.add_argument('--drop-bad-flags', action='store_true')
    p.add_argument('--flag-cols', nargs='+')
    pressure_map(**vars(p.parse_args()))


if __name__ == '__main__':
    main()
