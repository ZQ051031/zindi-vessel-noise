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
from .features import GROUPS, dbsum, window_features
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
        window_daily = meta.assign(day=meta.t.dt.floor('D'), inferred_share=share,
                                   excess_energy=window_features(cube).excess.to_numpy())
        for value in ['inferred_share', 'excess_energy']:
            daily = window_daily.pivot_table(index='site', columns='day', values=value, aggfunc='mean')
            daily.to_csv(out / f'daily_{value}.csv')
            fig, ax = plt.subplots(figsize=(10, 5))
            im = ax.imshow(daily, aspect='auto', vmin=0, vmax=1, cmap='viridis')
            ax.set_yticks(range(len(daily)), daily.index)
            ax.set_xticks(range(len(daily.columns)), [str(d.date()) for d in daily.columns])
            ax.set_title(title + 'Daily mean ' + value.replace('_', ' ') + ' — NOT measured vessel share')
            fig.colorbar(im, ax=ax, label='Window fraction proxy')
            fig.tight_layout()
            fig.savefig(out / f'daily_{value}.png', dpi=150)
            plt.close(fig)
    def scatter(column, label, filename):
        fig, ax = plt.subplots(figsize=(9, 6))
        size = lambda coverage: 30 + 500 * np.asarray(coverage) ** 2
        plot = ax.scatter(summary.lon, summary.lat, c=summary[column], s=size(summary.coverage), cmap='viridis')
        for _, r in summary.iterrows():
            ax.annotate(f'{r.site}\n{100 * r.coverage:.1f}% coverage', (r.lon, r.lat),
                        xytext=(10, 8), textcoords='offset points', fontsize=8, annotation_clip=False)
        ax.set(xlabel='Longitude (degrees)', ylabel='Latitude (degrees)', title=title + label)
        ax.set_aspect(1 / max(.1, np.cos(np.deg2rad(summary.lat.mean()))))
        dx = max(.1, summary.lon.max() - summary.lon.min())
        dy = max(.1, summary.lat.max() - summary.lat.min())
        ax.set_xlim(summary.lon.min() - .25 * dx, summary.lon.max() + .65 * dx)
        ax.set_ylim(summary.lat.min() - .6 * dy, summary.lat.max() + .4 * dy)
        handles = [ax.scatter([], [], s=size(c), color='gray', label=f'{c:.0%}') for c in [.25, .5, 1.]]
        ax.legend(handles=handles, title='Minute coverage', loc='upper left', framealpha=1, labelspacing=1.2)
        km = max(1, round(dx * 111 * np.cos(np.deg2rad(summary.lat.mean())) / 4))
        # A dedicated inset reserves a corner below all stations, including its opaque background.
        scale = ax.inset_axes([.58, .03, .36, .12], facecolor='white')
        scale.set_xlim(0, 1)
        scale.set_ylim(0, 1)
        span = ax.get_xlim()[1] - ax.get_xlim()[0]
        fraction = km / (111 * np.cos(np.deg2rad(summary.lat.mean()))) / span / .36
        scale.plot([.08, .08 + fraction], [.35, .35], color='black', lw=3)
        scale.text(.08, .55, f'{km} km', fontsize=9)
        scale.set_xticks([])
        scale.set_yticks([])
        fig.colorbar(plot, ax=ax, label=label)
        fig.tight_layout()
        fig.savefig(out / filename, dpi=150)
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
        fig.savefig(out / filename, dpi=150)
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
