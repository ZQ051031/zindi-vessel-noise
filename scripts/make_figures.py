"""Regenerate committed SYNTHETIC figures/tables: .venv/bin/python scripts/make_figures.py."""
from pathlib import Path
import json
import resource
import shutil
import subprocess
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score
from src.explain import explain, SYNTHETIC_TITLE
from src.ablation import ablate
from src.evaluate import bootstrap_ci
from src.synthetic import synthetic_background, add_controlled_passages
from src.schema import BAND_CENTRES


def main():
    start = time.perf_counter()
    root = Path(__file__).resolve().parents[1]
    work = root / 'outputs/synthetic_writeup'
    figures, tables = root / 'writeup/figures', root / 'writeup/tables'
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, '-m', 'src.run_synthetic_demo', '--out-dir', str(work),
                    '--days', '2', '--n-test', '600', '--augment-k', '4', '--seed', '42'], cwd=root, check=True)
    metrics = json.loads((work / 'models/metrics.json').read_text())
    heldout = json.loads((work / 'SYNTHETIC_evaluation.json').read_text())
    oof = pd.read_parquet(work / 'models/SYNTHETIC_addition_oof.parquet')
    x = pd.read_parquet(work / 'models/SYNTHETIC_addition_features.parquet')
    explanations = explain(work / 'models/bundle.joblib', x, work / 'explanations', synthetic=True)
    ablations = ablate(x, oof, tables)
    for file in (work / 'maps').glob('*.png'):
        shutil.copy(file, figures / file.name)
    for file in (work / 'maps').glob('*.csv'):
        shutil.copy(file, tables / file.name)
    for file in (work / 'explanations').glob('*.png'):
        shutil.copy(file, figures / file.name)
    for name, table in explanations.items():
        table.to_csv(tables / f'{name}.csv', index=False)
    def save(fig, name):
        fig.suptitle(SYNTHETIC_TITLE.strip(), fontsize=11)
        fig.tight_layout()
        fig.savefig(figures / name, dpi=150)
        plt.close(fig)
    rng = np.random.default_rng(2026)
    found = {}
    while len(found) < 3:
        bg = synthetic_background(1, rng)[0]
        mixture, share, presence, kind = add_controlled_passages(bg, rng)
        if kind in ['none', 'small_craft', 'large_ship'] and kind not in found:
            found[kind] = (bg, mixture, share)
    for kind, (bg, mixture, share) in found.items():
        added = np.maximum(10 ** (mixture / 10) - 10 ** (bg.astype(float) / 10), 0)
        added_db = 10 * np.log10(np.maximum(added, 1e-12))
        if kind == 'none':
            added_db[:] = np.nan
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
        for ax, values, name in zip(axes, [bg, mixture, added_db], ['Background', 'Mixture', 'Added reference']):
            im = ax.imshow(values.T, origin='lower', aspect='auto', vmin=40, vmax=95)
            ax.set(title=f'{name} ({kind}, share={share:.2f})', xlabel='Minute position')
            ax.set_xticks([0, 7, 14], [1, 8, 15])
            ax.set_yticks([0, 5, 10, 15, 20, 24], [f'{BAND_CENTRES[i]:.0f}' for i in [0, 5, 10, 15, 20, 24]])
            fig.colorbar(im, ax=ax, label='Synthetic export dB')
        axes[0].set_ylabel('Band centre (Hz)')
        if kind == 'none':
            axes[2].text(.5, .5, 'No added energy', transform=axes[2].transAxes, ha='center')
        save(fig, f'example_{kind}.png')
    comparison = []
    for name, record in metrics['oof'].items():
        ci = record['background_window_id_cluster_ci']
        comparison.append({'model': name, 'score': record['score'], 'rmse': record['rmse'], 'cluster_ci_low': ci[0], 'cluster_ci_high': ci[1]})
    comparison = pd.DataFrame(comparison)
    comparison.to_csv(tables / 'oof_comparison.csv', index=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(comparison.model, comparison.score)
    ax.errorbar(comparison.model, comparison.score, yerr=[comparison.score - comparison.cluster_ci_low, comparison.cluster_ci_high - comparison.score], fmt='none', color='black', capsize=5)
    ax.set(ylabel='1 − RMSE (background-window cluster 95% CI)', ylim=(.5, 1), title='Site-held-out OOF; ensemble weights selected on OOF')
    save(fig, 'oof_model_comparison.png')
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
    for ax, name in zip(axes, ['baseline', 'ensemble']):
        ax.scatter(oof.share_added_vessel, oof[name], s=5, alpha=.25)
        ax.plot([0, 1], [0, 1], color='black')
        ax.set(title=name, xlabel='True added fraction', ylabel='Predicted fraction')
    save(fig, 'predicted_vs_true.png')
    bins = []
    for lo, hi in [(0, .1), (.1, .3), (.3, .6), (.6, 1.01)]:
        part = oof[(oof.share_added_vessel >= lo) & (oof.share_added_vessel < hi)]
        for model in ['baseline', 'ensemble']:
            ci = bootstrap_ci(part.share_added_vessel, part[model], part.background_window_id)
            bins.append({'bin': f'{lo}–{min(hi, 1)}', 'model': model, 'n': len(part),
                         'rmse': float(np.sqrt(np.mean((part.share_added_vessel - part[model]) ** 2))),
                         'rmse_ci_low': 1 - ci[1], 'rmse_ci_high': 1 - ci[0]})
    bin_table = pd.DataFrame(bins)
    bin_table.to_csv(tables / 'error_by_share_bin.csv', index=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    for model in ['baseline', 'ensemble']:
        part = bin_table[bin_table.model == model]
        ax.errorbar(part['bin'], part.rmse, yerr=[part.rmse - part.rmse_ci_low, part.rmse_ci_high - part.rmse], marker='o', capsize=4, label=model)
    ax.set(xlabel='True share bin', ylabel='RMSE (background cluster CI)', title='Error by true-share bin')
    ax.legend()
    save(fig, 'error_by_share_bin.png')
    fig, ax = plt.subplots(figsize=(6, 5))
    fpr, tpr, _ = roc_curve(oof.presence, oof.score_added_vessel)
    ax.plot(fpr, tpr, label=f'OOF AUC={roc_auc_score(oof.presence, oof.score_added_vessel):.3f}')
    ax.plot([0, 1], [0, 1], '--', color='gray')
    ax.set(xlabel='False positive rate', ylabel='True positive rate', title='Controlled-addition diagnostic ROC')
    ax.legend()
    save(fig, 'presence_roc.png')
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('off')
    display = ablations[['control', 'n_features', 'score', 'cluster_ci_low', 'cluster_ci_high']].copy()
    for c in ['score', 'cluster_ci_low', 'cluster_ci_high']:
        display[c] = display[c].map(lambda v: f'{v:.4f}')
    ax.table(cellText=display.values, colLabels=display.columns, loc='center', cellLoc='center')
    ax.set_title('Site-held-out ablations (LightGBM except isotonic controls)')
    save(fig, 'ablation_table.png')
    for name, table in [('oof_comparison', comparison), ('ablation', ablations), ('error_by_share_bin', bin_table)]:
        headers = ' & '.join(c.replace('_', r'\_') for c in table.columns)
        lines = [r'\begin{tabular}{' + 'l' * len(table.columns) + '}', r'\toprule', headers + r' \\', r'\midrule']
        for row in table.itertuples(index=False, name=None):
            lines.append(' & '.join((f'{v:.4f}' if isinstance(v, float) else str(v).replace('_', r'\_')) for v in row) + r' \\')
        lines += [r'\bottomrule', r'\end{tabular}']
        (tables / f'{name}.tex').write_text('\n'.join(lines) + '\n')
    summary = {'SYNTHETIC': True, 'seed': 42, 'oof': metrics['oof'], 'heldout': heldout,
               'ablations': ablations.to_dict('records'), 'training_seconds': metrics['training_seconds'],
               'model_bytes': metrics['model_bytes'], 'total_seconds': time.perf_counter() - start,
               'peak_rss_mib': max(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss) / 1024,
               'note': 'Parametric synthetic references only; no claim of real vessel detection. OOF ensemble weights selected on OOF.'}
    (tables / 'metrics_summary.json').write_text(json.dumps(summary, indent=2))
    print('ABLATIONS\n' + ablations.to_string(index=False))
    print('FIGURE GENERATION', json.dumps({k: summary[k] for k in ['total_seconds', 'peak_rss_mib', 'model_bytes']}))


if __name__ == '__main__':
    main()
