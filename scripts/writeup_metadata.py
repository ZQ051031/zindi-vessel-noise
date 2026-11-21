"""Generate LaTeX values only from committed metrics and dependency pins."""
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    metrics = json.loads((root / 'writeup/tables/metrics_summary.json').read_text())
    values = {
        'HeldoutScore': f"{metrics['heldout']['leaderboard_score']:.5f}",
        'HeldoutAUC': f"{metrics['heldout']['auc']:.5f}",
        'HeldoutBaseline': f"{metrics['heldout']['baseline_score']:.5f}",
        'HeldoutCILow': f"{metrics['heldout']['site_cluster_ci'][0]:.5f}",
        'HeldoutCIHigh': f"{metrics['heldout']['site_cluster_ci'][1]:.5f}",
        'TrainSeconds': f"{metrics['training_seconds']:.2f}",
        'DemoSeconds': f"{metrics['heldout']['demo_seconds']:.2f}",
        'FigureSeconds': f"{metrics['total_seconds']:.2f}",
        'PeakMemory': f"{metrics['peak_rss_mib']:.2f}",
        'ModelBytes': str(metrics['model_bytes']),
        'InferenceSeconds': f"{metrics['inference_seconds']:.3f}",
        'InferenceMemory': f"{metrics['inference_peak_process_rss_mib']:.1f}",
        'HardwareCPUs': str(metrics['hardware_vcpus']),
        'HeldoutCount': str(sum(v['n'] for v in metrics['heldout']['by_share_bin'].values())),
    }
    lines = [f'\\newcommand{{\\{key}}}{{{value}}}' for key, value in values.items()]
    (root / 'writeup/tables/metadata.tex').write_text('\n'.join(lines) + '\n')
    pins = (root / 'requirements.txt').read_text().splitlines()
    text = '; '.join(pin.replace('==', ' ') for pin in pins if pin)
    (root / 'writeup/tables/dependencies.tex').write_text(text + '.\n')


if __name__ == '__main__':
    main()
