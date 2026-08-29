"""Reproduces fig_window_sweep.pdf: for each window size W, plots
trimmed_std/std minus IQR_scale/std across the outlier severity grid
(SKAB anomaly-free). Negative = trimmed wins; positive = IQR wins. Shows
the Table-II crossover (trimmed wins low severity, IQR wins high
severity) collapsing to "trimmed wins everywhere" as W grows.

Reads results/window_sweep/table2_by_window.csv (produced by
experiments/run_window_sweep.py). Does not re-run the sweep.
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(_REPO_ROOT, 'results', 'window_sweep', 'table2_by_window.csv')
OUT_PDF = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fig_window_sweep.pdf')
OUT_PNG = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fig_window_sweep.png')


def main():
    df = pd.read_csv(CSV)
    df['gap'] = df['trimmed_std_over_std'] - df['iqr_over_std']

    fig, ax = plt.subplots(figsize=(3.6, 3.0))
    colors = {15: 'tab:blue', 30: 'tab:orange', 60: 'tab:green', 120: 'tab:red'}
    for w, g in df.groupby('w'):
        g = g.sort_values('magnitude')
        ax.plot(g['magnitude'], g['gap'], marker='o', markersize=3.5, linewidth=1.4,
                color=colors[w], label=f'W={w}')

    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.text(0.3, 0.002, 'IQR beats trimmed', fontsize=7, color='dimgray', va='bottom')
    ax.text(0.3, -0.002, 'trimmed beats IQR', fontsize=7, color='dimgray', va='top')
    ax.set_xscale('log')
    ax.set_xticks([0.25, 0.5, 1, 2, 4, 8])
    ax.set_xticklabels(['0.25', '0.5', '1', '2', '4', '8'])
    ax.set_xlabel(r'Severity ($\sigma$)', fontsize=9)
    ax.set_ylabel('trimmed_std/std $-$ IQR/std', fontsize=8.5)
    ax.set_title('Crossover collapses as W grows\n(SKAB anomaly-free, outlier)', fontsize=9)
    ax.tick_params(labelsize=7.5)
    ax.legend(fontsize=7, loc='upper right')
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    plt.tight_layout()
    plt.savefig(OUT_PDF)
    plt.savefig(OUT_PNG, dpi=320)
    print(f"Wrote {OUT_PDF}")


if __name__ == '__main__':
    main()
