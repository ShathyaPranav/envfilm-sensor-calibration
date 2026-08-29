"""Reproduces fig_perturbation_magnitude.pdf: log(ratio) vs. log(normalized
perturbation), SKAB and UCI side by side, points colored by defect family
(structural vs. amplitude), with the naive-pooled fitted line per domain.

Reads results/perturbation/perturbation_valid.csv (produced by
experiments/run_perturbation_analysis.py). Does not re-run the sweep.
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as spstats

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VALID_CSV = os.path.join(_REPO_ROOT, 'results', 'perturbation', 'perturbation_valid.csv')
OUT_PDF = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fig_perturbation_magnitude.pdf')
OUT_PNG = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fig_perturbation_magnitude.png')


def main():
    df = pd.read_csv(VALID_CSV)
    df['logx'] = np.log(df['norm_pert'])
    df['logy'] = np.log(df['ratio'])

    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.1))
    colors = {'structural': 'tab:red', 'amplitude': 'tab:blue'}
    for ax, domain in zip(axes, ['SKAB', 'UCI']):
        sub = df[df.domain == domain]
        r = spstats.linregress(sub['logx'], sub['logy'])
        for fam, c in colors.items():
            g = sub[sub.family == fam]
            ax.scatter(g['logx'], g['logy'], s=3, alpha=0.2, color=c, label=fam, edgecolor='none')
        xs = np.linspace(sub['logx'].min(), sub['logx'].max(), 50)
        ax.plot(xs, r.intercept + r.slope * xs, color='black', linewidth=1.3,
                label=f'fit (slope={r.slope:.2f})')
        ax.axhline(0, color='gray', linewidth=0.7, linestyle=':')
        ax.set_title(domain, fontsize=10)
        ax.set_xlabel('log(normalized perturbation)', fontsize=8.5)
        ax.tick_params(labelsize=7.5)
        if domain == 'SKAB':
            ax.set_ylabel('log(ratio, MADn/std)', fontsize=8.5)
        ax.legend(fontsize=6.5, loc='upper right', markerscale=2)

    plt.tight_layout()
    plt.savefig(OUT_PDF)
    plt.savefig(OUT_PNG, dpi=320)
    print(f"Wrote {OUT_PDF}")


if __name__ == '__main__':
    main()
