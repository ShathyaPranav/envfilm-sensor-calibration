"""Reproduces fig_trim_sweep.pdf: trimmed_std/std vs. trim fraction tau,
one line per mechanism, SKAB, at Table I's fixed operating point. Shows
the optimal tau is not stable across mechanisms: monotonically improving
for amplitude/out_of_order, non-monotonic with a peak near tau=0.10 for
missing, and never beating 1.0 for stale_run.

Reads results/trim_sweep/table1_by_tau.csv (produced by
experiments/run_trim_sweep.py). Does not re-run the sweep.
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# DQ_CONSISTENCY=1 reads the sweep rerun with the consistency-scaled trimmed std,
# which is what the paper now reports.
CSV = os.path.join(_REPO_ROOT, 'results', 'trim_sweep' + ('_consistent' if os.environ.get('DQ_CONSISTENCY', '0') == '1' else ''), 'table1_by_tau.csv')
OUT_PDF = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fig_trim_sweep.pdf')
OUT_PNG = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fig_trim_sweep.png')

MECH_ORDER = ['outlier', 'spike_burst', 'out_of_order', 'duplicate',
              'sensor_dropout_burst', 'stale_run', 'missing']
STRUCTURAL = {'missing', 'duplicate', 'out_of_order', 'sensor_dropout_burst', 'stale_run'}


def main():
    df = pd.read_csv(CSV)
    df = df[df.domain == 'SKAB']

    fig, ax = plt.subplots(figsize=(4.6, 3.1))
    cmap = plt.get_cmap('tab10')
    for i, mech in enumerate(MECH_ORDER):
        g = df[df.mechanism == mech].sort_values('tau')
        style = '--' if mech in STRUCTURAL else '-'
        ax.plot(g['tau'] * 100, g['trimmed_std_over_std'], style, marker='o', markersize=3,
                linewidth=1.3, color=cmap(i), label=mech.replace('_', ' '))

    ax.axhline(1.0, color='black', linewidth=0.8, linestyle=':')
    ax.set_yscale('log')
    ax.set_xlabel(r'Trim fraction $\tau$ (%)', fontsize=9)
    ax.set_ylabel('trimmed_std / std', fontsize=8.5)
    ax.set_title('Optimal trim level is not stable\n(SKAB, fixed operating point)', fontsize=9)
    ax.tick_params(labelsize=7.5)
    ax.legend(fontsize=6.5, loc='center left', bbox_to_anchor=(1.02, 0.5), ncol=1, framealpha=0.9)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    plt.tight_layout()
    plt.savefig(OUT_PDF)
    plt.savefig(OUT_PNG, dpi=320)
    print(f"Wrote {OUT_PDF}")


if __name__ == '__main__':
    main()
