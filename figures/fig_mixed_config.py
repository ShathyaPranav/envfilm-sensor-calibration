"""Reproduces the paper's second figure (fig_mixed_config.pdf): frequency
of "mixed configuration" regimes (oracle pairs a robust location expert
with a non-robust spread expert, or vice versa), by mechanism and domain.

Reads results/phase2a_full/phase2a_report.json's
`mixed_configuration_frequency` field (produced by
experiments/run_phase2a_full_report.py, which calls
phase2a_full_analysis.mixed_configuration_frequency()). Does not
recompute anything -- pure plotting from an already-computed result.
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# DQ_CONSISTENCY=1 plots the report recomputed with the consistency-scaled trimmed
# standard deviation (results/consistency/), which is what the paper now uses.
REPORT_JSON = (os.path.join(_REPO_ROOT, 'results', 'consistency', 'phase2a_report_corrected.json')
               if os.environ.get('DQ_CONSISTENCY', '0') == '1'
               else os.path.join(_REPO_ROOT, 'results', 'phase2a_full', 'phase2a_report.json'))
OUT_PDF = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fig_mixed_config.pdf')
OUT_PNG = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fig_mixed_config.png')

# Mechanism display order/labels matching the paper's figure.
MECH_ORDER = ['duplicate', 'missing', 'out_of_order', 'sensor_dropout_burst',
              'stale_run', 'outlier', 'spike_burst']
MECH_LABELS = ['duplicate', 'missing', 'out_of_order', 'sensor_\ndropout',
               'stale_run', 'outlier', 'spike_burst']


def main():
    with open(REPORT_JSON) as f:
        report = json.load(f)
    mixed = report['mixed_configuration_frequency']

    skab_overall = mixed['SKAB']['overall_fraction_mixed'] * 100
    uci_overall = mixed['UCI']['overall_fraction_mixed'] * 100
    skab_per_mech = mixed['SKAB']['per_mechanism']
    uci_per_mech = mixed['UCI']['per_mechanism']

    skab = [skab_per_mech.get(m, 0.0) * 100 for m in MECH_ORDER]
    uci = [uci_per_mech.get(m, 0.0) * 100 for m in MECH_ORDER]
    print(f"SKAB overall: {skab_overall:.1f}%   UCI overall: {uci_overall:.1f}%")
    for m, s, u in zip(MECH_ORDER, skab, uci):
        print(f"  {m}: SKAB={s:.1f}%  UCI={u:.1f}%")

    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    x = np.arange(len(MECH_ORDER))
    w = 0.36
    ax.bar(x - w / 2, skab, width=w, label=f'SKAB ({skab_overall:.1f}% overall)', color='tab:orange')
    ax.bar(x + w / 2, uci, width=w, label=f'UCI ({uci_overall:.1f}% overall)', color='tab:green')
    ax.set_xticks(x)
    ax.set_xticklabels(MECH_LABELS, fontsize=6.5, rotation=35, ha='right')
    ax.set_ylabel('Mixed-config.\nregimes (%)', fontsize=9)
    ax.set_title('Robust-location + non-robust-spread\n(or vice versa): by mechanism', fontsize=8.8)
    ax.legend(fontsize=6.5, loc='upper right', framealpha=0.9)
    ax.tick_params(axis='y', labelsize=8)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    ax.set_ylim(0, 112)
    plt.tight_layout()
    plt.savefig(OUT_PDF)
    plt.savefig(OUT_PNG, dpi=320)
    print(f"Wrote {OUT_PDF}")


if __name__ == '__main__':
    main()
