"""Reproduces the paper's central figure (fig_contamination_flip.pdf):
per-segment UCI spread-error ratio (robust MADn / naive std) under the two
contamination models, side by side, on identical UCI data.

Both panels use the geometric mean of per-segment ratios as the summary
marker, matching Section III-B's stated pooling rule ("pooling across cells
uses the geometric mean of log-ratios"). Phase 1B originally reported the
isolated-incident side as an arithmetic mean (ratio 1.94); that number is
footnoted in the paper as the figure originally reported, not used as the
primary comparison.

Left panel (isolated-incident model): the 10 per-segment ratios from
results/phase1b/pooled_per_segment.csv (produced by
experiments/run_phase1b.py).

Right panel (ambient model): computed here from
results/phase2a_full/phase2a_cells.csv (produced by
experiments/run_phase2a_full.py) -- for each UCI run (segment), the
geometric mean of spr_mae_madn/spr_mae_std pooled across all 23 regimes and
4 channels for that run.

Both panels' 95% CIs are a segment-clustered percentile bootstrap (resample
the per-segment values with replacement, recompute the geometric mean,
10,000 resamples, seed 0; report the [2.5, 97.5] percentiles of the
resampled geometric means).

This script does not re-run either experiment; both results/ files must
already exist (see README.md for the two experiments that produce them).
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PHASE1B_CSV = os.path.join(_REPO_ROOT, 'results', 'phase1b', 'pooled_per_segment.csv')
CELLS_CSV = os.path.join(_REPO_ROOT, 'results', 'phase2a_full', 'phase2a_cells.csv')
OUT_PDF = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fig_contamination_flip.pdf')
OUT_PNG = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fig_contamination_flip.png')

N_BOOT = 10_000


def geomean(x):
    x = np.asarray(x, dtype=float)
    x = x[x > 0]
    return np.exp(np.mean(np.log(x)))


def segment_clustered_bootstrap_ci(values, seed=0, n_boot=N_BOOT):
    """Percentile bootstrap CI on the geometric mean, resampling segments
    with replacement -- the same procedure used throughout the paper for
    segment/run-clustered CIs."""
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(values, size=len(values), replace=True)
        boot[i] = geomean(sample)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return lo, hi


def main():
    isolated_df = pd.read_csv(PHASE1B_CSV)
    isolated = isolated_df['ratio_mae_spr'].to_numpy(dtype=float)
    isolated_arith_mean = isolated.mean()  # originally reported by Phase 1B; footnoted in the paper
    isolated_summary = geomean(isolated)  # primary comparison: matches the paper's stated pooling rule
    isolated_lo, isolated_hi = segment_clustered_bootstrap_ci(isolated)

    cells = pd.read_csv(CELLS_CSV)
    uci = cells[cells.domain == 'UCI']
    per_run = uci.groupby('run').apply(
        lambda g: geomean(g['spr_mae_madn'] / g['spr_mae_std']), include_groups=False)
    ambient = per_run.to_numpy(dtype=float)
    ambient_summary = geomean(ambient)  # geometric mean, matches Phase 2A's pooling rule
    ambient_lo, ambient_hi = segment_clustered_bootstrap_ci(ambient)

    print(f"isolated-incident: n={len(isolated)}, geomean={isolated_summary:.4f}, "
          f"95% CI [{isolated_lo:.4f}, {isolated_hi:.4f}] "
          f"(originally reported arithmetic mean: {isolated_arith_mean:.4f})")
    print(f"ambient:           n={len(ambient)}, geomean={ambient_summary:.4f}, "
          f"95% CI [{ambient_lo:.4f}, {ambient_hi:.4f}]")

    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    rng = np.random.default_rng(0)
    groups = [('Isolated-incident', isolated, isolated_summary, 'tab:red'),
              ('Ambient (5%)', ambient, ambient_summary, 'tab:blue')]
    for i, (label, vals, summary, color) in enumerate(groups):
        x = i + 1 + rng.uniform(-0.08, 0.08, size=len(vals))
        ax.scatter(x, vals, color=color, alpha=0.75, s=28, zorder=3, edgecolor='none')
        ax.hlines(summary, i + 1 - 0.22, i + 1 + 0.22, color=color, linewidth=2.4, zorder=4)
        ax.text(i + 1, summary * 1.16, f'{summary:.2f}', ha='center', fontsize=9,
                fontweight='bold', color=color)

    ax.axhline(1.0, color='black', linewidth=0.9, linestyle='--', zorder=1)
    ax.text(2.42, 1.0, 'parity', va='center', fontsize=7.5, color='black')
    ax.set_yscale('log')
    ax.set_xlim(0.5, 2.5)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['Isolated-incident', 'Ambient (5%)'], fontsize=9)
    ax.set_ylabel('Spread error ratio\n(robust MADn / naive std)', fontsize=9)
    ax.set_title('Same UCI data, same metric,\ndifferent contamination model', fontsize=9.5)
    ax.tick_params(axis='y', labelsize=8)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    ax.grid(axis='y', which='major', linestyle=':', linewidth=0.5, alpha=0.6)

    plt.tight_layout()
    plt.savefig(OUT_PDF)
    plt.savefig(OUT_PNG, dpi=320)
    print(f"Wrote {OUT_PDF}")


if __name__ == '__main__':
    main()
