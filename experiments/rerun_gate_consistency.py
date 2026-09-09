"""Re-run RQ3's headroom gate with the consistency-corrected trimmed_std.

Why this is necessary: the gate's decisive criterion (ii) is "oracle must beat
the best fixed INTERMEDIATE estimator by >=15%". The intermediate spread pool
is {trimmed_std, IQR_scale}. Uncorrected, trimmed_std carried a pure scale
advantage under the absolute error metric, so it could be selected as "best
intermediate" for reasons unrelated to robustness, which moves the denominator
of the decisive criterion. The gate's numeric thresholds (25% / 15%) stay
exactly as pre-specified -- only the estimator bug is fixed.

Location results MUST be unchanged (trimmed_mean needs no consistency factor);
that is used here as a built-in correctness check on the whole pipeline.

Uncorrected side is read from the existing results/phase2a_full/phase2a_cells.csv
rather than recomputed, so the comparison is against exactly what the paper
reports.
"""
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore', category=RuntimeWarning)

_EXP_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_EXP_DIR)
for p in (_EXP_DIR, _REPO_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from run_phase2a_full import load_series, build_regime_grid
from rerun_tables_consistency import sweep
import phase2a_full_analysis as pfa

OUT_DIR = os.path.join(_REPO_ROOT, 'results', 'consistency')
OLD_CELLS = os.path.join(_REPO_ROOT, 'results', 'phase2a_full', 'phase2a_cells.csv')


def summarize(results, tag):
    rows = []
    for (statistic, split, held_out, scope), res in results.items():
        s = res['summary']
        rows.append(dict(setting=tag, statistic=statistic, split=split, held_out=held_out,
                         scope=scope, n_units=s['n_units'],
                         best_fixed=s['best_fixed'], best_intermediate=s['best_intermediate'],
                         headroom_fixed=100 * s['headroom_fixed_mean'],
                         headroom_intermediate=100 * s['headroom_intermediate_mean']))
    return pd.DataFrame(rows)


def report(df, tag):
    print(f"\n{'='*78}\n=== HEADROOM GATE ({tag}) ===")
    for statistic in ('location', 'spread'):
        d = df[(df.statistic == statistic) & (df.split == 'leave_one_mechanism_out') &
               (df.scope == 'combined')]
        if d.empty:
            continue
        hf, hi = d['headroom_fixed'].mean(), d['headroom_intermediate'].mean()
        verdict = 'Pass' if (hf >= 25.0 and hi >= 15.0) else 'Reject'
        print(f"  PRIMARY (leave-one-mechanism-out, {len(d)} folds)  {statistic:9s} "
              f"vs best fixed {hf:6.1f}%   vs best intermediate {hi:6.1f}%   -> {verdict}")
        print(f"      best_fixed={sorted(d['best_fixed'].unique())}  "
              f"best_intermediate={sorted(d['best_intermediate'].unique())}")
    print("  --- generalization checks (spread) ---")
    for split, scope in [('unseen_severity', 'combined'), ('unseen_severity', 'UCI_only'),
                         ('leave_one_domain_out', 'cross_domain')]:
        d = df[(df.statistic == 'spread') & (df.split == split) & (df.scope == scope)]
        for _, r in d.iterrows():
            print(f"      {split:22s} {scope:11s} {r['held_out']:22s} "
                  f"vs fixed {r['headroom_fixed']:6.1f}%  vs intermediate {r['headroom_intermediate']:6.1f}%")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.time()

    print("=== uncorrected: reading the paper's existing cells table ===")
    old = pd.read_csv(OLD_CELLS)
    print(f"  {len(old)} cells")
    res_old = pfa.run_all_splits(old)
    df_old = summarize(res_old, 'uncorrected')
    report(df_old, 'uncorrected -- reproduces the paper')

    print(f"\n=== corrected: re-sweeping all regimes with consistency=True "
          f"(the expensive part) ===")
    series_list = load_series()
    regimes = build_regime_grid()
    new = sweep(series_list, regimes, consistency=True)
    new.to_csv(os.path.join(OUT_DIR, 'phase2a_cells_corrected.csv'), index=False)
    print(f"  {len(new)} cells, elapsed {time.time()-t0:.0f}s")
    res_new = pfa.run_all_splits(new)
    df_new = summarize(res_new, 'corrected')
    report(df_new, 'corrected')

    out = pd.concat([df_old, df_new], ignore_index=True)
    out.to_csv(os.path.join(OUT_DIR, 'gate_comparison.csv'), index=False)

    # built-in check: location must not move
    lo = df_old[df_old.statistic == 'location'].set_index(['split', 'held_out', 'scope'])
    ln = df_new[df_new.statistic == 'location'].set_index(['split', 'held_out', 'scope'])
    common = lo.index.intersection(ln.index)
    delta = (lo.loc[common, 'headroom_fixed'] - ln.loc[common, 'headroom_fixed']).abs().max()
    print(f"\n[check] max |location headroom change| = {delta:.6f}%  "
          f"({'OK, location is unaffected as expected' if delta < 1e-6 else 'UNEXPECTED -- investigate'})")
    print(f"\nTotal elapsed {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()
