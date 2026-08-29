"""Reproduces Table I and Table II from results/phase2a_full/phase2a_cells.csv
(produced by run_phase2a_full.py). Table III (the headroom gate) does not
need a separate script -- it is already written directly by
run_phase2a_full_report.py to results/phase2a_full/gate_verdicts.csv (see
README.md for which rows/columns map to Table III).

Table I: SKAB, ambient contamination (5%), at the fixed operating point
used in the paper (amplitude defects at 4-sigma, burst defects at length
15, pointwise defects at length 1). Geometric-mean ratios of robust to
naive error per mechanism, plus the two interior spread estimators
(trimmed_std, IQR_scale) over std, added to check whether a fixed interior
estimator beats both endpoints on every mechanism or only on the amplitude
ones already covered by Table II's severity sweep.

Table II: SKAB anomaly-free background only, ambient contamination (5%),
amplitude (outlier) defects, across the full magnitude sweep. Error ratios
relative to the standard deviation.

Does not re-run the sweep -- reads the already-computed per-cell table only.
"""
import os
import sys

import numpy as np
import pandas as pd

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

CELLS_CSV = os.path.join(_REPO_ROOT, 'results', 'phase2a_full', 'phase2a_cells.csv')
OUT_DIR = os.path.join(_REPO_ROOT, 'results', 'phase2a_full')

# Table I's fixed operating point, exactly as stated in the paper's caption.
TABLE1_MECH_SEV = {
    'missing': None, 'duplicate': None, 'out_of_order': None,  # pointwise, no severity axis
    'sensor_dropout_burst': ('burst_len', 15),
    'stale_run': ('burst_len', 15),
    'outlier': ('magnitude', 4),
    'spike_burst': ('magnitude', 4),
}


def geomean(x):
    x = np.asarray(x, dtype=float)
    x = x[x > 0]
    return np.exp(np.mean(np.log(x))) if x.size else np.nan


def table1(cells):
    skab = cells[cells.domain == 'SKAB']
    rows = []
    for mech, sev in TABLE1_MECH_SEV.items():
        g = skab[skab.mechanism == mech]
        if sev is not None:
            sev_type, sev_value = sev
            g = g[(g.sev_type == sev_type) & (g.sev_value == sev_value)]
        rows.append(dict(
            defect=mech,
            location=geomean(g['loc_mae_median'] / g['loc_mae_mean']),
            spread=geomean(g['spr_mae_madn'] / g['spr_mae_std']),
            loc_peak=geomean(g['loc_peak_median'] / g['loc_peak_mean']),
            spr_peak=geomean(g['spr_peak_madn'] / g['spr_peak_std']),
            # Interior spread estimators, added to check the cross-mechanism
            # "fixed interior estimator dominates both endpoints" claim
            # (Contribution 1 / Discussion / Conclusion) against mechanisms
            # beyond the amplitude sweep in Table II.
            trimmed_std_over_std=geomean(g['spr_mae_trimmed_std'] / g['spr_mae_std']),
            iqr_over_std=geomean(g['spr_mae_IQR_scale'] / g['spr_mae_std']),
            n_cells=len(g),
        ))
    return pd.DataFrame(rows)


def table2(cells):
    af = cells[(cells.domain == 'SKAB') & (cells.run == 'anomaly-free') & (cells.mechanism == 'outlier')]
    rows = []
    for mag in sorted(af['sev_value'].unique()):
        g = af[af['sev_value'] == mag]
        madn = geomean(g['spr_mae_madn'] / g['spr_mae_std'])
        trim = geomean(g['spr_mae_trimmed_std'] / g['spr_mae_std'])
        iqr = geomean(g['spr_mae_IQR_scale'] / g['spr_mae_std'])
        vals = {'madn': madn, 'trimmed_std': trim, 'IQR_scale': iqr, 'std': 1.0}
        oracle = min(vals, key=vals.get)
        rows.append(dict(magnitude=mag, madn_over_std=madn, trimmed_std_over_std=trim,
                          iqr_over_std=iqr, oracle_pick=oracle, n_channels=len(g)))
    return pd.DataFrame(rows)


def main():
    cells = pd.read_csv(CELLS_CSV)
    print(f"Loaded {len(cells)} cells from {CELLS_CSV}")

    t1 = table1(cells)
    t1.to_csv(os.path.join(OUT_DIR, 'table1.csv'), index=False)
    print("\n=== Table I: SKAB, fixed operating point ===")
    print(t1.to_string(index=False))

    t2 = table2(cells)
    t2.to_csv(os.path.join(OUT_DIR, 'table2.csv'), index=False)
    print("\n=== Table II: SKAB anomaly-free, outlier magnitude sweep ===")
    print(t2.to_string(index=False))

    return t1, t2


if __name__ == '__main__':
    main()
