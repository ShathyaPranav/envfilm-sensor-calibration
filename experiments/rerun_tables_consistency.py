"""Re-run Tables I and II with the consistency-corrected trimmed_std.

Everything else is held fixed: same series, same regimes, same seeds, same
error metric, same geometric-mean pooling. The ONLY change is that
trimmed_std is divided by its Gaussian consistency factor (see
phase2a_ext._trim_consistency), putting it on the same sigma scale as std,
MADn (x1.4826) and IQR_scale (/1.349).

Emits both the uncorrected and corrected numbers side by side so the size of
the scale artifact is visible per cell rather than asserted.
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

from run_phase2a_full import load_series, build_regime_grid, RATE, SEEDS
from harness import phase2a_ext as p2x

OUT_DIR = os.path.join(_REPO_ROOT, 'results', 'consistency')

TABLE1_REGIMES = {
    'missing': 'missing', 'duplicate': 'duplicate', 'out_of_order': 'out_of_order',
    'sensor_dropout_burst': 'sensor_dropout_burst_burst15',
    'stale_run': 'stale_run_burst15',
    'outlier': 'outlier_mag4', 'spike_burst': 'spike_burst_mag4',
}


def geomean(a):
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a) & (a > 0)]
    return float(np.exp(np.mean(np.log(a)))) if a.size else np.nan


def sweep(series_list, regimes, consistency):
    rows = []
    for s in series_list:
        cache = p2x.precompute_clean(s['x'], consistency=consistency)
        for r in regimes:
            kw = {}
            if r['sev_type'] == 'magnitude':
                kw['magnitude'] = r['sev_value']
            elif r['sev_type'] == 'burst_len':
                kw['burst_len'] = r['sev_value']
            per_seed = []
            for seed in SEEDS:
                per_seed.append(p2x.eval_cell_16(s['x'], r['mechanism'], RATE, seed,
                                                  precomputed_clean=cache,
                                                  consistency=consistency, **kw))
            agg = {}
            for key in per_seed[0]:
                v = np.array([p[key] for p in per_seed], dtype=float)
                v = v[np.isfinite(v) & (v > 0)]
                agg[key] = float(np.exp(np.mean(np.log(v)))) if v.size else np.nan
            rows.append(dict(domain=s['domain'], run=s['run'], channel=s['channel'],
                             mechanism=r['mechanism'], sev_type=r['sev_type'],
                             sev_value=r['sev_value'], regime=r['label'], **agg))
    return pd.DataFrame(rows)


def table1(cells):
    out = []
    for mech, regime in TABLE1_REGIMES.items():
        g = cells[(cells.domain == 'SKAB') & (cells.regime == regime)]
        if g.empty:
            continue
        madn = geomean(g['spr_mae_madn'] / g['spr_mae_std'])
        trim = geomean(g['spr_mae_trimmed_std'] / g['spr_mae_std'])
        iqr = geomean(g['spr_mae_IQR_scale'] / g['spr_mae_std'])
        loc = geomean(g['loc_mae_median'] / g['loc_mae_mean'])
        out.append(dict(mechanism=mech, loc_median_over_mean=loc, madn_over_std=madn,
                        trimmed_over_std=trim, iqr_over_std=iqr,
                        interior_beats_both=bool(min(trim, iqr) < min(1.0, madn))))
    return pd.DataFrame(out)


def table2(cells):
    af = cells[(cells.domain == 'SKAB') & (cells.run == 'anomaly-free') & (cells.mechanism == 'outlier')]
    out = []
    for mag in sorted(af['sev_value'].unique()):
        g = af[af['sev_value'] == mag]
        madn = geomean(g['spr_mae_madn'] / g['spr_mae_std'])
        trim = geomean(g['spr_mae_trimmed_std'] / g['spr_mae_std'])
        iqr = geomean(g['spr_mae_IQR_scale'] / g['spr_mae_std'])
        vals = {'madn': madn, 'trimmed_std': trim, 'IQR_scale': iqr, 'std': 1.0}
        out.append(dict(magnitude=mag, madn_over_std=madn, trimmed_over_std=trim,
                        iqr_over_std=iqr, oracle_pick=min(vals, key=vals.get),
                        interior_beats_both=bool(min(trim, iqr) < min(1.0, madn))))
    return pd.DataFrame(out)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.time()
    series_list = load_series()
    regimes = build_regime_grid()
    needed = set(TABLE1_REGIMES.values()) | {r['label'] for r in regimes if r['mechanism'] == 'outlier'}
    regimes = [r for r in regimes if r['label'] in needed]
    print(f"{len(series_list)} series x {len(regimes)} regimes x {len(SEEDS)} seeds, both settings")

    for consistency in (False, True):
        tag = 'corrected' if consistency else 'uncorrected'
        cells = sweep(series_list, regimes, consistency)
        cells.to_csv(os.path.join(OUT_DIR, f'cells_{tag}.csv'), index=False)
        t1, t2 = table1(cells), table2(cells)
        t1.to_csv(os.path.join(OUT_DIR, f'table1_{tag}.csv'), index=False)
        t2.to_csv(os.path.join(OUT_DIR, f'table2_{tag}.csv'), index=False)
        print(f"\n{'='*70}\n=== TABLE I ({tag}) ===")
        print(t1.to_string(index=False, float_format=lambda v: f"{v:8.3f}"))
        print(f"\n=== TABLE II ({tag}) ===")
        print(t2.to_string(index=False, float_format=lambda v: f"{v:8.4f}"))
        print(f"[elapsed {time.time()-t0:.0f}s]")


if __name__ == '__main__':
    main()
