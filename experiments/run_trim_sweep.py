"""Trim-level sensitivity sweep: tau in {0%, 5%, 10%, 20%, 30%} for the
trimmed spread estimator (all other estimators/parameters, including
W=30, held at their paper values), on both SKAB and UCI, at the paper's
existing operating points:

  - Table-I-style: all 7 mechanisms at the fixed severity operating point
    (amplitude at 4-sigma, burst at length 15, pointwise at rate 0.05),
    on SKAB (as in the paper) and, identically, on UCI (not in the paper,
    added here to test "on both datasets" per the sensitivity request).
  - Table-II-style: the full outlier severity grid (0.25-8 sigma), on
    SKAB anomaly-free (as in the paper) and, identically, pooled across
    all 10 UCI segments/channels.

Reports, per (domain, mechanism-or-magnitude) cell, which tau minimizes
trimmed_std/std -- i.e. whether the optimal trim level is stable across
datasets, mechanisms, and severities, or shifts. Uses 5 of the 10 paper
seeds (0-4) rather than all 10: this is a secondary sensitivity check,
not a primary result, and halving the seed count keeps five tau values
tractable. Flagged here, not silently done.

Requires the raw SKAB + UCI datasets in data/ (see README.md).
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import run_phase2a_full as r2a
from harness import phase2a_ext as p2x

OUT_DIR = os.path.join(_REPO_ROOT, 'results', 'trim_sweep')
SEEDS = list(range(5))  # half of the paper's 10 -- see module docstring
TAUS = [0.0, 0.05, 0.10, 0.20, 0.30]

TABLE1_MECH_SEV = {
    'missing': None, 'duplicate': None, 'out_of_order': None,
    'sensor_dropout_burst': ('burst_len', 15), 'stale_run': ('burst_len', 15),
    'outlier': ('magnitude', 4), 'spike_burst': ('magnitude', 4),
}


def geomean(x):
    x = np.asarray(x, dtype=float)
    x = x[x > 0]
    return np.exp(np.mean(np.log(x))) if x.size else np.nan


def sweep_one_tau(series_list, regimes, tau):
    rows = []
    for s in series_list:
        x = s['x']
        clean_cache = p2x.precompute_clean(x, trim_frac=tau)
        for r in regimes:
            kwargs = {}
            if r['sev_type'] == 'magnitude':
                kwargs['magnitude'] = r['sev_value']
            elif r['sev_type'] == 'burst_len':
                kwargs['burst_len'] = r['sev_value']
            for seed in SEEDS:
                res = p2x.eval_cell_16(x, r['mechanism'], r2a.RATE, seed,
                                        precomputed_clean=clean_cache, trim_frac=tau, **kwargs)
                rows.append(dict(
                    domain=s['domain'], run=s['run'], channel=s['channel'],
                    mechanism=r['mechanism'], sev_type=r['sev_type'], sev_value=r['sev_value'],
                    seed=seed, spr_mae_std=res['spr_mae_std'], spr_mae_trimmed_std=res['spr_mae_trimmed_std'],
                ))
    return pd.DataFrame(rows)


def table1_style(percell, tau):
    rows = []
    for domain in ['SKAB', 'UCI']:
        dsub = percell[percell.domain == domain]
        for mech, sev in TABLE1_MECH_SEV.items():
            g = dsub[dsub.mechanism == mech]
            if sev is not None:
                sev_type, sev_value = sev
                g = g[(g.sev_type == sev_type) & (g.sev_value == sev_value)]
            trim = geomean(g['spr_mae_trimmed_std'] / g['spr_mae_std'])
            rows.append(dict(tau=tau, domain=domain, mechanism=mech, trimmed_std_over_std=trim))
    return pd.DataFrame(rows)


def table2_style(percell, tau):
    rows = []
    skab_af = percell[(percell.domain == 'SKAB') & (percell.run == 'anomaly-free') & (percell.mechanism == 'outlier')]
    uci_all = percell[(percell.domain == 'UCI') & (percell.mechanism == 'outlier')]
    for domain, dsub in [('SKAB_anomaly-free', skab_af), ('UCI_pooled', uci_all)]:
        for mag in sorted(dsub['sev_value'].unique()):
            g = dsub[dsub['sev_value'] == mag]
            trim = geomean(g['spr_mae_trimmed_std'] / g['spr_mae_std'])
            rows.append(dict(tau=tau, domain=domain, magnitude=mag, trimmed_std_over_std=trim))
    return pd.DataFrame(rows)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.time()

    print("Loading raw series (SKAB + UCI)...")
    series_list = r2a.load_series()
    regimes = r2a.build_regime_grid()
    print(f"  {len(series_list)} series, {len(regimes)} regimes, {len(TAUS)} trim levels, {len(SEEDS)} seeds")

    all_t1, all_t2 = [], []
    for tau in TAUS:
        tt0 = time.time()
        print(f"\n=== tau={tau} ===")
        percell = sweep_one_tau(series_list, regimes, tau)
        percell.to_csv(os.path.join(OUT_DIR, f'percell_tau{tau}.csv'), index=False)
        print(f"  {len(percell)} cells, elapsed {time.time()-tt0:.1f}s")
        all_t1.append(table1_style(percell, tau))
        all_t2.append(table2_style(percell, tau))

    t1_all = pd.concat(all_t1, ignore_index=True)
    t2_all = pd.concat(all_t2, ignore_index=True)
    t1_all.to_csv(os.path.join(OUT_DIR, 'table1_by_tau.csv'), index=False)
    t2_all.to_csv(os.path.join(OUT_DIR, 'table2_by_tau.csv'), index=False)

    # optimal tau per (domain, mechanism) and per (domain, magnitude)
    print("\n=== Optimal tau, Table-I-style cells (domain, mechanism) ===")
    opt1 = t1_all.loc[t1_all.groupby(['domain', 'mechanism'])['trimmed_std_over_std'].idxmin()]
    print(opt1[['domain', 'mechanism', 'tau', 'trimmed_std_over_std']].to_string(index=False))

    print("\n=== Optimal tau, Table-II-style cells (domain, magnitude) ===")
    opt2 = t2_all.loc[t2_all.groupby(['domain', 'magnitude'])['trimmed_std_over_std'].idxmin()]
    print(opt2[['domain', 'magnitude', 'tau', 'trimmed_std_over_std']].to_string(index=False))

    with open(os.path.join(OUT_DIR, 'trim_sweep_summary.json'), 'w') as f:
        json.dump(dict(
            table1_by_tau=t1_all.to_dict(orient='records'),
            table2_by_tau=t2_all.to_dict(orient='records'),
            optimal_tau_table1=opt1.to_dict(orient='records'),
            optimal_tau_table2=opt2.to_dict(orient='records'),
        ), f, indent=2)

    print(f"\nTotal elapsed: {time.time()-t0:.1f}s. Wrote outputs to {OUT_DIR}")
    return t1_all, t2_all, opt1, opt2


if __name__ == '__main__':
    main()
