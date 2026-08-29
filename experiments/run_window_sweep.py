"""Window-size sensitivity sweep: W in {15, 30, 60, 120}, on both the main
SKAB comparisons (Table I/II-style dominance patterns) and the
perturbation-magnitude regression (Section V-B). All results use W=30
elsewhere in the paper; this checks whether that choice was load-bearing.

Reuses harness1c.rolling_stats/rolling_extended and phase2a_ext.eval_cell_16
unmodified except for an added optional `w` parameter that threads through
to those two functions (byte-identical at w=None/w=30 to every other use of
this code in the artifact). Does not touch the isolated-incident (Phase 1B)
pipeline: harness/pipeline.py's W_S is a module-level constant baked into
several incident-timing computations (not just the rolling window), and
reparametrizing it safely is a larger, separate job than the ambient-model
(Phase 2A/SKAB+UCI) sweep this script performs -- flagged as a scope
limitation, not silently skipped.

One per-seed sweep per window size covers both checks: Table I/II-style
ratios are seed-pooled geometric means of the same per-cell columns the
perturbation regression uses per-seed, so there is no need to sweep twice.

Requires the raw SKAB + UCI datasets in data/ (see README.md).
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats as spstats

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import run_phase2a_full as r2a
from harness import phase2a_ext as p2x

OUT_DIR = os.path.join(_REPO_ROOT, 'results', 'window_sweep')
SEEDS = list(range(10))
WINDOWS = [15, 30, 60, 120]

STRUCTURAL = {'missing', 'duplicate', 'out_of_order', 'sensor_dropout_burst', 'stale_run'}
AMPLITUDE = {'outlier', 'spike_burst'}

# Table I's fixed operating point (reproduce_tables.py, unchanged)
TABLE1_MECH_SEV = {
    'missing': None, 'duplicate': None, 'out_of_order': None,
    'sensor_dropout_burst': ('burst_len', 15), 'stale_run': ('burst_len', 15),
    'outlier': ('magnitude', 4), 'spike_burst': ('magnitude', 4),
}


def geomean(x):
    x = np.asarray(x, dtype=float)
    x = x[x > 0]
    return np.exp(np.mean(np.log(x))) if x.size else np.nan


def sweep_one_window(series_list, regimes, w):
    rows = []
    for s in series_list:
        x = s['x']
        finite = x[np.isfinite(x)]
        channel_sigma = float(np.nanstd(finite)) if finite.size else np.nan
        clean_cache = p2x.precompute_clean(x, w=w)
        for r in regimes:
            kwargs = {}
            if r['sev_type'] == 'magnitude':
                kwargs['magnitude'] = r['sev_value']
            elif r['sev_type'] == 'burst_len':
                kwargs['burst_len'] = r['sev_value']
            for seed in SEEDS:
                res = p2x.eval_cell_16(x, r['mechanism'], r2a.RATE, seed,
                                        precomputed_clean=clean_cache, w=w, **kwargs)
                rows.append(dict(
                    domain=s['domain'], run=s['run'], channel=s['channel'],
                    mechanism=r['mechanism'], sev_type=r['sev_type'], sev_value=r['sev_value'],
                    seed=seed, channel_sigma=channel_sigma,
                    spr_mae_std=res['spr_mae_std'], spr_mae_madn=res['spr_mae_madn'],
                    spr_mae_trimmed_std=res['spr_mae_trimmed_std'], spr_mae_IQR_scale=res['spr_mae_IQR_scale'],
                ))
    return pd.DataFrame(rows)


def table1_style(percell, w):
    """Seed-pooled (geomean) SKAB ratios at Table I's fixed operating point,
    for this window size."""
    skab = percell[percell.domain == 'SKAB']
    rows = []
    for mech, sev in TABLE1_MECH_SEV.items():
        g = skab[skab.mechanism == mech]
        if sev is not None:
            sev_type, sev_value = sev
            g = g[(g.sev_type == sev_type) & (g.sev_value == sev_value)]
        madn = geomean(g['spr_mae_madn'] / g['spr_mae_std'])
        trim = geomean(g['spr_mae_trimmed_std'] / g['spr_mae_std'])
        iqr = geomean(g['spr_mae_IQR_scale'] / g['spr_mae_std'])
        rows.append(dict(w=w, mechanism=mech, madn_over_std=madn,
                          trimmed_std_over_std=trim, iqr_over_std=iqr,
                          interior_beats_both=bool((trim < 1.0 and trim < madn) or (iqr < 1.0 and iqr < madn))))
    return pd.DataFrame(rows)


def table2_style(percell, w):
    """Seed-pooled SKAB anomaly-free outlier severity sweep, this window size."""
    af = percell[(percell.domain == 'SKAB') & (percell.run == 'anomaly-free') & (percell.mechanism == 'outlier')]
    rows = []
    for mag in sorted(af['sev_value'].unique()):
        g = af[af['sev_value'] == mag]
        madn = geomean(g['spr_mae_madn'] / g['spr_mae_std'])
        trim = geomean(g['spr_mae_trimmed_std'] / g['spr_mae_std'])
        iqr = geomean(g['spr_mae_IQR_scale'] / g['spr_mae_std'])
        vals = {'madn': madn, 'trimmed_std': trim, 'IQR_scale': iqr, 'std': 1.0}
        rows.append(dict(w=w, magnitude=mag, madn_over_std=madn, trimmed_std_over_std=trim,
                          iqr_over_std=iqr, oracle_pick=min(vals, key=vals.get)))
    return pd.DataFrame(rows)


def perturbation_fit(percell, w):
    df = percell.copy()
    df['norm_pert'] = df['spr_mae_std'] / df['channel_sigma']
    df['ratio'] = df['spr_mae_madn'] / df['spr_mae_std']
    valid = df[(df['norm_pert'] > 0) & (df['ratio'] > 0) & np.isfinite(df['norm_pert']) & np.isfinite(df['ratio'])]
    out = {}
    for domain in ['SKAB', 'UCI']:
        sub = valid[valid.domain == domain]
        logx, logy = np.log(sub['norm_pert']), np.log(sub['ratio'])
        r = spstats.linregress(logx, logy)
        out[domain] = dict(w=w, n=len(sub), slope=float(r.slope), r2=float(r.rvalue ** 2),
                            crossing=float(np.exp(-r.intercept / r.slope)))
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.time()

    print("Loading raw series (SKAB + UCI)...")
    series_list = r2a.load_series()
    regimes = r2a.build_regime_grid()
    print(f"  {len(series_list)} series, {len(regimes)} regimes, {len(WINDOWS)} window sizes")

    all_table1, all_table2, all_pert = [], [], []
    for w in WINDOWS:
        tw0 = time.time()
        print(f"\n=== W={w} ===")
        percell = sweep_one_window(series_list, regimes, w)
        percell.to_csv(os.path.join(OUT_DIR, f'percell_w{w}.csv'), index=False)
        print(f"  {len(percell)} cells, elapsed {time.time()-tw0:.1f}s")

        t1 = table1_style(percell, w)
        t2 = table2_style(percell, w)
        pert = perturbation_fit(percell, w)
        all_table1.append(t1)
        all_table2.append(t2)
        for domain, d in pert.items():
            all_pert.append(dict(domain=domain, **d))

        print(f"  Table-I-style: {t1[['mechanism','interior_beats_both']].to_dict(orient='records')}")
        print(f"  Table-II-style oracle picks: {t2[['magnitude','oracle_pick']].to_dict(orient='records')}")
        print(f"  perturbation fit: {pert}")

    table1_all = pd.concat(all_table1, ignore_index=True)
    table2_all = pd.concat(all_table2, ignore_index=True)
    pert_all = pd.DataFrame(all_pert)
    table1_all.to_csv(os.path.join(OUT_DIR, 'table1_by_window.csv'), index=False)
    table2_all.to_csv(os.path.join(OUT_DIR, 'table2_by_window.csv'), index=False)
    pert_all.to_csv(os.path.join(OUT_DIR, 'perturbation_fit_by_window.csv'), index=False)

    with open(os.path.join(OUT_DIR, 'window_sweep_summary.json'), 'w') as f:
        json.dump(dict(
            table1=table1_all.to_dict(orient='records'),
            table2=table2_all.to_dict(orient='records'),
            perturbation=pert_all.to_dict(orient='records'),
        ), f, indent=2)

    print(f"\nTotal elapsed: {time.time()-t0:.1f}s. Wrote outputs to {OUT_DIR}")
    return table1_all, table2_all, pert_all


if __name__ == '__main__':
    main()
