"""Perturbation-magnitude analysis: tests whether the robust/naive spread
ratio is governed primarily by how much a defect perturbs the naive
estimator, in units of the channel's own scale, rather than by defect
mechanism per se.

For each (domain, run, channel, mechanism, severity, seed) cell:
    normalized_perturbation = spr_mae_std / channel_sigma
    ratio                   = spr_mae_madn / spr_mae_std
channel_sigma = np.nanstd(x_clean) on the full native-units series -- the
same quantity harness1c.inject() already uses to scale amplitude defects,
not a new definition introduced for this analysis.

Unlike phase2a_cells.csv (which is pre-pooled across the 10 seeds via
geometric mean), this script keeps every seed as a separate observation,
because the regression is specified at per-cell, not per-condition,
granularity: fitting on condition-level means instead of raw cells
suppresses within-condition variance and inflates r^2 (this was checked
against an earlier, informal run that fit 11 condition-level geometric
means and got r^2=0.795, crossing=0.026 -- both artifacts of averaging
before fitting, not reproducible from per-cell data, and not used
anywhere in the paper).

Fits log(ratio) ~ log(normalized_perturbation) two ways:
  - naive pooled: one OLS line per domain, ignoring series identity.
  - (run, channel)-demeaned: removes each series' own mean log-level
    before fitting, isolating the common within-series slope. Individual
    series fit this relationship much more tightly (r^2 0.6-0.9 each)
    than the naive pooled fit suggests, because different channels sit
    at different baseline levels even after normalizing by their own
    sigma -- a common-slope, varying-intercept relationship, not a
    single global line.

Also checks the RQ2 connection: restricted to the 4 mechanisms Phase 1B
and Phase 2A both tested (missing, duplicate, outlier, out_of_order),
does UCI's isolated-incident normalized perturbation fall below the
ambient-fitted parity-crossing point, and ambient fall above it?

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

import phase1b_real as p1b
import run_phase2a_full as r2a
from harness import phase2a_ext as p2x

OUT_DIR = os.path.join(_REPO_ROOT, 'results', 'perturbation')
PHASE1B_INCIDENT_CSV = os.path.join(_REPO_ROOT, 'results', 'phase1b', 'incident_level.csv')

SEEDS = list(range(10))
STRUCTURAL = {'missing', 'duplicate', 'out_of_order', 'sensor_dropout_burst', 'stale_run'}
AMPLITUDE = {'outlier', 'spike_burst'}
SHARED_MECH = {'missing', 'duplicate', 'outlier', 'out_of_order'}  # tested by both Phase 1B and Phase 2A


def run_sweep():
    series_list = r2a.load_series()
    regimes = r2a.build_regime_grid()
    rows = []
    for s in series_list:
        x = s['x']
        finite = x[np.isfinite(x)]
        channel_sigma = float(np.nanstd(finite)) if finite.size else np.nan
        clean_cache = p2x.precompute_clean(x)
        for r in regimes:
            kwargs = {}
            if r['sev_type'] == 'magnitude':
                kwargs['magnitude'] = r['sev_value']
            elif r['sev_type'] == 'burst_len':
                kwargs['burst_len'] = r['sev_value']
            for seed in SEEDS:
                res = p2x.eval_cell_16(x, r['mechanism'], r2a.RATE, seed,
                                        precomputed_clean=clean_cache, **kwargs)
                rows.append(dict(
                    domain=s['domain'], run=s['run'], channel=s['channel'],
                    mechanism=r['mechanism'], sev_type=r['sev_type'], sev_value=r['sev_value'],
                    seed=seed, channel_sigma=channel_sigma,
                    spr_mae_std=res['spr_mae_std'], spr_mae_madn=res['spr_mae_madn'],
                ))
    return pd.DataFrame(rows)


def fit_domain(df, domain):
    sub = df[df.domain == domain].copy()
    sub['logx'] = np.log(sub['norm_pert'])
    sub['logy'] = np.log(sub['ratio'])

    pooled = spstats.linregress(sub['logx'], sub['logy'])
    pooled_crossing = float(np.exp(-pooled.intercept / pooled.slope))

    sub['logx_c'] = sub['logx'] - sub.groupby(['run', 'channel'])['logx'].transform('mean')
    sub['logy_c'] = sub['logy'] - sub.groupby(['run', 'channel'])['logy'].transform('mean')
    demeaned = spstats.linregress(sub['logx_c'], sub['logy_c'])

    per_series = []
    for (run, ch), g in sub.groupby(['run', 'channel']):
        if len(g) < 5:
            continue
        rr = spstats.linregress(g['logx'], g['logy'])
        per_series.append(dict(run=run, channel=ch, n=len(g), slope=rr.slope, r2=rr.rvalue ** 2))
    per_series_df = pd.DataFrame(per_series)

    struct_pct = {}
    for mech in sorted(STRUCTURAL):
        g = sub[sub.mechanism == mech]
        if not g.empty:
            struct_pct[mech] = float((g['norm_pert'] < pooled_crossing).mean())

    return dict(
        domain=domain, n=len(sub),
        n_runs=int(sub['run'].nunique()), n_channels=int(sub['channel'].nunique()),
        pooled_slope=float(pooled.slope), pooled_r2=float(pooled.rvalue ** 2),
        pooled_crossing=pooled_crossing,
        demeaned_slope=float(demeaned.slope), demeaned_r2=float(demeaned.rvalue ** 2),
        per_series_mean_r2=float(per_series_df['r2'].mean()),
        per_series_min_r2=float(per_series_df['r2'].min()),
        per_series_max_r2=float(per_series_df['r2'].max()),
        per_series_mean_slope=float(per_series_df['slope'].mean()),
        structural_pct_below_crossing=struct_pct,
    )


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.time()

    print("=== Loading raw series (SKAB + UCI) and running per-seed sweep ===")
    df = run_sweep()
    print(f"  {len(df)} per-seed cells, elapsed {time.time()-t0:.1f}s")
    df.to_csv(os.path.join(OUT_DIR, 'perturbation_percell.csv'), index=False)

    df['norm_pert'] = df['spr_mae_std'] / df['channel_sigma']
    df['ratio'] = df['spr_mae_madn'] / df['spr_mae_std']
    df['family'] = df['mechanism'].apply(lambda m: 'amplitude' if m in AMPLITUDE else 'structural')
    valid = df[(df['norm_pert'] > 0) & (df['ratio'] > 0)
               & np.isfinite(df['norm_pert']) & np.isfinite(df['ratio'])].copy()
    valid.to_csv(os.path.join(OUT_DIR, 'perturbation_valid.csv'), index=False)

    results = {'SKAB': fit_domain(valid, 'SKAB'), 'UCI': fit_domain(valid, 'UCI')}
    for domain, r in results.items():
        print(f"\n--- {domain} ---")
        print(f"  pooled:   slope={r['pooled_slope']:.4f}  r2={r['pooled_r2']:.4f}  "
              f"crossing={r['pooled_crossing']:.5f}")
        print(f"  demeaned: slope={r['demeaned_slope']:.4f}  r2={r['demeaned_r2']:.4f}")
        print(f"  per-series mean r2={r['per_series_mean_r2']:.4f} "
              f"(range {r['per_series_min_r2']:.3f}-{r['per_series_max_r2']:.3f})")
        print(f"  structural pct below pooled crossing: {r['structural_pct_below_crossing']}")

    # RQ2 connection
    uci_valid = valid[valid.domain == 'UCI']
    crossing_uci = results['UCI']['pooled_crossing']
    ambient = uci_valid[uci_valid.mechanism.isin(SHARED_MECH)]
    ambient_frac_above = float((ambient['norm_pert'] > crossing_uci).mean())

    isolated = pd.read_csv(PHASE1B_INCIDENT_CSV)
    isolated_naive = isolated[(isolated.estimator == 'naive') & (isolated.type.isin(SHARED_MECH))]
    isolated_frac_below = float((isolated_naive['mae_spr'] < crossing_uci).mean())

    rq2 = dict(
        uci_crossing=crossing_uci,
        ambient_n=len(ambient), ambient_median_norm_pert=float(ambient['norm_pert'].median()),
        ambient_frac_above_crossing=ambient_frac_above,
        isolated_n=len(isolated_naive), isolated_median_norm_pert=float(isolated_naive['mae_spr'].median()),
        isolated_frac_below_crossing=isolated_frac_below,
    )
    print(f"\n--- RQ2 connection (UCI, shared mechanisms {sorted(SHARED_MECH)}) ---")
    print(f"  ambient:  n={rq2['ambient_n']}  median={rq2['ambient_median_norm_pert']:.5f}  "
          f"frac above crossing={ambient_frac_above:.3f}")
    print(f"  isolated: n={rq2['isolated_n']}  median={rq2['isolated_median_norm_pert']:.5f}  "
          f"frac below crossing={isolated_frac_below:.3f}")

    with open(os.path.join(OUT_DIR, 'perturbation_summary.json'), 'w') as f:
        json.dump({'fits': results, 'rq2_connection': rq2}, f, indent=2)

    print(f"\nTotal elapsed: {time.time()-t0:.1f}s. Wrote outputs to {OUT_DIR}")
    return valid, results, rq2


if __name__ == '__main__':
    main()
