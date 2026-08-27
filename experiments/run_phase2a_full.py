"""Phase 2A (full): two-domain (SKAB + UCI) adaptive-selection headroom
gate. Diagnostic experiment only -- no online selector, no feature
extraction, no delayed-feedback simulation.

Reuses BYTE-IDENTICAL: harness1c.rolling_stats, harness1c.inject (extended
additively with magnitude/burst_len, see harness1c.py docstring), and the
err(t)=|E_W(x')(t)-E_W(x)(t)| / MAE=mean / PEAK=max error definition
(phase2a_ext._stats, copied verbatim from eval_cell's inner closure).

SCOPE LIMITATION (declared before running, not discovered after):
missing/duplicate/out_of_order have no burst/contiguous-run concept in the
reused harness (they scatter k independent point corruptions via
rng.choice) -- generalizing them to burst_len>1 would mean inventing new
corruption semantics not present in harness1c.py, not parameterizing an
existing hardcoded value. They are therefore evaluated as a single
(pointwise, rate=0.05) regime each, with no severity axis. The severity
grid (burst_len in {1,5,15,30}) is applied only to sensor_dropout_burst and
stale_run, which already have a burst_len concept (hardcoded at 15).
Amplitude defects (outlier, spike_burst) get the magnitude grid
({0.25,...,8}sigma) per the brief. This gives 3 + 2*6 + 2*4 = 23 regimes,
not ~40 -- reported explicitly, not silently reconciled.
"""
import json
import os
import sys
import time
import warnings
from itertools import product

import numpy as np
import pandas as pd
from scipy import stats

# burst_len=30 (full window) legitimately produces some all-NaN windows for
# sensor_dropout_burst; rolling_stats/rolling_extended already propagate
# these as NaN correctly (filtered out downstream by _stats' isfinite mask)
# -- suppressing the resulting numpy RuntimeWarnings, not the behavior.
warnings.filterwarnings('ignore', category=RuntimeWarning)

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import phase1b_real as p1b
from harness import phase2a_ext as p2x
from harness.harness1c import W

OUT_DIR = os.path.join(_REPO_ROOT, 'results', 'phase2a_full')
RATE = 0.05
SEEDS = list(range(10))
MAGNITUDES = [0.25, 0.5, 1, 2, 4, 8]
BURST_LENS = [1, 5, 15, 30]
AMPLITUDE_MECH = ['outlier', 'spike_burst']
BURST_MECH = ['sensor_dropout_burst', 'stale_run']
POINTWISE_MECH = ['missing', 'duplicate', 'out_of_order']
ALL_MECH = POINTWISE_MECH + AMPLITUDE_MECH + BURST_MECH

SKAB_DIR = os.path.join(_REPO_ROOT, 'data', 'skab_data')
SKAB_RUNS = {
    'anomaly-free': f'{SKAB_DIR}/anomaly-free/anomaly-free.csv',
    'other5_rotor_imbalance': f'{SKAB_DIR}/other/5.csv',
    'other10_water_increase': f'{SKAB_DIR}/other/10.csv',
    'valve1_1_inlet_close': f'{SKAB_DIR}/valve1/1.csv',
    'valve2_1_outlet_close': f'{SKAB_DIR}/valve2/1.csv',
}
SKAB_CHANNELS = ['Accelerometer1RMS', 'Accelerometer2RMS', 'Current', 'Pressure',
                  'Temperature', 'Thermocouple', 'Voltage', 'Volume Flow RateRMS']
UCI_CHANNELS = ['Global_active_power', 'Global_reactive_power', 'Voltage', 'Global_intensity']


def build_regime_grid():
    regimes = []
    for mech in POINTWISE_MECH:
        regimes.append(dict(mechanism=mech, sev_type=None, sev_value=None, label=mech))
    for mech in AMPLITUDE_MECH:
        for mag in MAGNITUDES:
            regimes.append(dict(mechanism=mech, sev_type='magnitude', sev_value=mag, label=f'{mech}_mag{mag}'))
    for mech in BURST_MECH:
        for bl in BURST_LENS:
            regimes.append(dict(mechanism=mech, sev_type='burst_len', sev_value=bl, label=f'{mech}_burst{bl}'))
    return regimes


def load_series():
    """Returns list of dicts: domain, run, channel, x (raw, native units)."""
    series = []
    for run_name, path in SKAB_RUNS.items():
        df = pd.read_csv(path, sep=';')
        for ch in SKAB_CHANNELS:
            if ch not in df.columns:
                continue
            x = df[ch].to_numpy(dtype=float)
            if len(x) < 5 * W or not np.isfinite(x).all():
                x = x[np.isfinite(x)]
            if len(x) < 5 * W or np.nanstd(x) <= 0:
                continue
            series.append(dict(domain='SKAB', run=run_name, channel=ch, x=x))

    segments = p1b.select_segments()
    for seg in segments:
        for j, ch in enumerate(UCI_CHANNELS):
            raw = seg['x'][:, j] * seg['raw_std'][j] + seg['raw_mean'][j]  # undo z-score -> native units
            series.append(dict(domain='UCI', run=seg['label'], channel=ch, x=raw))
    return series


def run_sweep(series_list, regimes):
    rows = []
    for s in series_list:
        clean_cache = p2x.precompute_clean(s['x'])
        for r in regimes:
            kwargs = {}
            if r['sev_type'] == 'magnitude':
                kwargs['magnitude'] = r['sev_value']
            elif r['sev_type'] == 'burst_len':
                kwargs['burst_len'] = r['sev_value']
            per_seed = []
            for seed in SEEDS:
                res = p2x.eval_cell_16(s['x'], r['mechanism'], RATE, seed,
                                        precomputed_clean=clean_cache, **kwargs)
                per_seed.append(res)
            # geometric mean across seeds within this cell x regime (matches Phase 1C pooling)
            agg = {}
            for key in per_seed[0]:
                vals = np.array([p[key] for p in per_seed], dtype=float)
                vals = vals[np.isfinite(vals) & (vals > 0)]
                agg[key] = float(np.exp(np.mean(np.log(vals)))) if vals.size else np.nan
            rows.append(dict(domain=s['domain'], run=s['run'], channel=s['channel'],
                              mechanism=r['mechanism'], sev_type=r['sev_type'], sev_value=r['sev_value'],
                              regime=r['label'], **agg))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Gate machinery
# ---------------------------------------------------------------------------

RUN_COL = 'run'  # independent unit within a domain


def cell_config_mae(df, statistic):
    """df: cells table (one row per domain,run,channel,regime). Returns long
    form: domain, run, channel, regime, estimator, mae."""
    ests = p2x.LOCATION_ESTIMATORS if statistic == 'location' else p2x.SPREAD_ESTIMATORS
    prefix = 'loc_mae_' if statistic == 'location' else 'spr_mae_'
    out = []
    for e in ests:
        col = prefix + e
        sub = df[['domain', 'run', 'channel', 'regime', col]].rename(columns={col: 'mae'})
        sub['estimator'] = e
        out.append(sub)
    return pd.concat(out, ignore_index=True)


def run_level_mae(long_df, regime_filter):
    """Mean over channels within each (domain,run), for the given set of
    regimes -- respects the run/segment as the independent unit."""
    sub = long_df[long_df.regime.isin(regime_filter)]
    return sub.groupby(['domain', 'run', 'estimator'])['mae'].mean().reset_index()


def select_best(run_mae_df, estimators):
    """Best estimator = lowest mean-across-(domain,run) MAE, TRAINING data only."""
    scores = run_mae_df[run_mae_df.estimator.isin(estimators)].groupby('estimator')['mae'].mean()
    return scores.idxmin(), scores


def oracle_per_run(run_mae_df):
    """Oracle MAE per (domain,run): min over estimators."""
    return run_mae_df.groupby(['domain', 'run'])['mae'].min().reset_index().rename(columns={'mae': 'oracle_mae'})


def oracle_estimator_per_run(run_mae_df):
    idx = run_mae_df.groupby(['domain', 'run'])['mae'].idxmin()
    return run_mae_df.loc[idx, ['domain', 'run', 'estimator']].rename(columns={'estimator': 'oracle_estimator'})


T_CRIT = {}


def t_crit_95(n):
    if n not in T_CRIT:
        T_CRIT[n] = stats.t.ppf(0.975, df=max(n - 1, 1))
    return T_CRIT[n]


def ci95_t(values):
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return (np.nan, np.nan)
    sem = values.std(ddof=1) / np.sqrt(len(values))
    m = values.mean()
    tc = t_crit_95(len(values))
    return m - tc * sem, m + tc * sem


def headroom_for_split(long_df, statistic, train_regimes, test_regimes, split_name, held_out_label):
    ests = p2x.LOCATION_ESTIMATORS if statistic == 'location' else p2x.SPREAD_ESTIMATORS
    intermediate = p2x.INTERMEDIATE_LOCATION if statistic == 'location' else p2x.INTERMEDIATE_SPREAD

    train_run_mae = run_level_mae(long_df, train_regimes)
    if train_run_mae.empty:
        return None
    best_fixed, _ = select_best(train_run_mae, ests)
    best_intermediate, _ = select_best(train_run_mae, intermediate)

    test_run_mae = run_level_mae(long_df, test_regimes)
    if test_run_mae.empty:
        return None
    oracle = oracle_per_run(test_run_mae)
    oracle_est = oracle_estimator_per_run(test_run_mae)
    fixed = test_run_mae[test_run_mae.estimator == best_fixed][['domain', 'run', 'mae']].rename(columns={'mae': 'fixed_mae'})
    intermed = test_run_mae[test_run_mae.estimator == best_intermediate][['domain', 'run', 'mae']].rename(columns={'mae': 'intermediate_mae'})

    m = oracle.merge(fixed, on=['domain', 'run']).merge(intermed, on=['domain', 'run']).merge(oracle_est, on=['domain', 'run'])
    m['headroom_fixed'] = (m['fixed_mae'] - m['oracle_mae']) / m['fixed_mae']
    m['headroom_intermediate'] = (m['intermediate_mae'] - m['oracle_mae']) / m['intermediate_mae']
    m['statistic'] = statistic
    m['split'] = split_name
    m['held_out'] = held_out_label
    m['best_fixed'] = best_fixed
    m['best_intermediate'] = best_intermediate

    lo_f, hi_f = ci95_t(m['headroom_fixed'].values)
    lo_i, hi_i = ci95_t(m['headroom_intermediate'].values)
    summary = dict(
        statistic=statistic, split=split_name, held_out=held_out_label,
        best_fixed=best_fixed, best_intermediate=best_intermediate,
        n_units=len(m), oracle_mae_mean=float(m['oracle_mae'].mean()),
        fixed_mae_mean=float(m['fixed_mae'].mean()), intermediate_mae_mean=float(m['intermediate_mae'].mean()),
        headroom_fixed_mean=float(m['headroom_fixed'].mean()), headroom_fixed_ci_lo=lo_f, headroom_fixed_ci_hi=hi_f,
        headroom_intermediate_mean=float(m['headroom_intermediate'].mean()),
        headroom_intermediate_ci_lo=lo_i, headroom_intermediate_ci_hi=hi_i,
    )
    return dict(units=m, summary=summary)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.time()

    regimes = build_regime_grid()
    print(f"=== Phase 2A (full): {len(regimes)} regimes (see scope-limitation note in module docstring) ===")
    for r in regimes:
        print(' ', r['label'])

    print("\n=== Loading series (SKAB + UCI) ===")
    series_list = load_series()
    n_skab = sum(1 for s in series_list if s['domain'] == 'SKAB')
    n_uci = sum(1 for s in series_list if s['domain'] == 'UCI')
    print(f"  SKAB series (run x channel): {n_skab}   UCI series (segment x channel): {n_uci}")

    print("\n=== Running sweep (this is the expensive part) ===")
    cells = run_sweep(series_list, regimes)
    cells.to_csv(os.path.join(OUT_DIR, 'phase2a_cells.csv'), index=False)
    print(f"  {len(cells)} (series x regime) cells written. elapsed: {time.time()-t0:.1f}s")

    return cells, regimes, series_list


if __name__ == '__main__':
    main()
