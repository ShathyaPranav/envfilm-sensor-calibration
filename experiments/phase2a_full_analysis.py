"""Phase 2A (full): gate computation, diagnostics, and report generation
from phase2a_cells.csv. Split from run_phase2a_full.py (the sweep) so the
analysis can be re-run/edited without repeating the expensive sweep.
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from harness import phase2a_ext as p2x
from run_phase2a_full import (AMPLITUDE_MECH, BURST_MECH, POINTWISE_MECH, ALL_MECH,
                                MAGNITUDES, BURST_LENS)

T_CRIT = {}


def t_crit_95(n):
    if n not in T_CRIT:
        T_CRIT[n] = stats.t.ppf(0.975, df=max(n - 1, 1))
    return T_CRIT[n]


def ci95_t(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return (np.nan, np.nan)
    sem = values.std(ddof=1) / np.sqrt(len(values))
    m = values.mean()
    tc = t_crit_95(len(values))
    return m - tc * sem, m + tc * sem


def cell_config_mae(df, statistic):
    ests = p2x.LOCATION_ESTIMATORS if statistic == 'location' else p2x.SPREAD_ESTIMATORS
    prefix = 'loc_mae_' if statistic == 'location' else 'spr_mae_'
    out = []
    for e in ests:
        col = prefix + e
        sub = df[['domain', 'run', 'channel', 'regime', col]].rename(columns={col: 'mae'})
        sub['estimator'] = e
        out.append(sub)
    return pd.concat(out, ignore_index=True)


def run_level_mae(long_df, domain_run_filter=None, regime_filter=None):
    sub = long_df
    if regime_filter is not None:
        sub = sub[sub.regime.isin(regime_filter)]
    if domain_run_filter is not None:
        sub = sub[sub.set_index(['domain', 'run']).index.isin(domain_run_filter)]
    return sub.groupby(['domain', 'run', 'estimator'])['mae'].mean().reset_index()


def select_best(run_mae_df, estimators):
    scores = run_mae_df[run_mae_df.estimator.isin(estimators)].groupby('estimator')['mae'].mean()
    if scores.empty:
        return None, scores
    return scores.idxmin(), scores


def oracle_per_run(run_mae_df):
    return run_mae_df.groupby(['domain', 'run'])['mae'].min().reset_index().rename(columns={'mae': 'oracle_mae'})


def oracle_estimator_per_run(run_mae_df):
    idx = run_mae_df.groupby(['domain', 'run'])['mae'].idxmin()
    return run_mae_df.loc[idx, ['domain', 'run', 'estimator']].rename(columns={'estimator': 'oracle_estimator'})


def headroom(long_df, statistic, train_regimes=None, test_regimes=None,
             train_domain_runs=None, test_domain_runs=None,
             split_name='', held_out_label=''):
    """General held-out headroom computation: TRAIN side selects best_fixed/
    best_intermediate (regime and/or domain-run filtered), TEST side
    evaluates oracle/fixed/intermediate per (domain,run) -- respecting the
    run/segment as the independent unit throughout."""
    ests = p2x.LOCATION_ESTIMATORS if statistic == 'location' else p2x.SPREAD_ESTIMATORS
    intermediate = p2x.INTERMEDIATE_LOCATION if statistic == 'location' else p2x.INTERMEDIATE_SPREAD

    train_run_mae = run_level_mae(long_df, domain_run_filter=train_domain_runs, regime_filter=train_regimes)
    if train_run_mae.empty:
        return None
    best_fixed, _ = select_best(train_run_mae, ests)
    best_intermediate, _ = select_best(train_run_mae, intermediate)
    if best_fixed is None or best_intermediate is None:
        return None

    test_run_mae = run_level_mae(long_df, domain_run_filter=test_domain_runs, regime_filter=test_regimes)
    if test_run_mae.empty:
        return None
    oracle = oracle_per_run(test_run_mae)
    oracle_est = oracle_estimator_per_run(test_run_mae)
    fixed = test_run_mae[test_run_mae.estimator == best_fixed][['domain', 'run', 'mae']].rename(columns={'mae': 'fixed_mae'})
    intermed = test_run_mae[test_run_mae.estimator == best_intermediate][['domain', 'run', 'mae']].rename(columns={'mae': 'intermediate_mae'})

    m = oracle.merge(fixed, on=['domain', 'run']).merge(intermed, on=['domain', 'run']).merge(oracle_est, on=['domain', 'run'])
    if m.empty:
        return None
    m['headroom_fixed'] = (m['fixed_mae'] - m['oracle_mae']) / m['fixed_mae']
    m['headroom_intermediate'] = (m['intermediate_mae'] - m['oracle_mae']) / m['intermediate_mae']
    m['statistic'] = statistic
    m['split'] = split_name
    m['held_out'] = held_out_label

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


# ---------------------------------------------------------------------------
# Split builders
# ---------------------------------------------------------------------------

SEV_TRAIN_MAG = [0.25, 1, 4]
SEV_TEST_MAG = [0.5, 2, 8]
SEV_TRAIN_BURST = [1, 15]
SEV_TEST_BURST = [5, 30]


def severity_split_regimes():
    train = ([f'outlier_mag{m}' for m in SEV_TRAIN_MAG] + [f'spike_burst_mag{m}' for m in SEV_TRAIN_MAG] +
             [f'sensor_dropout_burst_burst{b}' for b in SEV_TRAIN_BURST] + [f'stale_run_burst{b}' for b in SEV_TRAIN_BURST])
    test = ([f'outlier_mag{m}' for m in SEV_TEST_MAG] + [f'spike_burst_mag{m}' for m in SEV_TEST_MAG] +
            [f'sensor_dropout_burst_burst{b}' for b in SEV_TEST_BURST] + [f'stale_run_burst{b}' for b in SEV_TEST_BURST])
    return train, test


def mechanism_regimes(cells_df, mechanism):
    return sorted(cells_df.loc[cells_df.mechanism == mechanism, 'regime'].unique().tolist())


def run_all_splits(cells_df):
    all_regimes = sorted(cells_df['regime'].unique().tolist())
    results = {}  # (statistic, split, held_out, scope) -> dict(units, summary)

    for statistic in ('location', 'spread'):
        long_df = cell_config_mae(cells_df, statistic)

        # --- A. unseen severity ---
        train_r, test_r = severity_split_regimes()
        for scope, domruns in [('combined', None), ('UCI_only', set(zip(cells_df[cells_df.domain == 'UCI'].domain, cells_df[cells_df.domain == 'UCI'].run)))]:
            res = headroom(long_df, statistic, train_regimes=train_r, test_regimes=test_r,
                            train_domain_runs=domruns, test_domain_runs=domruns,
                            split_name='unseen_severity', held_out_label='unseen_severity')
            if res:
                res['summary']['scope'] = scope
                results[(statistic, 'unseen_severity', 'unseen_severity', scope)] = res

        # --- B. leave-one-mechanism-out (7 folds) ---
        for mech in ALL_MECH:
            held_regimes = mechanism_regimes(cells_df, mech)
            train_regimes = [r for r in all_regimes if r not in held_regimes]
            for scope, domruns in [('combined', None), ('UCI_only', set(zip(cells_df[cells_df.domain == 'UCI'].domain, cells_df[cells_df.domain == 'UCI'].run)))]:
                res = headroom(long_df, statistic, train_regimes=train_regimes, test_regimes=held_regimes,
                                train_domain_runs=domruns, test_domain_runs=domruns,
                                split_name='leave_one_mechanism_out', held_out_label=mech)
                if res:
                    res['summary']['scope'] = scope
                    results[(statistic, 'leave_one_mechanism_out', mech, scope)] = res

        # --- C. leave-one-domain-out ---
        skab_runs = set(zip(cells_df[cells_df.domain == 'SKAB'].domain, cells_df[cells_df.domain == 'SKAB'].run))
        uci_runs = set(zip(cells_df[cells_df.domain == 'UCI'].domain, cells_df[cells_df.domain == 'UCI'].run))
        for train_dom, test_dom, train_set, test_set, label in [
            ('SKAB', 'UCI', skab_runs, uci_runs, 'train_SKAB_test_UCI'),
            ('UCI', 'SKAB', uci_runs, skab_runs, 'train_UCI_test_SKAB'),
        ]:
            res = headroom(long_df, statistic, train_regimes=None, test_regimes=None,
                            train_domain_runs=train_set, test_domain_runs=test_set,
                            split_name='leave_one_domain_out', held_out_label=label)
            if res:
                res['summary']['scope'] = 'cross_domain'
                results[(statistic, 'leave_one_domain_out', label, 'cross_domain')] = res

    return results


# ---------------------------------------------------------------------------
# Required diagnostics
# ---------------------------------------------------------------------------

ROBUST_LOCATION = {'median', 'trimmed_mean', 'huber_M'}
ROBUST_SPREAD = {'madn', 'trimmed_std', 'IQR_scale'}
NON_ROBUST_LOCATION = {'mean'}
NON_ROBUST_SPREAD = {'std'}


def regime_oracle_estimators(cells_df, domain=None):
    """Per regime (pooled hierarchically: cell -> run -> regime mean),
    the oracle's winning location estimator and winning spread estimator.
    domain=None pools both; domain='SKAB'/'UCI' restricts to one."""
    df = cells_df if domain is None else cells_df[cells_df.domain == domain]
    out = []
    for regime in sorted(df['regime'].unique()):
        row = dict(regime=regime, domain=domain if domain else 'combined')
        for statistic in ('location', 'spread'):
            long_df = cell_config_mae(df[df.regime == regime], statistic)
            run_mae = run_level_mae(long_df)
            if run_mae.empty:
                row[f'oracle_{statistic}'] = None
                continue
            scores = run_mae.groupby('estimator')['mae'].mean()
            row[f'oracle_{statistic}'] = scores.idxmin() if len(scores) else None
        out.append(row)
    return pd.DataFrame(out)


def mixed_configuration_frequency(cells_df):
    """Fraction of the 23 regimes (pooled across both domains, and
    separately per domain) where the oracle pairs a robust location with a
    non-robust spread or vice versa. Per-mechanism breakdown included."""
    results = {}
    for domain_label, domain in [('combined', None), ('SKAB', 'SKAB'), ('UCI', 'UCI')]:
        m = regime_oracle_estimators(cells_df, domain=domain)
        m['loc_robust'] = m['oracle_location'].apply(lambda e: e in ROBUST_LOCATION if e else np.nan)
        m['spr_robust'] = m['oracle_spread'].apply(lambda e: e in ROBUST_SPREAD if e else np.nan)
        m['mixed'] = m['loc_robust'] != m['spr_robust']
        m['mechanism'] = m['regime'].apply(lambda r: next(mech for mech in ALL_MECH if r == mech or r.startswith(mech + '_')))
        overall_frac = float(m['mixed'].mean())
        per_mech = m.groupby('mechanism')['mixed'].mean().to_dict()
        results[domain_label] = dict(overall_fraction_mixed=overall_frac, n_regimes=len(m),
                                      per_mechanism=per_mech, detail=m)
    return results


def oracle_configuration_map(cells_df):
    """Winning config (location estimator, spread estimator) per
    (mechanism, severity) regime, SKAB and UCI side by side."""
    skab = regime_oracle_estimators(cells_df, domain='SKAB').set_index('regime')
    uci = regime_oracle_estimators(cells_df, domain='UCI').set_index('regime')
    rows = []
    for regime in sorted(cells_df['regime'].unique()):
        rows.append(dict(
            regime=regime,
            skab_oracle_location=skab.loc[regime, 'oracle_location'] if regime in skab.index else None,
            skab_oracle_spread=skab.loc[regime, 'oracle_spread'] if regime in skab.index else None,
            uci_oracle_location=uci.loc[regime, 'oracle_location'] if regime in uci.index else None,
            uci_oracle_spread=uci.loc[regime, 'oracle_spread'] if regime in uci.index else None,
        ))
    m = pd.DataFrame(rows)
    m['location_agrees'] = m['skab_oracle_location'] == m['uci_oracle_location']
    m['spread_agrees'] = m['skab_oracle_spread'] == m['uci_oracle_spread']
    return m


def crossover_localization(cells_df):
    """For outlier and spike_burst, the magnitude at which the oracle
    SPREAD estimator flips from non-robust (std) to robust (madn/trimmed_
    std/IQR_scale), per dataset."""
    out = {}
    for mech in AMPLITUDE_MECH:
        out[mech] = {}
        for domain in ('SKAB', 'UCI'):
            df = cells_df[(cells_df.domain == domain) & (cells_df.mechanism == mech)]
            long_df = cell_config_mae(df, 'spread')
            run_mae = run_level_mae(long_df)
            seq = []
            for mag in MAGNITUDES:
                regime = f'{mech}_mag{mag}'
                sub = run_level_mae(long_df, regime_filter=[regime])
                if sub.empty:
                    seq.append((mag, None))
                    continue
                scores = sub.groupby('estimator')['mae'].mean()
                winner = scores.idxmin()
                seq.append((mag, winner, winner in ROBUST_SPREAD))
            out[mech][domain] = seq
            # find first flip point
            flip = None
            for i in range(1, len(seq)):
                prev_robust = seq[i - 1][2] if seq[i - 1][1] else None
                cur_robust = seq[i][2] if seq[i][1] else None
                if prev_robust is not None and cur_robust is not None and prev_robust != cur_robust:
                    flip = (seq[i - 1][0], seq[i][0])
                    break
            out[mech][f'{domain}_flip_between'] = flip
    return out


def per_channel_breakdown(cells_df):
    """Spread ratio (madn/std, geomean pooled across all 23 regimes) per
    channel, both domains -- replicates Phase 1C's per-channel check
    (specifically: is Pressure anomalous again?)."""
    rows = []
    for domain in ('SKAB', 'UCI'):
        df = cells_df[cells_df.domain == domain]
        for ch in sorted(df['channel'].unique()):
            sub = df[df.channel == ch]
            std_vals = sub['spr_mae_std'].to_numpy(dtype=float)
            madn_vals = sub['spr_mae_madn'].to_numpy(dtype=float)
            ratio = madn_vals / np.maximum(std_vals, 1e-300)
            ratio = ratio[np.isfinite(ratio) & (ratio > 0)]
            geo = float(np.exp(np.mean(np.log(ratio)))) if ratio.size else np.nan
            loc_ratio_vals = sub['loc_mae_median'].to_numpy(dtype=float) / np.maximum(sub['loc_mae_mean'].to_numpy(dtype=float), 1e-300)
            loc_ratio_vals = loc_ratio_vals[np.isfinite(loc_ratio_vals) & (loc_ratio_vals > 0)]
            loc_geo = float(np.exp(np.mean(np.log(loc_ratio_vals)))) if loc_ratio_vals.size else np.nan
            rows.append(dict(domain=domain, channel=ch, spr_madn_over_std_geomean=geo,
                              loc_median_over_mean_geomean=loc_geo, n_regimes=len(sub['regime'].unique())))
    return pd.DataFrame(rows)


def per_run_breakdown(cells_df):
    rows = []
    for domain in ('SKAB', 'UCI'):
        df = cells_df[cells_df.domain == domain]
        for run in sorted(df['run'].unique()):
            sub = df[df.run == run]
            std_vals = sub['spr_mae_std'].to_numpy(dtype=float)
            madn_vals = sub['spr_mae_madn'].to_numpy(dtype=float)
            ratio = madn_vals / np.maximum(std_vals, 1e-300)
            ratio = ratio[np.isfinite(ratio) & (ratio > 0)]
            geo = float(np.exp(np.mean(np.log(ratio)))) if ratio.size else np.nan
            rows.append(dict(domain=domain, run=run, spr_madn_over_std_geomean=geo))
    return pd.DataFrame(rows)
