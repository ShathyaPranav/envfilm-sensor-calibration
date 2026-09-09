"""Part B: proposed adaptive-trim-level methods, evaluated against the
corrected (direct-form) perturbation-magnitude finding of the paper's
Section V-A/VII-A, and against RQ3's held-out split protocol.

Two candidate online methods, both restricted to features computable from
the CURRENT (possibly dirty) window alone -- no clean-side reference is
used at inference time, only for offline oracle-label generation:

  ATS (rule-based): map an observable, ground-truth-free perturbation
    PROXY (a gap between two already-computed window statistics) onto a
    trim level tau via a monotonic, quantile-calibrated lookup fit on
    TRAINING rows only.

  LTP (learned): HistGradientBoostingRegressor and a shallow MLPRegressor,
    both predicting continuous tau* from 9 online-computable window
    features, trained on TRAINING rows only, snapped to the nearest of the
    5 tested trim levels at evaluation time (the granularity we have
    precomputed errors for).

Both are evaluated on the SAME 3 held-out splits as RQ3 (unseen severity,
leave-one-mechanism-out x7, leave-one-domain-out x2), reusing the split
builders in phase2a_full_analysis.py so the partition is identical to the
one already used for the paper's headroom gate, not a new ad hoc split.

Compute-budget disclosure (reported honestly in output, not silent):
  - 2 seeds (0,1), not the paper's usual 10 -- a full per-window sweep at
    10 seeds and stride 1 is not tractable in the time available for this
    check; this is a secondary sensitivity/feasibility experiment, not
    a primary result.
  - stride = 2*W = 60 (non-overlapping-ish sampling of window positions)
    rather than every single window position, since adjacent windows
    overlap in 29/30 points and are highly redundant.
  - timestamp-monotonicity-violation feature is well-defined (a companion
    position array run through the identical out_of_order permutation)
    ONLY for the out_of_order mechanism; it is reported as 0 for every
    other mechanism by construction, since this harness has no separate
    timestamp channel for the other mechanisms to violate.
"""
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from scipy import stats as spstats

warnings.filterwarnings('ignore', category=RuntimeWarning)

_EXP_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_EXP_DIR)
for p in (_EXP_DIR, _REPO_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from run_phase2a_full import load_series, build_regime_grid, RATE, ALL_MECH
from phase2a_full_analysis import severity_split_regimes, mechanism_regimes
from harness.harness1c import W, inject
from harness import phase2a_ext as p2x

OUT_DIR = os.path.join(_REPO_ROOT, 'results', 'adaptive_trim')
SEEDS = [0, 1]
STRIDE = 2 * W  # 60
TAUS = [0.0, 0.05, 0.10, 0.20, 0.30]
EPS = 1e-8

# Consistency-correct trimmed_std so every tau level estimates the same
# quantity (sigma). Without this, higher tau reads systematically lower
# (0.69x std at tau=0.10, 0.30x at tau=0.30), and the absolute error metric
# rewards that shrinkage directly -- which is what made tau=0.30 win every
# split in the first run. See phase2a_ext._trim_consistency.
CONSISTENCY = '--no-consistency' not in sys.argv
SUFFIX = '' if not CONSISTENCY else '_corrected'


# ---------------------------------------------------------------------------
# Per-window feature + label extraction for one (series, regime, seed) cell
# ---------------------------------------------------------------------------

def _window_moments(sw, mean, std, n_valid, w):
    """NaN-aware population skewness / excess kurtosis per row."""
    z = (sw - mean[:, None]) / np.maximum(std[:, None], EPS)
    skew = np.nanmean(z ** 3, axis=1)
    kurt = np.nanmean(z ** 4, axis=1) - 3.0
    return skew, kurt


def _unique_ratio(sw_sorted, n_valid, w):
    d = np.diff(sw_sorted, axis=1)
    d = np.where(np.isfinite(d), d, 1.0)  # NaN gaps (past end of valid data) don't count as new uniques
    changes = (d != 0).sum(axis=1)
    n_unique = 1 + changes
    return n_unique / np.maximum(n_valid, 1)


def _flat_run_length(sw, w):
    """Longest run of consecutive identical values per row (NaN never
    joins a run: NaN==NaN is False)."""
    same = (sw[:, 1:] == sw[:, :-1])
    idx = np.arange(w - 1)[None, :]
    reset_at = ~same
    group_id = np.where(reset_at, idx, -1)
    group_id = np.maximum.accumulate(group_id, axis=1)
    run_len_pairs = np.where(same, idx - group_id, 0)
    max_run_pairs = run_len_pairs.max(axis=1)
    return max_run_pairs + 1  # k equal adjacent pairs = a run of k+1 identical values


def _lag1_autocorr(sw, mean, var):
    a = sw[:, :-1] - mean[:, None]
    b = sw[:, 1:] - mean[:, None]
    prod = a * b
    num = np.nanmean(prod, axis=1)
    return num / np.maximum(var, EPS)


def _timestamp_violations(pos_bad_full, w, sample_idx):
    posw = sliding_window_view(pos_bad_full, w)[sample_idx]
    finite_pair = np.isfinite(posw[:, 1:]) & np.isfinite(posw[:, :-1])
    inv = (posw[:, 1:] < posw[:, :-1]) & finite_pair
    return inv.sum(axis=1).astype(float)


def build_cell(x_clean, channel_sigma, clean_cache, mechanism, sev_type, sev_value, seed,
               domain, run, channel, regime_label):
    n = len(x_clean)
    kwargs = {}
    if sev_type == 'magnitude':
        kwargs['magnitude'] = sev_value
    elif sev_type == 'burst_len':
        kwargs['burst_len'] = sev_value

    rng = np.random.default_rng(seed)
    x_bad = inject(x_clean, mechanism, RATE, rng, **kwargs)

    # --- clean side (cached per series) ---
    std0, madn0, iqr0, trimmed_std0_by_tau = clean_cache

    # --- dirty side ---
    m1, md1, std1, mad1 = p2x.rolling_stats(x_bad, W)
    trimmed_std1_by_tau = {}
    iqr1 = None
    for tau in TAUS:
        tm1, hu1, ts1, iq1 = p2x.rolling_extended(x_bad, W, trim_frac=tau, consistency=CONSISTENCY)
        trimmed_std1_by_tau[tau] = ts1
        if iqr1 is None:
            iqr1 = iq1

    n_windows = len(std1)
    sample_idx = np.arange(0, n_windows, STRIDE)
    if sample_idx.size == 0:
        return None

    err_std = np.abs(std1 - std0)[sample_idx]
    err_madn = np.abs(mad1 - madn0)[sample_idx]
    err_iqr = np.abs(iqr1 - iqr0)[sample_idx]
    err_tau = {tau: np.abs(trimmed_std1_by_tau[tau] - trimmed_std0_by_tau[tau])[sample_idx] for tau in TAUS}

    tau_arr = np.array(TAUS)
    err_matrix = np.stack([err_tau[tau] for tau in TAUS], axis=1)  # (n_sample, 5)
    oracle_idx = np.nanargmin(np.where(np.isfinite(err_matrix), err_matrix, np.inf), axis=1)
    oracle_tau = tau_arr[oracle_idx]
    oracle_err = err_matrix[np.arange(len(oracle_idx)), oracle_idx]

    # --- features (dirty side only, no clean reference) ---
    sw = sliding_window_view(x_bad, W)[sample_idx]
    valid = np.isfinite(sw)
    n_valid = valid.sum(axis=1)
    mean_w = np.nanmean(sw, axis=1)
    std_w = np.nanstd(sw, axis=1)
    var_w = std_w ** 2
    med_w = np.nanmedian(sw, axis=1)
    sw_sorted = np.sort(sw, axis=1)

    gap_trim_untrim = (std_w - trimmed_std1_by_tau[0.10][sample_idx]) / np.maximum(std_w, EPS)
    gap_mean_median = np.abs(mean_w - med_w) / np.maximum(std_w, EPS)
    gap_iqr_std = (std_w - iqr1[sample_idx]) / np.maximum(std_w, EPS)

    skew, kurt = _window_moments(sw, mean_w, std_w, n_valid, W)
    flat_run = _flat_run_length(sw, W)
    nan_frac = 1.0 - n_valid / W
    uniq_ratio = _unique_ratio(sw_sorted, n_valid, W)
    ac1 = _lag1_autocorr(sw, mean_w, var_w)

    if mechanism == 'out_of_order':
        rng_pos = np.random.default_rng(seed)
        pos = np.arange(n, dtype=float)
        pos_bad = inject(pos, mechanism, RATE, rng_pos, **kwargs)
        ts_viol = _timestamp_violations(pos_bad, W, sample_idx)
    else:
        ts_viol = np.zeros(len(sample_idx))

    m = len(sample_idx)
    df = pd.DataFrame(dict(
        domain=domain, run=run, channel=channel, mechanism=mechanism,
        sev_type=sev_type, sev_value=sev_value, regime=regime_label, seed=seed,
        t=sample_idx, channel_sigma=channel_sigma,
        err_std=err_std, err_madn=err_madn, err_iqr=err_iqr,
        err_tau00=err_tau[0.0], err_tau05=err_tau[0.05], err_tau10=err_tau[0.10],
        err_tau20=err_tau[0.20], err_tau30=err_tau[0.30],
        oracle_tau=oracle_tau, oracle_err=oracle_err,
        gap_trim_untrim=gap_trim_untrim, gap_mean_median=gap_mean_median, gap_iqr_std=gap_iqr_std,
        skew=skew, kurt=kurt, flat_run=flat_run, nan_frac=nan_frac,
        uniq_ratio=uniq_ratio, ac1=ac1, ts_viol=ts_viol,
    ))
    return df


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.time()
    series_list = load_series()
    regimes = build_regime_grid()
    print(f"=== {len(series_list)} series, {len(regimes)} regimes, seeds={SEEDS}, stride={STRIDE}, "
          f"consistency={CONSISTENCY} ===")

    all_dfs = []
    for si, s in enumerate(series_list):
        x = s['x']
        finite = x[np.isfinite(x)]
        channel_sigma = float(np.nanstd(finite)) if finite.size else np.nan
        std0, md0, s0, mad0 = p2x.rolling_stats(x, W)
        trimmed_std0_by_tau = {}
        iqr0 = None
        for tau in TAUS:
            tm0, hu0, ts0, iq0 = p2x.rolling_extended(x, W, trim_frac=tau, consistency=CONSISTENCY)
            trimmed_std0_by_tau[tau] = ts0
            if iqr0 is None:
                iqr0 = iq0
        clean_cache = (s0, mad0, iqr0, trimmed_std0_by_tau)

        for r in regimes:
            for seed in SEEDS:
                df = build_cell(x, channel_sigma, clean_cache, r['mechanism'], r['sev_type'], r['sev_value'],
                                 seed, s['domain'], s['run'], s['channel'], r['label'])
                if df is not None:
                    all_dfs.append(df)
        if (si + 1) % 10 == 0:
            print(f"  series {si+1}/{len(series_list)}  rows so far: {sum(len(d) for d in all_dfs)}  "
                  f"elapsed {time.time()-t0:.1f}s")

    full = pd.concat(all_dfs, ignore_index=True)
    print(f"\nTotal rows: {len(full)}  elapsed: {time.time()-t0:.1f}s")
    out_path = os.path.join(OUT_DIR, f'adaptive_trim_percell{SUFFIX}.csv.gz')
    full.to_csv(out_path, index=False, compression='gzip')
    print(f"Wrote {out_path}")


if __name__ == '__main__':
    main()
