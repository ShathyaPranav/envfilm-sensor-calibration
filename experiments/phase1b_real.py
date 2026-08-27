"""Phase 1B: does the Phase 0 location/spread dissociation survive on a real
streaming dataset? UCI "Individual Household Electric Power Consumption"
(archive.ics.uci.edu, dataset #235) -- 1-minute-cadence smart-meter
telemetry, Dec 2006 - Nov 2010, 2,075,259 rows.

Reuses UNMODIFIED: pilot.incidents (defect sampling/injection -- identical
functions/definitions to Phase 0/1A) and pilot.pipeline.LanePipeline
(feature-operator math, W_S=30, both branches). No SGD learner, no drift
monitor/reset, no downstream-harm metric. No snapshot/restore (same
from-scratch local priming as phase0_dq.py, same reasoning).

DEVIATIONS from Phase 0 (reported explicitly, not silently, per
instruction):

  1. Ground truth is REAL data, not an analytically-known AR(1) generator.
     "Seeds" are 10 non-overlapping ~1-week (10,080-minute) contiguous,
     fully non-missing segments of the real series, deterministically
     selected to spread across the full ~4-year span (the raw timestamp
     index has ZERO gaps -- every minute has a row -- but 1.25% of rows
     have NaN readings; segments are chosen from the runs between NaN
     rows), rather than 30 independent RNG realizations of a generator.
  2. Each segment's 4 raw channels (Global_active_power,
     Global_reactive_power, Voltage, Global_intensity -- the dataset's 4
     independently-metered numeric channels, playing the role of Phase 0's
     D=4 synthetic dims) are z-scored using THAT SEGMENT's own mean/std,
     computed once over the whole clean segment, BEFORE any defect
     injection. This is necessary only so cross-dimension MAE aggregation
     for duplicate/out_of_order (which corrupt all 4 dims at once, on
     physically incommensurable units -- kW, kVAR, V, A) is meaningful; it
     does not change the naive/robust RATIO for single-dim incidents
     (missing/outlier), which is scale-invariant under this transform.
  3. No analytic population truth (mean=0, std=1) exists for real data, so
     the clean-stream baseline uses each segment's own full-segment
     mean/std (naive reference) or median/MAD (robust reference) --
     computed once, held fixed -- as the reference level. This is the
     real-data analog of Phase 0's known population constants, not a new
     free parameter (no tuning: it's fixed by the data before any incident
     injection or estimator comparison happens).
  4. STALE excluded, exactly as Phase 0/1A (same rationale: no frozen
     duration definition independent of the excluded drift-monitor
     calibration).
  5. LanePipeline's edges/ref_pct constructor arguments are populated from
     a throwaway fit (drift monitor output is never read anywhere in this
     module).

Everything else -- W_S=30 (unchanged), incidents.py's defect
definitions/RNG-driven positions (unchanged), severity ~
LogNormal(0, 0.5) median-1x-local-std outlier magnitude (unchanged, same
call as Phase 0), the paired-difference MAE/peak methodology (unchanged) --
is identical to Phase 0.
"""
import os
import sys

import numpy as np
import pandas as pd

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from harness import incidents as inc_mod
from harness.pipeline import LanePipeline, fit_psi_reference, D, W_S
from harness import phase0_dq as p0  # LOCAL_H, TAU_DRIFT_DUMMY, NOMINAL_SEVERITY_SCALE reused

RAW_PATH = os.path.join(_REPO_ROOT, 'data', 'household_power_consumption.txt')
COLS = ['Global_active_power', 'Global_reactive_power', 'Voltage', 'Global_intensity']
SEG_LEN = 10080          # 1 week at 1-minute cadence
N_SEGMENTS = 10
N_INCIDENTS = 250        # -> ~50/type after STALE is dropped (round-robin over 5 types)
PRIME_MARGIN = 5          # incident positions must be >= W_S + margin into the segment
END_MARGIN = 5


def _load_clean_runs():
    df = pd.read_csv(RAW_PATH, sep=';', na_values=['?'], usecols=['Date', 'Time'] + COLS)
    dt = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H:%M:%S')
    assert dt.is_monotonic_increasing and (dt.diff().dropna() == pd.Timedelta(minutes=1)).all(), \
        "timestamp index is not perfectly regular -- assumption behind fixed-length segments broke"
    clean = ~df[COLS].isna().any(axis=1)
    clean = clean.values
    runs = []
    start = None
    for i, v in enumerate(clean):
        if v and start is None:
            start = i
        elif not v and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(clean)))
    need = SEG_LEN + W_S + PRIME_MARGIN + p0.LOCAL_H + END_MARGIN
    long_runs = [(s, e) for s, e in runs if e - s >= need]
    return df, dt, long_runs


def select_segments():
    """Deterministic: pick N_SEGMENTS runs evenly spread by position across
    the full sorted list of long-enough clean runs (spans the whole ~4-year
    timeline, not just its most-populated stretch), take the first SEG_LEN
    minutes of each."""
    df, dt, long_runs = _load_clean_runs()
    idx = np.unique(np.round(np.linspace(0, len(long_runs) - 1, N_SEGMENTS)).astype(int))
    segments = []
    for k, i in enumerate(idx):
        s, e = long_runs[i]
        seg_end = s + W_S + PRIME_MARGIN + SEG_LEN + p0.LOCAL_H + END_MARGIN
        raw = df[COLS].values[s:seg_end].astype(np.float64)
        seg_mean = raw.mean(axis=0)
        seg_std = raw.std(axis=0, ddof=1)
        x_std = (raw - seg_mean) / seg_std
        segments.append(dict(
            label=f'seg{k+1}', start_time=str(dt.iloc[s]), end_time=str(dt.iloc[seg_end - 1]),
            x=x_std, raw_mean=seg_mean, raw_std=seg_std, n_rows=seg_end - s,
        ))
    return segments


# ---------------------------------------------------------------------------
# incident experiment (mirrors phase0_dq.run_incident_phase0, real-data source)
# ---------------------------------------------------------------------------

def run_incident_real(segment, feature_mode, N=N_INCIDENTS, seed_for_rng=0):
    assert feature_mode in ('meanvol', 'robust')
    x_true = segment['x']
    T = len(x_true)
    span_lo = W_S + PRIME_MARGIN
    span_hi = T - p0.LOCAL_H - END_MARGIN

    rng = np.random.default_rng(seed_for_rng * 1_000_003 + 7)
    spec = inc_mod.sample_incidents(rng, N, span_lo, span_hi, D, p0.NOMINAL_SEVERITY_SCALE)
    mask = spec['type'] != inc_mod.STALE
    spec = {k: v[mask] for k, v in spec.items()}
    n = len(spec['type'])
    pos = spec['pos']
    dim_mask = p0._affected_dim_mask(spec)

    edges, ref_pct = fit_psi_reference(x_true[:min(500, T // 2)])  # dummy: drift stat never read
    n_lanes = 2 * n
    pipe = LanePipeline(n_lanes, edges, ref_pct, p0.TAU_DRIFT_DUMMY,
                         reset_enabled_mask=np.zeros(n_lanes, dtype=bool),
                         freeze_learner=True, feature_mode=feature_mode)

    pos2 = np.tile(pos, 2)
    y_dummy = np.zeros(n_lanes)
    for k in range(W_S):
        idx = pos2 + (k - W_S)
        pipe.step(x_true[idx], y_dummy)

    sum_x0, sumsq_x0 = pipe.sum_x[:n].copy(), pipe.sumsq_x[:n].copy()
    local_std = np.sqrt(np.maximum(sumsq_x0 / W_S - (sum_x0 / W_S) ** 2, 1e-8))
    local_std_at_arrival = local_std[np.arange(n), spec['dim']]

    y_true_dummy = np.zeros(T)
    e_obs_trace = np.empty((p0.LOCAL_H, n_lanes, 2 * D))
    for s in range(p0.LOCAL_H):
        x_obs_def, _ = inc_mod.build_obs(s, spec, pos, x_true, y_true_dummy, local_std_at_arrival)
        x_obs_clean = x_true[pos + s]
        x_obs = np.concatenate([x_obs_def, x_obs_clean], axis=0)
        loss, pred, reset_fired, drift_stat, e_obs, grad_e = pipe.step(x_obs, y_dummy)
        e_obs_trace[s] = e_obs

    e_obs_def = e_obs_trace[:, :n, :]
    e_obs_clean_traj = e_obs_trace[:, n:, :]
    err = np.abs(e_obs_def - e_obs_clean_traj)

    dm = dim_mask[None, :, :].astype(float)
    denom = np.maximum(dm.sum(axis=2), 1.0)
    loc_err_perstep = (err[:, :, :D] * dm).sum(axis=2) / denom
    spr_err_perstep = (err[:, :, D:] * dm).sum(axis=2) / denom

    mae_loc = loc_err_perstep.mean(axis=0)
    peak_loc = loc_err_perstep.max(axis=0)
    mae_spr = spr_err_perstep.mean(axis=0)
    peak_spr = spr_err_perstep.max(axis=0)

    dwell_s = W_S + spec['dur'] - 1
    post_exit_mask = np.arange(p0.LOCAL_H)[:, None] >= (dwell_s[None, :] + 1)
    residual = np.where(post_exit_mask, np.maximum(loc_err_perstep, spr_err_perstep), 0.0)
    max_residual = float(residual.max()) if residual.size else 0.0

    df = pd.DataFrame(dict(
        segment=segment['label'], incident_id=np.arange(n),
        type=[inc_mod.TYPE_NAMES[t] for t in spec['type']],
        dim=spec['dim'], dur=spec['dur'], position=pos,
        mae_loc=mae_loc, peak_loc=peak_loc, mae_spr=mae_spr, peak_spr=peak_spr,
    ))
    return df, max_residual


# ---------------------------------------------------------------------------
# clean-baseline (real-data analog of Phase 0's population-truth comparison)
# ---------------------------------------------------------------------------

def run_clean_baseline_real(segment, feature_mode, warm_skip=W_S):
    assert feature_mode in ('meanvol', 'robust')
    x_true = segment['x']
    T = len(x_true)
    edges, ref_pct = fit_psi_reference(x_true[:min(500, T // 2)])  # dummy
    pipe = LanePipeline(1, edges, ref_pct, p0.TAU_DRIFT_DUMMY,
                         reset_enabled_mask=np.zeros(1, dtype=bool),
                         freeze_learner=True, feature_mode=feature_mode)
    y_dummy = np.zeros(1)

    # segment-level reference (naive: mean/std; robust: median/MAD), computed
    # once over the WHOLE segment, held fixed -- the real-data analog of
    # Phase 0's known population truth (0, 1).
    if feature_mode == 'meanvol':
        ref_loc = x_true.mean(axis=0)
        ref_spr = x_true.std(axis=0, ddof=1)
    else:
        ref_loc = np.median(x_true, axis=0)
        ref_spr = 1.4826 * np.median(np.abs(x_true - ref_loc[None, :]), axis=0)

    loc_err = np.empty((T, D))
    spr_err = np.empty((T, D))
    for t in range(T):
        loss, pred, reset_fired, drift_stat, e_obs, grad_e = pipe.step(x_true[t:t + 1], y_dummy)
        loc_err[t] = np.abs(e_obs[0, :D] - ref_loc)
        spr_err[t] = np.abs(e_obs[0, D:] - ref_spr)
    return loc_err[warm_skip:].mean(axis=1), spr_err[warm_skip:].mean(axis=1)  # (T-warm_skip,)
