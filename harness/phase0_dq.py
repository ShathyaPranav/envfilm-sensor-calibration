"""Phase 0: pure DQ / query-accuracy comparison of the naive (rolling
mean+vol) vs robust (rolling median+MAD) feature-operator estimators under
streaming defects, on the corrected (post-snapshot-bug-fix) harness.

Scope discipline (per explicit instruction): this module reuses, UNMODIFIED,
`pilot.stream` (background generation), `pilot.incidents` (defect sampling
and injection -- identical RNG construction `seed*1_000_003+7`, identical
defect definitions/positions), and `pilot.pipeline.LanePipeline` (the
feature-operator math itself, W_S=30, both 'meanvol' and 'robust' branches).
It does NOT use the SGD learner, the PSI drift monitor/reset, or any of the
old project's downstream-harm metrics (Delta_total/inwindow/postwindow, A_q,
persistence, cascade). Those are ImpactDaQ-project concepts and are out of
scope for Phase 0, whose only question is: how accurately does each
estimator recover the TRUE underlying statistic, at each timestep, under
(a) defects and (b) no defects at all.

No snapshot/restore is used anywhere in this module -- this side-steps the
bug's entire code path rather than merely relying on its fix. Every
incident's local window is primed from scratch, directly from W_S known-true
preceding raw values, via a fresh (non-snapshot) LanePipeline. A fresh
LanePipeline starts every lane's circular-buffer pointer at 0
(pipeline.py __init__), which is unconditionally correct for a from-scratch
run -- the fixed bug only ever existed in the load_snapshot path, which is
never called here. (For the record: the 'robust' feature branch reads
`buf_x[:, :, :n_eff_s]` as an unordered SET once the window is fully warmed,
so median/MAD are pointer-order-independent even had a snapshot been used --
only the incremental mean/vol running sums along the old buggy path actually
depended on eviction order. This module doesn't rely on that fact -- it
avoids snapshotting altogether -- it's noted only to explain why the two
estimators are on equal, correct footing here.)
"""
import numpy as np
import pandas as pd

from . import stream, incidents as inc_mod
from .pipeline import LanePipeline, fit_psi_reference, D, W_S

PRIMARY_TYPES = ['missing', 'duplicate', 'outlier', 'out_of_order']

# severity_scale only ever parametrizes STALE's median duration (see
# PREREGISTRATION.md amendment 9) via a deterministic post-hoc transform of
# an already-drawn `severity` value -- it consumes no RNG draws of its own,
# so ANY value passed here leaves pos/type/dim/sign/severity (and therefore
# every primary-type incident) bit-identical. STALE is excluded from Phase 0
# (see run_phase0.py docstring for why), so this value is never read
# downstream of the boolean type filter; it's passed only because
# sample_incidents() requires a value.
NOMINAL_SEVERITY_SCALE = 1.0

TAU_DRIFT_DUMMY = 0.51875  # unused: no drift-monitor output is ever read here

# LOCAL_H: steps to simulate from an incident's arrival. Primary-type max
# window residency is dwell_s = W_S + dur - 1 = 30 + 2 - 1 = 31 (out_of_order,
# dur=2; missing/outlier/duplicate have dur=1 -> residency 30). LOCAL_H=40
# gives >= 8 steps of confirmed post-exit zero-error as a data-integrity
# check, at negligible extra compute cost.
LOCAL_H = 40


def sample_primary_incidents(seed, N=1000, span_lo=2000, span_hi=152000, d=D):
    """Regenerates the identical incident set a full pilot run would draw
    for this seed (same RNG construction, same N, same span), then drops
    STALE rows. Positions/types/dims/signs/severities of the remaining
    (up to) 4*N/5 primary-type incidents are therefore identical to what
    ARM3 / the original pilot drew for the same seed."""
    rng = np.random.default_rng(seed * 1_000_003 + 7)
    spec = inc_mod.sample_incidents(rng, N, span_lo, span_hi, d, NOMINAL_SEVERITY_SCALE)
    mask = spec['type'] != inc_mod.STALE
    return {k: v[mask] for k, v in spec.items()}


def _affected_dim_mask(spec, d=D):
    """True where dimension j is actually corrupted by incident i. Single-dim
    types (missing, outlier) touch only their drawn dim; duplicate/out_of_order
    overwrite the whole feature vector (see incidents.build_obs)."""
    n = len(spec['type'])
    mask = np.zeros((n, d), dtype=bool)
    single = np.isin(spec['type'], [inc_mod.MISSING, inc_mod.OUTLIER])
    mask[single, spec['dim'][single]] = True
    allmask = np.isin(spec['type'], [inc_mod.DUPLICATE, inc_mod.OOO])
    mask[allmask, :] = True
    return mask


def run_incident_phase0(seed, feature_mode, N=1000, span=150000, burn_in=2000):
    """For every primary-type incident drawn for `seed`: prime a fresh
    LanePipeline with the W_S true preceding raw values, then run LOCAL_H
    steps in parallel for a defect-observing lane and a defect-free
    (clean-continuation) lane sharing the identical priming. Returns one row
    per incident: mean/peak absolute error of the engineered-feature
    location statistic (mean or median) and spread statistic (vol or MAD),
    restricted to the incident's own affected dimension(s), between the
    defect-observing and clean-continuation trajectories.
    """
    assert feature_mode in ('meanvol', 'robust')
    span_hi = burn_in + span
    T = span_hi + LOCAL_H + 5
    x_true, _, _ = stream.generate_background(seed, T)
    spec = sample_primary_incidents(seed, N=N, span_lo=burn_in, span_hi=span_hi, d=D)
    n = len(spec['type'])
    pos = spec['pos']
    dim_mask = _affected_dim_mask(spec)

    edges, ref_pct = fit_psi_reference(x_true[:burn_in])  # dummy: drift stat is never read
    n_lanes = 2 * n  # first n = defect-observing, last n = clean-continuation
    pipe = LanePipeline(n_lanes, edges, ref_pct, TAU_DRIFT_DUMMY,
                         reset_enabled_mask=np.zeros(n_lanes, dtype=bool),
                         freeze_learner=True, feature_mode=feature_mode)

    pos2 = np.tile(pos, 2)
    y_dummy = np.zeros(n_lanes)
    # priming: W_S steps of shared true history -- identical for both blocks,
    # since no corruption has been injected yet.
    for k in range(W_S):
        idx = pos2 + (k - W_S)
        pipe.step(x_true[idx], y_dummy)

    sum_x0, sumsq_x0 = pipe.sum_x[:n].copy(), pipe.sumsq_x[:n].copy()
    local_std = np.sqrt(np.maximum(sumsq_x0 / W_S - (sum_x0 / W_S) ** 2, 1e-8))
    local_std_at_arrival = local_std[np.arange(n), spec['dim']]

    y_true_dummy = np.zeros(T)
    e_obs_trace = np.empty((LOCAL_H, n_lanes, 2 * D))
    for s in range(LOCAL_H):
        x_obs_def, _ = inc_mod.build_obs(s, spec, pos, x_true, y_true_dummy, local_std_at_arrival)
        x_obs_clean = x_true[pos + s]
        x_obs = np.concatenate([x_obs_def, x_obs_clean], axis=0)
        loss, pred, reset_fired, drift_stat, e_obs, grad_e = pipe.step(x_obs, y_dummy)
        e_obs_trace[s] = e_obs

    e_obs_def = e_obs_trace[:, :n, :]
    e_obs_clean_traj = e_obs_trace[:, n:, :]
    err = np.abs(e_obs_def - e_obs_clean_traj)  # (LOCAL_H, n, 2D)

    dm = dim_mask[None, :, :].astype(float)  # (1, n, D)
    denom = np.maximum(dm.sum(axis=2), 1.0)  # (1, n)

    loc_err_perstep = (err[:, :, :D] * dm).sum(axis=2) / denom  # (LOCAL_H, n)
    spr_err_perstep = (err[:, :, D:] * dm).sum(axis=2) / denom

    mae_loc = loc_err_perstep.mean(axis=0)
    peak_loc = loc_err_perstep.max(axis=0)
    mae_spr = spr_err_perstep.mean(axis=0)
    peak_spr = spr_err_perstep.max(axis=0)

    # data-integrity check: post-full-exit residual should be exactly zero
    dwell_s = W_S + spec['dur'] - 1
    post_exit_mask = np.arange(LOCAL_H)[:, None] >= (dwell_s[None, :] + 1)
    residual = np.where(post_exit_mask, np.maximum(loc_err_perstep, spr_err_perstep), 0.0)
    max_residual = float(residual.max()) if residual.size else 0.0

    df = pd.DataFrame(dict(
        seed=seed, incident_id=np.arange(n),
        type=[inc_mod.TYPE_NAMES[t] for t in spec['type']],
        dim=spec['dim'], dur=spec['dur'], position=pos,
        mae_loc=mae_loc, peak_loc=peak_loc, mae_spr=mae_spr, peak_spr=peak_spr,
    ))
    return df, max_residual


def run_clean_stream_seed_batch(seeds, feature_mode, T=60000, warm_skip=W_S):
    """No incidents. Vectorizes across `seeds` as independent lanes of a
    single LanePipeline (one Python-level time loop of length T total,
    regardless of seed count). For each t past warm_skip, records the
    absolute error of the location statistic (mean or median) against the
    AR(1) process's exact stationary marginal mean (0.0, by construction --
    see stream.py) and of the spread statistic (vol or MAD) against its
    exact stationary marginal std (1.0). Returns (loc_err, spr_err), each
    shape (T-warm_skip, n_seeds)."""
    assert feature_mode in ('meanvol', 'robust')
    n_seeds = len(seeds)
    x_stack = np.stack([stream.generate_background(sd, T)[0] for sd in seeds])  # (n_seeds, T, D)
    edges, ref_pct = fit_psi_reference(x_stack[0, :2000])  # dummy: drift stat never read
    pipe = LanePipeline(n_seeds, edges, ref_pct, TAU_DRIFT_DUMMY,
                         reset_enabled_mask=np.zeros(n_seeds, dtype=bool),
                         freeze_learner=True, feature_mode=feature_mode)
    y_dummy = np.zeros(n_seeds)
    loc_err = np.empty((T, n_seeds))
    spr_err = np.empty((T, n_seeds))
    for t in range(T):
        loss, pred, reset_fired, drift_stat, e_obs, grad_e = pipe.step(x_stack[:, t, :], y_dummy)
        loc_err[t] = np.mean(np.abs(e_obs[:, :D] - 0.0), axis=1)
        spr_err[t] = np.mean(np.abs(e_obs[:, D:] - 1.0), axis=1)
    return loc_err[warm_skip:], spr_err[warm_skip:]
