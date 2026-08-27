"""
Phase 1C: external validity check on a second real-world domain.

Domain: SKAB (Skoltech Anomaly Benchmark) -- industrial water-circulation
pump testbed telemetry. 8 physical sensor channels (vibration RMS x2,
motor current, loop pressure, engine-body temperature, fluid thermocouple,
motor voltage, volumetric flow rate), 1 Hz sampling.

Deliberately different from UCI household smart-meter power: different
physical process, different sensing modality, different sampling rate,
different failure physics.

METRIC (identical in form to Phase 0 / Phase 1B):
  For a background signal x (real, uncorrupted-by-us) and a corrupted
  version x' (same signal + injected defects):
      err_E(t) = | E_W(x')(t) - E_W(x)(t) |
  where E_W is a rolling estimator over window W.
  MAE_E   = mean_t err_E(t)
  PEAK_E  = max_t err_E(t)   (also p99.9 reported: max is fragile)

  location ratio = MAE_median / MAE_mean
  spread   ratio = MAE_MADn   / MAE_std      (MADn = 1.4826*MAD, std-scaled)

  Ratios are computed WITHIN a (run, channel) cell, so the channel's
  physical units cancel. Pooling is done on log-ratios (geometric mean),
  which is the correct aggregation for ratio-valued quantities.

  The comparison is PAIRED on the same background: any real dynamics or
  pre-existing anomalies in the background appear identically in x and x'
  and therefore cancel in the paired difference.
"""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

W = 30
MAD_SCALE = 1.4826


# ---------------------------------------------------------------------------
# Rolling estimators
# ---------------------------------------------------------------------------

def rolling_stats(x, w=W):
    """Return (mean, median, std, mad_scaled) as rolling arrays, NaN-aware."""
    sw = sliding_window_view(x, w)
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(sw, axis=1)
        med = np.nanmedian(sw, axis=1)
        std = np.nanstd(sw, axis=1)
        mad = np.nanmedian(np.abs(sw - med[:, None]), axis=1) * MAD_SCALE
    return mean, med, std, mad


# ---------------------------------------------------------------------------
# Defect injection
# ---------------------------------------------------------------------------
# Family A: standardized mechanisms, identical in form to Phase 0 / 1B.
# Family B: domain-grounded mechanisms for industrial pump telemetry.
#   Realism justification (documented BEFORE running, per protocol):
#   - sensor_dropout_burst: industrial DAQ/telemetry links fail in bursts
#     (loose connector, network outage, buffer overflow), not as independent
#     per-sample losses. Contiguous gaps are the characteristic failure mode.
#   - stale_run: "stuck-at" sensor failure. Temperature and pressure
#     transducers commonly freeze and repeat their last reading; this is a
#     standard, widely-documented industrial instrumentation fault.
#   - spike_burst: motor switching transients / EMI produce CLUSTERED
#     amplitude spikes on electrical channels, not isolated single spikes.

def inject(x, kind, rate, rng, magnitude=None, burst_len=None):
    """Return corrupted copy of x. rate = fraction of records affected.

    magnitude: sigma-multiplier for amplitude defects (outlier, spike_burst).
        None -> ORIGINAL behavior, rng.uniform(5.0, 10.0) per affected point
        (unchanged, byte-identical to Phase 1C). A number -> that fixed
        sigma-multiplier applied to every affected point in this call (no
        per-point random draw), for a controlled severity grid.
    burst_len: contiguous run length for the 3 mechanisms that already have
        a burst concept (sensor_dropout_burst, stale_run, spike_burst).
        None -> ORIGINAL hardcoded defaults (15, 15, 5 respectively,
        byte-identical to Phase 1C). missing/duplicate/outlier/out_of_order
        have no burst concept in this harness and are unaffected by this
        argument (Phase 2A does not extend them -- see run script for why).
    Both new arguments are additive: every existing call site that doesn't
    pass them reproduces the original function exactly.
    """
    y = x.copy()
    n = len(y)
    k = max(1, int(round(rate * n)))

    if kind == "missing":
        idx = rng.choice(n, size=k, replace=False)
        y[idx] = np.nan

    elif kind == "duplicate":
        # a retransmitted/duplicated record: value repeats the previous one
        idx = rng.choice(np.arange(1, n), size=k, replace=False)
        y[idx] = y[idx - 1]

    elif kind == "outlier":
        # amplitude corruption scaled to the signal's own variability
        idx = rng.choice(n, size=k, replace=False)
        scale = np.nanstd(x)
        signs = rng.choice([-1.0, 1.0], size=k)
        mag = rng.uniform(5.0, 10.0, size=k) if magnitude is None else np.full(k, float(magnitude))
        y[idx] = y[idx] + signs * mag * scale

    elif kind == "out_of_order":
        # timestamp disorder: adjacent records swapped
        idx = rng.choice(np.arange(n - 1), size=k // 2 + 1, replace=False)
        for i in idx:
            y[i], y[i + 1] = y[i + 1], y[i]

    elif kind == "sensor_dropout_burst":
        # same total budget k, delivered as contiguous bursts
        burst = 15 if burst_len is None else int(burst_len)
        nb = max(1, k // burst)
        starts = rng.choice(max(1, n - burst), size=nb, replace=False)
        for s in starts:
            y[s:s + burst] = np.nan

    elif kind == "stale_run":
        # stuck-at sensor: value freezes for a run of samples
        burst = 15 if burst_len is None else int(burst_len)
        nb = max(1, k // burst)
        starts = rng.choice(np.arange(1, max(2, n - burst)), size=nb, replace=False)
        for s in starts:
            y[s:s + burst] = y[s - 1]

    elif kind == "spike_burst":
        # clustered electrical transients
        burst = 5 if burst_len is None else int(burst_len)
        nb = max(1, k // burst)
        scale = np.nanstd(x)
        starts = rng.choice(max(1, n - burst), size=nb, replace=False)
        for s in starts:
            signs = rng.choice([-1.0, 1.0], size=burst)
            mag = rng.uniform(5.0, 10.0, size=burst) if magnitude is None else np.full(burst, float(magnitude))
            y[s:s + burst] = y[s:s + burst] + signs * mag * scale

    else:
        raise ValueError(kind)

    return y


STANDARDIZED = ["missing", "duplicate", "outlier", "out_of_order"]
DOMAIN = ["sensor_dropout_burst", "stale_run", "spike_burst"]
ALL_DEFECTS = STANDARDIZED + DOMAIN


# ---------------------------------------------------------------------------
# Single cell evaluation
# ---------------------------------------------------------------------------

def eval_cell(x_clean, kind, rate, seed):
    rng = np.random.default_rng(seed)
    x_bad = inject(x_clean, kind, rate, rng)

    m0, md0, s0, mad0 = rolling_stats(x_clean)
    m1, md1, s1, mad1 = rolling_stats(x_bad)

    def stats(a, b):
        e = np.abs(a - b)
        e = e[np.isfinite(e)]
        if e.size == 0:
            return np.nan, np.nan, np.nan
        return float(np.mean(e)), float(np.max(e)), float(np.quantile(e, 0.999))

    mae_mean, pk_mean, p999_mean = stats(m1, m0)
    mae_med, pk_med, p999_med = stats(md1, md0)
    mae_std, pk_std, p999_std = stats(s1, s0)
    mae_mad, pk_mad, p999_mad = stats(mad1, mad0)

    def safe_ratio(a, b):
        if not np.isfinite(a) or not np.isfinite(b) or b <= 0:
            return np.nan
        return a / b

    return {
        "loc_mae_ratio": safe_ratio(mae_med, mae_mean),
        "spr_mae_ratio": safe_ratio(mae_mad, mae_std),
        "loc_peak_ratio": safe_ratio(pk_med, pk_mean),
        "spr_peak_ratio": safe_ratio(pk_mad, pk_std),
        "loc_p999_ratio": safe_ratio(p999_med, p999_mean),
        "spr_p999_ratio": safe_ratio(p999_mad, p999_std),
        "mae_mean": mae_mean, "mae_med": mae_med,
        "mae_std": mae_std, "mae_mad": mae_mad,
    }
