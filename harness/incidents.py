"""Incident type definitions, position sampling, and (D(q), D(-q)) observed
raw-value/label reconstruction for a batch of lanes at a given local step.

All five types are value-level corruptions at fixed event indices -- no
insertion or deletion -- so D(q) and D(-q) stay index-aligned for every t.
"""
import numpy as np

MISSING, OUTLIER, STALE, DUPLICATE, OOO = range(5)
TYPE_NAMES = ['missing', 'outlier', 'stale', 'duplicate', 'out_of_order']
MAX_DUR = 5


def sample_incidents(rng, n, span_lo, span_hi, D, severity_scale):
    """Returns a dict of (n,)-shaped arrays describing each incident.

    severity_scale is the calibration knob (see PREREGISTRATION.md amendment
    9): it sets the MEDIAN duration (in events) of STALE incidents, not their
    magnitude -- magnitude was tried first (amendments 1-8) and shown to have
    no measurable reach on reset-disagreement probability for a bucket-count
    PSI statistic, because a single-event perturbation's bucket-count
    contribution saturates almost immediately regardless of how far out it
    lands. Duration, by contrast, is not diluted the same way: more affected
    events in the W_l window means more bucket-count shift, linearly, up to
    W_l itself -- confirmed empirically monotonic (3.5% at dur=5 to 58% at
    dur=60 on the calibration seed).
    """
    pos = rng.integers(span_lo, span_hi, size=n)
    types = np.tile(np.arange(5), n // 5 + 1)[:n]
    rng.shuffle(types)
    dim = rng.integers(0, D, size=n)
    sign = rng.choice([-1.0, 1.0], size=n)
    # severity draw: median 1.0, shape sigma=0.5 fixed (median of
    # LogNormal(mu,sigma) = exp(mu), independent of sigma).
    severity = rng.lognormal(mean=0.0, sigma=0.5, size=n)

    dur = np.ones(n, dtype=int)
    stale_mask = types == STALE
    stale_dur = np.round(severity_scale * severity[stale_mask]).astype(int)
    dur[stale_mask] = np.clip(stale_dur, 2, 250)
    dur[types == OOO] = 2
    return dict(pos=pos, type=types, dim=dim, dur=dur, sign=sign, severity=severity)


def build_obs(step_s, incidents, p_arr, x_true, y_true, local_std_at_arrival):
    """Vectorized observed (x_obs, y_obs) for all lanes at local step `step_s`.

    x_true, y_true: full-length global arrays (T,D)/(T,).
    p_arr: (n_lanes,) incident position for each lane (an incident index is
           shared by the ON and OFF lane copies).
    local_std_at_arrival: (n_lanes,) pre-incident rolling std of the
           corrupted dim, used only to scale outlier severity (frozen at
           arrival time, not re-measured later -- part of the incident spec).
    """
    n = len(p_arr)
    gpos = p_arr + step_s
    x_obs = x_true[gpos].copy()
    y_obs = y_true[gpos].copy()

    typ = incidents['type']
    dim = incidents['dim']
    dur = incidents['dur']
    sign = incidents['sign']
    severity = incidents['severity']
    lane = np.arange(n)

    in_span = step_s < dur

    m = (typ == MISSING) & (step_s == 0)
    if m.any():
        x_obs[m, dim[m]] = x_true[p_arr[m] - 1, dim[m]]

    o = (typ == OUTLIER) & (step_s == 0)
    if o.any():
        x_obs[o, dim[o]] = x_true[p_arr[o], dim[o]] + sign[o] * severity[o] * np.maximum(local_std_at_arrival[o], 1e-3)

    s = (typ == STALE) & in_span
    if s.any():
        # freeze at pre-incident true value plus a severity-scaled bias, held
        # constant for the whole dwell (unlike outlier's single-event spike,
        # this compounds with dur -- see PREREGISTRATION.md amendment 4).
        x_obs[s, dim[s]] = x_true[p_arr[s] - 1, dim[s]] + sign[s] * severity[s] * np.maximum(local_std_at_arrival[s], 1e-3)

    d = (typ == DUPLICATE) & (step_s == 0)
    if d.any():
        x_obs[d, :] = x_true[p_arr[d] - 1, :]

    oo0 = (typ == OOO) & (step_s == 0)
    if oo0.any():
        x_obs[oo0, :] = x_true[p_arr[oo0] + 1, :]
        y_obs[oo0] = y_true[p_arr[oo0] + 1]
    oo1 = (typ == OOO) & (step_s == 1)
    if oo1.any():
        x_obs[oo1, :] = x_true[p_arr[oo1], :]
        y_obs[oo1] = y_true[p_arr[oo1]]

    return x_obs, y_obs


def last_affected_offset(incidents):
    """Local-step offset (relative to incident start) of the last event whose
    raw value differs between D(q) and D(-q). Needed for t_expire."""
    typ = incidents['type']
    dur = incidents['dur']
    off = np.zeros(len(typ), dtype=int)
    off[typ == MISSING] = 0
    off[typ == OUTLIER] = 0
    off[typ == STALE] = dur[typ == STALE] - 1
    off[typ == DUPLICATE] = 0
    off[typ == OOO] = 1
    return off
