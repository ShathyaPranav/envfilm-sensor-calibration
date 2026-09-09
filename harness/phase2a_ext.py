"""Phase 2A (full, two-domain) headroom gate: 16-configuration estimator
pool built on top of the UNMODIFIED harness1c.rolling_stats/eval_cell error
definition.

Reuses BYTE-IDENTICAL from harness1c.py: rolling_stats() (mean/median/std/
madn), inject() (extended with magnitude/burst_len -- see harness1c.py's
own docstring for that additive change), and the error formula --
err(t) = |E_W(x')(t) - E_W(x)(t)|, MAE = mean_t, PEAK = max_t -- copied
verbatim from eval_cell()'s inner `stats()` closure (duplicated here only
because it's a nested function and can't be imported directly; the
arithmetic is identical, not redefined).

NEW here: 4 additional rolling estimators (trimmed_mean, huber_M,
trimmed_std, IQR_scale), fully vectorized (no Python per-window loop) and
NaN-aware, computed over the SAME W=30 sliding windows harness1c.rolling_stats
uses. Combined with the 4 existing estimators this gives the 4 (location) x
4 (spread) = 16 configuration pool.
"""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.special import ndtri

from .harness1c import W, MAD_SCALE, inject, rolling_stats

TRIM_FRAC = 0.10
HUBER_C = 1.345
HUBER_ITERS = 5

LOCATION_ESTIMATORS = ['mean', 'median', 'trimmed_mean', 'huber_M']
SPREAD_ESTIMATORS = ['std', 'madn', 'trimmed_std', 'IQR_scale']
INTERMEDIATE_LOCATION = ['trimmed_mean', 'huber_M']
INTERMEDIATE_SPREAD = ['trimmed_std', 'IQR_scale']


def _percentile_from_sorted(sw_sorted, n_valid, p, w):
    """Linear-interpolation percentile (matches numpy's default method),
    computed from an already row-sorted (NaNs-last) array -- avoids a
    second internal sort inside np.nanpercentile, which otherwise dominates
    runtime (profiled: >100x the cost of every other step combined)."""
    idx = np.clip((p / 100.0) * (n_valid - 1), 0, None)
    lo = np.clip(np.floor(idx).astype(int), 0, w - 1)
    hi = np.clip(np.ceil(idx).astype(int), 0, w - 1)
    frac = idx - lo
    rows = np.arange(sw_sorted.shape[0])
    val = sw_sorted[rows, lo] * (1 - frac) + sw_sorted[rows, hi] * frac
    return np.where(n_valid < 1, np.nan, val)


def _trim_consistency(alpha):
    """Gaussian consistency factor for a symmetrically trimmed standard
    deviation, i.e. E[nanstd(retained)] / sigma when the underlying data is
    N(mu, sigma^2) and a fraction `alpha` is removed from EACH tail.

    Derivation: the retained points are a doubly-truncated normal on
    [-z, +z] with z = Phi^-1(1-alpha), whose variance is
        sigma^2 * (1 - 2*z*phi(z)/(1-2*alpha)),
    so the plain std of the retained values reads low by the square root of
    that factor. Dividing by it puts trimmed_std on the same scale as the
    standard deviation.

    This is the exact analogue of the 1.4826 applied to MAD and the 1/1.349
    applied to the IQR elsewhere in this harness -- constants that exist so
    those estimators are directly comparable to std (see the paper's
    Section III). trimmed_std was the one spread estimator in the 16-config
    pool without such a factor, which gave it a scale advantage (not a
    robustness advantage) under the absolute error metric
    err = |E(x') - E(x)|: at TRIM_FRAC=0.10 it reads ~0.69x std, so ~31% of
    its measured error reduction was scale alone.

    alpha is the EFFECTIVE trim fraction k/n_valid (k = floor(trim_frac *
    n_valid)), not the nominal trim_frac -- flooring means e.g. trim_frac
    0.05 removes only 1 of 30 points (3.3%), and the correction must match
    what was actually removed.
    """
    alpha = np.asarray(alpha, dtype=float)
    out = np.ones_like(alpha)
    m = (alpha > 0) & (alpha < 0.5)
    if np.any(m):
        a = alpha[m]
        z = ndtri(1.0 - a)
        phi = np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)
        var_frac = 1.0 - 2.0 * z * phi / (1.0 - 2.0 * a)
        out[m] = np.sqrt(np.maximum(var_frac, 1e-12))
    out[alpha >= 0.5] = np.nan          # nothing retained; estimator undefined
    return np.maximum(out, 1e-6)


def rolling_extended(x, w=W, trim_frac=None, consistency=False):
    """Vectorized, NaN-aware trimmed_mean/huber_M/trimmed_std/IQR_scale over
    the same rolling windows rolling_stats() uses.

    trim_frac: fraction trimmed from each tail for trimmed_mean/trimmed_std.
    None -> module default (TRIM_FRAC=0.10, byte-identical to every prior
    use). 0.0 makes trimmed_mean/trimmed_std reduce to plain mean/std (no
    points dropped); IQR_scale and huber_M are unaffected by this parameter.

    consistency: False -> ORIGINAL behavior, byte-identical to every prior
    use (trimmed_std is the raw np.nanstd of the retained points, which
    reads systematically low and lower the more you trim). True -> divide
    trimmed_std by the Gaussian consistency factor above so it estimates
    sigma, matching how MAD and IQR are already scaled. trimmed_mean needs
    no such factor (it is already unbiased for a symmetric distribution)
    and is returned unchanged either way.
    """
    trim_frac = TRIM_FRAC if trim_frac is None else trim_frac
    sw = sliding_window_view(x, w)              # (N, w)
    N = sw.shape[0]
    valid = np.isfinite(sw)
    n_valid = valid.sum(axis=1)                  # (N,)

    with np.errstate(invalid='ignore'):
        sw_sorted = np.sort(sw, axis=1)           # NaNs sort to the end
        ranks = np.arange(w)[None, :]
        k = np.floor(trim_frac * n_valid)[:, None]
        keep = (ranks >= k) & (ranks < (n_valid[:, None] - k)) & (n_valid[:, None] - 2 * k > 0)
        trimmed_vals = np.where(keep, sw_sorted, np.nan)
        trimmed_mean = np.nanmean(trimmed_vals, axis=1)
        trimmed_std = np.nanstd(trimmed_vals, axis=1)
        if consistency:
            alpha_eff = k[:, 0] / np.maximum(n_valid, 1)
            trimmed_std = trimmed_std / _trim_consistency(alpha_eff)

        q25 = _percentile_from_sorted(sw_sorted, n_valid, 25, w)
        q75 = _percentile_from_sorted(sw_sorted, n_valid, 75, w)
        iqr_scale = np.maximum((q75 - q25) / 1.349, 1e-8)

        med = np.nanmedian(sw, axis=1)
        madn = np.nanmedian(np.abs(sw - med[:, None]), axis=1) * MAD_SCALE
        scale = np.maximum(madn, 1e-8)
        sw_filled = np.where(valid, sw, 0.0)
        mu = med.copy()
        for _ in range(HUBER_ITERS):
            resid = (sw_filled - mu[:, None]) / scale[:, None]
            absr = np.abs(resid)
            wgt = np.where(absr <= HUBER_C, 1.0, HUBER_C / np.maximum(absr, 1e-12))
            wgt = np.where(valid, wgt, 0.0)
            wsum = wgt.sum(axis=1)
            mu = np.where(wsum > 0, (wgt * sw_filled).sum(axis=1) / np.maximum(wsum, 1e-12), np.nan)
        huber_M = mu

    # rows with too few valid points for a trim to make sense fall back to NaN
    # (propagates through MAE/PEAK via the same np.isfinite filter eval_cell uses)
    return trimmed_mean, huber_M, trimmed_std, iqr_scale


def _stats(a, b):
    """Copied verbatim from harness1c.eval_cell's inner closure -- same
    err(t)=|a-b|, MAE=mean, PEAK=max, plus p99.9. Not a redefinition."""
    e = np.abs(a - b)
    e = e[np.isfinite(e)]
    if e.size == 0:
        return np.nan, np.nan, np.nan
    return float(np.mean(e)), float(np.max(e)), float(np.quantile(e, 0.999))


def eval_cell_16(x_clean, kind, rate, seed, magnitude=None, burst_len=None,
                  precomputed_clean=None, w=None, trim_frac=None, consistency=False):
    """Like harness1c.eval_cell, but over all 16 (location x spread)
    configurations. Returns a flat dict: mae_<est>, peak_<est> for each of
    the 8 estimator names, plus mae/peak for all 16 pairwise combinations
    is not needed -- location and spread errors are estimator-specific and
    combine freely at analysis time (a "configuration" = one location
    estimator's series + one spread estimator's series, independently
    correct/incorrect).

    precomputed_clean: optional (mean0,med0,std0,mad0,tm0,hu0,ts0,iqr0) tuple
    to skip recomputation of the clean side across many regime/seed calls
    against the same background (the clean side never changes).
    w: rolling window size. None -> module default (W=30, byte-identical to
    every prior use of this function). Passing a value threads it through
    to rolling_stats/rolling_extended, which already accept w explicitly;
    nothing else in the estimator math changes.
    """
    w = W if w is None else w
    rng = np.random.default_rng(seed)
    x_bad = inject(x_clean, kind, rate, rng, magnitude=magnitude, burst_len=burst_len)

    if precomputed_clean is None:
        m0, md0, s0, mad0 = rolling_stats(x_clean, w)
        tm0, hu0, ts0, iqr0 = rolling_extended(x_clean, w, trim_frac, consistency)
    else:
        m0, md0, s0, mad0, tm0, hu0, ts0, iqr0 = precomputed_clean

    m1, md1, s1, mad1 = rolling_stats(x_bad, w)
    tm1, hu1, ts1, iqr1 = rolling_extended(x_bad, w, trim_frac, consistency)

    loc_series = dict(mean=(m1, m0), median=(md1, md0), trimmed_mean=(tm1, tm0), huber_M=(hu1, hu0))
    spr_series = dict(std=(s1, s0), madn=(mad1, mad0), trimmed_std=(ts1, ts0), IQR_scale=(iqr1, iqr0))

    out = {}
    for name, (a, b) in loc_series.items():
        mae, pk, p999 = _stats(a, b)
        out[f'loc_mae_{name}'] = mae
        out[f'loc_peak_{name}'] = pk
    for name, (a, b) in spr_series.items():
        mae, pk, p999 = _stats(a, b)
        out[f'spr_mae_{name}'] = mae
        out[f'spr_peak_{name}'] = pk
    return out


def precompute_clean(x_clean, w=None, trim_frac=None, consistency=False):
    w = W if w is None else w
    m0, md0, s0, mad0 = rolling_stats(x_clean, w)
    tm0, hu0, ts0, iqr0 = rolling_extended(x_clean, w, trim_frac, consistency)
    return (m0, md0, s0, mad0, tm0, hu0, ts0, iqr0)
