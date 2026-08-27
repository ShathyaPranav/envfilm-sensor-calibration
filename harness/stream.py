"""Generative process for the background multivariate temporal stream.

x_true: (T, D) AR(1) Gaussian process per raw feature dimension.
y_true: (T,) Bernoulli labels generated from a FIXED ground-truth logistic
         function of the engineered (rolling mean/vol) features of x_true.

w_true/b_true are drawn once (seed 0) and frozen across every seed and every
run of the pilot, so the "task" being learned is identical everywhere.
"""
import numpy as np

D = 4          # raw feature dimensions
W_S = 30       # feature-operator rolling window
PHI = 0.6      # AR(1) autocorrelation


def _ar1(rng, T, d, phi=PHI):
    eps = rng.standard_normal((T, d))
    x = np.empty((T, d))
    x[0] = eps[0]
    coef = np.sqrt(1 - phi ** 2)
    for t in range(1, T):
        x[t] = phi * x[t - 1] + coef * eps[t]
    return x


def rolling_mean_vol(x, w=W_S):
    """Causal rolling mean/vol, no lookahead. x: (T, d) -> (T, 2d)."""
    T, d = x.shape
    mean = np.empty((T, d))
    vol = np.empty((T, d))
    csum = np.cumsum(x, axis=0)
    csumsq = np.cumsum(x ** 2, axis=0)
    for t in range(T):
        lo = max(0, t - w + 1)
        n = t - lo + 1
        s = csum[t] - (csum[lo - 1] if lo > 0 else 0.0)
        sq = csumsq[t] - (csumsq[lo - 1] if lo > 0 else 0.0)
        m = s / n
        v = np.sqrt(np.maximum(sq / n - m ** 2, 1e-8))
        mean[t] = m
        vol[t] = v
    return np.concatenate([mean, vol], axis=1)


def get_ground_truth_weights():
    rng = np.random.default_rng(0)
    w = rng.normal(0, 1.0, size=2 * D)
    b = rng.normal(0, 0.2)
    return w, b


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def generate_background(seed, T):
    """Returns x_true (T,D), e_true (T,2D) engineered from x_true, y_true (T,)."""
    rng = np.random.default_rng(seed)
    x_true = _ar1(rng, T, D)
    e_true = rolling_mean_vol(x_true, W_S)
    w_true, b_true = get_ground_truth_weights()
    p_true = sigmoid(e_true @ w_true + b_true)
    y_true = (rng.random(T) < p_true).astype(np.float64)
    return x_true, e_true, y_true
