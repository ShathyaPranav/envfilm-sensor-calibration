"""CCT-Norm: Consistency-Corrected Trimmed normalization.

A normalization layer whose ROBUSTNESS LEVEL is a learned parameter.

Every standard normalization -- BatchNorm, LayerNorm, GroupNorm, InstanceNorm,
sklearn's StandardScaler -- centres on a mean and scales by a standard
deviation. This paper's RQ1 shows those are the two estimators that degrade
fastest under contamination. The obvious repair is to trim, but then the trim
fraction is a hyperparameter, and this paper's Section VII-B shows the right
trim level is regime-dependent, so it should be learned.

THE OBSTACLE, AND THE POINT OF THIS LAYER. A trim fraction cannot simply be
made learnable. An uncorrected trimmed standard deviation reads systematically
low, and lower the more you trim -- measured in this paper at 0.69x sigma for
10% trimming and 0.30x for 30%. So alpha has a direct, monotone effect on the
layer's OUTPUT SCALE that is entirely independent of any robustness benefit.
Gradient descent will therefore move alpha to manipulate activation scale for
whatever the loss happens to prefer, and the learned value carries no
information about contamination. The robustness parameter is unlearnable in
principle.

CCT-Norm removes that degree of freedom by dividing out the Gaussian
consistency factor INSIDE the layer, differentiably:

    c(alpha)^2 = 1 - 2 z phi(z) / (1 - 2 alpha),      z = Phi^{-1}(1 - alpha)

so the emitted scale estimates sigma for EVERY setting of alpha. Changing alpha
then leaves output scale (approximately) invariant under the null, and alpha can
only move for genuine robustness reasons. That is the claim, and
run_cctnorm_mechanism.py is built to falsify it: with the correction the learned
alpha should track the true contamination rate; without it, alpha should be
driven by loss-scale manipulation and be roughly blind to contamination.

Soft-trimming uses a differentiable rank: a pairwise-sigmoid empirical CDF, so
gradients reach alpha through the weights rather than through a hard order
statistic (which has zero gradient almost everywhere).
"""
import numpy as np
import torch
import torch.nn as nn

ALPHA_MAX = 0.45          # 0.5 would trim the entire window away at any W
ALPHA_MIN = 1e-4
EPS = 1e-8


def consistency_factor(alpha):
    """c(alpha) such that (trimmed std)/c(alpha) is Fisher-consistent for sigma
    under a Gaussian null. Differentiable in alpha via torch.special.ndtri.

    Derivation: retaining the central (1-2 alpha) mass of N(0,1) truncates to
    [-z, z] with z = Phi^{-1}(1-alpha); that truncated distribution has variance
    1 - 2 z phi(z)/(1 - 2 alpha).
    """
    a = alpha.clamp(ALPHA_MIN, ALPHA_MAX)
    z = torch.special.ndtri(1.0 - a)
    phi = torch.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)
    var_frac = 1.0 - 2.0 * z * phi / (1.0 - 2.0 * a)
    return torch.sqrt(var_frac.clamp_min(1e-6))


class CCTNorm(nn.Module):
    """Consistency-Corrected Trimmed normalization over the sample axis.

    forward(v): v is (n, C). Returns (m, s) per channel, each (C,), where m is a
    soft-trimmed location and s a soft-trimmed, consistency-corrected scale.

    Parameters
    ----------
    n_ch      : number of channels (per-channel trim logits if per_channel)
    correct   : if False, the consistency division is REMOVED. This is the
                ablation arm -- the layer then behaves like a naive learnable
                trimmed statistic, and alpha becomes a scale knob.
    """

    def __init__(self, n_ch, correct=True, per_channel=True, init_alpha=0.10):
        super().__init__()
        self.correct = correct
        # invert alpha = ALPHA_MAX * sigmoid(a) to initialise at init_alpha
        p = init_alpha / ALPHA_MAX
        a0 = float(np.log(p / (1 - p)))
        shape = (n_ch,) if per_channel else (1,)
        self.a_loc = nn.Parameter(torch.full(shape, a0))
        self.a_sc = nn.Parameter(torch.full(shape, a0))
        self.log_Tr = nn.Parameter(torch.tensor(np.log(0.15), dtype=torch.float32))
        self.log_Tw = nn.Parameter(torch.tensor(np.log(0.05), dtype=torch.float32))

    def alphas(self):
        return (ALPHA_MAX * torch.sigmoid(self.a_loc),
                ALPHA_MAX * torch.sigmoid(self.a_sc))

    def _soft_weights(self, v, alpha):
        """v: (n, C). Returns normalised soft-trim weights (n, C)."""
        n = v.shape[0]
        Tr = self.log_Tr.exp().clamp(1e-3, 10.0)
        Tw = self.log_Tw.exp().clamp(1e-3, 10.0)
        # robust pre-scaling so the sigmoid temperature is in comparable units
        med = v.median(dim=0, keepdim=True).values
        mad = (v - med).abs().median(dim=0, keepdim=True).values * 1.4826 + EPS
        u = (v - med) / mad
        # soft empirical CDF: F_i = mean_j sigmoid((u_i - u_j)/Tr)
        F = torch.sigmoid((u.unsqueeze(1) - u.unsqueeze(0)) / Tr).mean(dim=1)  # (n, C)
        al = alpha.unsqueeze(0)
        w = torch.sigmoid((F - al) / Tw) * torch.sigmoid(((1.0 - al) - F) / Tw)
        return w / (w.sum(dim=0, keepdim=True) + EPS)

    def _operator_consistency(self, a_sc, n_ch, device, dtype):
        """Consistency factor for the ACTUAL soft operator, not for an idealised
        hard trim.

        The analytic Gaussian hard-trim factor is the wrong constant here: soft
        boundaries retain mass a hard trim would discard, so dividing by the
        hard-trim c(alpha) over-corrects badly at large alpha (measured: 2.75x
        at alpha=0.44). Instead evaluate this layer's own weighting scheme on a
        deterministic standard-normal reference grid and read off the weighted
        variance it produces under the null. That IS the factor, by definition,
        for whatever alpha and whatever temperatures the layer currently holds,
        and it stays differentiable in all three.

        Nodes are the fixed quantiles Phi^{-1}((i-0.5)/M) -- deterministic, so
        the layer has no stochastic component.
        """
        M = 256
        p = (torch.arange(M, device=device, dtype=dtype) + 0.5) / M
        z = torch.special.ndtri(p).unsqueeze(1).expand(M, n_ch)
        w = self._soft_weights(z, a_sc)
        mz = (w * z).sum(dim=0)
        var = (w * (z - mz.unsqueeze(0)) ** 2).sum(dim=0)
        return torch.sqrt(var.clamp_min(1e-6))

    def forward(self, v):
        a_loc, a_sc = self.alphas()
        w_loc = self._soft_weights(v, a_loc)
        m = (w_loc * v).sum(dim=0)
        w_sc = self._soft_weights(v, a_sc)
        u2 = (w_sc * (v - m.unsqueeze(0)) ** 2).sum(dim=0)
        s_raw = torch.sqrt(u2 + EPS)
        if self.correct:
            c = self._operator_consistency(a_sc, v.shape[1], v.device, v.dtype)
            s = s_raw / c
        else:
            s = s_raw
        return m, s.clamp_min(1e-6)


def winsorize(r, log_kappa):
    """Point-level bounded-influence transform, applied BEFORE any temporal
    aggregation -- required by the finding that a 5% point-contamination rate
    puts a defect in 95.4% of length-60 windows, past the breakdown point of any
    window-level robust statistic. Redescends smoothly rather than clipping, and
    is rescaled so the transform is variance-preserving under the null.
    """
    k = log_kappa.exp().clamp(0.5, 20.0)
    return k * torch.tanh(r / k)
