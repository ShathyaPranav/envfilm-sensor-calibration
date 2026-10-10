"""Training and scoring on the rebuilt TESLA benchmark (see tsl_data.py).

Windows exactly as in TESLA's get_window_dataset: input = input channel over rows
[i-360, i), target = reference channel at row i-1, inside one unit. Values are divided by
300 as in their Normalizer. Every model gets the SAME training recipe here (Adam, MSE,
checkpoint on the validation unit), so differences come from the model, not the recipe.
"""
import math
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

_EXP = os.path.dirname(os.path.abspath(__file__))
if _EXP not in sys.path:
    sys.path.insert(0, _EXP)
from tsl_data import CITIES, OPC, PMS, PUBLISHED, load

DEV = 'cuda'
N = int(os.environ.get('TSL_N', '360'))
SCALE = 300.0
OUT = os.path.join(os.path.dirname(_EXP), 'results', 'tesla')
RECIPE = dict(lr=1e-3, batch=512, epochs=10, wd=0.0)
CURRENT = {}                                   # pm size of the split being fitted (for pretrained models)
RECIPE.update(eval(os.environ.get('TSL_RECIPE', '{}')))


class Split:
    """All units of one city, one PM size, as GPU tensors with window indices."""

    def __init__(self, city, pm):
        self.city = city
        U = load(city)
        xs, ys, uid, starts, cs = [], [], [], [], []
        off = 0
        for k, (ase, f) in enumerate(U):
            xs.append(f[PMS[pm]].to_numpy(np.float32) / SCALE)
            cs.append(context(f[PMS[pm]]))
            ys.append(f[OPC[pm]].to_numpy(np.float32) / SCALE)
            uid.append(np.full(len(f), k))
            starts.append(off)
            off += len(f)
        self.x = torch.tensor(np.concatenate(xs), device=DEV)
        self.y = torch.tensor(np.concatenate(ys), device=DEV)
        self.c = torch.tensor(np.concatenate(cs), device=DEV)
        from tsl_data import EXTRA
        if EXTRA:
            ex = np.concatenate([f[EXTRA].to_numpy(np.float32) for _, f in U])
            lo, hi = np.nanpercentile(ex, 0.5, axis=0), np.nanpercentile(ex, 99.5, axis=0)
            ex = np.clip(np.nan_to_num(ex, nan=0.0), lo, hi)             # sensor glitches (e.g. 20865 degC)
            self.ex_norm = (lo, hi, ex.mean(0), ex.std(0) + 1e-6)
            self.ex = torch.tensor((ex - self.ex_norm[2]) / self.ex_norm[3], device=DEV)
        self.uid = torch.tensor(np.concatenate(uid), device=DEV)
        ends = np.array(starts[1:] + [off])
        idx = [np.arange(s + N, e + 1) for s, e in zip(starts, ends)]   # window [i-N, i), target i-1
        lab = int(os.environ.get('TSL_LABEL_MINUTES', '0'))   # label-scarce setting: first L labelled rows per unit
        self.train = np.concatenate([i[:lab] if lab else i for i in idx[:-2]])
        self.val = idx[-2][:lab] if lab else idx[-2]
        self.test = idx[-1]
        self.units = [a for a, _ in U]
        self.offs = torch.arange(-N, 0, device=DEV)

    def batch(self, i):
        i = torch.as_tensor(i, device=DEV)
        return self.x[i[:, None] + self.offs[None, :]], self.y[i - 1]

    def extra(self, i):
        i = torch.as_tensor(i, device=DEV)
        return self.ex[i[:, None] + self.offs[None, :]]                 # B, N, E

    def unit_of(self, i):
        return self.uid[torch.as_tensor(i, device=DEV) - 1]

    def ctx(self, i):
        return self.c[torch.as_tensor(i, device=DEV) - 1]


CTX_ROWS = int(os.environ.get('TSL_CTX_ROWS', '4320'))      # three days of one-minute rows
CTX_Q = (0.1, 0.5, 0.9)


def context(series):
    """Label-free, causal summary of the unit's own recent readings: the 10/50/90th
    percentile of its input channel over the previous CTX_ROWS rows (log scale)."""
    v = series.clip(lower=0).astype(np.float64)
    r = v.rolling(CTX_ROWS, min_periods=N)
    c = np.stack([r.quantile(q).to_numpy() for q in CTX_Q], 1)
    c = pd.DataFrame(c).bfill().to_numpy()
    return (c / SCALE).astype(np.float32)          # raw scale, so a unit gain g multiplies it


class Linear(nn.Module):
    def __init__(self):
        super().__init__()
        self.l = nn.Linear(N, 1)

    def forward(self, x):
        return self.l(x).squeeze(-1)


def tesla_bins(n=N):
    z = int(math.log(n, 2))
    sizes = [2 ** i for i in range(z + 1)]
    cs = np.cumsum(sizes)
    sizes = sizes[:int((cs <= n).sum())]
    if n - sum(sizes) > 0:
        sizes.append(n - sum(sizes))
    sizes = sizes[::-1]                       # oldest bin first, the latest single step last
    edges = np.concatenate([[0], np.cumsum(sizes)])
    return [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:])]


class Tesla(nn.Module):
    """Re-implementation of lbt/models/tesla.py (TensorFlow) in PyTorch, same sizes."""

    def __init__(self, D=8, H=4, K=8):
        super().__init__()
        self.emb = nn.Linear(1, D)
        self.emb2 = nn.Linear(N, D)
        self.bins = tesla_bins()
        self.binw = nn.ParameterList([nn.Parameter(torch.randn(b - a) * (1 / math.sqrt(b - a)))
                                      for a, b in self.bins])
        self.binb = nn.Parameter(torch.zeros(len(self.bins), D))
        self.q, self.k, self.v = nn.Linear(D, H * K), nn.Linear(D, H * K), nn.Linear(D, H * K)
        self.o = nn.Linear(H * K, D)
        self.H, self.K = H, K
        self.ln = nn.LayerNorm(D, eps=1e-6)
        self.ffn = nn.Linear(len(self.bins), 1)
        self.ws = nn.Linear(D, 1)

    def forward(self, x):
        out = self.emb(x[..., None]) + self.emb2(x)[:, None, :]                     # B, S, D
        b = torch.stack([torch.einsum('bsd,s->bd', out[:, a:e], w) for (a, e), w in zip(self.bins, self.binw)],
                        1) + self.binb                                              # B, T, D
        B, T, _ = b.shape
        q = self.q(b).view(B, T, self.H, self.K).transpose(1, 2)
        k = self.k(b).view(B, T, self.H, self.K).transpose(1, 2)
        v = self.v(b).view(B, T, self.H, self.K).transpose(1, 2)
        a = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(self.K), -1) @ v
        a = self.o(a.transpose(1, 2).reshape(B, T, -1))
        h = self.ln(b + a)                                                          # B, T, D
        h = self.ffn(h.transpose(1, 2)).squeeze(-1)                                 # B, D
        return self.ws(h).squeeze(-1)


class TCN(nn.Module):
    """Dilated causal convolutions over the window (log scale), optional context."""

    def __init__(self, use_ctx=False, C=32):
        super().__init__()
        self.use_ctx = use_ctx
        layers, ch = [], 1
        for d in (1, 2, 4, 8, 16, 32, 64):
            layers += [nn.Conv1d(ch, C, 3, padding=d, dilation=d), nn.GELU()]
            ch = C
        self.conv = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Linear(C * 2 + 1 + 3 * use_ctx, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, x, c=None):
        lx = torch.log1p(torch.clamp(x * SCALE, min=0))
        h = self.conv(lx[:, None, :])
        f = [h[..., -1], h.mean(-1), lx[:, -1:]]
        if self.use_ctx:
            f.append(torch.log1p(torch.clamp(c * SCALE, min=0)))
        return self.head(torch.cat(f, -1)).squeeze(-1)


def trailing_means(lx, widths):
    """Trailing moving averages of the log signal (causal within the window)."""
    out = [lx]
    c = torch.cumsum(torch.nn.functional.pad(lx, (1, 0)), -1)
    for w in widths:
        idx = torch.arange(1, lx.shape[-1] + 1, device=lx.device)
        lo = torch.clamp(idx - w, min=0)
        out.append((c[..., idx] - c[..., lo]) / (idx - lo).float())
    return torch.stack(out, 1)


class ResTCN(nn.Module):
    """Calibration TCN built from residual blocks (two dilated convolutions, GELU, dropout,
    skip connection), on the log signal; optional multi-scale input (trailing 5- and
    30-minute means as extra channels) and optional log-normal output head."""

    def __init__(self, C=32, ms=False, ln=False, drop=0.1):
        super().__init__()
        self.ms, self.ln = ms, ln
        cin = 3 if ms else 1
        self.inp = nn.Conv1d(cin, C, 1)
        self.blocks = nn.ModuleList()
        for d in (1, 2, 4, 8, 16, 32, 64):
            self.blocks.append(nn.Sequential(nn.Conv1d(C, C, 3, padding=d, dilation=d), nn.GELU(), nn.Dropout(drop),
                                             nn.Conv1d(C, C, 3, padding=d, dilation=d), nn.GELU()))
        self.head = nn.Sequential(nn.Linear(C * 2 + 1, 64), nn.GELU(), nn.Linear(64, 2 if ln else 1))

    def trunk(self, x):
        lx = torch.log1p(torch.clamp(x * SCALE, min=0))
        z = trailing_means(lx, (5, 30)) if self.ms else lx[:, None, :]
        h = self.inp(z)
        for b in self.blocks:
            h = h + b(h)
        return self.head(torch.cat([h[..., -1], h.mean(-1), lx[:, -1:]], -1))

    def forward(self, x):
        o = self.trunk(x)
        if not self.ln:
            return o.squeeze(-1)
        mu, ls = o.unbind(-1)
        s2 = torch.exp(2 * torch.clamp(ls, -5, 2))
        return (torch.expm1(mu + s2 / 2)) / SCALE                       # mean of a log-normal

    def train_loss(self, x, y):
        if not self.ln:
            return torch.mean((self(x) - y) ** 2)
        o = self.trunk(x)
        mu, ls = o.unbind(-1)
        ls = torch.clamp(ls, -5, 2)
        t = torch.log1p(torch.clamp(y * SCALE, min=0))
        nll = torch.mean(ls + 0.5 * ((t - mu) / torch.exp(ls)) ** 2)
        mean = torch.expm1(mu + torch.exp(2 * ls) / 2) / SCALE
        return nll * 1e-3 + torch.mean((mean - y) ** 2)


class TCNgain(TCN):
    """LogTCN with a latent gain per training unit's TARGET sensor.

    In this benchmark the 'reference' is itself a low-cost instrument whose unit-to-unit
    gain varies by +/-10-15% (measured across co-located units), while the input sensors
    agree within +/-3%. Training labels therefore carry a unit-specific multiplicative
    error. The model learns y_u = g_u * f(x) with one free log-gain per training unit
    (centred to mean zero), so f learns the fleet-standard response from cleaner labels;
    at validation and test time it predicts f(x), the fleet standard (g = 1)."""

    def __init__(self, n_units=64):
        super().__init__()
        self.loggain = nn.Parameter(torch.zeros(n_units))

    def train_loss_u(self, x, y, u):
        lg = self.loggain - self.loggain.mean()
        return torch.mean((torch.exp(lg[u]) * self(x) - y) ** 2)


class TCNx(nn.Module):
    """LogTCN with extra on-board channels (e.g. humidity) as additional input channels."""
    uses_extra = True

    def __init__(self, E=1, C=32):
        super().__init__()
        self.E = E
        layers, ch = [], 1 + E
        for d in (1, 2, 4, 8, 16, 32, 64):
            layers += [nn.Conv1d(ch, C, 3, padding=d, dilation=d), nn.GELU()]
            ch = C
        self.conv = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Linear(C * 2 + 1 + E, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, x, e):
        e = e[..., :self.E]
        lx = torch.log1p(torch.clamp(x * SCALE, min=0))
        h = self.conv(torch.cat([lx[:, None, :], e.transpose(1, 2)], 1))
        return self.head(torch.cat([h[..., -1], h.mean(-1), lx[:, -1:], e[:, -1, :]], -1)).squeeze(-1)


class HumidGate(nn.Module):
    """Humidity-gated calibrator. A LogTCN on the particle channel gives a base estimate
    f(x); a gate driven by humidity multiplies it, y = f(x) * exp(G), with
        G = a * log(1 + kappa * w / (1 - w)) + h(...)
    where w is the water activity (relative humidity / 100, clipped below 0.98), the
    first term is the kappa-Koehler growth law (kappa and a learned, positive) and h is a
    small bounded correction from the recent humidity trajectory, temperature proxy and
    the particle level. Humidity can only act multiplicatively, as hygroscopic growth does."""
    uses_extra = True

    def __init__(self, C=32):
        super().__init__()
        self.base = TCN(C=C)
        self.rh_mu = 0.0
        self.k = nn.Parameter(torch.tensor(-1.0))
        self.a = nn.Parameter(torch.tensor(0.0))
        self.h = nn.Sequential(nn.Linear(4, 16), nn.Tanh(), nn.Linear(16, 1))
        nn.init.zeros_(self.h[-1].weight)
        nn.init.zeros_(self.h[-1].bias)

    def forward(self, x, e):
        f = self.base(x)
        rh = e[..., 0]                                         # standardised humidity window
        w = torch.clamp(torch.sigmoid(rh[:, -1] * 1.2), 0.05, 0.98)   # monotone map to (0, 1)
        kappa = nn.functional.softplus(self.k)
        g = torch.tanh(self.a) * torch.log1p(kappa * w / (1 - w))
        lx = torch.log1p(torch.clamp(x * SCALE, min=0))
        z = torch.stack([rh[:, -1], rh.mean(-1), rh[:, -1] - rh[:, -60:].mean(-1), lx[:, -1] / 5], -1)
        g = g + 0.3 * torch.tanh(self.h(z).squeeze(-1))
        return f * torch.exp(g)


class EnvFiLM(nn.Module):
    """State-conditioned calibrator. The particle stream goes through the LogTCN convolution
    stack; the slow on-board state (humidity, temperature, the target sensor's internal
    humidity and temperature) is summarised by a separate encoder (last value, window mean,
    1-hour change, window min and max per channel) and sets a per-channel scale and shift
    for every convolution layer (FiLM, initialised to identity). The environment changes how
    particle readings are mapped instead of being mixed in as another signal."""
    uses_extra = True

    def __init__(self, E=4, C=32):
        super().__init__()
        self.E = E
        self.convs = nn.ModuleList()
        ch = 1
        self.dil = (1, 2, 4, 8, 16, 32, 64)
        for d in self.dil:
            self.convs.append(nn.Conv1d(ch, C, 3, padding=d, dilation=d))
            ch = C
        self.env = nn.Sequential(nn.Linear(5 * E, 64), nn.GELU(), nn.Linear(64, 2 * C * len(self.dil)))
        nn.init.zeros_(self.env[-1].weight)
        nn.init.zeros_(self.env[-1].bias)
        self.head = nn.Sequential(nn.Linear(C * 2 + 1 + 5 * E, 64), nn.GELU(), nn.Linear(64, 1))
        self.C = C

    def forward(self, x, e):
        e = e[..., :self.E]
        z = torch.cat([e[:, -1], e.mean(1), e[:, -1] - e[:, -60:].mean(1), e.amin(1), e.amax(1)], -1)
        film = self.env(z).view(len(x), len(self.dil), 2, self.C)
        lx = torch.log1p(torch.clamp(x * SCALE, min=0))
        h = lx[:, None, :]
        for k, conv in enumerate(self.convs):
            h = conv(h) * (1 + film[:, k, 0, :, None]) + film[:, k, 1, :, None]
            h = nn.functional.gelu(h)
        return self.head(torch.cat([h[..., -1], h.mean(-1), lx[:, -1:], z], -1)).squeeze(-1)


class TeslaEnv(Tesla):
    """Equal-information baseline: TESLA with the on-board state channels concatenated to
    the particle value at every time step before its token embedding."""
    uses_extra = True

    def __init__(self, E=4):
        super().__init__()
        self.E = E
        self.emb = nn.Linear(1 + E, 8)

    def forward(self, x, e):
        e = e[..., :self.E]
        out = self.emb(torch.cat([x[..., None], e], -1)) + self.emb2(x)[:, None, :]
        b = torch.stack([torch.einsum('bsd,s->bd', out[:, a:f], w) for (a, f), w in zip(self.bins, self.binw)],
                        1) + self.binb
        B_, T, _ = b.shape
        q = self.q(b).view(B_, T, self.H, self.K).transpose(1, 2)
        k = self.k(b).view(B_, T, self.H, self.K).transpose(1, 2)
        v = self.v(b).view(B_, T, self.H, self.K).transpose(1, 2)
        a = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(self.K), -1) @ v
        a = self.o(a.transpose(1, 2).reshape(B_, T, -1))
        h = self.ln(b + a)
        h = self.ffn(h.transpose(1, 2)).squeeze(-1)
        return self.ws(h).squeeze(-1)


class EnvFiLMb(EnvFiLM):
    """EnvFiLM with bounded conditioning (pre-specified robustness fix, 2026-10-07): the
    on-board state entering the encoder is clipped to +/-3 standard deviations of the
    training data, and every FiLM scale is 1 + 0.5 tanh(.) and every shift 0.5 tanh(.), so
    an unusual environment can change the particle pathway by at most a factor 0.5-1.5."""

    def forward(self, x, e):
        e = torch.clamp(e[..., :self.E], -3, 3)
        z = torch.cat([e[:, -1], e.mean(1), e[:, -1] - e[:, -60:].mean(1), e.amin(1), e.amax(1)], -1)
        film = torch.tanh(self.env(z)).view(len(x), len(self.dil), 2, self.C) * 0.5
        lx = torch.log1p(torch.clamp(x * SCALE, min=0))
        h = lx[:, None, :]
        for k, conv in enumerate(self.convs):
            h = conv(h) * (1 + film[:, k, 0, :, None]) + film[:, k, 1, :, None]
            h = nn.functional.gelu(h)
        return self.head(torch.cat([h[..., -1], h.mean(-1), lx[:, -1:], z], -1)).squeeze(-1)


class EnvLate(EnvFiLMb):
    """Ablation of EnvFiLM-b: identical, but the FiLM modulation is switched off; the
    on-board state summary reaches the model only through the output head (late fusion)."""

    def forward(self, x, e):
        e = torch.clamp(e[..., :self.E], -3, 3)
        z = torch.cat([e[:, -1], e.mean(1), e[:, -1] - e[:, -60:].mean(1), e.amin(1), e.amax(1)], -1)
        lx = torch.log1p(torch.clamp(x * SCALE, min=0))
        h = lx[:, None, :]
        for conv in self.convs:
            h = nn.functional.gelu(conv(h))
        return self.head(torch.cat([h[..., -1], h.mean(-1), lx[:, -1:], z], -1)).squeeze(-1)


class TCNwide(TCN):
    """Capacity control: the plain TCN widened to the ResTCN parameter count."""

    def __init__(self):
        super().__init__(C=46)


class TCNctx(TCN):
    def __init__(self):
        super().__init__(use_ctx=True)


AUG = float(os.environ.get('TSL_AUG', '0.3'))   # sd of log unit gain used for augmentation


class GainTCN(TCN):
    """Multiplicative calibration: the network does not predict the concentration, it
    predicts a gain applied to a learned short average of the sensor's own latest
    readings, y = (sum_j a_j x_{t-j}) * exp(r) + b, with r and b from the same trunk.
    A proportional sensor needs r constant, so large readings are not pulled toward the
    training mean the way an additive regressor pulls them."""

    def __init__(self, use_ctx=False, C=32, L=16):
        super().__init__(use_ctx=use_ctx, C=C)
        self.L = L
        self.head = nn.Sequential(nn.Linear(C * 2 + 1 + 3 * use_ctx, 64), nn.GELU(), nn.Linear(64, 2))
        self.mix = nn.Parameter(torch.zeros(L))

    def forward(self, x, c=None):
        lx = torch.log1p(torch.clamp(x * SCALE, min=0))
        h = self.conv(lx[:, None, :])
        f = [h[..., -1], h.mean(-1), lx[:, -1:]]
        if self.use_ctx:
            f.append(torch.log1p(torch.clamp(c * SCALE, min=0)))
        r, b = self.head(torch.cat(f, -1)).unbind(-1)
        base = (x[:, -self.L:] * torch.softmax(self.mix, 0)).sum(-1)
        return base * torch.exp(r) + 0.01 * b


class GainTCNctx(GainTCN):
    def __init__(self):
        super().__init__(use_ctx=True)


class GainTCNctxAug(GainTCNctx):
    """Same network; during training every window and its context are multiplied by a
    random unit gain, the target is not. The only way to stay accurate is to read the
    unit's gain off its own week of readings (all units sit next to each other, so their
    true distributions agree) and divide it out."""
    aug = AUG


class TCNctxAug(TCNctx):
    aug = AUG


class LIC(nn.Module):
    """Leaky-integrator calibrator. A bank of K exponential moving averages of the
    log reading, each with its own learned time constant (initialised log-spaced from 1 to
    720 minutes), plus the matching moving averages of the squared deviation, feed a tiny
    MLP. On a device every average is one multiply-add per new reading: constant memory,
    no window buffer. In training the averages are computed over the 360-row window."""

    def __init__(self, K=8, H=24):
        super().__init__()
        tau = torch.logspace(0, math.log10(720), K)
        self.raw = nn.Parameter(torch.log(torch.expm1(1.0 / tau)))       # softplus^-1 of the rate
        self.head = nn.Sequential(nn.Linear(2 * K + 1, H), nn.GELU(), nn.Linear(H, 1))
        self.lin = nn.Parameter(torch.ones(1))
        self.register_buffer('age', torch.arange(N - 1, -1, -1, dtype=torch.float32))

    def forward(self, x):
        lx = torch.log1p(torch.clamp(x * SCALE, min=0))
        rate = torch.clamp(nn.functional.softplus(self.raw), max=1.0)
        w = (1 - rate)[None, :] ** self.age[:, None] * rate[None, :]     # N, K
        w = w / w.sum(0, keepdim=True)
        m = lx @ w                                                        # B, K
        v = (lx ** 2) @ w - m ** 2
        f = torch.cat([m, torch.sqrt(torch.clamp(v, min=1e-6)), lx[:, -1:]], -1)
        return self.head(f).squeeze(-1) + 0.0 * self.lin


class GRUsmall(nn.Module):
    """Streaming baseline: GRU over the log window, hidden size chosen to match LIC."""

    def __init__(self, H=12):
        super().__init__()
        self.g = nn.GRU(1, H, batch_first=True)
        self.o = nn.Linear(H, 1)

    def forward(self, x):
        lx = torch.log1p(torch.clamp(x * SCALE, min=0))
        h, _ = self.g(lx[..., None])
        return self.o(h[:, -1]).squeeze(-1)


class TCNsmall(TCN):
    def __init__(self):
        super().__init__(C=8)


class TCNraw(TCN):
    """Ablation: the same TCN on the raw (/300) scale instead of log."""

    def forward(self, x, c=None):
        lx = x * 10
        h = self.conv(lx[:, None, :])
        return self.head(torch.cat([h[..., -1], h.mean(-1), lx[:, -1:]], -1)).squeeze(-1)


class TeslaLog(Tesla):
    """Ablation: TESLA fed log-scaled input (output still on the raw scale)."""

    def forward(self, x):
        return super().forward(torch.log1p(torch.clamp(x * SCALE, min=0)) / 6)


class PatchTST(nn.Module):
    """PatchTST (Nie et al., ICLR 2023) adapted to a scalar target: channel-independent
    Transformer encoder over patches (length 16, stride 8) of each channel, instance
    normalisation per window and channel (RevIN without affine), flatten head. The
    particle channel enters on the log scale as for LogTCN; the removed window mean and
    spread and the last log value go to the output MLP, so the level is not lost."""

    def __init__(self, E=0, P=16, S=8, D=64, L=3, H=4, F=128, drop=0.2):
        super().__init__()
        self.E, self.P, self.S = E, P, S
        self.T = (N - P) // S + 2                              # end padding by S, as in the paper
        self.patch = nn.Linear(P, D)
        self.pos = nn.Parameter(torch.randn(self.T, D) * 0.02)
        layer = nn.TransformerEncoderLayer(D, H, F, drop, activation='gelu', batch_first=True, norm_first=False)
        self.enc = nn.TransformerEncoder(layer, L)
        self.drop = nn.Dropout(drop)
        self.flat = nn.Linear(self.T * D, 64)
        self.head = nn.Sequential(nn.Linear(64 * (1 + E) + 3 + E, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, x, e=None):
        lx = torch.log1p(torch.clamp(x * SCALE, min=0))
        ch = [lx[:, None, :]]
        if self.E:
            ch.append(e[..., :self.E].transpose(1, 2))
        z = torch.cat(ch, 1)                                  # B, C, N
        mu, sd = z.mean(-1, keepdim=True), z.std(-1, keepdim=True) + 1e-3
        z = torch.nn.functional.pad((z - mu) / sd, (0, self.S), mode='replicate')
        p = z.unfold(-1, self.P, self.S)                      # B, C, T, P
        B_, C, T, _ = p.shape
        h = self.enc(self.patch(p).reshape(B_ * C, T, -1) + self.pos)
        h = self.flat(self.drop(h.reshape(B_, C, -1)))        # B, C, 64
        f = [h.reshape(B_, -1), mu[:, 0], sd[:, 0], lx[:, -1:]]
        if self.E:
            f.append(e[:, -1, :self.E])
        return self.head(torch.cat(f, -1)).squeeze(-1)


class PatchTSTenv(PatchTST):
    uses_extra = True

    def __init__(self):
        super().__init__(E=4)


MODELS = {'PatchTST': PatchTST, 'PatchTST+ENV': PatchTSTenv, 'Linear': Linear, 'TESLA': Tesla, 'TCN+gain': TCNgain, 'TCN+RH': TCNx, 'HumidGate': HumidGate, 'TCN+ENV': lambda: TCNx(E=4), 'EnvFiLM': EnvFiLM, 'EnvFiLM-b': EnvFiLMb, 'EnvLate': EnvLate, 'TESLA+ENV': TeslaEnv, 'TCN+ENVwide': lambda: TCNx(E=4, C=50), 'ResTCN': ResTCN, 'TCNwide': TCNwide,
          'ResTCN+MS': lambda: ResTCN(ms=True), 'ResTCN+LN': lambda: ResTCN(ln=True),
          'ResTCN+MS+LN': lambda: ResTCN(ms=True, ln=True), 'TCNraw': TCNraw, 'TESLAlog': TeslaLog, 'LIC': LIC, 'GRUsmall': GRUsmall, 'TCNsmall': TCNsmall, 'TCN': TCN, 'TCN+ctx': TCNctx, 'GainTCN': GainTCN,
          'GainTCN+ctx': GainTCNctx, 'GainTCN+ctx+aug': GainTCNctxAug, 'TCN+ctx+aug': TCNctxAug}
USES_CTX = {'TCN+ctx', 'GainTCN+ctx'}


@torch.no_grad()
def predict(model, S, idx):
    model.eval()
    out = []
    for j in range(0, len(idx), 8192):
        x, _ = S.batch(idx[j:j + 8192])
        if getattr(model, 'uses_extra', False):
            out.append(model(x, S.extra(idx[j:j + 8192])))
        else:
            out.append(model(x, S.ctx(idx[j:j + 8192])) if getattr(model, 'use_ctx', False) else model(x))
    return torch.cat(out)


def rmse_on(model, S, idx):
    p = predict(model, S, idx)
    y = S.y[torch.as_tensor(idx, device=DEV) - 1]
    return float(torch.sqrt(torch.mean((p - y) ** 2)) * SCALE)


def fit(name, S, seed=0, make=None, verbose=False):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = (make or MODELS[name])().to(DEV)
    opt = torch.optim.Adam(model.param_groups() if hasattr(model, 'param_groups') else model.parameters(),
                           lr=RECIPE['lr'], weight_decay=RECIPE['wd'])
    best, state, bad = np.inf, None, 0
    for ep in range(RECIPE['epochs']):
        model.train()
        perm = rng.permutation(S.train)
        for j in range(0, len(perm), RECIPE['batch']):
            x, y = S.batch(perm[j:j + RECIPE['batch']])
            c = S.ctx(perm[j:j + RECIPE['batch']])
            if getattr(model, 'aug', 0):
                g = torch.exp(torch.randn(len(x), 1, device=DEV) * model.aug)
                x, c = x * g, c * g
            if hasattr(model, 'train_loss_u'):
                loss = model.train_loss_u(x, y, S.unit_of(perm[j:j + RECIPE['batch']]))
            elif hasattr(model, 'train_loss'):
                loss = model.train_loss(x, y)
            elif getattr(model, 'uses_extra', False):
                loss = torch.mean((model(x, S.extra(perm[j:j + RECIPE['batch']])) - y) ** 2)
            else:
                p = model(x, c) if getattr(model, 'use_ctx', False) else model(x)
                loss = torch.mean((p - y) ** 2)
            if hasattr(model, 'aux_loss'):
                loss = loss + model.aux_loss()
            opt.zero_grad()
            loss.backward()
            opt.step()
            if hasattr(model, 'after_step'):
                model.after_step()
        v = rmse_on(model, S, S.val)
        if verbose:
            print('     epoch %d val %.3f' % (ep, v), flush=True)
        if v < best:
            best, state, bad = v, {k: t.detach().clone() for k, t in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= RECIPE.get('patience', 10 ** 9):
                break
    model.load_state_dict(state)
    return model, best, rmse_on(model, S, S.test)


def main():
    os.makedirs(OUT, exist_ok=True)
    names = [a for a in sys.argv[1:] if a in MODELS] or list(MODELS)
    pms = [a for a in sys.argv[1:] if a in PMS] or list(PMS)
    cities = [a for a in sys.argv[1:] if a in CITIES] or list(CITIES)
    seeds = int(os.environ.get('TSL_SEEDS', '1'))
    tag = os.environ.get('TSL_TAG', 'run')
    rows = []
    for pm in pms:
        for ci, city in enumerate(cities):
            S = Split(city, pm)
            CURRENT['pm'] = pm
            for name in names:
                for s in range(seeds):
                    t = time.time()
                    _, v, te = fit(name, S, seed=s)
                    pub = PUBLISHED.get(name, {}).get(pm, (np.nan,) * 3)[CITIES.index(city)]
                    rows.append(dict(pm=pm, city=city, model=name, seed=s, val=v, test=te, published=pub))
                    print('%-5s %-8s %-10s seed %d  val %.3f  test %.3f  (published %.2f)  %.0f s'
                          % (pm, city, name, s, v, te, pub, time.time() - t), flush=True)
                    pd.DataFrame(rows).to_csv(os.path.join(OUT, 'bench_%s.csv' % tag), index=False)


if __name__ == '__main__':
    main()
