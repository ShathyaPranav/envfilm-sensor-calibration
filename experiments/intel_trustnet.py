"""TRUST-Net: Telemetry-conditioned Reliability for Unbiased Streaming sTatistics.

PROBLEM (the paper's own): a streaming data-quality monitor must report window
location and spread that stay faithful to the true signal while its sensors
degrade -- without knowing which readings are bad.

WHY PRIOR ADAPTIVE METHODS FAILED, AND WHAT IS DIFFERENT HERE. Four adaptive
schemes in this project (per-regime, per-window, per-channel, supervised) all
lost to a single fixed estimator. Every one of them (a) saw only features
computable from the window itself and (b) chose ONE aggregation rule per window.
TRUST-Net changes both:
  - it reads an INDEPENDENT reliability signal -- the sensor's own battery
    voltage -- measured on Intel Lab to predict saturation 100% of the time at
    2.40 V with a median 340 h of warning, but with sensor-specific onset
    (global threshold F1 0.92, worst sensor 0.08);
  - it assigns reliability PER READING, so a window can keep its 27 good readings
    and drop its 3 bad ones -- which no per-window estimator choice can express.

ARCHITECTURE. For reading i:

    logit_i = b + g(telemetry_i) + h(value_context_i)
    w_i     = sigmoid(logit_i)
    mu_W    = sum_i w_i x_i / sum_i w_i
    sd_W    = sqrt( sum_i w_i (x_i - mu_W)^2 / sum_i w_i )

  g  MONOTONE telemetry prior: a positive-weighted sum of increasing sigmoids of
     absolute voltage and of voltage RELATIVE to the sensor's own early-deployment
     baseline, plus a small trend term. Monotone because a falling battery never
     makes a reading more trustworthy; relative because onset varies per sensor
     and the early baseline is observable for an unseen sensor without labels.
  h  value evidence: small MLP over causal local outlyingness, exact-repeat run
     length, long-context level shift, and deviation from the sensor's early
     baseline. Catches faults voltage cannot see.

TRAINED END-TO-END on window-statistic error against the reference computed from
label-clean readings (voltage-independent labels: saturation, implausibility, or
>1.5 C deviation from healthy spatial neighbours). Because the target is the TRUE
clean scale, shrinking the scale on clean data raises the loss -- the objective
itself rules out the scale-shrinkage degeneracy that CCT-Norm needed an explicit
consistency correction to prevent.

EVALUATION: leave-sensors-out K-fold; every number is on sensors never trained on.
All features are causal and use only the sensor's own stream -- no neighbours, no
labels, no future readings at inference.

ABLATIONS: full (monotone), full (non-monotone), telemetry-only, value-only.
"""
import math
import os
import pickle
import sys
import time
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.lib.stride_tricks import sliding_window_view

warnings.filterwarnings('ignore', category=RuntimeWarning)
torch.set_num_threads(max(1, (os.cpu_count() or 4) - 2))

DQ = r"C:\Users\luckf\OneDrive\Desktop\dq-estimator-eval"
OUT = os.path.join(DQ, 'results', 'intel')
W = 30
FOLDS = int(os.environ.get('FOLDS', '5'))
EPOCHS = int(os.environ.get('EPOCHS', '30'))
BASE_N = 2000
GLOBAL_T = 2.31
CONFIGS = os.environ.get('CONFIGS', 'full_mono,full_free,tel_only,val_only').split(',')
# Label set to train and evaluate on. The original deviation-from-neighbours labels were
# shown to be mostly spatial-reference artifacts (99.8% at healthy voltage, 3.4x concentrated
# by hour of day), so the default is now the unambiguous saturated-or-implausible set.
LABEL_FILE = os.environ.get('LABELS', 'organic_sequences_satimpl.pkl')
TAG = os.path.splitext(LABEL_FILE)[0].replace('organic_sequences', '').strip('_') or 'dev15'


# ------------------------------------------------------------------ features
def causal_features(x, v):
    s = pd.Series(x)
    med = s.rolling(61, min_periods=10).median()
    mad = (s - med).abs().rolling(61, min_periods=10).median() * 1.4826
    z = np.log1p(((s - med).abs() / (mad + 1e-3)).clip(upper=1e3)).fillna(0).to_numpy()
    same = np.zeros(len(x), dtype=bool)
    same[1:] = x[1:] == x[:-1]
    rid = np.cumsum(~same)
    runlen = np.minimum(np.bincount(rid)[rid] / 30.0, 5.0) * same
    medL = s.rolling(601, min_periods=50).median()
    sdL = s.rolling(601, min_periods=50).std()
    lvl = np.log1p(((s - medL).abs() / (sdL + 1e-3)).clip(upper=1e3)).fillna(0).to_numpy()
    base_x = np.median(x[:BASE_N])
    # Robust spread of the early baseline: a sensor whose first readings already include
    # saturated values would otherwise get an inflated scale that masks its later faults.
    base_sd = max(np.median(np.abs(x[:BASE_N] - base_x)) * 1.4826, 0.05)
    dev = np.log1p(np.abs(x - base_x) / base_sd)
    vb = np.median(v[:BASE_N])
    vrel = v / vb
    dv = pd.Series(v).diff(300).fillna(0).to_numpy() / 300.0 * 1e5
    dv = np.clip(dv, -5, 5)
    f = np.stack([v, vrel, dv, z, same.astype(float), runlen, lvl, dev], axis=1)
    return np.nan_to_num(f, nan=0.0).astype(np.float32), base_sd


def window_refs(x, lab, v):
    sw = sliding_window_view(x, W)
    lw = sliding_window_view(lab, W)
    vw = sliding_window_view(v, W)
    clean = np.where(~lw, sw, np.nan)
    valid = (~lw).sum(1) >= W // 2
    ref_mu = np.nanmean(clean, 1)
    ref_sd = np.nanstd(clean, 1)
    medc = np.nanmedian(clean, 1)
    ref_madn = np.nanmedian(np.abs(clean - medc[:, None]), 1) * 1.4826
    med = np.nanmedian(sw, 1)
    madn = np.nanmedian(np.abs(sw - med[:, None]), 1) * 1.4826
    thr = np.where(vw >= GLOBAL_T, sw, np.nan)
    thr_mu, thr_sd = np.nanmean(thr, 1), np.nanstd(thr, 1)
    # A deployed threshold rule that discards every reading in a window still has to
    # report something. Fall back to the plain estimate instead of emitting NaN, which
    # would silently drop the threshold baseline's hardest windows from its median.
    empty = ~(vw >= GLOBAL_T).any(1)
    thr_mu = np.where(empty, sw.mean(1), thr_mu)
    thr_sd = np.where(empty, sw.std(1), thr_sd)
    return dict(valid=valid, fault=lw.any(1), dying=np.nanmedian(vw, 1) < 2.5,
                ref_mu=ref_mu, ref_sd=ref_sd, ref_madn=ref_madn,
                plain_mu=sw.mean(1), plain_sd=sw.std(1), madn=madn, median=med,
                ref_median=medc, thr_mu=thr_mu, thr_sd=thr_sd, thr_empty=empty)


# --------------------------------------------------------------------- model
class MonoSig(nn.Module):
    """Monotone increasing: positive-weighted sum of increasing sigmoids."""

    def __init__(self, lo, hi, K=8, init_total=6.0):
        super().__init__()
        self.c = nn.Parameter(torch.linspace(lo, hi, K))
        amp = init_total / K
        self.a = nn.Parameter(torch.full((K,), math.log(math.expm1(amp))))
        self.log_tau = nn.Parameter(torch.full((K,), math.log((hi - lo) / K)))

    def forward(self, u):
        tau = self.log_tau.exp().clamp_min(1e-3)
        return (F.softplus(self.a)[None, :] * torch.sigmoid((u[:, None] - self.c[None, :]) / tau[None, :])).sum(1)


class TrustNet(nn.Module):
    def __init__(self, use_tel=True, use_val=True, monotone=True):
        super().__init__()
        self.use_tel, self.use_val, self.monotone = use_tel, use_val, monotone
        self.g_v = MonoSig(2.10, 2.65)
        self.g_vrel = MonoSig(0.75, 1.05)
        self.w_dv = nn.Parameter(torch.zeros(1))
        self.g_free = nn.Sequential(nn.Linear(3, 16), nn.ReLU(), nn.Linear(16, 1))
        self.h = nn.Sequential(nn.Linear(5, 16), nn.ReLU(), nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 1))
        nn.init.zeros_(self.h[-1].weight); nn.init.zeros_(self.h[-1].bias)
        nn.init.zeros_(self.g_free[-1].weight); nn.init.zeros_(self.g_free[-1].bias)
        self.b = nn.Parameter(torch.tensor(-2.0 if (use_tel and monotone) else 2.0))

    def forward(self, f):
        logit = self.b.expand(f.shape[0])
        if self.use_tel:
            if self.monotone:
                logit = logit + self.g_v(f[:, 0]) + self.g_vrel(f[:, 1]) + self.w_dv * f[:, 2]
            else:
                tel = torch.stack([f[:, 0] - 2.4, (f[:, 1] - 0.9) * 10, f[:, 2]], 1)
                logit = logit + self.g_free(tel).squeeze(1)
        if self.use_val:
            logit = logit + self.h(f[:, 3:8]).squeeze(1)
        return torch.sigmoid(logit)


def wstats(x, w):
    xw, ww = x.unfold(0, W, 1), w.unfold(0, W, 1)
    sw = ww.sum(1).clamp_min(1e-6)
    mu = (ww * xw).sum(1) / sw
    var = (ww * (xw - mu[:, None]) ** 2).sum(1) / sw
    return mu, torch.sqrt(var.clamp_min(1e-10))


CFG = {'full_mono': (True, True, True), 'full_free': (True, True, False),
       'tel_only': (True, False, True), 'val_only': (False, True, True)}


# ----------------------------------------------------------------- pipeline
def prepare():
    seqs = pickle.load(open(os.path.join(DQ, 'data', 'intel_lab', LABEL_FILE), 'rb'))
    data = {}
    for m, s in seqs.items():
        x, v, lab = s['temp'].astype(np.float64), s['volt'].astype(np.float64), s['label'].astype(bool)
        if len(x) < BASE_N + 5 * W:
            continue
        feat, scale = causal_features(x, v)
        refs = window_refs(x, lab, v)
        if refs['valid'].sum() < 200:
            continue
        data[m] = dict(x=x, v=v, lab=lab, feat=feat, scale=scale, refs=refs,
                       xt=torch.tensor(x, dtype=torch.float32),
                       ft=torch.tensor(feat),
                       rmu=torch.tensor(np.nan_to_num(refs['ref_mu']), dtype=torch.float32),
                       rsd=torch.tensor(np.nan_to_num(refs['ref_sd']), dtype=torch.float32),
                       valid=torch.tensor(refs['valid']),
                       fault=torch.tensor(refs['fault']))
    return data


def train(train_motes, data, cfg):
    torch.manual_seed(0)
    model = TrustNet(*CFG[cfg])
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    rng = np.random.default_rng(0)
    for ep in range(EPOCHS):
        for m in rng.permutation(train_motes):
            d = data[m]
            w = model(d['ft'])
            mu, sd = wstats(d['xt'], w)
            err = ((mu - d['rmu']).abs() + (sd - d['rsd']).abs()) / d['scale']
            fm = d['valid'] & d['fault']
            cm = d['valid'] & ~d['fault']
            parts = [err[k].mean() for k in (fm, cm) if k.any()]
            if not parts:
                continue
            loss = sum(parts)
            opt.zero_grad(); loss.backward(); opt.step()
    return model


def evaluate(model, test_motes, data, cfg, fold):
    rows = []
    with torch.no_grad():
        for m in test_motes:
            d = data[m]
            w = model(d['ft'])
            mu, sd = wstats(d['xt'], w)
            mu, sd = mu.numpy(), sd.numpy()
            r = d['refs']
            idx = np.where(r['valid'])[0]
            rows.append(pd.DataFrame(dict(
                cfg=cfg, fold=fold, mote=m, dying=r['dying'][idx], fault=r['fault'][idx],
                e_trust_sd=np.abs(sd[idx] - r['ref_sd'][idx]),
                e_trust_mu=np.abs(mu[idx] - r['ref_mu'][idx]),
                e_std=np.abs(r['plain_sd'][idx] - r['ref_sd'][idx]),
                e_mean=np.abs(r['plain_mu'][idx] - r['ref_mu'][idx]),
                e_madn=np.abs(r['madn'][idx] - r['ref_madn'][idx]),
                e_median=np.abs(r['median'][idx] - r['ref_median'][idx]),
                e_thr_sd=np.abs(r['thr_sd'][idx] - r['ref_sd'][idx]),
                e_thr_mu=np.abs(r['thr_mu'][idx] - r['ref_mu'][idx]))))
            wn = w.numpy()
            lab = d['lab']
            if lab.any() and (~lab).any():
                from sklearn.metrics import roc_auc_score
                rows[-1]['weight_auc'] = roc_auc_score(~lab, wn)
    return pd.concat(rows, ignore_index=True)


def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    data = prepare()
    motes = sorted(data)
    print(f"labels={LABEL_FILE}  sensors usable: {len(motes)}  folds={FOLDS} epochs={EPOCHS} configs={CONFIGS}  "
          f"[prep {time.time()-t0:.0f}s]")
    folds = {m: i % FOLDS for i, m in enumerate(motes)}
    allres = []
    for cfg in CONFIGS:
        for k in range(FOLDS):
            tr = [m for m in motes if folds[m] != k]
            te = [m for m in motes if folds[m] == k]
            model = train(tr, data, cfg)
            allres.append(evaluate(model, te, data, cfg, k))
            print(f"  {cfg:10s} fold {k}: train {len(tr)} sensors, test {len(te)}  [{time.time()-t0:.0f}s]")
        pd.concat(allres, ignore_index=True).to_csv(os.path.join(OUT, f'trustnet_results_{TAG}.csv'), index=False)

    res = pd.concat(allres, ignore_index=True)

    def dist(v):
        v = np.asarray(v, float)
        return np.median(v), v.mean(), np.percentile(v, 90), np.percentile(v, 99)

    # The error distribution is heavy-tailed. In most fault windows one plausible-but-wrong
    # reading barely moves std; in a few, one saturated 122 C reading moves it by ~18 C. The
    # organic baseline run showed median and mean ranking std and MADn in OPPOSITE orders
    # because of this, so a lone median is not an honest summary. Report the whole
    # distribution on fault windows, paired win rates, and the cost paid on clean windows.
    for stat, tcol, bases in (
            ('SPREAD', 'e_trust_sd', (('std', 'e_std'), ('MADn', 'e_madn'), ('V>=2.31 std', 'e_thr_sd'))),
            ('LOCATION', 'e_trust_mu', (('mean', 'e_mean'), ('median', 'e_median'), ('V>=2.31 mean', 'e_thr_mu')))):
        f = res[res.fault]
        print(f"\n=== {stat} |error| on FAULT windows, held-out sensors "
              f"({f.drop_duplicates(['fold', 'mote']).shape[0]} sensors) ===")
        print(f"{'method':24s}{'median':>10s}{'mean':>10s}{'p90':>10s}{'p99':>10s}")
        b0 = f[f.cfg == CONFIGS[0]]
        for name, col in bases:
            print(f"{name:24s}" + ''.join(f"{q:10.4f}" for q in dist(b0[col])))
        for cfg in CONFIGS:
            c = f[f.cfg == cfg]
            if c.empty:
                continue
            wins = '  '.join(f"beats {n} {100 * (c[tcol] < c[col]).mean():.0f}%" for n, col in bases)
            print(f"{'TRUST ' + cfg:24s}" + ''.join(f"{q:10.4f}" for q in dist(c[tcol])) + f"   {wins}")
        cl = res[~res.fault]
        print(f"  cost on CLEAN windows, mean |error| (plain estimators are exactly 0 here by construction):")
        print(f"    {bases[2][0]:22s}{cl[cl.cfg == CONFIGS[0]][bases[2][1]].mean():.5f}")
        for cfg in CONFIGS:
            c = cl[cl.cfg == cfg]
            if not c.empty:
                print(f"    {'TRUST ' + cfg:22s}{c[tcol].mean():.5f}")
    if 'weight_auc' in res:
        print("\nper-reading weight AUC, clean vs faulty (held-out sensors, unweighted mean over sensors):")
        print(res.drop_duplicates(['cfg', 'fold', 'mote']).groupby('cfg')['weight_auc']
              .mean().round(4).to_string())
    print(f"\nElapsed {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()
