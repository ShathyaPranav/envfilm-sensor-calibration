"""SARN: a retention network trained THROUGH the statistic it feeds.

THE CLAIM, at its true size. Standard practice is detect-then-act: train a classifier on
point labels, threshold it, drop the flagged points, recompute the statistic. Every part
of that pipeline exists. What does not exist, on the evidence of three adversarial
prior-art searches, is fitting the detector by differentiating THROUGH the estimator, so
that the training signal is the error in the reported statistic rather than agreement
with point labels. That is the one idea here, and it is tested against the identical
network trained the standard way, at identical capacity, on identical features.

WHY THIS DATA JUSTIFIES IT. Measured on Logan River (see results/lro/FINDINGS.md):
  - the published ARIMA detector scores 0.89-0.97 on the benchmark's own event F1 and
    leaves the monitored statistic unchanged or slightly worse, because it flags event
    onsets only. Detection score and downstream usefulness are decoupled.
  - masking exactly the technician's labels is NOT optimal for the statistic: on
    MainStreet conductivity a rule screen reaches 0.0441 against the label oracle's
    0.0443. So there is headroom that no amount of label-fitting can reach, and it is
    only reachable by optimising the statistic directly.

WHAT IT CANNOT DO, established before building it: 83% of the conductivity error is
calibration drift, whose size is a number a technician measures in the field, and which
is unpredictable from the data (negative out-of-sample R^2 under two model classes, and
worse than assuming zero even when given the true drift up to 30 days ago). No retention
weighting can recover a level offset. The model is therefore expected to win on the
deletion channel and to be flat where drift dominates, and the results are reported that
way rather than pooled into one flattering average.

ARCHITECTURE
  features   per point, strictly causal, raw only: robust z against a trailing window,
             first and second differences over a robust scale, flatline run length,
             missingness and gap age, the strongest co-located variable's robust z, and a
             spatial residual against the other sites. No corrected value, no label, no
             future value, no neighbour's corrected value.
  backbone   dilated convolutions over the window (1, 2, 4), 32 channels, ~25k parameters
  head       one retention weight w in [0, 1] per point in the window
  decoder    the paper's own trimmed mean, made weight-aware and differentiable: sort the
             window, accumulate weights, keep the mass between alpha and 1-alpha of the
             total. With hard 0/1 weights this reduces exactly to dropping points and
             recomputing, so the two arms below are comparable by construction.
  loss       Huber on (predicted statistic - technician-corrected statistic) / scale,
             plus a small penalty on how much weight is discarded.

ARMS
  sarn       trained through the decoder on the statistic
  pointnet   identical backbone and features, trained with BCE on technician labels, then
             thresholded and hard-masked (the standard pipeline, and the control that
             decides whether the idea is worth anything)
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

_EXP = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_EXP)
for p in (_ROOT, _EXP):
    if p not in sys.path:
        sys.path.insert(0, p)
from lro_killtest import (ALLVARS, OUT, SITES, TEST_YEARS, TRAIN_YEARS, apply_rules, fit_rules,
                          load_site, rolling_all, run_lengths, series_scale)

W = 30
TRIM = 0.10
VAL_YEAR = 2017
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
FEATS = ['z', 'd1', 'd2', 'flat', 'miss', 'gap_age', 'cross', 'spatial',
         'global_z', 'clim_z', 'clim_spread']


def causal_features(raw, neigh, others, win=96, glob=None):
    """Causal, raw-only features. Nothing here may see a corrected value or the future."""
    n = len(raw)
    x = pd.Series(raw)
    med = x.rolling(win, min_periods=4).median()
    madn = (x - med).abs().rolling(win, min_periods=4).median() * 1.4826
    sc = madn.replace(0, np.nan).ffill().bfill().fillna(1.0).to_numpy()
    sc = np.maximum(sc, 1e-6)
    d1 = np.diff(raw, prepend=raw[0])
    d2 = np.diff(raw, n=2, prepend=[raw[0], raw[0]])
    finite = np.isfinite(raw)
    gap_age = np.zeros(n)
    age = 0
    for i in range(n):
        age = 0 if finite[i] else age + 1
        gap_age[i] = age
    f = {
        'z': (raw - med.to_numpy()) / sc,
        'd1': d1 / sc,
        'd2': d2 / sc,
        'flat': np.log1p(np.minimum(run_lengths(raw), 1000)),
        'miss': (~finite).astype(float),
        'gap_age': np.log1p(gap_age),
        'cross': np.zeros(n),
        'spatial': np.zeros(n),
    }
    if others:
        zs = []
        for o in others:
            om = pd.Series(o).rolling(win, min_periods=4).median()
            osd = (pd.Series(o) - om).abs().rolling(win, min_periods=4).median() * 1.4826
            zs.append(np.abs((o - om.to_numpy()) / np.maximum(osd.to_numpy(), 1e-6)))
        f['cross'] = np.nanmax(np.vstack(zs), axis=0)
    if neigh is not None:
        f['spatial'] = (raw - neigh) / sc
    for k in ('global_z', 'clim_z', 'clim_spread'):
        f[k] = np.zeros(n)
    if glob is not None:
        f.update(glob)
    F = np.vstack([np.nan_to_num(f[k], nan=0.0, posinf=0.0, neginf=0.0) for k in FEATS]).T
    return np.clip(F, -50, 50).astype(np.float32)


def global_features(raw, doy, mask_train):
    """Context a trailing window cannot see.

    A multi-day excursion drags any trailing median along with it, so a local robust z
    stays near zero throughout and the point looks normal. Measured consequence: the
    local-only model left MainStreet temperature at 0.327 where a fixed range rule
    reached 0.0018. These features fix the reference to the TRAINING years: how unusual
    the value is against the whole training distribution, and against the training
    climatology for this day of the year.
    """
    tr = raw[mask_train]
    tr = tr[np.isfinite(tr)]
    if tr.size < 200:
        return None
    med = float(np.median(tr))
    sc = max(float(np.median(np.abs(tr - med)) * 1.4826), 1e-6)
    clim_med = np.full(367, med)
    clim_sc = np.full(367, sc)
    dtr = doy[mask_train]
    for day in range(1, 367):
        sel = np.isfinite(raw) & mask_train & (doy == day)
        v = raw[sel]
        if v.size >= 20:
            m = float(np.median(v))
            clim_med[day] = m
            clim_sc[day] = max(float(np.median(np.abs(v - m)) * 1.4826), 1e-6)
    cm, cs = clim_med[doy], clim_sc[doy]
    return {'global_z': (raw - med) / sc,
            'clim_z': (raw - cm) / cs,
            'clim_spread': np.full(len(raw), 0.0) + cs / sc}


def weighted_trimmed_mean(vals, w, valid, trim=TRIM):
    """Weight-aware trimmed mean, differentiable in w, exact for hard 0/1 weights.

    Sort the window, walk the cumulative weight, and keep the mass lying between trim and
    1 - trim of the total. Each point contributes the overlap of its own weight interval
    with the kept band, which is what trimming means when points carry weight.
    """
    w = w * valid
    order = torch.argsort(torch.where(valid > 0, vals, torch.full_like(vals, float('inf'))), dim=1)
    v = torch.gather(vals, 1, order)
    ww = torch.gather(w, 1, order)
    tot = ww.sum(dim=1, keepdim=True).clamp_min(1e-6)
    hi = torch.cumsum(ww, dim=1)
    lo = hi - ww
    a, b = trim * tot, (1.0 - trim) * tot
    eff = (torch.minimum(hi, b) - torch.maximum(lo, a)).clamp_min(0.0)
    denom = eff.sum(dim=1, keepdim=True)
    safe = denom.squeeze(1) > 1e-6
    out = (eff * torch.nan_to_num(v)).sum(dim=1) / denom.squeeze(1).clamp_min(1e-6)
    return out, safe


class Net(nn.Module):
    def __init__(self, n_feat=len(FEATS), ch=32):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv1d(n_feat, ch, 5, padding=2, dilation=1), nn.ReLU(),
            nn.Conv1d(ch, ch, 5, padding=4, dilation=2), nn.ReLU(),
            nn.Conv1d(ch, ch, 5, padding=8, dilation=4), nn.ReLU(),
            nn.Conv1d(ch, 1, 1))
        # start near "keep", but not saturated: sigmoid(1.7) = 0.85 leaves usable gradient
        nn.init.constant_(self.body[-1].bias, 1.7)

    def forward(self, x):                      # x: (B, W, F)
        return self.body(x.transpose(1, 2)).squeeze(1)    # (B, W) logits


def physical_screen(raw, site, var):
    """Drop values outside the instrument's published physical range.

    Necessary because raw carries sensor error codes: MainStreet temperature contains 99
    points of 7999.0 in one contiguous block, and when a whole window is 7999 no amount of
    trimming or reweighting helps, so the statistic becomes 7999. Measured, those windows
    are 0.1% of the record and carried 99.8% of the total error, which made the benchmark
    a test of one event. Every arm, baselines included, gets this same screen, using
    pyhydroqc's own published max_range and min_range rather than anything fitted here. It
    is preprocessing that any deployed pipeline performs, not a contribution.
    """
    try:
        from lro_bench_protocol import PARAMS, SITE_KEY
        p = PARAMS[SITE_KEY.get(site, site)][var]
    except Exception:
        return raw.copy(), 0
    x = raw.copy()
    with np.errstate(invalid='ignore'):
        bad = np.isfinite(x) & ((x > p.max_range) | (x < p.min_range))
    x[bad] = np.nan
    return x, int(bad.sum())


def build_series(site_frames, site, var, prescreen=True, rule_screen=False):
    """Features, windows and targets for one (site, variable) series.

    rule_screen=True removes what the standard rule screen (physical range, rate of change,
    flatline runs, thresholds fitted on training years only) would remove, BEFORE the model
    sees anything. The model then decides what else to drop, and the comparison becomes
    "rules" against "rules plus the model", which is the question that matters: a screen and
    a learned head are not competitors, and measured on pH the screen removes long blocks
    that a per-point head only partially recalls, where partial recall buys nothing because
    the rest of the block still dominates the window.
    """
    d = site_frames[site]
    raw = d[var].to_numpy(float)
    n_screened = 0
    if prescreen:
        raw, n_screened = physical_screen(raw, site, var)
    if rule_screen:
        from lro_killtest import apply_rules, fit_rules
        yr_all = d.datetime.dt.year.to_numpy()
        mtr_all = (yr_all >= TRAIN_YEARS[0]) & (yr_all <= TRAIN_YEARS[1])
        raw = apply_rules(raw, fit_rules(raw, mtr_all))
    cor = d['%s_cor' % var].to_numpy(float)
    qual = d['%s_qual' % var].astype(str).str.strip()
    labeled = (~qual.isin(['NULL', 'nan', 'None', '', '<NA>'])).to_numpy()
    deleted = np.isfinite(raw) & ~np.isfinite(cor)
    target_point = labeled | deleted
    yr = d.datetime.dt.year.to_numpy()

    others = [d[v].to_numpy(float) for v in ALLVARS if v != var]
    neigh = None
    ref = []
    for s2, d2 in site_frames.items():
        if s2 == site:
            continue
        v = d2.set_index('datetime')[var].reindex(d.datetime).to_numpy(float)
        ref.append(v)
    if ref:
        R = np.vstack(ref)
        mtr = (yr >= TRAIN_YEARS[0]) & (yr <= TRAIN_YEARS[1])
        zs = []
        for v in ref:
            mu, sd = np.nanmedian(v[mtr]), max(np.nanstd(v[mtr]), 1e-6)
            zs.append((v - mu) / sd)
        zn = np.nanmedian(np.vstack(zs), axis=0)
        neigh = zn * max(np.nanstd(raw[mtr]), 1e-6) + np.nanmedian(raw[mtr])
    doy = d.datetime.dt.dayofyear.to_numpy()
    mtr = (yr >= TRAIN_YEARS[0]) & (yr <= TRAIN_YEARS[1])
    F = causal_features(raw, neigh, others, glob=global_features(raw, doy, mtr))
    return dict(raw=raw, cor=cor, year=yr, feat=F, point=target_point,
                labeled=labeled, deleted=deleted, datetime=d.datetime, n_screened=n_screened)


def windows_for(S, years, stride=1):
    n = len(S['raw'])
    ends = np.arange(W - 1, n, stride)
    yr = S['year'][ends]
    keep = (yr >= years[0]) & (yr <= years[1])
    return ends[keep]


def trimmed_mean_np(win):
    """The harness's trimmed mean, vectorised over a matrix of windows."""
    with np.errstate(invalid='ignore'):
        v = np.sort(win, axis=1)                    # NaNs sort to the end
        nv = np.isfinite(win).sum(axis=1)
        ranks = np.arange(win.shape[1])[None, :]
        k = np.floor(TRIM * nv)[:, None]
        keep = (ranks >= k) & (ranks < (nv[:, None] - k)) & (nv[:, None] - 2 * k > 0)
        out = np.where(keep, v, np.nan)
        cnt = keep.sum(axis=1)
        res = np.full(len(win), np.nan)
        good = cnt > 0
        res[good] = np.nansum(out[good], axis=1) / cnt[good]
    return res


def window_index(ends):
    return (ends - W + 1)[:, None] + np.arange(W)[None, :]


def make_tensors(S, ends, center, scale, target='corrected'):
    """Windows as tensors. Values are centred and scaled per series; the trimmed mean is
    equivariant under that, so the loss is comparable across series.

    target='corrected'  the statistic of the technician-corrected series. This is the end
        goal, but on conductivity and pH most of it is calibration drift that no retention
        weighting can remove, and the measured consequence is that the learnable signal is
        buried: trained this way the model abstained on pH even though a rule screen cuts
        the error there by 41%.
    target='retention'  the statistic over the points the technician KEPT, with raw values
        elsewhere. This isolates exactly the channel a retention model can affect. It
        changes what the model is taught, never what it is scored on: evaluation stays on
        the corrected statistic.
    """
    idx = window_index(ends)
    raw = (S['raw'][idx] - center) / scale
    valid = np.isfinite(raw)
    if target == 'retention':
        kept = np.where(S['point'][idx], np.nan, S['raw'][idx])
        tgt = trimmed_mean_np((kept - center) / scale)
    else:
        tgt = trimmed_mean_np((S['cor'][idx] - center) / scale)
    ok = np.isfinite(tgt) & valid.any(axis=1)
    return (torch.tensor(np.nan_to_num(S['feat'][idx][ok]), dtype=torch.float32),
            torch.tensor(np.nan_to_num(raw[ok]), dtype=torch.float32),
            torch.tensor(valid[ok].astype(np.float32)),
            torch.tensor(tgt[ok], dtype=torch.float32),
            torch.tensor(S['point'][idx][ok].astype(np.float32)),
            ends[ok])


def statistic_errors(S, ends, scale, weights=None, drop=None):
    """Per-window absolute error in the reported statistic, in robust-scale units.

    Returned per window rather than averaged so that arms can be compared with a PAIRED
    test on the same windows, which is the only honest way to compare two methods that
    see identical data.
    """
    idx = window_index(ends)
    raw = S['raw'][idx]
    tgt = trimmed_mean_np(S['cor'][idx])
    if weights is not None:
        v = torch.tensor(np.nan_to_num(raw), dtype=torch.float32)
        va = torch.tensor(np.isfinite(raw).astype(np.float32))
        pred, safe = weighted_trimmed_mean(v, torch.tensor(weights, dtype=torch.float32), va)
        pred = pred.numpy()
        pred[~safe.numpy()] = np.nan
    else:
        x = raw.copy()
        if drop is not None:
            x = np.where(drop[idx], np.nan, x)
        pred = trimmed_mean_np(x)
    err = np.abs(pred - tgt) / scale
    return err, np.isfinite(pred) & np.isfinite(tgt)


def statistic_mae(S, ends, scale, weights=None, drop=None):
    """Error in the reported statistic on the given windows.

    weights: soft per-point retention weights, used through the weight-aware decoder.
    drop:    a boolean per-point mask, dropped then recomputed (the standard pipeline).
    Neither: the raw series untouched.
    """
    idx = window_index(ends)
    raw = S['raw'][idx]
    tgt = trimmed_mean_np(S['cor'][idx])
    if weights is not None:
        v = torch.tensor(np.nan_to_num(raw), dtype=torch.float32)
        va = torch.tensor(np.isfinite(raw).astype(np.float32))
        pred, safe = weighted_trimmed_mean(v, torch.tensor(weights, dtype=torch.float32), va)
        pred = pred.numpy()
        pred[~safe.numpy()] = np.nan
    else:
        x = raw.copy()
        if drop is not None:
            x = np.where(drop[idx], np.nan, x)
        pred = trimmed_mean_np(x)
    m = np.isfinite(pred) & np.isfinite(tgt)
    if not m.any():
        return np.nan, 0
    return float(np.abs(pred[m] - tgt[m]).mean() / scale), int(m.sum())


def infer_weights(net, X, batch=4096):
    """Per-point retention weights for a batch of windows."""
    net.eval()
    with torch.no_grad():
        logit = torch.cat([net(X[i:i + batch].to(DEV)).cpu() for i in range(0, len(X), batch)])
    return torch.sigmoid(logit).numpy()
