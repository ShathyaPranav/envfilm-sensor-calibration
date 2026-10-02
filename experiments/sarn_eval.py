"""SARN versus its controls, scored so that abstaining cannot look like accuracy.

WHY THIS REPLACES sarn_train.py's SCORING. The earlier scoring averaged each arm's error
over the windows where that arm still produced an estimate. An arm that removes every
point in a window produces no estimate there, so the window left its average. Measured
with lro_coverage_audit.py, that mattered a great deal: on MainStreet pH the rule screen
appeared to cut the error from 0.7716 to 0.4539, but it covered only 85% of the windows,
and on the windows every arm covered the figures were 0.4747 and 0.4497. Arms were being
compared on different window sets, and the ones that dropped the most were flattered the
most. The label oracle benefited the same way.

Every arm is therefore scored three ways, on every test window where the technician's
corrected statistic exists (whether or not any raw point survives in the window):

  coverage  share of those windows where the arm produces an estimate
  common    mean error over the windows where EVERY core arm produces an estimate
            (like for like)
  hold      mean error over all of them, a window with no estimate carrying the arm's
            last available estimate forward. This is what a deployed monitor displays, so
            it is the full-coverage figure and the headline.

Model selection is changed to match: on the validation year an arm that produces no
estimate for a window falls back to the untouched window statistic, so a threshold or an
epoch cannot be chosen for abstaining.

The comparison that decides the method is unchanged: SARN against 'pointnet', the same
backbone, features and capacity trained with binary cross entropy on the technician's
point labels and then thresholded. Splits are by time (fit 2014-2016, select 2017, report
2018-2020).

Also reported, because the earlier identifiability gate used a linear model on local
features only: how well each network's per-point output separates the technician's flags
from everything else (average precision against the base rate).
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

_EXP = os.path.dirname(os.path.abspath(__file__))
if _EXP not in sys.path:
    sys.path.insert(0, _EXP)
from lro_killtest import OUT, SITES, TEST_YEARS, TRAIN_YEARS, apply_rules, fit_rules, load_site, series_scale
from sarn import (DEV, FEATS, Net, VAL_YEAR, build_series, infer_weights, make_tensors,
                  trimmed_mean_np, weighted_trimmed_mean, window_index, windows_for)
from sarn_train import published_detections, sampling_weights

CORE = ['no_action', 'rules', 'oracle', 'pointnet', 'sarn_hard']


def validate(net, arm, data):
    """Validation-year error under the same full-coverage rule used at test time.

    A window for which the arm yields no estimate carries that arm's previous estimate
    forward within the same series (windows are in time order within each series); only a
    gap at the very start of a series falls back to the untouched statistic. So neither an
    epoch nor a threshold can be selected for abstaining. Thresholds are tried from the
    least aggressive, so a tie never selects the one that drops more.
    """
    net.eval()
    Xv, Vv, Mv, Tv, _ = data['val']
    with torch.no_grad():
        logit = torch.cat([net(Xv[i:i + 4096].to(DEV)).cpu() for i in range(0, len(Xv), 4096)])
    prob = torch.sigmoid(logit)
    base, base_ok = weighted_trimmed_mean(Vv, Mv.clone(), Mv)
    base, base_ok, tv = base.numpy(), base_ok.numpy(), Tv.numpy()
    bounds = np.cumsum([0] + list(data['val_sizes']))

    def score(weights):
        pred, safe = weighted_trimmed_mean(Vv, weights, Mv)
        pred = pred.numpy().astype(float)
        pred[~safe.numpy()] = np.nan
        final = np.empty_like(pred)
        for lo, hi in zip(bounds[:-1], bounds[1:]):
            seg = pd.Series(pred[lo:hi]).ffill().to_numpy()
            lead = ~np.isfinite(seg)
            seg[lead] = base[lo:hi][lead]
            final[lo:hi] = seg
        ok = np.isfinite(final) & base_ok
        return float(np.abs(final[ok] - tv[ok]).mean())

    if arm == 'sarn':
        return score((prob >= 0.5).float()), 0.5
    best = (np.inf, 0.9)
    for thr in (0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1):
        v = score((prob < thr).float())
        if v < best[0]:
            best = (v, thr)
    return best


def train_arm(arm, data, epochs, lr=1e-3, drop_penalty=0.02, seed=0, sampler=None):
    torch.manual_seed(seed)
    net = Net().to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    Xtr, Vtr, Mtr, Ttr, Ptr = data['train']
    n = len(Xtr)
    pos = float(np.clip(float(Ptr.mean()), 1e-4, 1 - 1e-4))
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((1 - pos) / pos, device=DEV))
    best = (np.inf, None, 0.5)
    for ep in range(epochs):
        net.train()
        perm = (torch.multinomial(sampler, n, replacement=True) if sampler is not None
                else torch.randperm(n))
        for i in range(0, n, 256):
            b = perm[i:i + 256]
            x, v, m = Xtr[b].to(DEV), Vtr[b].to(DEV), Mtr[b].to(DEV)
            t, p = Ttr[b].to(DEV), Ptr[b].to(DEV)
            logit = net(x)
            if arm == 'sarn':
                w = torch.sigmoid(logit)
                pred, safe = weighted_trimmed_mean(v, w, m)
                if safe.sum() == 0:
                    continue
                loss = (pred[safe] - t[safe]).abs().mean() + drop_penalty * (1 - w).mean()
            else:
                sel = m > 0
                if sel.sum() == 0:
                    continue
                loss = bce(logit[sel], p[sel])
            opt.zero_grad()
            loss.backward()
            opt.step()
        val, thr = validate(net, arm, data)
        if val < best[0]:
            best = (val, {k: t.detach().clone() for k, t in net.state_dict().items()}, thr)
    net.load_state_dict(best[1])
    return net, best[0], best[2]


def test_windows(S):
    """Every test-year window where the corrected statistic exists."""
    ends = windows_for(S, TEST_YEARS, 1)
    tgt = trimmed_mean_np(S['cor'][window_index(ends)])
    keep = np.isfinite(tgt)
    return ends[keep], tgt[keep]


def score_arms(preds, tgt, scale):
    """coverage / common / hold for each arm, plus the common-window count."""
    common = np.ones(len(tgt), bool)
    for name in CORE:
        common &= np.isfinite(preds[name])
    out = {'n_target': int(len(tgt)), 'n_common': int(common.sum())}
    base = pd.Series(preds['no_action']).ffill().to_numpy()
    for name, p in preds.items():
        ok = np.isfinite(p)
        # carry the arm's last estimate forward; before its first estimate, the untouched
        # statistic stands in, exactly as in model selection
        held = pd.Series(p).ffill().to_numpy()
        lead = ~np.isfinite(held)
        held[lead] = base[lead]
        full = np.isfinite(held)
        cm = common & ok
        out['cov_' + name] = float(ok.mean())
        out['common_' + name] = float(np.abs(p[cm] - tgt[cm]).mean() / scale) if cm.any() else np.nan
        out['hold_' + name] = float(np.abs(held[full] - tgt[full]).mean() / scale) if full.any() else np.nan
    return out, common


def main():
    from sklearn.metrics import average_precision_score
    ap = argparse.ArgumentParser()
    ap.add_argument('--vars', default='temp,cond,ph,do')
    ap.add_argument('--epochs', type=int, default=15)
    ap.add_argument('--seeds', default='0,1,2')
    ap.add_argument('--stride', type=int, default=6)
    ap.add_argument('--target', default='retention', choices=['corrected', 'retention'])
    args = ap.parse_args()
    tag = 'sarn_v2_%s' % args.target
    os.makedirs(os.path.join(OUT, 'models'), exist_ok=True)

    frames = {}
    for s in SITES:
        loaded = load_site(s)
        if loaded is not None:
            frames[s] = loaded[0]
    print('loaded %d sites, device %s, target %s' % (len(frames), DEV, args.target), flush=True)
    rows = []
    for var in args.vars.split(','):
        print('\n=== %s ===' % var, flush=True)
        series, packs = {}, {}
        for site in frames:
            S = build_series(frames, site, var)
            mtr = (S['year'] >= TRAIN_YEARS[0]) & (S['year'] <= TRAIN_YEARS[1])
            if not np.isfinite(S['cor'][mtr]).any():
                continue
            center = float(np.nanmedian(S['cor'][mtr]))
            scale = series_scale(S['cor'], mtr)
            if not np.isfinite(scale) or scale <= 0:
                continue
            tr = make_tensors(S, windows_for(S, TRAIN_YEARS, args.stride), center, scale, args.target)
            va = make_tensors(S, windows_for(S, (VAL_YEAR, VAL_YEAR), args.stride), center, scale,
                              args.target)
            ends, tgt = test_windows(S)
            if len(tr[0]) < 500 or len(va[0]) < 100 or len(ends) < 100:
                print('  %-14s skipped (windows: train %d val %d test %d)'
                      % (site, len(tr[0]), len(va[0]), len(ends)), flush=True)
                continue
            series[site] = (S, center, scale, ends, tgt)
            packs[site] = dict(train=tr, val=va)
        if not packs:
            continue

        def cat(part, j):
            return torch.cat([packs[s][part][j] for s in packs])

        flat = cat('train', 0).reshape(-1, len(FEATS))
        mu, sd = flat.mean(0), flat.std(0).clamp_min(1e-3)
        data = {part: ((cat(part, 0) - mu) / sd, cat(part, 1), cat(part, 2), cat(part, 3),
                       cat(part, 4)) for part in ('train', 'val')}
        data['val_sizes'] = [len(packs[s]['val'][0]) for s in packs]
        sampler, hit = sampling_weights(data)
        print('  %d sites | train windows %d | val %d | flagged points %.4f | moved windows %.3f'
              % (len(packs), len(data['train'][0]), len(data['val'][0]),
                 float(data['train'][4].mean()), hit), flush=True)

        for seed in [int(x) for x in args.seeds.split(',')]:
            arms = {}
            for arm in ('sarn', 'pointnet'):
                net, val, thr = train_arm(arm, data, args.epochs, seed=seed, sampler=sampler)
                arms[arm] = (net, thr)
                torch.save({'state': net.state_dict(), 'mu': mu, 'sd': sd, 'thr': thr},
                           os.path.join(OUT, 'models', '%s_%s_%s_seed%d.pt' % (tag, var, arm, seed)))
                print('  [%s] seed %d  val %.5f  threshold %.2f' % (arm, seed, val, thr), flush=True)
            for site, (S, center, scale, ends, tgt) in series.items():
                idx = window_index(ends)
                raw = S['raw'][idx]
                valid = np.isfinite(raw)
                X = (torch.tensor(np.nan_to_num(S['feat'][idx]), dtype=torch.float32) - mu) / sd
                w_sarn = infer_weights(arms['sarn'][0], X)
                p_pn = infer_weights(arms['pointnet'][0], X)
                del X
                mtr = (S['year'] >= TRAIN_YEARS[0]) & (S['year'] <= TRAIN_YEARS[1])
                scr = apply_rules(S['raw'], fit_rules(S['raw'], mtr))
                rule_mask = ~np.isfinite(scr) & np.isfinite(S['raw'])
                preds = {
                    'no_action': trimmed_mean_np(raw),
                    'rules': trimmed_mean_np(np.where(rule_mask[idx], np.nan, raw)),
                    'oracle': trimmed_mean_np(np.where(S['point'][idx], np.nan, raw)),
                    'pointnet': trimmed_mean_np(np.where(p_pn >= arms['pointnet'][1], np.nan, raw)),
                    'sarn_hard': trimmed_mean_np(np.where(w_sarn < 0.5, np.nan, raw)),
                }
                v = torch.tensor(np.nan_to_num(raw), dtype=torch.float32)
                va_t = torch.tensor(valid.astype(np.float32))
                ps, safe = weighted_trimmed_mean(v, torch.tensor(w_sarn), va_t)
                ps = ps.numpy().astype(float)
                ps[~safe.numpy()] = np.nan
                preds['sarn_soft'] = ps
                for name, mask in published_detections(site, var, S['datetime']).items():
                    preds[name] = trimmed_mean_np(np.where(mask[idx], np.nan, raw))
                res, common = score_arms(preds, tgt, scale)
                res.update(var=var, site=site, seed=seed, scale=scale,
                           kept_sarn=float((w_sarn[valid] >= 0.5).mean()),
                           kept_pointnet=float((p_pn[valid] < arms['pointnet'][1]).mean()))
                # how well each network's per-point output separates technician flags
                y = S['point'][idx][valid]
                res['base_rate'] = float(y.mean())
                if 0 < y.sum() < y.size:
                    res['ap_pointnet'] = float(average_precision_score(y, p_pn[valid]))
                    res['ap_sarn'] = float(average_precision_score(y, 1.0 - w_sarn[valid]))
                # paired difference on common windows
                e_s = np.abs(preds['sarn_hard'][common] - tgt[common]) / scale
                e_p = np.abs(preds['pointnet'][common] - tgt[common]) / scale
                res['paired_diff'] = float((e_s - e_p).mean()) if common.any() else np.nan
                rows.append(res)
                print('   %-14s cov none %.3f rules %.3f pn %.3f sarn %.3f | common: none %.4f '
                      'rules %.4f pn %.4f SARN %.4f oracle %.4f | hold: none %.4f rules %.4f pn '
                      '%.4f SARN %.4f | AP pn %.3f (base %.3f)'
                      % (site, res['cov_no_action'], res['cov_rules'], res['cov_pointnet'],
                         res['cov_sarn_hard'], res['common_no_action'], res['common_rules'],
                         res['common_pointnet'], res['common_sarn_hard'], res['common_oracle'],
                         res['hold_no_action'], res['hold_rules'], res['hold_pointnet'],
                         res['hold_sarn_hard'], res.get('ap_pointnet', np.nan), res['base_rate']),
                      flush=True)
                pd.DataFrame(rows).to_csv(os.path.join(OUT, tag + '.csv'), index=False)

    r = pd.DataFrame(rows)
    if not len(r):
        return
    r.to_csv(os.path.join(OUT, tag + '.csv'), index=False)
    for kind in ('common', 'hold', 'cov'):
        cols = ['%s_%s' % (kind, a) for a in CORE + ['sarn_soft']]
        print('\n=== %s, mean over sites and seeds ===' % kind)
        print(r.groupby('var')[cols].mean().to_string(float_format='%.4f'))
    print('\nwrote %s' % os.path.join(OUT, tag + '.csv'))


if __name__ == '__main__':
    main()
