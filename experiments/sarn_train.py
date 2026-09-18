"""Train and evaluate SARN against its controls on Logan River.

The decisive comparison is SARN versus 'pointnet': the SAME backbone, the SAME features,
the SAME capacity, trained the standard way (binary cross entropy on the technician's
point labels, then thresholded and hard-masked). Both arms select their epoch on the SAME
validation criterion, the error in the reported statistic on a held-out year, and the
control additionally gets to choose its threshold on that validation year. If SARN does
not beat it, training through the statistic buys nothing and the claim is withdrawn.

Splits are by time: fit on 2014-2016, select on 2017, report on 2018-2020. Features are
standardised with training-window statistics only.
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
                  statistic_mae, weighted_trimmed_mean, windows_for)


def validate(net, arm, data):
    """Validation-year statistic MAE; for the control, also pick its best threshold."""
    net.eval()
    Xv, Vv, Mv, Tv, _ = data['val']
    with torch.no_grad():
        logit = torch.cat([net(Xv[i:i + 4096].to(DEV)).cpu() for i in range(0, len(Xv), 4096)])
    prob = torch.sigmoid(logit)
    if arm == 'sarn':
        pred, safe = weighted_trimmed_mean(Vv, prob, Mv)
        return float((pred[safe] - Tv[safe]).abs().mean()), 0.5
    best = (np.inf, 0.5)
    for thr in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        keep = (prob < thr).float()                 # points the classifier flags are dropped
        pred, safe = weighted_trimmed_mean(Vv, keep, Mv)
        v = float((pred[safe] - Tv[safe]).abs().mean())
        if v < best[0]:
            best = (v, thr)
    return best


def sampling_weights(data, boost=0.2):
    """Half of each batch is drawn from windows the technician's correction actually moves.

    Without this, 99% of windows are clean, 'keep everything' minimises the average loss,
    and the model converges to a no-op (measured: mean retention weight 1.000). But
    oversampling too hard is its own failure: at boost 0.5 both arms learned to discard a
    third of ALL data, because in training half the windows were contaminated while at
    test time about 1% are. Boost is kept mild for that reason. It is computed from
    TRAINING windows only and applied identically to both arms.
    """
    _, V, M, T, _ = data['train']
    pred, safe = weighted_trimmed_mean(V, M.clone(), M)
    err = (pred - T).abs().numpy()
    err[~safe.numpy()] = 0.0
    hit = err > 0.01
    w = np.full(len(err), (1.0 - boost) / max((~hit).sum(), 1))
    if hit.any():
        w[hit] = boost / hit.sum()
    return torch.tensor(w / w.sum(), dtype=torch.double), float(hit.mean())


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
        tot = 0.0
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
                # L1 matches the metric being reported; Huber with a wide delta made
                # typical errors (far below one scale unit) almost gradient-free
                loss = ((pred[safe] - t[safe]).abs().mean()
                        + drop_penalty * (1 - w).mean())
            else:
                sel = m > 0
                if sel.sum() == 0:
                    continue
                loss = bce(logit[sel], p[sel])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss) * len(b)
        val, thr = validate(net, arm, data)
        star = ''
        if val < best[0]:
            best = (val, {k: t.detach().clone() for k, t in net.state_dict().items()}, thr)
            star = '  *'
        print('    epoch %2d  train loss %.5f  val statistic MAE %.5f%s'
              % (ep + 1, tot / max(n, 1), val, star), flush=True)
    net.load_state_dict(best[1])
    return net, best[0], best[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--vars', default='temp,cond,ph,do')
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--seeds', default='0')
    ap.add_argument('--stride', type=int, default=3)
    args = ap.parse_args()

    frames = {}
    for s in SITES:
        loaded = load_site(s)
        if loaded is not None:
            frames[s] = loaded[0]
    print('loaded %d sites, device %s' % (len(frames), DEV))
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
            tr = make_tensors(S, windows_for(S, TRAIN_YEARS, args.stride), center, scale)
            va = make_tensors(S, windows_for(S, (VAL_YEAR, VAL_YEAR), args.stride), center, scale)
            te = make_tensors(S, windows_for(S, TEST_YEARS, 1), center, scale)
            if len(tr[0]) < 500 or len(va[0]) < 100 or len(te[0]) < 100:
                print('  %-14s skipped (windows: train %d val %d test %d)'
                      % (site, len(tr[0]), len(va[0]), len(te[0])))
                continue
            series[site] = (S, center, scale)
            packs[site] = dict(train=tr, val=va, test=te)
        if not packs:
            continue

        def cat(part, j):
            return torch.cat([packs[s][part][j] for s in packs])

        flat = cat('train', 0).reshape(-1, len(FEATS))
        mu, sd = flat.mean(0), flat.std(0).clamp_min(1e-3)
        data = {}
        for part in ('train', 'val'):
            data[part] = ((cat(part, 0) - mu) / sd, cat(part, 1), cat(part, 2),
                          cat(part, 3), cat(part, 4))
        print('  %d sites | train windows %d | val %d | flagged points %.4f'
              % (len(packs), len(data['train'][0]), len(data['val'][0]),
                 float(data['train'][4].mean())), flush=True)

        sampler, hit_rate = sampling_weights(data)
        print('  windows the correction actually moves: %.3f of training windows' % hit_rate,
              flush=True)
        for seed in [int(x) for x in args.seeds.split(',')]:
            arms = {}
            for arm in ('sarn', 'pointnet'):
                print('  [%s] seed %d' % (arm, seed), flush=True)
                net, val, thr = train_arm(arm, data, args.epochs, seed=seed, sampler=sampler)
                print('    best val statistic MAE %.5f  threshold %.2f' % (val, thr), flush=True)
                arms[arm] = (net, val, thr)
            for site, (S, center, scale) in series.items():
                Xt, _, _, _, _, ends = packs[site]['test']
                Xn = (Xt - mu) / sd
                mtr = (S['year'] >= TRAIN_YEARS[0]) & (S['year'] <= TRAIN_YEARS[1])
                scr = apply_rules(S['raw'], fit_rules(S['raw'], mtr))
                w_sarn = infer_weights(arms['sarn'][0], Xn)
                w_pn = infer_weights(arms['pointnet'][0], Xn)
                res = dict(var=var, site=site, seed=seed, n_test=len(ends),
                           n_screened=S['n_screened'])
                res['raw'] = statistic_mae(S, ends, scale)[0]
                res['rules'] = statistic_mae(S, ends, scale,
                                             drop=~np.isfinite(scr) & np.isfinite(S['raw']))[0]
                res['oracle_labels'] = statistic_mae(S, ends, scale, drop=S['point'])[0]
                res['pointnet'] = statistic_mae(S, ends, scale,
                                                weights=(w_pn < arms['pointnet'][2]).astype(float))[0]
                res['sarn'] = statistic_mae(S, ends, scale, weights=w_sarn)[0]
                res['sarn_hard'] = statistic_mae(S, ends, scale,
                                                 weights=(w_sarn >= 0.5).astype(float))[0]
                res['mean_weight'] = float(w_sarn.mean())
                rows.append(res)
                print('   %-14s screened %5d | prescreened %.4f | rules %.4f | label-oracle '
                      '%.4f | pointnet %.4f | SARN %.4f (hard %.4f, keeps %.3f)'
                      % (site, res['n_screened'], res['raw'], res['rules'], res['oracle_labels'],
                         res['pointnet'], res['sarn'], res['sarn_hard'], res['mean_weight']),
                      flush=True)
                pd.DataFrame(rows).to_csv(os.path.join(OUT, 'sarn_results.csv'), index=False)

    r = pd.DataFrame(rows)
    if not len(r):
        print('no results')
        return
    r.to_csv(os.path.join(OUT, 'sarn_results.csv'), index=False)
    print('\n=== test-year statistic MAE (2018-2020), mean over sites and seeds ===')
    print(r.groupby('var')[['raw', 'rules', 'oracle_labels', 'pointnet', 'sarn', 'sarn_hard']]
          .mean().to_string(float_format='%.4f'))
    r['vs_pointnet'] = (r.sarn - r.pointnet) / r.pointnet
    r['vs_rules'] = (r.sarn - r.rules) / r.rules
    print('\nrelative change from SARN (negative means SARN is better):')
    print(r.groupby(['var', 'site'])[['vs_pointnet', 'vs_rules']].mean().to_string(float_format='%+.3f'))
    print('\nSARN beats the point-label control on %d of %d series-seeds; median relative change '
          '%+.3f' % (int((r.sarn < r.pointnet).sum()), len(r), float(r.vs_pointnet.median())))
    print('SARN beats the rules screen on %d of %d; median relative change %+.3f'
          % (int((r.sarn < r.rules).sum()), len(r), float(r.vs_rules.median())))


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    main()
