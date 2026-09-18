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
                  statistic_errors, statistic_mae, weighted_trimmed_mean, windows_for)

PYHQC_DIR = os.path.join(os.path.dirname(_EXP), 'data', 'lro', 'pyhydroqc_results')


def published_detections(site, var, when):
    """pyhydroqc's own published detections for this series, aligned to our index.

    Returns a dict of model name to boolean mask. Empty when the resource ships no
    detections for this series (it covers conductivity at two sites and pH at one).
    """
    out = {}
    if not os.path.isdir(PYHQC_DIR):
        return out
    for f in sorted(os.listdir(PYHQC_DIR)):
        if 'detections' not in f or not f.endswith('.csv'):
            continue
        parts = f[:-4].replace('_detections', '').split('_')
        if parts[-2] != site or parts[-1] != var:
            continue
        dd = pd.read_csv(os.path.join(PYHQC_DIR, f), parse_dates=['datetime'])
        m = dd.set_index('datetime')['anomaly'].reindex(when).fillna(False).to_numpy(bool)
        out['pyhydroqc_' + '_'.join(parts[:-2])] = m
    return out


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
    ap.add_argument('--target', default='corrected', choices=['corrected', 'retention'])
    ap.add_argument('--rule-screen', action='store_true',
                    help='apply the standard rule screen first, so arms measure what the '
                         'model ADDS on top of it')
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
            S = build_series(frames, site, var, rule_screen=args.rule_screen)
            mtr = (S['year'] >= TRAIN_YEARS[0]) & (S['year'] <= TRAIN_YEARS[1])
            if not np.isfinite(S['cor'][mtr]).any():
                continue
            center = float(np.nanmedian(S['cor'][mtr]))
            scale = series_scale(S['cor'], mtr)
            if not np.isfinite(scale) or scale <= 0:
                continue
            tr = make_tensors(S, windows_for(S, TRAIN_YEARS, args.stride), center, scale,
                              args.target)
            va = make_tensors(S, windows_for(S, (VAL_YEAR, VAL_YEAR), args.stride), center, scale,
                              args.target)
            # The TEST split is always built on the corrected target, whatever the model was
            # trained against. make_tensors drops windows whose target is undefined, and the
            # retention target is undefined exactly where the technician flagged every point
            # in a window. Those are the worst windows, so letting the training target pick
            # the test set silently removed them and flattered the run (measured: the
            # no-action baseline on MainStreet pH read 0.4874 instead of 0.7716).
            te = make_tensors(S, windows_for(S, TEST_YEARS, 1), center, scale, 'corrected')
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
                # soft-weighting control: the point-label network's own probabilities used
                # as weights, separating "trained through the statistic" from "soft weights
                # instead of a hard mask"
                res['pointnet_soft'] = statistic_mae(S, ends, scale, weights=1.0 - w_pn)[0]
                for name, mask in published_detections(site, var, S['datetime']).items():
                    res[name] = statistic_mae(S, ends, scale, drop=mask)[0]
                e_sarn, ok_a = statistic_errors(S, ends, scale, weights=w_sarn)
                e_pn, ok_b = statistic_errors(S, ends, scale,
                                              weights=(w_pn < arms['pointnet'][2]).astype(float))
                both = ok_a & ok_b & np.isfinite(e_sarn) & np.isfinite(e_pn)
                if both.sum() > 100:
                    diff = e_sarn[both] - e_pn[both]
                    rng = np.random.default_rng(0)
                    draws = rng.integers(0, len(diff), size=(1000, len(diff)))
                    boot = diff[draws].mean(axis=1)
                    res['diff_mean'] = float(diff.mean())
                    res['diff_lo'] = float(np.percentile(boot, 2.5))
                    res['diff_hi'] = float(np.percentile(boot, 97.5))
                    res['win_rate'] = float((e_sarn[both] < e_pn[both]).mean())
                rows.append(res)
                print('   %-14s screened %5d | prescreened %.4f | rules %.4f | label-oracle '
                      '%.4f | pointnet %.4f | SARN %.4f (hard %.4f, keeps %.3f)'
                      % (site, res['n_screened'], res['raw'], res['rules'], res['oracle_labels'],
                         res['pointnet'], res['sarn'], res['sarn_hard'], res['mean_weight']),
                      flush=True)
                pd.DataFrame(rows).to_csv(os.path.join(
                    OUT, 'sarn_results_%s.csv'
                    % (args.target + ('_onrules' if args.rule_screen else ''))), index=False)

    r = pd.DataFrame(rows)
    if not len(r):
        print('no results')
        return
    r.to_csv(os.path.join(OUT, 'sarn_results.csv'), index=False)
    print('\n=== test-year statistic MAE (2018-2020), mean over sites and seeds ===')
    cols = [c for c in ['raw', 'rules', 'oracle_labels', 'pointnet', 'pointnet_soft', 'sarn',
                        'sarn_hard'] if c in r.columns]
    print(r.groupby('var')[cols].mean().to_string(float_format='%.4f'))
    pub = [c for c in r.columns if c.startswith('pyhydroqc_')]
    if pub:
        print()
        print('where published detections exist, on identical windows:')
        sub = r[r[pub].notna().any(axis=1)]
        print(sub.groupby(['var', 'site'])[['raw', 'rules', 'sarn', 'sarn_hard'] + pub]
              .mean().to_string(float_format='%.4f'))
    if 'diff_lo' in r.columns:
        print()
        print('paired bootstrap over test windows, SARN minus the point-label control:')
        print(r[['var', 'site', 'seed', 'diff_mean', 'diff_lo', 'diff_hi', 'win_rate']]
              .to_string(index=False, float_format='%+.5f'))
        print('series-seeds whose 95%% interval lies entirely below zero: %d of %d'
              % (int((r.diff_hi < 0).sum()), len(r)))
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
