"""Kill test for the anchor idea, with simple models only.

In each city a handful of units spent the whole deployment next to a reference station
("anchors"); the rest stood in streets with no reference. A real network looks the same:
a few nodes at regulatory stations, most elsewhere. The idea: the anchors show, month by
month, how the calibration of this sensor model is going wrong, and that knowledge is
passed to units that have no reference.

It only works if the calibration error is SHARED across units and sites. That is tested
here before any network is designed. Each anchor in turn plays the unit without a
reference (its own reference is used for scoring only); the anchors at OTHER stations are
the ones allowed to supply labels.

Arms, scored on 30-day blocks of the deployment:
  own-first        fitted on the target unit's first co-location, never updated
  fleet-first      one model on every unit's first co-location
  anchor-recent    one model on the other anchors' last 30 days of (sensor, reference)
  ...-z            the same arms after each unit's channels are scaled by their own
                   first-co-location mean and spread (all units saw the same air then)
  anchor-all-z     one model on every first co-location plus all anchor labels so far
  anchor-residual  own-first plus a correction learned from the other anchors' last-30-day
                   residuals, as a function of temperature, humidity and the prediction
  [oracle] own-recent  the target's own last 30 days of labels (not available in practice)

PASS MARKS, fixed before running, on the median over target units of
      closed = (own-first - arm) / (own-first - oracle)
  closed >= 0.30 for the better anchor arm: the error is shared, the idea is alive
  closed <  0.10: the error is unit- or site-specific, the idea is dead
  closed >= 0.80: a simple procedure already does it, a network has little to add
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

_EXP = os.path.dirname(os.path.abspath(__file__))
if _EXP not in sys.path:
    sys.path.insert(0, _EXP)
from sec_common import TASKS, hourly, meta, periods, unit_names

warnings.filterwarnings('ignore')
OUT = os.path.join(os.path.dirname(_EXP), 'results', 'sec')
BLOCK = pd.Timedelta(days=30)
KIND = os.environ.get('SEC_MODEL', 'mlp')


def make():
    if KIND == 'linear':
        return LinearRegression()
    return make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=400,
                                                        early_stopping=True, random_state=0))


def rmse(e):
    return float(np.sqrt(np.mean(np.square(e))))


def run_city(city, m, tasks):
    names = unit_names(city)
    H = {n: hourly(n) for n in names}
    P = {n: periods(n, m) for n in names}
    anchors = [n for n in names if isinstance(P[n][1][2], str) and 'REF' in P[n][1][2]]
    print('%s: %d units on disk, %d anchors: %s' % (city, len(names), len(anchors),
                                                    {a.split('_')[1]: P[a][1][2] for a in anchors}), flush=True)
    rows = []
    for task in tasks:
        ref, feats = TASKS[task]
        ok = {n: H[n][H[n][feats + [ref]].notna().all(axis=1)] for n in names}
        first = {n: ok[n][(ok[n].hour >= P[n][0][0]) & (ok[n].hour <= P[n][0][1])] for n in names}
        first = {n: f for n, f in first.items() if len(f) >= 300}
        own = {n: make().fit(first[n][feats], first[n][ref]) for n in anchors if n in first}
        fleet = make().fit(pd.concat([f[feats] for f in first.values()]), pd.concat([f[ref] for f in first.values()]))
        # every unit saw the same air during the first co-location, so scaling each channel by
        # its own first-co-location mean and spread puts the units on a common footing
        stat = {n: (first[n][feats].mean(), first[n][feats].std() + 1e-9) for n in first}

        def z(n, df):
            return (df[feats] - stat[n][0]) / stat[n][1]

        fleet_z = make().fit(pd.concat([z(n, f) for n, f in first.items()]),
                             pd.concat([f[ref] for f in first.values()]))
        for u in anchors:
            if u not in own:
                continue
            d0, d1, site = P[u][1]
            du = ok[u][(ok[u].hour >= d0) & (ok[u].hour <= d1) & (ok[u].site == site)]
            if len(du) < 500:
                continue
            others = [a for a in anchors if a != u and P[a][1][2] != site and a in own]
            t = d0
            while t < d1:
                te = du[(du.hour >= t) & (du.hour < t + BLOCK)]
                t0 = t
                t = t + BLOCK
                if len(te) < 100:
                    continue
                y = te[ref].to_numpy()
                p_own = own[u].predict(te[feats])
                res = dict(unit=u, task=task, block=t0, n=len(te), sd=float(np.std(y)), n_anchor=0)
                res['own-first'] = rmse(p_own - y)
                res['fleet-first'] = rmse(fleet.predict(te[feats]) - y)
                res['fleet-first-z'] = rmse(fleet_z.predict(z(u, te)) - y)
                # labelled data the other anchors produced in the 30 days before this block
                rec = {a: ok[a][(ok[a].hour >= t0 - BLOCK) & (ok[a].hour < t0)] for a in others}
                rec = {a: r for a, r in rec.items() if len(r) >= 50}
                if rec:
                    X = pd.concat([r[feats] for r in rec.values()])
                    Y = pd.concat([r[ref] for r in rec.values()])
                    res['n_anchor'] = len(rec)
                    res['anchor-recent'] = rmse(make().fit(X, Y).predict(te[feats]) - y)
                    Xz = pd.concat([z(a, r) for a, r in rec.items()])
                    res['anchor-recent-z'] = rmse(make().fit(Xz, Y).predict(z(u, te)) - y)
                    # everything the network has: all first co-locations plus the anchors' labels so far
                    past = {a: ok[a][(ok[a].hour >= P[a][1][0]) & (ok[a].hour < t0)] for a in others}
                    Xa = pd.concat([z(n, f) for n, f in first.items()] + [z(a, r) for a, r in past.items()])
                    Ya = pd.concat([f[ref] for f in first.values()] + [r[ref] for r in past.values()])
                    res['anchor-all-z'] = rmse(make().fit(Xa, Ya).predict(z(u, te)) - y)
                    C, R = [], []
                    for a, r in rec.items():
                        pa = own[a].predict(r[feats])
                        C.append(np.column_stack([r['SHT31TE'], r['SHT31HE'], pa]))
                        R.append(r[ref].to_numpy() - pa)
                    corr = make_pipeline(StandardScaler(), Ridge(1.0)).fit(np.vstack(C), np.concatenate(R))
                    cu = np.column_stack([te['SHT31TE'], te['SHT31HE'], p_own])
                    res['anchor-residual'] = rmse(p_own + corr.predict(cu) - y)
                    res['anchor-offset'] = rmse(p_own + float(np.mean(np.concatenate(R))) - y)
                mine = ok[u][(ok[u].hour >= t0 - BLOCK) & (ok[u].hour < t0)]
                if len(mine) >= 200:
                    res['[oracle] own-recent'] = rmse(make().fit(mine[feats], mine[ref]).predict(te[feats]) - y)
                rows.append(res)
            print('   %s %s done' % (task, u), flush=True)
    return pd.DataFrame(rows)


def main():
    os.makedirs(OUT, exist_ok=True)
    m = meta()
    cities = [a for a in sys.argv[1:] if a in ('Antwerp', 'Oslo', 'Zagreb')] or ['Antwerp']
    tasks = [a for a in sys.argv[1:] if a in TASKS] or ['NO2', 'NO', 'PM2.5']
    r = pd.concat([run_city(c, m, tasks) for c in cities], ignore_index=True)
    r.to_csv(os.path.join(OUT, 'anchor_test_%s.csv' % KIND), index=False)
    arms = ['own-first', 'fleet-first', 'fleet-first-z', 'anchor-recent', 'anchor-recent-z', 'anchor-all-z',
            'anchor-offset', 'anchor-residual', '[oracle] own-recent']
    full = r.dropna(subset=arms)
    print('\nmodel class: %s. blocks with every arm available: %d of %d' % (KIND, len(full), len(r)))
    for task, g in full.groupby('task'):
        per = g.groupby('unit')[arms].apply(lambda x: np.sqrt((x ** 2).mean()))
        print('\n%s: RMSE per target unit over its deployment (reference sd %.2f)' % (task, g.sd.median()))
        print(per.round(2).to_string())
        gap = per['own-first'] - per['[oracle] own-recent']
        print('  median RMSE: ' + '   '.join('%s %.2f' % (a, per[a].median()) for a in arms))
        for a in arms[1:-1]:
            closed = ((per['own-first'] - per[a]) / gap)[gap > 0]
            print('  %-16s closes %+.0f%% of the gap (median over %d units); better than own-first on %d of %d'
                  % (a, 100 * closed.median(), len(closed), int((per[a] < per['own-first']).sum()), len(per)))
        by = g.assign(month=g.block.dt.to_period('M')).groupby('month')[arms].median()
        print('  by month (median over units):')
        print(by.round(2).to_string())


if __name__ == '__main__':
    main()
