"""The benchmark protocol on SensEURCity, and the simple arms any model has to beat.

A network as it is actually run:
  * every unit is calibrated once, next to a reference station (first co-location);
  * a few units then stay at reference stations for the whole deployment (anchors);
  * all other units (targets) go to streets and never see a reference again.
Test: the targets' second co-location, eight to ten months later, where a reference is
available again. It is used for scoring only.

What a method may use: every unit's first co-location, the anchors' labelled deployment
data (at stations other than the test station, up to the end of the deployment), and the
target's own unlabelled sensor readings. Nothing labelled from the test period.

Simple arms:
  own-first        per-unit model on its first co-location (standard practice)
  fleet-first-z    one model on all first co-locations, units aligned by z-scoring
  anchor-all-z     one model on all first co-locations plus all anchor deployment labels
  anchor-late-z    one model on the anchors' last 60 deployment days
  own+anchor-res   own-first plus a ridge correction learned from anchor residuals
                   (anchor's own-first model against its deployment labels, last 60 days)
  ...+ctx          the same pooled arms with a label-free context: the 5th and 50th
                   percentile of each of the unit's own aligned channels over the past week
                   (added 2026-10-03; PASS MARK fixed before its first run: a +ctx arm beats
                   its plain twin by at least 5% in median RMSE and on at least two thirds
                   of target units, for at least two gases, with both MLP and boosting)
Reference lines that use test-period labels (not deployable):
  [oracle] offset+gain   own-first with 2 numbers refitted on even days of the test period
  [oracle] in-period     model refitted on even days of the test period
Oracles are scored on odd days; so every arm is reported on odd days.
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
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
LATE = pd.Timedelta(days=60)


def build(city, task, same_site_anchors=False):
    """Everything a method may touch, and the held-out test hours."""
    m = meta()
    ref, feats = TASKS[task]
    D = dict(city=city, task=task, ref=ref, feats=feats, first={}, anchor={}, test={}, test_anchor={}, stream={},
             raw={})
    for n in unit_names(city):
        h = hourly(n)
        (a0, a1, s1), (d0, d1, sd), (b0, b1, s2) = periods(n, m)
        if pd.isna(a0) or pd.isna(b0) or not isinstance(s2, str):
            continue
        ok = h[h[feats + [ref]].notna().all(axis=1)]
        first = ok[(ok.hour >= a0) & (ok.hour <= a1) & (ok.site == s1)]
        second = ok[(ok.hour >= b0) & (ok.hour <= b1) & (ok.site == s2)]
        if len(first) < 300:
            continue
        D['first'][n] = first
        D['raw'][n] = h[['hour'] + feats].set_index('hour').sort_index()
        is_anchor = isinstance(sd, str) and 'REF' in sd
        if is_anchor:
            dep = ok[(ok.hour >= d0) & (ok.hour <= d1) & (ok.site == sd)]
            if len(dep) >= 500 and (same_site_anchors or sd != s2):
                D['anchor'][n] = dep
            if len(second) >= 300:
                D['test_anchor'][n] = second
        elif len(second) >= 300:
            D['test'][n] = second
            D['stream'][n] = h[(h.hour >= d0) & (h.hour <= b1)][['hour'] + feats].dropna()
    D['stat'] = {n: (f[feats].mean(), f[feats].std() + 1e-9) for n, f in D['first'].items()}
    D['ctx'] = {n: context(D, n) for n in D['first']}
    return D


CTX_WINDOW = '7D'


def context(D, n):
    """Label-free summary of the unit's own recent readings: 5th and 50th percentile of each
    aligned channel over the past week. Needs no reference, so it exists for every unit."""
    mu, sd = D['stat'][n]
    r = ((D['raw'][n] - mu) / sd).rolling(CTX_WINDOW, min_periods=24)
    c = pd.concat([r.quantile(0.05).add_suffix('_q05'), r.quantile(0.5).add_suffix('_q50')], axis=1)
    return c.bfill()


def zc(D, n, df):
    """Aligned channels plus the unit's context at the same hours."""
    c = D['ctx'][n].reindex(df.hour.to_numpy())
    c.index = df.index
    return pd.concat([z(D, n, df), c], axis=1).fillna(0.0)


def z(D, n, df):
    mu, sd = D['stat'][n]
    return (df[D['feats']] - mu) / sd


def make(kind):
    if kind == 'linear':
        return LinearRegression()
    if kind == 'boosting':
        return HistGradientBoostingRegressor(max_iter=300, random_state=0)
    return make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=400,
                                                        early_stopping=True, random_state=0))


def simple_arms(D, kind):
    """dict arm -> dict unit -> predictions on that unit's test hours."""
    feats, ref = D['feats'], D['ref']
    P = {}
    own = {n: make(kind).fit(f[feats], f[ref]) for n, f in D['first'].items()}
    P['own-first'] = {u: own[u].predict(t[feats]) for u, t in D['test'].items()}
    Xf = pd.concat([z(D, n, f) for n, f in D['first'].items()])
    Yf = pd.concat([f[ref] for f in D['first'].values()])
    fleet = make(kind).fit(Xf, Yf)
    P['fleet-first-z'] = {u: fleet.predict(z(D, u, t)) for u, t in D['test'].items()}
    if D['anchor']:
        Xa = pd.concat([z(D, a, d) for a, d in D['anchor'].items()])
        Ya = pd.concat([d[ref] for d in D['anchor'].values()])
        allm = make(kind).fit(pd.concat([Xf, Xa]), pd.concat([Yf, Ya]))
        P['anchor-all-z'] = {u: allm.predict(z(D, u, t)) for u, t in D['test'].items()}
        late = {a: d[d.hour >= d.hour.max() - LATE] for a, d in D['anchor'].items()}
        latem = make(kind).fit(pd.concat([z(D, a, d) for a, d in late.items()]),
                               pd.concat([d[ref] for d in late.values()]))
        P['anchor-late-z'] = {u: latem.predict(z(D, u, t)) for u, t in D['test'].items()}
        Xfc = pd.concat([zc(D, n, f) for n, f in D['first'].items()])
        Xac = pd.concat([zc(D, a, d) for a, d in D['anchor'].items()])
        allc = make(kind).fit(pd.concat([Xfc, Xac]), pd.concat([Yf, Ya]))
        P['anchor-all-z+ctx'] = {u: allc.predict(zc(D, u, t)) for u, t in D['test'].items()}
        latec = make(kind).fit(pd.concat([zc(D, a, d) for a, d in late.items()]),
                               pd.concat([d[ref] for d in late.values()]))
        P['anchor-late-z+ctx'] = {u: latec.predict(zc(D, u, t)) for u, t in D['test'].items()}
        C, R = [], []
        for a, d in late.items():
            pa = own[a].predict(d[feats])
            C.append(np.column_stack([d['SHT31TE'], d['SHT31HE'], pa]))
            R.append(d[ref].to_numpy() - pa)
        corr = make_pipeline(StandardScaler(), Ridge(1.0)).fit(np.vstack(C), np.concatenate(R))
        P['own+anchor-res'] = {u: P['own-first'][u] + corr.predict(
            np.column_stack([t['SHT31TE'], t['SHT31HE'], P['own-first'][u]])) for u, t in D['test'].items()}
    og, ip = {}, {}
    for u, t in D['test'].items():
        e = even(t)
        y = t[ref].to_numpy()
        p = P['own-first'][u]
        b, a = np.polyfit(p[e], y[e], 1)
        og[u] = a + b * p
        ip[u] = make(kind).fit(t[feats][e], y[e]).predict(t[feats])
    P['[oracle] offset+gain'] = og
    P['[oracle] in-period'] = ip
    return P


def even(t):
    return (t.hour.dt.dayofyear % 2 == 0).to_numpy()


def score(D, P):
    """Per-unit RMSE on odd test days for every arm, plus peak bias (top decile of the reference)."""
    ref = D['ref']
    rows = []
    for arm, pu in P.items():
        for u, p in pu.items():
            t = D['test'][u]
            o = ~even(t)
            y = t[ref].to_numpy()[o]
            q = np.asarray(p)[o]
            top = y >= np.quantile(y, 0.9)
            rows.append(dict(arm=arm, unit=u, rmse=float(np.sqrt(np.mean((q - y) ** 2))),
                             mae=float(np.mean(np.abs(q - y))), bias=float(np.mean(q - y)),
                             peak_bias=float(np.mean(q[top] - y[top]) / np.mean(y[top])), sd=float(np.std(y))))
    return pd.DataFrame(rows)


def report(D, S, base='own-first'):
    w = S.pivot(index='unit', columns='arm', values='rmse')
    order = [a for a in S.arm.unique()]
    w = w[order]
    print('\n%s %s: %d target units, %d anchors, reference sd %.2f' % (D['city'], D['task'], len(w), len(D['anchor']),
                                                                      S.sd.median()))
    pk = S.groupby('arm').peak_bias.median()
    for a in order:
        gain = 100 * (1 - (w[a] / w[base])).median()
        print('   %-24s median RMSE %6.2f   vs %s %+6.1f%%   better on %2d of %2d   peak bias %+5.1f%%'
              % (a, w[a].median(), base, gain, int((w[a] < w[base]).sum()), len(w), 100 * pk[a]))
    return w


def main():
    os.makedirs(OUT, exist_ok=True)
    cities = [a for a in sys.argv[1:] if a in ('Antwerp', 'Oslo', 'Zagreb')] or ['Antwerp']
    tasks = [a for a in sys.argv[1:] if a in TASKS] or list(TASKS)
    kinds = [a for a in sys.argv[1:] if a in ('linear', 'mlp', 'boosting')] or ['linear', 'mlp']
    out = []
    for city in cities:
        for task in tasks:
            D = build(city, task)
            if len(D['test']) < 3:
                print('%s %s: only %d target units, skipped' % (city, task, len(D['test'])))
                continue
            for kind in kinds:
                S = score(D, simple_arms(D, kind))
                print('\n--- model class: %s' % kind, end='')
                report(D, S)
                out.append(S.assign(city=city, task=task, kind=kind))
    if out:
        pd.concat(out).to_csv(os.path.join(OUT, 'protocol_simple_%s.csv' % '_'.join(cities)), index=False)


if __name__ == '__main__':
    main()
