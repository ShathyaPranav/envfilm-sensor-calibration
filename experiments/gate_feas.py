"""Feasibility: a support-gated residual calibrator.

  y = linear(x) + g(x) * r(x)
linear(x) extrapolates; r(x) is a boosted correction learned on the training data; g(x)
shrinks the correction where x lies outside the training support. g is label-free and has
no tuned parameter: with d(x) the mean distance to the 10 nearest training points
(standardised inputs) and d0 the same quantity for training points among themselves
(95th percentile), g = 1 inside the support and decays as exp(-(d - d0) / d0) outside.

Same split as phys_feas.py: every unit's first co-location trains (pooled per city), every
unit's second co-location tests.
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

_EXP = os.path.dirname(os.path.abspath(__file__))
if _EXP not in sys.path:
    sys.path.insert(0, _EXP)
from sec_common import TASKS, hourly, meta, periods, unit_names

warnings.filterwarnings('ignore')


def frames(city, ref, feats):
    m = meta()
    tr, te = [], []
    for n in unit_names(city):
        h = hourly(n)
        (a0, a1, s1), _, (b0, b1, s2) = periods(n, m)
        if pd.isna(a0) or pd.isna(b0):
            continue
        ok = h[feats + [ref]].notna().all(axis=1)
        f = h[ok & (h.hour >= a0) & (h.hour <= a1) & (h.site == s1)].assign(unit=n)
        s = h[ok & (h.hour >= b0) & (h.hour <= b1) & (h.site == s2)].assign(unit=n)
        if len(f) > 200 and len(s) > 200:
            tr.append(f)
            te.append(s)
    return (pd.concat(tr), pd.concat(te)) if tr else (None, None)


def gate(Xtr, Xte, k=10, cap=20000, seed=0):
    sc = StandardScaler().fit(Xtr)
    A = sc.transform(Xtr)
    rng = np.random.default_rng(seed)
    if len(A) > cap:
        A = A[rng.choice(len(A), cap, replace=False)]
    nn_ = NearestNeighbors(n_neighbors=k + 1).fit(A)
    d_in = nn_.kneighbors(A)[0][:, 1:].mean(1)
    d0 = np.quantile(d_in, 0.95)
    d = nn_.kneighbors(sc.transform(Xte), n_neighbors=k)[0].mean(1)
    return np.exp(-np.clip(d - d0, 0, None) / d0)


def main():
    rows = []
    for task in ('PM2.5', 'PM10', 'NO2', 'NO', 'O3'):
        ref, feats = TASKS[task]
        for city in ('Antwerp', 'Oslo', 'Zagreb'):
            tr, te = frames(city, ref, feats)
            if tr is None:
                continue
            y = te[ref].to_numpy()
            lin = LinearRegression().fit(tr[feats], tr[ref])
            res = HistGradientBoostingRegressor(max_iter=300, random_state=0).fit(tr[feats], tr[ref] - lin.predict(tr[feats]))
            g = gate(tr[feats].to_numpy(), te[feats].to_numpy())
            P = {
                'linear': lin.predict(te[feats]),
                'boosting': HistGradientBoostingRegressor(max_iter=300, random_state=0).fit(tr[feats], tr[ref]).predict(te[feats]),
                'MLP': make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=400,
                                                                    early_stopping=True, random_state=0)).fit(
                    tr[feats], tr[ref]).predict(te[feats]),
                'linear+residual': lin.predict(te[feats]) + res.predict(te[feats]),
                'linear+gated residual': lin.predict(te[feats]) + g * res.predict(te[feats]),
            }
            line = []
            for name, p in P.items():
                per = pd.DataFrame(dict(u=te.unit.to_numpy(), e=(p - y) ** 2)).groupby('u').e.mean() ** 0.5
                rows.append(dict(task=task, city=city, model=name, rmse=float(np.sqrt(np.mean((p - y) ** 2))),
                                 unit_median=float(per.median())))
                line.append('%s %.2f' % (name, per.median()))
            print('%-5s %-8s mean gate %.2f | %s' % (task, city, g.mean(), ' | '.join(line)), flush=True)
    r = pd.DataFrame(rows)
    w = r.pivot_table(index=['task', 'city'], columns='model', values='unit_median')
    best_other = w[['linear', 'boosting', 'MLP']].min(axis=1)
    w['gated vs best of 3'] = 100 * (1 - w['linear+gated residual'] / best_other)
    print('\nmedian per-unit RMSE; last column: % gain of the gated model over the best of linear/boosting/MLP')
    print(w.round(2).to_string())


if __name__ == '__main__':
    main()
