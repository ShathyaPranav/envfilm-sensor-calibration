"""Feasibility: varying-coefficient calibration for electrochemical gas sensors across seasons.

Per unit: fit on its first co-location (spring 2020), score on its second co-location
(8-10 months later), odd days of the second co-location only (comparable with sec_protocol).
Sensor signals s = the four Alphasense cells (NO2, NO, OX, CO) and the CO2 cell.
Environment z = ambient and internal temperature and humidity, pressure.

  linear     y = a's + c'z + b
  boosting   gradient boosting on (s, z)
  MLP        two-layer network on (s, z)
  RESPIRE    y = sum_k w_k(T) s_k + b(T), w and b smooth in ambient temperature (RBF basis,
             ridge); re-implementation of the 2025 semi-parametric model without its
             outlier-trimming loop
  NVC        neural varying coefficients: y = sum_k w_k(z) s_k + b(z), w and b from a small
             network over all of z; linear in the sensor signals, nonlinear in the environment
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

_EXP = os.path.dirname(os.path.abspath(__file__))
if _EXP not in sys.path:
    sys.path.insert(0, _EXP)
from sec_common import hourly, meta, periods, unit_names

warnings.filterwarnings('ignore')
torch.set_num_threads(4)
SIG = ['NO2_B43F_P1', 'NO_B4_P1', 'OX_A431_P1', 'CO_A4_P1', 'D300']
ENV = ['SHT31TE', 'SHT31HE', 'SHT31TI', 'SHT31HI', 'BMP280']
TARGETS = {'NO2': 'Ref.NO2', 'NO': 'Ref.NO', 'O3': 'Ref.O3', 'CO': 'Ref.CO_ppm'}


def rbf(t, centers, width):
    return np.exp(-0.5 * ((t[:, None] - centers[None, :]) / width) ** 2)


def respire(tr, te, ref):
    t_all = tr['SHT31TE'].to_numpy()
    centers = np.linspace(np.percentile(t_all, 1) - 10, np.percentile(t_all, 99) + 10, 8)
    width = (centers[1] - centers[0]) * 1.5

    def design(d):
        P = rbf(d['SHT31TE'].to_numpy(), centers, width)
        S = d[SIG].to_numpy()
        return np.column_stack([(S[:, k:k + 1] * P) for k in range(S.shape[1])] + [P])
    sc = StandardScaler().fit(design(tr))
    m = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(sc.transform(design(tr)), tr[ref])
    return m.predict(sc.transform(design(te)))


class NVC(nn.Module):
    def __init__(self, k, e, H=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(e, H), nn.Tanh(), nn.Linear(H, H), nn.Tanh(), nn.Linear(H, k + 1))

    def forward(self, s, z):
        c = self.net(z)
        return (c[:, :-1] * s).sum(-1) + c[:, -1]


def nvc(tr, te, ref, seed=0):
    torch.manual_seed(seed)
    ss, sz = StandardScaler().fit(tr[SIG]), StandardScaler().fit(tr[ENV])
    ym, ysd = tr[ref].mean(), tr[ref].std()
    T = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32)
    s, z, y = T(ss.transform(tr[SIG])), T(sz.transform(tr[ENV])), T((tr[ref] - ym) / ysd)
    m = NVC(len(SIG), len(ENV))
    opt = torch.optim.Adam(m.parameters(), lr=3e-3, weight_decay=1e-3)
    for _ in range(1500):
        loss = torch.mean((m(s, z) - y) ** 2)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        return m(T(ss.transform(te[SIG])), T(sz.transform(te[ENV]))).numpy() * ysd + ym


def main():
    m = meta()
    rows = []
    for city in ('Antwerp', 'Oslo', 'Zagreb'):
        for n in unit_names(city):
            h = hourly(n)
            (a0, a1, s1), _, (b0, b1, s2) = periods(n, m)
            if pd.isna(a0) or pd.isna(b0):
                continue
            for gas, ref in TARGETS.items():
                ok = h[SIG + ENV + [ref]].notna().all(axis=1)
                tr = h[ok & (h.hour >= a0) & (h.hour <= a1) & (h.site == s1)]
                te = h[ok & (h.hour >= b0) & (h.hour <= b1) & (h.site == s2)]
                te = te[(te.hour.dt.dayofyear % 2) == 1]
                if len(tr) < 300 or len(te) < 150:
                    continue
                y = te[ref].to_numpy()
                X = SIG + ENV
                P = {'linear': LinearRegression().fit(tr[X], tr[ref]).predict(te[X]),
                     'boosting': HistGradientBoostingRegressor(max_iter=300, random_state=0).fit(tr[X], tr[ref]).predict(te[X]),
                     'MLP': make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=400,
                                                                         early_stopping=True, random_state=0)).fit(
                         tr[X], tr[ref]).predict(te[X]),
                     'RESPIRE': respire(tr, te, ref),
                     'NVC': nvc(tr, te, ref)}
                for k, p in P.items():
                    rows.append(dict(city=city, unit=n, gas=gas, model=k, rmse=float(np.sqrt(np.mean((p - y) ** 2))),
                                     sd=float(np.std(y))))
            print(n, 'done', flush=True)
    r = pd.DataFrame(rows)
    out = os.path.join(os.path.dirname(_EXP), 'results', 'sec', 'vc_feas.csv')
    r.to_csv(out, index=False)
    w = r.pivot_table(index=['gas', 'city', 'unit'], columns='model', values='rmse')
    print('\nmedian per-unit RMSE (second co-location, odd days):')
    print(w.groupby(level=[0, 1]).median().round(2).to_string())
    print('\nNVC better than X on share of units:')
    for x in ['linear', 'boosting', 'MLP', 'RESPIRE']:
        print('  vs %-8s ' % x + '  '.join('%s/%s %.0f%%' % (g, c[:3], 100 * v) for (g, c), v in
                                          (w['NVC'] < w[x]).groupby(level=[0, 1]).mean().items()))


if __name__ == '__main__':
    main()
