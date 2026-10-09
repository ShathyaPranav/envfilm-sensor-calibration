"""Feasibility: does humidity physics transfer across seasons better than black boxes?

Task: Plantower PM2.5 (5325CAT) -> reference PM2.5 (FIDAS/TEOM/sequential sampler),
hourly. Train on every unit's first co-location in a city, test on every unit's second
co-location (8-10 months later, another season). Models:
  linear       x, RH, T
  boosting     all Plantower channels, number counts, OPC, met
  MLP          same inputs
  kappa        x / (1 + kappa * aw / (1 - aw)) * a   (kappa-Koehler growth, 2 numbers)
  kappa-net    same law, but kappa and a predicted per hour by a small network from the
               Plantower size channels (count ratios) and temperature: hygroscopicity
               depends on what the aerosol is, and size distribution is a proxy for that
Question for this file only: is there a city where kappa or kappa-net beats the black
boxes at the second co-location? If physics never helps, the direction is dropped.
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.optimize import least_squares
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

_EXP = os.path.dirname(os.path.abspath(__file__))
if _EXP not in sys.path:
    sys.path.insert(0, _EXP)
from sec_common import MET, OPC, PMS, hourly, meta, periods, unit_names

warnings.filterwarnings('ignore')
REF = 'Ref.PM2.5'
X = '5325CAT'
RH = 'SHT31HE'
COUNTS = ['53PT003', '53PT005', '53PT010', '53PT025', '53PT050', '53PT100']


def frames(city):
    m = meta()
    tr, te = [], []
    for n in unit_names(city):
        h = hourly(n)
        (a0, a1, s1), _, (b0, b1, s2) = periods(n, m)
        if pd.isna(a0) or pd.isna(b0):
            continue
        cols = PMS + OPC + MET + [REF]
        ok = h[cols].notna().all(axis=1)
        f = h[ok & (h.hour >= a0) & (h.hour <= a1) & (h.site == s1)].assign(unit=n)
        s = h[ok & (h.hour >= b0) & (h.hour <= b1) & (h.site == s2)].assign(unit=n)
        if len(f) > 200 and len(s) > 200:
            tr.append(f)
            te.append(s)
    return pd.concat(tr), pd.concat(te)


def aw(d):
    return np.clip(d[RH].to_numpy() / 100.0, 0.0, 0.97)


def kappa_fit(d):
    x, y, w = d[X].to_numpy(), d[REF].to_numpy(), aw(d)

    def res(p):
        return p[1] * x / (1 + p[0] * w / (1 - w)) - y
    return least_squares(res, [0.2, 1.0], bounds=([0, 0.05], [3, 10])).x


class KappaNet(nn.Module):
    def __init__(self, f):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(f, 32), nn.GELU(), nn.Linear(32, 32), nn.GELU(), nn.Linear(32, 2))

    def forward(self, z, x, w):
        k, a = self.net(z).unbind(-1)
        kappa = nn.functional.softplus(k)
        gain = torch.exp(a)
        return gain * x / (1 + kappa * w / (1 - w))


def kappa_feats(d):
    c = d[COUNTS].clip(lower=0).to_numpy() + 1.0
    frac = np.log(c / c.sum(1, keepdims=True))
    return np.column_stack([frac, d['SHT31TE'].to_numpy() / 20.0])


def kappa_net(tr, te, seed=0):
    torch.manual_seed(seed)
    zt, zs = kappa_feats(tr), kappa_feats(te)
    mu, sd = zt.mean(0), zt.std(0) + 1e-6
    T = lambda a: torch.tensor(a, dtype=torch.float32)
    zt, zs = T((zt - mu) / sd), T((zs - mu) / sd)
    xt, wt, yt = T(tr[X].to_numpy()), T(aw(tr)), T(tr[REF].to_numpy())
    net = KappaNet(zt.shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    for _ in range(1500):
        loss = torch.mean((net(zt, xt, wt) - yt) ** 2)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        return net(zs, T(te[X].to_numpy()), T(aw(te))).numpy()


def main():
    feats = PMS + OPC + MET
    for city in ('Antwerp', 'Oslo', 'Zagreb'):
        tr, te = frames(city)
        y = te[REF].to_numpy()
        P = {}
        P['raw Plantower'] = te[X].to_numpy()
        P['linear'] = LinearRegression().fit(tr[[X, RH, 'SHT31TE']], tr[REF]).predict(te[[X, RH, 'SHT31TE']])
        P['boosting'] = HistGradientBoostingRegressor(max_iter=300, random_state=0).fit(tr[feats], tr[REF]).predict(te[feats])
        P['MLP'] = make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=400,
                                                                 early_stopping=True, random_state=0)).fit(
            tr[feats], tr[REF]).predict(te[feats])
        k, a = kappa_fit(tr)
        w = aw(te)
        P['kappa (2 numbers)'] = a * te[X].to_numpy() / (1 + k * w / (1 - w))
        P['kappa-net'] = np.mean([kappa_net(tr, te, s) for s in range(3)], axis=0)
        print('\n%s: train %d h (first co-location, RH median %.0f%%), test %d h (second, RH median %.0f%%), '
              'kappa fitted %.2f' % (city, len(tr), tr[RH].median(), len(te), te[RH].median(), k))
        for name, p in P.items():
            per = pd.DataFrame(dict(u=te.unit.to_numpy(), e=(p - y) ** 2)).groupby('u').e.mean() ** 0.5
            hi = w > 0.8
            print('   %-18s RMSE %6.2f   median per unit %6.2f   RMSE when RH > 80%% %6.2f'
                  % (name, np.sqrt(np.mean((p - y) ** 2)), per.median(), np.sqrt(np.mean((p[hi] - y[hi]) ** 2))))


if __name__ == '__main__':
    main()
