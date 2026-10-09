"""Triage oracles O1-O4 (results/tesla/PREREG_oracles.md) on held-out training units."""
import os
import sys

import numpy as np
import pandas as pd
import torch

_EXP = os.path.dirname(os.path.abspath(__file__))
if _EXP not in sys.path:
    sys.path.insert(0, _EXP)
import tsl_bench as B
from tsl_data import CITIES, PMS
from tsl_rotate import unit_index
from sklearn.neural_network import MLPRegressor


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def main():
    rows = []
    for pm in PMS:
        for city in CITIES:
            S = B.Split(city, pm)
            B.CURRENT['pm'] = pm
            runs = unit_index(S)[:-2]
            n = len(runs)
            held = [k for k in range(n) if k % 4 == 0]
            rest = [k for k in range(n) if k not in held]
            S.train = np.concatenate([runs[k][::4] for k in rest[:-2]])
            S.val = np.concatenate([runs[k] for k in rest[-2:]])
            model, _, _ = B.fit('TCN', S, seed=0)
            for k in held:
                idx = runs[k]
                p = B.predict(model, S, idx).cpu().numpy() * B.SCALE
                y = S.y[torch.as_tensor(idx, device=B.DEV) - 1].cpu().numpy() * B.SCALE
                f = load_dates(S, k, idx)
                day = f.dt.floor('D').to_numpy()
                even = (f.dt.dayofyear % 2 == 0).to_numpy()
                g = (y @ p) / (p @ p)
                r0, r1 = rmse(y, p), rmse(y, g * p)
                # O2 on odd days with fits on even days
                ge = (y[even] @ p[even]) / (p[even] @ p[even])
                A = np.column_stack([p[even], np.ones(even.sum())])
                a, b = np.linalg.lstsq(A, y[even], rcond=None)[0]
                F = lambda q: np.column_stack([q, np.log1p(np.clip(q, 0, None))])
                mlp = MLPRegressor(hidden_layer_sizes=(16, 16), max_iter=500, random_state=0).fit(F(p[even]), y[even])
                o = ~even
                s_g, s_a, s_m = rmse(y[o], ge * p[o]), rmse(y[o], a * p[o] + b), rmse(y[o], mlp.predict(F(p[o])))
                # O3 daily gains
                dg = pd.DataFrame(dict(d=day, yp=y * p, pp=p * p)).groupby('d').sum()
                dg = dg[dg.pp > 0]
                daily = (dg.yp / dg.pp)
                spread = float((daily.quantile(.75) - daily.quantile(.25)) / daily.median()) if len(daily) > 3 else np.nan
                dy = np.diff(y)
                floor = float(np.sqrt(np.var(dy) / 2))
                rows.append(dict(pm=pm, city=city, unit=S.units[k], rmse=r0, rmse_O1=r1, gain=g,
                                 gain_share=1 - r1 ** 2 / r0 ** 2, O2_scalar=s_g, O2_affine=s_a, O2_mlp=s_m,
                                 O3_daily_spread=spread, floor=floor))
            print(pm, city, 'done', flush=True)
    r = pd.DataFrame(rows)
    r.to_csv(os.path.join(B.OUT, 'oracles.csv'), index=False)
    s = r.groupby(['pm', 'city']).median(numeric_only=True)
    s['O1/floor'] = s.rmse_O1 / s.floor
    s['mlp_vs_scalar'] = 1 - s.O2_mlp / s.O2_scalar
    print(s[['rmse', 'rmse_O1', 'gain_share', 'floor', 'O1/floor', 'O2_scalar', 'O2_affine', 'O2_mlp', 'mlp_vs_scalar',
             'O3_daily_spread']].round(3).to_string())
    print('\nmedians over all units: O1/floor %.2f  MLP vs scalar %.3f  daily gain spread %.3f' % (
        (r.rmse_O1 / r.floor).median(), (1 - r.O2_mlp / r.O2_scalar).median(), r.O3_daily_spread.median()))


def load_dates(S, k, idx):
    from tsl_data import load
    if not hasattr(S, '_dates'):
        S._dates = pd.concat([f.date for _, f in load(S.city)], ignore_index=True)
    return S._dates.iloc[idx - 1].reset_index(drop=True)


if __name__ == '__main__':
    main()
