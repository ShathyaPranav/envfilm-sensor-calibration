"""Headroom on SensEURCity before any model is proposed.

Every unit was calibrated next to a reference station in spring 2020 (first co-location),
spent about eight months in the field, and came back to the same station in early 2021
(second co-location). That is the deployment question in its plainest form: a calibration
learned at the start, used many months later.

For each unit and pollutant:
  transfer   fit on the first co-location, score on the second
  in-period  fit on the even days of the second co-location, score on its odd days
             (what the same model class reaches when it is calibrated in the test period)
Both are scored on the same odd days. The ratio transfer / in-period is the price of the
eight months.

PASS MARK, fixed before running: the direction has headroom on a pollutant if, for the
BEST of the simple models, the median ratio across units is at least 1.5.
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

_EXP = os.path.dirname(os.path.abspath(__file__))
if _EXP not in sys.path:
    sys.path.insert(0, _EXP)
from sec_common import TASKS, hourly, meta, periods, unit_names

warnings.filterwarnings('ignore')
OUT = os.path.join(os.path.dirname(_EXP), 'results', 'sec')


def models():
    return {
        'linear': lambda: LinearRegression(),
        'boosting': lambda: HistGradientBoostingRegressor(max_iter=300, random_state=0),
        'MLP': lambda: make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=500,
                                                                    early_stopping=True, random_state=0)),
    }


def rmse(y, p):
    return float(np.sqrt(np.mean((p - y) ** 2)))


def main():
    os.makedirs(OUT, exist_ok=True)
    m = meta()
    rows = []
    for name in unit_names():
        h = hourly(name)
        (a0, a1, s1), _, (b0, b1, s2) = periods(name, m)
        if pd.isna(a0) or pd.isna(b0):
            continue
        for task, (ref, feats) in TASKS.items():
            feats = [f for f in feats if f in h.columns]
            ok = h[feats + [ref]].notna().all(axis=1)
            first = h[ok & (h.hour >= a0) & (h.hour <= a1) & (h.site == s1)]
            second = h[ok & (h.hour >= b0) & (h.hour <= b1) & (h.site == s2)]
            if len(first) < 300 or len(second) < 300:
                continue
            even = (second.hour.dt.dayofyear % 2 == 0).to_numpy()
            thr = float(np.quantile(second[ref], 0.9))
            for mname, make in models().items():
                p = make().fit(first[feats], first[ref]).predict(second[feats])
                q = make().fit(second[feats][even], second[ref][even]).predict(second[feats][~even])
                y = second[ref].to_numpy()
                top = y >= thr
                rows.append(dict(unit=name, task=task, model=mname, n_first=len(first), n_second=len(second),
                                 transfer=rmse(y, p), transfer_odd=rmse(y[~even], p[~even]),
                                 inperiod=rmse(y[~even], q), sd=float(np.std(y)),
                                 peak_bias=float(np.mean(p[top] - y[top]) / np.mean(y[top]))))
        print(name, 'done', flush=True)
    r = pd.DataFrame(rows)
    r['ratio'] = r.transfer_odd / r.inperiod
    r.to_csv(os.path.join(OUT, 'headroom_units.csv'), index=False)
    s = r.groupby(['task', 'model']).agg(units=('unit', 'nunique'), sd=('sd', 'median'),
                                         transfer=('transfer_odd', 'median'), inperiod=('inperiod', 'median'),
                                         ratio=('ratio', 'median'), peak_bias=('peak_bias', 'median')).round(3)
    print('\nmedian over units (RMSE in reference units; sd = spread of the reference in the test period):')
    print(s.to_string())
    print('\nbest simple model per unit, chosen separately for each column, then median over units:')
    for task, g in r.groupby('task'):
        tr = g.groupby('unit').transfer_odd.min()
        ip = g.groupby('unit').inperiod.min()
        ratio = float((tr / ip).median())
        who = g.loc[g.groupby('unit').transfer_odd.idxmin()].model.value_counts().to_dict()
        print('  %-6s units %2d  transfer %.2f  in-period %.2f  ratio %.2f  -> %s   (best transfer model: %s)'
              % (task, len(tr), tr.median(), ip.median(), ratio, 'HEADROOM' if ratio >= 1.5 else 'no headroom', who))


if __name__ == '__main__':
    main()
