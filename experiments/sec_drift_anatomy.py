"""What does eight months do to a calibration, and is it the same for every unit?

Each unit's first-co-location model is applied to its second co-location. The error is
then taken apart:
  offset+gain   refit only y = a + b * prediction on the second co-location (2 numbers)
  in-period     refit the whole model on the second co-location
Both refits use even days and are scored on odd days, as is the untouched model.

Then the question that decides whether a few reference-sited units can repair the others:
are a and b alike across units? "fleet a,b" applies to each unit the MEDIAN a and b of
the OTHER units (leave-one-out), which is what a perfect anchor scheme with a 2-number
correction could deliver.

PASS MARK, fixed before running: the shared 2-number correction is worth building on if
"fleet a,b" closes at least 30% of the gap between the untouched model and the in-period
refit, in the median over units, on at least one gas.
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
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
KIND = os.environ.get('SEC_MODEL', 'mlp')


def make():
    if KIND == 'linear':
        return LinearRegression()
    return make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=400,
                                                        early_stopping=True, random_state=0))


def rmse(e):
    return float(np.sqrt(np.mean(np.square(e))))


def main():
    os.makedirs(OUT, exist_ok=True)
    m = meta()
    city = sys.argv[1] if len(sys.argv) > 1 else 'Antwerp'
    rows, keep = [], {}
    for name in unit_names(city):
        h = hourly(name)
        (a0, a1, s1), _, (b0, b1, s2) = periods(name, m)
        if pd.isna(a0) or pd.isna(b0):
            continue
        for task, (ref, feats) in TASKS.items():
            ok = h[feats + [ref]].notna().all(axis=1)
            first = h[ok & (h.hour >= a0) & (h.hour <= a1) & (h.site == s1)]
            second = h[ok & (h.hour >= b0) & (h.hour <= b1) & (h.site == s2)]
            if len(first) < 300 or len(second) < 300:
                continue
            even = (second.hour.dt.dayofyear % 2 == 0).to_numpy()
            y = second[ref].to_numpy()
            p = make().fit(first[feats], first[ref]).predict(second[feats])
            b, a = np.polyfit(p[even], y[even], 1)
            q = make().fit(second[feats][even], y[even]).predict(second[feats][~even])
            rows.append(dict(unit=name, task=task, a=a, b=b, untouched=rmse(p[~even] - y[~even]),
                             offset_only=rmse(p[~even] + np.mean(y[even] - p[even]) - y[~even]),
                             offset_gain=rmse(a + b * p[~even] - y[~even]), inperiod=rmse(q - y[~even]),
                             bias=float(np.mean(p - y)), sd=float(np.std(y)),
                             dT=float(second['SHT31TE'].mean() - first['SHT31TE'].mean())))
            keep[(name, task)] = (p[~even], y[~even])
        print(name, 'done', flush=True)
    r = pd.DataFrame(rows)
    fl = []
    for i, row in r.iterrows():
        o = r[(r.task == row.task) & (r.unit != row.unit)]
        p, y = keep[(row.unit, row.task)]
        fl.append(rmse(o.a.median() + o.b.median() * p - y) if len(o) >= 2 else np.nan)
    r['fleet_ab'] = fl
    r.to_csv(os.path.join(OUT, 'drift_anatomy_%s_%s.csv' % (city, KIND)), index=False)
    print('\n%s, model class %s, median over units' % (city, KIND))
    for task, g in r.groupby('task'):
        gap = g.untouched - g.inperiod
        cl = lambda c: 100 * float(((g.untouched - g[c]) / gap)[gap > 0].median())
        print('  %-6s units %2d | RMSE untouched %.2f  offset %.2f  offset+gain %.2f  fleet a,b %.2f  in-period %.2f '
              '(sd %.2f)' % (task, len(g), g.untouched.median(), g.offset_only.median(), g.offset_gain.median(),
                             g.fleet_ab.median(), g.inperiod.median(), g.sd.median()))
        print('         gap closed: offset %+.0f%%  offset+gain %+.0f%%  fleet a,b %+.0f%%  (fleet a,b beats untouched '
              'on %d of %d units)' % (cl('offset_only'), cl('offset_gain'), cl('fleet_ab'),
                                      int((g.fleet_ab < g.untouched).sum()), int(g.fleet_ab.notna().sum())))
        print('         across units: gain b median %.2f (IQR %.2f to %.2f)   offset a median %+.2f (IQR %+.2f to '
              '%+.2f)   mean bias %+.2f' % (g.b.median(), g.b.quantile(.25), g.b.quantile(.75), g.a.median(),
                                           g.a.quantile(.25), g.a.quantile(.75), g.bias.median()))


if __name__ == '__main__':
    main()
