"""Headroom test before any model: low-cost sensor calibration on UCI Air Quality.

Three separate 2025-2026 papers report the same weakness in learned calibration: models
regress toward the mean, overestimating the baseline and UNDERESTIMATING PEAKS, and
overall RMSE or mean bias hides it (Frischmon et al. 2026; the OxAria LSTM paper 2026; the
"mean trap" in Tesla, AAAI 2025). Peaks are what air-quality limits are about, so this is
the error that matters to the data's consumer.

Before building anything, three questions with the answers decided by measurement:

  Q1  Is the weakness present here? Error and bias in the top decile of the reference,
      against the overall figures, for a linear and a strong tree model.
  Q2  Does a learned model beat the linear baseline at all, on a split by TIME (train on
      the first months, test on the rest)? Logan River failed this kind of question.
  Q3  Does a trivial fix already close the gap? Sample weights proportional to
      concentration, and a quantile loss. If either works, no new model is needed, which
      is the lesson of the rule screen beating SARN.

A direction is worth pursuing only if Q1 shows a clear peak penalty, Q2 shows the learned
model ahead of linear, and Q3 leaves a real gap.
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(_ROOT, 'data', 'uci_airquality', 'AirQualityUCI.csv')
OUT = os.path.join(_ROOT, 'results', 'calib')
SENS = ['PT08.S1(CO)', 'PT08.S2(NMHC)', 'PT08.S3(NOx)', 'PT08.S4(NO2)', 'PT08.S5(O3)', 'T', 'RH', 'AH']
TARGETS = ['CO(GT)', 'NO2(GT)', 'NOx(GT)', 'C6H6(GT)']
TRAIN_FRAC = 0.5          # first half of the year trains, second half tests


def metrics(y, p, thr):
    top = y >= thr
    return dict(rmse=float(np.sqrt(np.mean((p - y) ** 2))),
                rmse_top=float(np.sqrt(np.mean((p[top] - y[top]) ** 2))),
                bias_top=float(np.mean(p[top] - y[top])),
                rel_bias_top=float(np.mean(p[top] - y[top]) / np.mean(y[top])),
                recall_top=float((p[top] >= thr).mean()),
                bias_low=float(np.mean(p[y <= np.quantile(y, 0.1)] - y[y <= np.quantile(y, 0.1)])))


def main():
    os.makedirs(OUT, exist_ok=True)
    d = pd.read_csv(CSV, sep=';', decimal=',').dropna(how='all', axis=1).dropna(how='all')
    d = d.replace(-200, np.nan)
    rows = []
    for tgt in TARGETS:
        ok = d[SENS + [tgt]].notna().all(axis=1).to_numpy()
        X = d.loc[ok, SENS].to_numpy(float)
        y = d.loc[ok, tgt].to_numpy(float)
        n = len(y)
        cut = int(TRAIN_FRAC * n)
        Xtr, ytr, Xte, yte = X[:cut], y[:cut], X[cut:], y[cut:]
        thr = float(np.quantile(ytr, 0.9))               # "peak": top decile of training reference
        w = ytr / ytr.mean()
        models = {
            'linear': LinearRegression().fit(Xtr, ytr),
            'boosting': HistGradientBoostingRegressor(max_iter=400, random_state=0).fit(Xtr, ytr),
            'boosting, peak-weighted': HistGradientBoostingRegressor(max_iter=400, random_state=0)
            .fit(Xtr, ytr, sample_weight=w ** 2),
            'boosting, q=0.7 loss': HistGradientBoostingRegressor(loss='quantile', quantile=0.7, max_iter=400,
                                                                  random_state=0).fit(Xtr, ytr),
        }
        # how much of the test error is drift: the same model, with a random split instead
        rng = np.random.default_rng(0)
        perm = rng.permutation(n)
        a, b = perm[:cut], perm[cut:]
        rnd = HistGradientBoostingRegressor(max_iter=400, random_state=0).fit(X[a], y[a])
        rmse_random = float(np.sqrt(np.mean((rnd.predict(X[b]) - y[b]) ** 2)))
        print('\n%s  n=%d  train %d (first half)  test %d  | peak threshold %.2f (test share above: %.3f)'
              % (tgt, n, cut, n - cut, thr, float((yte >= thr).mean())))
        for name, m in models.items():
            r = metrics(yte, m.predict(Xte), thr)
            rows.append(dict(target=tgt, model=name, **r))
            print('   %-26s RMSE %7.3f | top decile: RMSE %7.3f  bias %+7.3f (%+.1f%%)  exceedance recall '
                  '%.3f | bottom-decile bias %+.3f'
                  % (name, r['rmse'], r['rmse_top'], r['bias_top'], 100 * r['rel_bias_top'], r['recall_top'],
                     r['bias_low']))
        base = [x for x in rows if x['target'] == tgt and x['model'] == 'boosting'][0]
        print('   same boosting model on a RANDOM split: RMSE %.3f  -> time-split penalty x%.2f'
              % (rmse_random, base['rmse'] / rmse_random))
    r = pd.DataFrame(rows)
    r.to_csv(os.path.join(OUT, 'uci_headroom.csv'), index=False)
    piv = r.pivot(index='target', columns='model', values='rel_bias_top')
    print('\nrelative bias in the top decile (negative = peaks underestimated):')
    print((100 * piv).round(1).to_string())
    print('\nexceedance recall:')
    print(r.pivot(index='target', columns='model', values='recall_top').round(3).to_string())
    print('\noverall RMSE:')
    print(r.pivot(index='target', columns='model', values='rmse').round(3).to_string())


if __name__ == '__main__':
    main()
