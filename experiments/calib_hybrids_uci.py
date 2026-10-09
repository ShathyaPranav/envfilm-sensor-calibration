"""Do obvious hybrids close the time-split gap on UCI Air Quality?

calib_headroom_uci.py found that a boosted model's NO2 error on a split by time is 3.3x
its random-split error and worse than a plain linear fit. Before proposing any model, the
cheap candidates get their chance, because twice in this project a simple method has beaten
the learned one. If one of these closes most of the gap, there is nothing to propose.

Candidates (all fitted on the first half of the year, scored on the second):
  linear                     the standard baseline
  boosting                   the standard ML approach
  linear + boosted residual  linear extrapolates, trees fix what is left
  ridge on quadratic terms   mild nonlinearity that can still extrapolate
  MLP                        a small neural network
  log-target linear          multiplicative sensor response
Reference lines, not deployable as stated:
  random split               the same boosting model when test data look like training data
  monthly refit, 30-day lag  linear model refitted each month on all labels older than 30
                             days: what periodic recalibration with reference data buys
"""
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(_ROOT, 'data', 'uci_airquality', 'AirQualityUCI.csv')
OUT = os.path.join(_ROOT, 'results', 'calib')
SENS = ['PT08.S1(CO)', 'PT08.S2(NMHC)', 'PT08.S3(NOx)', 'PT08.S4(NO2)', 'PT08.S5(O3)', 'T', 'RH', 'AH']
TARGETS = ['NO2(GT)', 'NOx(GT)', 'CO(GT)', 'C6H6(GT)']


def rmse(y, p):
    return float(np.sqrt(np.mean((p - y) ** 2)))


def main():
    os.makedirs(OUT, exist_ok=True)
    d = pd.read_csv(CSV, sep=';', decimal=',').dropna(how='all', axis=1).dropna(how='all').replace(-200, np.nan)
    d['when'] = pd.to_datetime(d['Date'] + ' ' + d['Time'], format='%d/%m/%Y %H.%M.%S')
    rows = []
    for tgt in TARGETS:
        ok = d[SENS + [tgt]].notna().all(axis=1).to_numpy()
        X, y, when = d.loc[ok, SENS].to_numpy(float), d.loc[ok, tgt].to_numpy(float), d.loc[ok, 'when'].to_numpy()
        n = len(y)
        cut = n // 2
        Xtr, ytr, Xte, yte = X[:cut], y[:cut], X[cut:], y[cut:]
        thr = float(np.quantile(ytr, 0.9))
        preds = {}
        lin = LinearRegression().fit(Xtr, ytr)
        preds['linear'] = lin.predict(Xte)
        preds['boosting'] = HistGradientBoostingRegressor(max_iter=400, random_state=0).fit(Xtr, ytr).predict(Xte)
        res = HistGradientBoostingRegressor(max_iter=300, random_state=0).fit(Xtr, ytr - lin.predict(Xtr))
        preds['linear + boosted residual'] = lin.predict(Xte) + res.predict(Xte)
        preds['ridge, quadratic'] = make_pipeline(StandardScaler(), PolynomialFeatures(2), RidgeCV(np.logspace(-2, 3, 12))
                                                 ).fit(Xtr, ytr).predict(Xte)
        mlps = [make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=600, early_stopping=True,
                                                             random_state=s)).fit(Xtr, ytr).predict(Xte)
                for s in range(3)]
        preds['MLP (3 seeds)'] = np.mean(mlps, axis=0)
        if (ytr > 0).all():
            preds['log-target linear'] = np.exp(LinearRegression().fit(np.log(np.maximum(Xtr, 1e-6)), np.log(ytr))
                                                .predict(np.log(np.maximum(Xte, 1e-6))))
        # reference: monthly recalibration using labels at least 30 days old
        p = np.full(len(yte), np.nan)
        t_te = when[cut:]
        months = pd.Series(t_te).dt.to_period('M')
        for mth in months.unique():
            sel = (months == mth).to_numpy()
            avail = when < (np.datetime64(mth.start_time) - np.timedelta64(30, 'D'))
            if avail.sum() < 500:
                avail = np.arange(n) < cut
            p[sel] = LinearRegression().fit(X[avail], y[avail]).predict(Xte[sel])
        preds['[ref] monthly refit, 30-day lag'] = p
        rng = np.random.default_rng(0)
        perm = rng.permutation(n)
        a, b = perm[:cut], perm[cut:]
        rand = rmse(y[b], HistGradientBoostingRegressor(max_iter=400, random_state=0).fit(X[a], y[a]).predict(X[b]))
        print('\n%s   (random-split boosting RMSE %.2f)' % (tgt, rand))
        for name, pr in preds.items():
            top = yte >= thr
            r = dict(target=tgt, model=name, rmse=rmse(yte, pr), bias_top=float(np.mean(pr[top] - yte[top]) / np.mean(yte[top])),
                     recall=float((pr[top] >= thr).mean()), random_split=rand)
            rows.append(r)
            print('   %-34s RMSE %8.3f   peak bias %+6.1f%%   exceedance recall %.3f'
                  % (name, r['rmse'], 100 * r['bias_top'], r['recall']))
    r = pd.DataFrame(rows)
    r.to_csv(os.path.join(OUT, 'uci_hybrids.csv'), index=False)
    print('\nRMSE by model:')
    print(r.pivot(index='model', columns='target', values='rmse').round(2).to_string())
    gap = r[~r.model.str.startswith('[ref]')].groupby('target').apply(
        lambda g: pd.Series(dict(best=g.loc[g.rmse.idxmin(), 'model'], best_rmse=g.rmse.min(),
                                 random_split=g.random_split.iloc[0],
                                 remaining_gap=g.rmse.min() / g.random_split.iloc[0])))
    print('\nbest deployable simple model per target, and how far it still is from the random-split figure:')
    print(gap.round(3).to_string())


if __name__ == '__main__':
    main()
