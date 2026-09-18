"""G5b: two follow-ups to the failed drift gate, both pre-specified before running.

G5 asked a model to predict the technician's drift correction from neighbouring sites
with no knowledge of any past correction, and it failed on 17 of 18 series. Two
objections to that verdict deserve a test, and only these two:

PART A, was it the model class? G5 used a Huber linear fit, and one series scored
R^2 = -4273, which is extrapolation exploding rather than absence of signal. Gradient
boosting cannot extrapolate outside the training range, so if signal exists it should
appear. Same features, same splits, same pre-registered bar (R^2 >= 0.25 on >= 2 primary
series). This can only CONFIRM the negative or overturn it; it is not a retry for a pass.

PART B, was the question posed fairly? Asking for drift with no access to any past
correction is harsher than deployment. A real monitoring system receives technician
corrections on a lag: today you know what was corrected up to L days ago, and you must
report a statistic for the current window now. That is exactly the streaming setting the
paper is about. So:

  known      the true drift up to t - LAG days (LAG = 30), plus all raw data up to t
  predict    the drift at t
  baselines  HOLD, carry the last known drift forward; TREND, extend the recent drift
             slope linearly; ZERO, assume no drift
  model      gradient boosting on last-known drift, its recent slope, days since that
             observation, and the neighbour features from G5

  PASS (fixed before running): the model reduces mean absolute drift error by >= 20%
       against the BEST of the three baselines, on at least half of the primary series.
  FAIL: even with lagged ground truth, drift tracking adds nothing over carrying the
       last correction forward, and no drift head is built.

Nothing here uses a corrected value newer than t - LAG, or any neighbour's corrected
value, or any future value.
"""
import os
import sys

import numpy as np
import pandas as pd

_EXP = os.path.dirname(os.path.abspath(__file__))
if _EXP not in sys.path:
    sys.path.insert(0, _EXP)
from lro_killtest import ALLVARS, OUT, PRIMARY, SITES, TEST_YEARS, TRAIN_YEARS, load_site
from lro_gate5_drift import daily_frame

LAG_DAYS = 30
G5B_MIN_R2 = 0.25
G5B_MIN_SERIES = 2
G5B_MIN_GAIN = 0.20


def build(frames, var):
    daily = {s: daily_frame(d, var) for s, d in frames.items()}
    out = {}
    for site in frames:
        g = daily[site].copy()
        g = g[g['n'] >= 24]
        if len(g) < 300:
            continue
        for s2 in frames:
            if s2 != site:
                g['nb_%s' % s2] = g['raw'] - daily[s2]['raw'].reindex(g.index)
        nb = [c for c in g.columns if c.startswith('nb_')]
        g['nb_med'] = g[nb].median(axis=1)
        yr = g.index.year
        tr = (yr >= TRAIN_YEARS[0]) & (yr <= TRAIN_YEARS[1])
        g['nb_anom'] = g['nb_med'] - np.nanmedian(g['nb_med'].to_numpy()[tr])
        g['doy_s'] = np.sin(2 * np.pi * g.index.dayofyear / 365.25)
        g['doy_c'] = np.cos(2 * np.pi * g.index.dayofyear / 365.25)
        # lagged knowledge of the technician's own correction
        g['drift_lag'] = g['delta'].shift(LAG_DAYS)
        g['drift_lag2'] = g['delta'].shift(LAG_DAYS + 7)
        g['drift_slope'] = (g['drift_lag'] - g['drift_lag2']) / 7.0
        g['drift_lag_ffill'] = g['drift_lag'].ffill()
        g['age'] = g['drift_lag'].notna().cumsum()
        g['age'] = g.groupby(g['age'])['age'].cumcount()
        out[site] = g
    return out


def score(y, p):
    return float(np.mean(np.abs(y - p)))


def main():
    from sklearn.ensemble import HistGradientBoostingRegressor
    frames = {}
    for site in SITES:
        loaded = load_site(site)
        if loaded is not None:
            frames[site] = loaded[0]
    rows = []
    for var in ALLVARS:
        built = build(frames, var)
        for site, g in built.items():
            yr = g.index.year
            tr = (yr >= TRAIN_YEARS[0]) & (yr <= TRAIN_YEARS[1])
            te = (yr >= TEST_YEARS[0]) & (yr <= TEST_YEARS[1])
            y = g['delta'].to_numpy(float)
            nb_feats = ['nb_anom', 'nb_med', 'doy_s', 'doy_c'] + [v for v in ALLVARS if v != var]
            lag_feats = nb_feats + ['drift_lag_ffill', 'drift_slope', 'age']
            okA = np.isfinite(g[nb_feats].to_numpy(float)).all(axis=1) & np.isfinite(y)
            okB = np.isfinite(g[lag_feats].to_numpy(float)).all(axis=1) & np.isfinite(y)
            if (tr & okA).sum() < 150 or (te & okA).sum() < 100 or (te & okB).sum() < 100:
                continue
            # PART A: no knowledge of any past correction, non-linear learner
            mA = HistGradientBoostingRegressor(max_iter=300, random_state=0)
            mA.fit(g[nb_feats].to_numpy(float)[tr & okA], y[tr & okA])
            pA = mA.predict(g[nb_feats].to_numpy(float)[te & okA])
            yA = y[te & okA]
            ssA = float(np.sum((yA - np.mean(yA)) ** 2))
            r2A = 1 - float(np.sum((yA - pA) ** 2)) / ssA if ssA > 0 else np.nan
            # PART B: drift known up to t - LAG
            Xb = g[lag_feats].to_numpy(float)
            mB = HistGradientBoostingRegressor(max_iter=300, random_state=0)
            mB.fit(Xb[tr & okB], y[tr & okB])
            pB = mB.predict(Xb[te & okB])
            yB = y[te & okB]
            hold = g['drift_lag_ffill'].to_numpy(float)[te & okB]
            trend = hold + g['drift_slope'].to_numpy(float)[te & okB] * LAG_DAYS
            e_model, e_hold = score(yB, pB), score(yB, hold)
            e_trend, e_zero = score(yB, trend), score(yB, np.zeros_like(yB))
            best_base = min(e_hold, e_trend, e_zero)
            gain = 1 - e_model / best_base if best_base > 0 else np.nan
            rows.append(dict(site=site, var=var, n_test=int((te & okB).sum()),
                             mean_abs_drift=float(np.mean(np.abs(yB))), r2_nolag_gbt=r2A,
                             mae_model=e_model, mae_hold=e_hold, mae_trend=e_trend,
                             mae_zero=e_zero, gain_vs_best_baseline=gain))
            r = rows[-1]
            print('  %-14s %-4s |drift| %7.3f | A: GBT no-lag R2 %+.3f | B: model %.3f  hold %.3f  '
                  'trend %.3f  zero %.3f  -> gain %+.3f'
                  % (site, var, r['mean_abs_drift'], r['r2_nolag_gbt'], e_model, e_hold, e_trend,
                     e_zero, gain), flush=True)
    r = pd.DataFrame(rows)
    if not len(r):
        print('G5b: no series usable. FAIL.')
        return
    r.to_csv(os.path.join(OUT, 'g5b_drift_lagged.csv'), index=False)
    prim = r[r['var'].isin(PRIMARY)]
    nA = int((prim.r2_nolag_gbt >= G5B_MIN_R2).sum())
    okA = nA >= G5B_MIN_SERIES
    share = float((prim.gain_vs_best_baseline >= G5B_MIN_GAIN).mean())
    okB = share >= 0.5
    print('\nPART A (no past correction, gradient boosting): series with R2 >= %.2f: %d of %d '
          '(need >= %d)  ->  %s' % (G5B_MIN_R2, nA, len(prim), G5B_MIN_SERIES,
                                    'OVERTURNS G5' if okA else 'CONFIRMS G5 FAILURE'))
    print('PART B (drift known up to t-%dd): model beats best baseline by >= %.0f%% on %.0f%% of '
          'primary series (need 50%%)  ->  %s'
          % (LAG_DAYS, G5B_MIN_GAIN * 100, share * 100, 'PASS' if okB else 'FAIL'))
    print('   median MAE: model %.4f  hold %.4f  trend %.4f  zero %.4f'
          % (prim.mae_model.median(), prim.mae_hold.median(), prim.mae_trend.median(),
             prim.mae_zero.median()))
    if not okA and not okB:
        print('   CONCLUSION: the drift channel is not learnable here, with or without lagged '
              'ground truth. Report it; build no drift head.')


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    main()
