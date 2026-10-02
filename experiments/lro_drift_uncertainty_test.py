"""Kill test: does uncertainty about the drift correction grow with time since calibration?

The idea under test is a monitor that reports an honest error bar instead of a corrected
value: tight just after a calibration, wide long after one. It only makes sense if the
size of the correction actually depends on how long ago the sensor was calibrated.

MainStreet is the only site with calibration records (start and end of each drift period
and the offset applied). A deployed monitor would know when it was last calibrated, so
"age" = time since the current drift period began is legitimately available.

PASS MARKS, fixed before running:
  A  across documented periods, |offset| rises with period length:
     Spearman rho >= 0.3 with p < 0.05, pooled over cond, pH and DO.
  B  an age-aware 90% interval, fitted on periods ending by 2016 and scored on periods in
     2018-2020, is at least 15% narrower than a constant-width interval AT THE SAME
     achieved coverage, on at least 2 of the 3 variables.
If A and B both fail, time since calibration carries no usable information about the
correction and the idea is dead on this data.
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

_EXP = os.path.dirname(os.path.abspath(__file__))
if _EXP not in sys.path:
    sys.path.insert(0, _EXP)
from lro_gates2 import calib_periods
from lro_killtest import OUT, load_site

SITE = 'MainStreet'
BINS = [0, 3, 7, 14, 28, 60, 10000]      # days since the period began


def main():
    d, step = load_site(SITE)
    t = d.datetime
    rowsA, resB = [], []
    for var in ('cond', 'ph', 'do'):
        per = calib_periods(SITE, var)
        per['days'] = (per['end'] - per['start']).dt.total_seconds() / 86400.0
        per = per[per.days > 0]
        for _, p in per.iterrows():
            rowsA.append(dict(var=var, days=p['days'], gap=abs(float(p['gap'])), year=p['end'].year))
        raw, cor = d[var].to_numpy(float), d['%s_cor' % var].to_numpy(float)
        delta = np.abs(cor - raw)
        age = np.full(len(t), np.nan)
        pyear = np.zeros(len(t), int)
        tt = t.to_numpy()
        for _, p in per.iterrows():
            a = int(np.searchsorted(tt, np.datetime64(p['start'])))
            b = int(np.searchsorted(tt, np.datetime64(p['end'])))
            age[a:b] = (np.arange(b - a) * step) / 86400.0
            pyear[a:b] = p['end'].year
        ok = np.isfinite(age) & np.isfinite(delta)
        tr = ok & (pyear <= 2016)
        te = ok & (pyear >= 2018)
        if tr.sum() < 500 or te.sum() < 500:
            print('  %s: too few records in documented periods (train %d, test %d)' % (var, tr.sum(), te.sum()))
            continue
        const_w = np.quantile(delta[tr], 0.90)
        b_tr, b_te = np.digitize(age[tr], BINS) - 1, np.digitize(age[te], BINS) - 1
        q = np.array([np.quantile(delta[tr][b_tr == k], 0.90) if (b_tr == k).sum() > 50 else const_w
                      for k in range(len(BINS) - 1)])
        w_age = q[b_te]
        cov_const = float((delta[te] <= const_w).mean())
        cov_age = float((delta[te] <= w_age).mean())
        # compare sharpness at EQUAL achieved coverage: rescale the constant width so it
        # covers exactly as much of the test data as the age-aware interval does
        matched_const = float(np.quantile(delta[te], cov_age))
        saving = 1 - float(w_age.mean()) / matched_const if matched_const > 0 else np.nan
        med_by_bin = [float(np.median(delta[te][b_te == k])) if (b_te == k).sum() else np.nan
                      for k in range(len(BINS) - 1)]
        resB.append(dict(var=var, n_train=int(tr.sum()), n_test=int(te.sum()), cov_const=cov_const,
                         cov_age=cov_age, width_age=float(w_age.mean()), width_const_matched=matched_const,
                         saving=saving))
        print('  %-4s periods %2d | 90%% interval coverage on test: constant %.3f  age-aware %.3f | '
              'mean width age-aware %.4f vs constant at same coverage %.4f  -> %+.1f%% narrower'
              % (var, len(per), cov_const, cov_age, w_age.mean(), matched_const, 100 * saving))
        print('       median |correction| on test by days since calibration %s: %s'
              % (BINS[:-1], ['%.3f' % m for m in med_by_bin]))
    A = pd.DataFrame(rowsA)
    rho, p = stats.spearmanr(A.days, A.gap)
    print('\nA: |offset| vs period length, %d documented periods: Spearman rho %+.3f (p = %.4f)' % (len(A), rho, p))
    for var, g in A.groupby('var'):
        r2, p2 = stats.spearmanr(g.days, g.gap)
        print('     %-4s n=%2d  rho %+.3f (p = %.3f)  median period %.0f days' % (var, len(g), r2, p2, g.days.median()))
    passA = (rho >= 0.3) and (p < 0.05)
    B = pd.DataFrame(resB)
    nB = int((B.saving >= 0.15).sum()) if len(B) else 0
    passB = nB >= 2
    print('   -> A %s' % ('PASS' if passA else 'FAIL'))
    print('B: age-aware interval at least 15%% narrower at equal coverage on %d of %d variables -> %s'
          % (nB, len(B), 'PASS' if passB else 'FAIL'))
    A.to_csv(os.path.join(OUT, 'drift_uncertainty_periods.csv'), index=False)
    B.to_csv(os.path.join(OUT, 'drift_uncertainty_intervals.csv'), index=False)


if __name__ == '__main__':
    main()
