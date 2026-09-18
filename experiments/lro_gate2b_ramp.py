"""G2b: test the technicians' ACTUAL correction procedure, not my approximation of it.

G2 fitted one constant offset per calibration period and explained 45% of the error at
MainStreet, the only site with documented calibration dates, and 7-16% elsewhere where
period boundaries had to be guessed from the raw series. But the technicians' procedure
is published, and it is not a constant: pyhydroqc.calibration.lin_drift_cor applies a
LINEAR RAMP of the measured gap across each drift period. So G2 tested the wrong
functional form. This tests the right one.

Three reconstructions of the corrected series, each recomputed through the paper's
estimator and compared against the technician-corrected statistic:

  deletions            only the technician's removals, values otherwise raw
  deletions + ramp     removals, plus the documented gap applied as a linear ramp from
                       the start to the end of each documented drift period, exactly as
                       lin_drift_cor does
  deletions + oracle   removals, plus the technicians' realised per-point delta. This is
                       the ceiling: it must explain ~100%, and if it does not, something
                       outside (delete, drift) is happening and the whole factorisation
                       is incomplete.

Only MainStreet has calibration records (cond, ph, do), so that is the only site where
this can be run, and the result is reported as evidence about the procedure, not as a
site-general claim.

WHY IT MATTERS: if the ramp explains most of the error, then the whole difference between
raw and validated data on this dataset is (removals) + (a ramp whose size is a number a
person measures in the field). G5 and G5b already showed that number cannot be predicted
in advance. That bounds what ANY causal model can achieve here, which is the thing worth
knowing before building one.
"""
import os
import sys

import numpy as np
import pandas as pd

_EXP = os.path.dirname(os.path.abspath(__file__))
if _EXP not in sys.path:
    sys.path.insert(0, _EXP)
from lro_killtest import OUT, TRAIN_YEARS, WINDOWS, load_site, paired_error, rolling_all, series_scale
from lro_gates2 import calib_periods

SITE = 'MainStreet'
VARS = ['cond', 'ph', 'do']
EST = ['trimmed_mean', 'median']


def main():
    d, step = load_site(SITE)
    idx = d.datetime.to_numpy()
    yr = d.datetime.dt.year.to_numpy()
    mtr = (yr >= TRAIN_YEARS[0]) & (yr <= TRAIN_YEARS[1])
    rows = []
    for var in VARS:
        raw = d[var].to_numpy(float)
        cor = d['%s_cor' % var].to_numpy(float)
        per = calib_periods(SITE, var)
        if per is None or not len(per):
            print('  %s: no calibration records' % var)
            continue
        scale = series_scale(cor, mtr)
        deleted = np.isfinite(raw) & ~np.isfinite(cor)

        masked = raw.copy()
        masked[deleted] = np.nan

        ramp = raw.copy()
        applied = 0
        for _, p in per.iterrows():
            a = int(np.searchsorted(idx, np.datetime64(p['start'])))
            b = int(np.searchsorted(idx, np.datetime64(p['end'])))
            if b <= a or b > len(raw):
                continue
            frac = np.linspace(0.0, 1.0, b - a)
            ramp[a:b] = raw[a:b] + float(p['gap']) * frac
            applied += 1
        ramp[deleted] = np.nan

        oracle = np.where(np.isfinite(cor), cor, np.nan)

        for est in EST:
            for w in WINDOWS:
                ec = rolling_all(cor, w)[est]
                base = paired_error(rolling_all(raw, w)[est], ec)[0]
                if not np.isfinite(base) or base <= 0:
                    continue
                e_mask = paired_error(rolling_all(masked, w)[est], ec)[0]
                e_ramp = paired_error(rolling_all(ramp, w)[est], ec)[0]
                e_orac = paired_error(rolling_all(oracle, w)[est], ec)[0]
                rows.append(dict(site=SITE, var=var, est=est, w=w, periods=applied,
                                 identity_mae=base / scale,
                                 expl_deletions=1 - e_mask / base,
                                 expl_del_ramp=1 - e_ramp / base,
                                 expl_del_oracle=1 - e_orac / base))
                r = rows[-1]
                print('  %-4s %-12s W%-3d periods %3d  identity %.4f  explained: deletions %+.3f  '
                      'deletions+ramp %+.3f  deletions+oracle delta %+.3f'
                      % (var, est, w, applied, r['identity_mae'], r['expl_deletions'],
                         r['expl_del_ramp'], r['expl_del_oracle']), flush=True)
    r = pd.DataFrame(rows)
    r.to_csv(os.path.join(OUT, 'g2b_ramp.csv'), index=False)
    m = r[r.w == WINDOWS[0]]
    print('\nMainStreet, W=%d, mean over %d series-estimator combinations:' % (WINDOWS[0], len(m)))
    print('   deletions alone            %+.3f' % m.expl_deletions.mean())
    print('   deletions + documented ramp %+.3f' % m.expl_del_ramp.mean())
    print('   deletions + realised delta  %+.3f  (the ceiling; near 1.0 means the raw-to-'
          'validated difference really is removals plus a drift correction)'
          % m.expl_del_oracle.mean())
    print('\nwrote %s' % os.path.join(OUT, 'g2b_ramp.csv'))


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    main()
