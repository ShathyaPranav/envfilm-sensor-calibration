"""G5: is the calibration DRIFT predictable from causal, raw-only evidence?

G0 measured that the technicians' edit on this data is overwhelmingly slow drift
correction (50-68% of all conductivity points modified, of which under 3% are point
substitutions) rather than spike removal. A drift is a level offset applied to a whole
period, and the paper's own result says a level offset is invisible to every
within-window estimator: it moves every point in the window equally. So the headroom
here can only be reached from OUTSIDE the window, and the natural outside evidence is
the other monitoring sites on the same river.

THIS GATE IS THE CRUX AND ITS THRESHOLDS ARE FIXED BEFORE IT IS RUN:

  target   the daily median of (corrected - raw), the drift the technician removed
  features causal and raw-only: this site's daily median minus the neighbouring sites'
           daily medians of the same variable, the same for the co-located variables,
           day of year, and days since the last long data gap. Neighbour CORRECTED
           values are never used, and neither is any future value.
  fit      robust linear model on TRAIN years only, scored on TEST years only
  compare  against doing nothing (predict zero) and against persistence (predict the
           previous period's mean drift), both scored the same way

  PASS  out-of-sample R^2 >= 0.25 on at least 2 primary series, OR median reduction in
        absolute drift >= 25% versus doing nothing, on the primary variables.
  FAIL  the drift channel is not learnable from neighbours. Since G0 says drift is most
        of the error, that would mean the achievable ceiling on this data is small, and
        the honest finding is that real correction here is dominated by information that
        does not exist in the sensor network at all (a technician's field calibration).

A pass does NOT establish a model. It establishes that signal exists for one to learn.
"""
import os
import sys

import numpy as np
import pandas as pd

_EXP = os.path.dirname(os.path.abspath(__file__))
if _EXP not in sys.path:
    sys.path.insert(0, _EXP)
from lro_killtest import ALLVARS, OUT, PRIMARY, SITES, TEST_YEARS, TRAIN_YEARS, load_site

G5_MIN_R2 = 0.25
G5_MIN_SERIES = 2
G5_MIN_REDUCTION = 0.25


def daily_frame(d, var):
    """Daily medians of raw, of the technician delta, and of the co-located variables."""
    raw = d[var].to_numpy(float)
    cor = d['%s_cor' % var].to_numpy(float)
    both = np.isfinite(raw) & np.isfinite(cor)
    t = pd.DataFrame({'day': d.datetime.dt.floor('D'),
                      'raw': raw,
                      'delta': np.where(both, cor - raw, np.nan)})
    for v2 in ALLVARS:
        if v2 != var:
            t[v2] = d[v2].to_numpy(float)
    g = t.groupby('day').median(numeric_only=True)
    g['n'] = t.groupby('day')['raw'].count()
    return g


def main():
    frames = {}
    for site in SITES:
        loaded = load_site(site)
        if loaded is not None:
            frames[site] = loaded[0]
    rows = []
    for var in ALLVARS:
        daily = {s: daily_frame(d, var) for s, d in frames.items()}
        for site in frames:
            g = daily[site].copy()
            g = g[g['n'] >= 24]
            if len(g) < 200:
                continue
            # causal neighbour features: this site's daily level minus each other site's
            for s2 in frames:
                if s2 == site:
                    continue
                other = daily[s2]['raw'].reindex(g.index)
                g['nb_%s' % s2] = g['raw'] - other
            nb_cols = [c for c in g.columns if c.startswith('nb_')]
            if not nb_cols:
                continue
            g['nb_med'] = g[nb_cols].median(axis=1)
            # how far the site has moved against its neighbours since a stable reference
            yr = g.index.year
            tr = (yr >= TRAIN_YEARS[0]) & (yr <= TRAIN_YEARS[1])
            te = (yr >= TEST_YEARS[0]) & (yr <= TEST_YEARS[1])
            if tr.sum() < 100 or te.sum() < 100:
                continue
            ref = np.nanmedian(g['nb_med'].to_numpy()[tr])
            g['nb_anom'] = g['nb_med'] - ref
            g['doy'] = np.sin(2 * np.pi * g.index.dayofyear / 365.25)
            g['doy2'] = np.cos(2 * np.pi * g.index.dayofyear / 365.25)
            feats = ['nb_anom', 'nb_med', 'doy', 'doy2'] + [v for v in ALLVARS if v != var]
            X = g[feats].to_numpy(float)
            y = g['delta'].to_numpy(float)
            ok = np.isfinite(X).all(axis=1) & np.isfinite(y)
            itr, ite = tr & ok, te & ok
            if itr.sum() < 100 or ite.sum() < 100 or np.nanstd(y[ite]) < 1e-9:
                continue
            from sklearn.linear_model import HuberRegressor
            from sklearn.preprocessing import StandardScaler
            sc = StandardScaler().fit(X[itr])
            m = HuberRegressor(max_iter=500).fit(sc.transform(X[itr]), y[itr])
            pred = m.predict(sc.transform(X[ite]))
            yt = y[ite]
            ss_res = float(np.sum((yt - pred) ** 2))
            ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
            # against doing nothing, and against carrying the training mean forward
            r2_zero = 1 - float(np.sum(yt ** 2)) / ss_tot if ss_tot > 0 else np.nan
            persist = float(np.mean(y[itr]))
            red_model = 1 - float(np.mean(np.abs(yt - pred))) / float(np.mean(np.abs(yt)))
            red_persist = 1 - float(np.mean(np.abs(yt - persist))) / float(np.mean(np.abs(yt)))
            rows.append(dict(site=site, var=var, n_train=int(itr.sum()), n_test=int(ite.sum()),
                             mean_abs_drift=float(np.mean(np.abs(yt))), r2=r2, r2_of_zero=r2_zero,
                             reduction_model=red_model, reduction_persistence=red_persist))
            r = rows[-1]
            print('  %-14s %-4s  mean |drift| %8.3f   R2 %+.3f (predicting zero scores %+.3f)   '
                  'reduction: model %+.3f  persistence %+.3f'
                  % (site, var, r['mean_abs_drift'], r['r2'], r['r2_of_zero'],
                     r['reduction_model'], r['reduction_persistence']), flush=True)
    r = pd.DataFrame(rows)
    if not len(r):
        print('G5: no series had enough paired days. FAIL by default.')
        return
    r.to_csv(os.path.join(OUT, 'g5_drift.csv'), index=False)
    prim = r[r['var'].isin(PRIMARY)]
    n_r2 = int((prim.r2 >= G5_MIN_R2).sum())
    med_red = prim.reduction_model.median()
    ok = (n_r2 >= G5_MIN_SERIES) or (med_red >= G5_MIN_REDUCTION)
    print('\nG5 (primary variables, %d series): series with out-of-sample R2 >= %.2f: %d '
          '(need >= %d); median reduction in absolute drift %.3f (need >= %.2f)  ->  %s'
          % (len(prim), G5_MIN_R2, n_r2, G5_MIN_SERIES, med_red, G5_MIN_REDUCTION,
             'PASS' if ok else 'FAIL'))
    print('   persistence baseline median reduction %.3f' % prim.reduction_persistence.median())
    if not ok:
        print('   FAIL ACTION: the drift the technicians remove is not recoverable from the '
              'sensor network. Report that as the finding; do not build a drift head.')


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    main()
