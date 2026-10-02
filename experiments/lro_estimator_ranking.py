"""Which estimator wins when the contamination is REAL?

Sections on injected defects rank the same eight estimators (four location, four spread)
under manufactured contamination. Here the corrupted signal is the raw sensor stream and
the clean signal is the technician-corrected stream, so the same paired metric gives the
ranking under real expert-identified defects.

A prediction is stated before running, because it follows from what the technicians do.
On conductance, pH and dissolved oxygen most of their edit is a calibration-drift
correction: a level offset that changes over weeks, so within a 7.5-hour window it moves
every point by nearly the same amount. All four location estimators are translation
equivariant, E(x + c) = E(x) + c, so each inherits the offset in full and their errors
should be nearly IDENTICAL: choosing among them cannot help. All four spread estimators
are translation invariant, so the offset should leave them almost untouched and their
errors should be small relative to the location errors.

Series are prescreened with the published physical range, exactly as in the model
comparison. No points are dropped by any arm here, so every estimator is scored on the
same windows.
"""
import os
import sys

import numpy as np
import pandas as pd

_EXP = os.path.dirname(os.path.abspath(__file__))
if _EXP not in sys.path:
    sys.path.insert(0, _EXP)
from lro_killtest import ALLVARS, OUT, SITES, TRAIN_YEARS, load_site, rolling_all, series_scale
from sarn import physical_screen

LOC = ['mean', 'median', 'trimmed_mean', 'huber_M']
SPR = ['std', 'madn', 'trimmed_std', 'iqr_scale']
W = 30


def main():
    rows = []
    for site in SITES:
        loaded = load_site(site)
        if loaded is None:
            continue
        d, _ = loaded
        yr = d.datetime.dt.year.to_numpy()
        mtr = (yr >= TRAIN_YEARS[0]) & (yr <= TRAIN_YEARS[1])
        for var in ALLVARS:
            raw, _ = physical_screen(d[var].to_numpy(float), site, var)
            cor = d['%s_cor' % var].to_numpy(float)
            scale = series_scale(cor, mtr)
            if not np.isfinite(scale) or scale <= 0:
                continue
            er, ec = rolling_all(raw, W), rolling_all(cor, W)
            ok = np.ones(len(er['mean']), bool)
            for k in LOC + SPR:
                ok &= np.isfinite(er[k]) & np.isfinite(ec[k])
            rec = dict(site=site, var=var, n_windows=int(ok.sum()))
            for k in LOC + SPR:
                e = np.abs(er[k][ok] - ec[k][ok])
                rec['mae_' + k] = float(e.mean() / scale)
                rec['peak_' + k] = float(e.max() / scale)
            rows.append(rec)
            print('  %-14s %-4s n=%7d | location: %s | spread: %s'
                  % (site, var, rec['n_windows'],
                     ' '.join('%s %.4f' % (k[:6], rec['mae_' + k]) for k in LOC),
                     ' '.join('%s %.4f' % (k[:6], rec['mae_' + k]) for k in SPR)), flush=True)
    r = pd.DataFrame(rows)
    r.to_csv(os.path.join(OUT, 'estimator_ranking_real.csv'), index=False)
    print('\n=== mean normalised MAE over sites, by variable ===')
    print(r.groupby('var')[['mae_' + k for k in LOC + SPR]].mean().to_string(float_format='%.4f'))
    r['loc_spread'] = r[['mae_' + k for k in LOC]].max(axis=1) / r[['mae_' + k for k in LOC]].min(axis=1)
    r['median_over_mean'] = r.mae_median / r.mae_mean
    r['madn_over_std'] = r.mae_madn / r.mae_std
    r['best_loc'] = r[['mae_' + k for k in LOC]].idxmin(axis=1).str.replace('mae_', '')
    r['best_spr'] = r[['mae_' + k for k in SPR]].idxmin(axis=1).str.replace('mae_', '')
    print('\nworst/best location error ratio within a series, by variable (1.0 = identical):')
    print(r.groupby('var').loc_spread.agg(['min', 'median', 'max']).to_string(float_format='%.3f'))
    print('\ngeometric-mean ratios, robust/naive (the quantities reported for injected defects):')
    for v, g in r.groupby('var'):
        print('  %-4s median/mean %.3f   MADn/std %.3f'
              % (v, float(np.exp(np.log(g.median_over_mean).mean())),
                 float(np.exp(np.log(g.madn_over_std).mean()))))
    print('\nbest location estimator per series:', r.best_loc.value_counts().to_dict())
    print('best spread estimator per series:', r.best_spr.value_counts().to_dict())
    r.to_csv(os.path.join(OUT, 'estimator_ranking_real.csv'), index=False)
    print('\nwrote %s' % os.path.join(OUT, 'estimator_ranking_real.csv'))


if __name__ == '__main__':
    main()
