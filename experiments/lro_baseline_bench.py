"""The score to beat: pyhydroqc's published detections on Logan River, scored two ways.

pyhydroqc (Jones, Jones and Horsburgh 2022, Environmental Modelling and Software) is the
published method for this dataset, and the HydroShare resource ships its per-point
detections. This script scores them, so the target is fixed before any new model exists.

TWO SCORINGS, because they answer different questions:

  POINT DETECTION  precision, recall, F1 against the technician's qualifier labels,
                   which is pyhydroqc's own target and its own definition of an anomaly
                   (anomaly_utilities.get_data: labeled_anomaly = qualifier not null).

  WINDOW STATISTIC the error in the rolling statistic that results from ACTING on those
                   detections (drop detected points, recompute), measured against the
                   statistic of the technician-corrected series. This is the paper's
                   metric, and it is what a data-quality monitor actually reports.
                   A detector can score well on the first and poorly on the second.

FAIRNESS NOTE, stated rather than buried: these detections come from models whose
parameters were tuned per site and per sensor over the whole record, so the published
output has seen the test years. Numbers are reported on the full record and on the test
years separately. Any new model here is trained on training years only, so the
comparison is tilted in pyhydroqc's favour, not ours.
"""
import os
import sys

import numpy as np
import pandas as pd

_EXP = os.path.dirname(os.path.abspath(__file__))
if _EXP not in sys.path:
    sys.path.insert(0, _EXP)
from lro_killtest import (DATA, OUT, TEST_YEARS, TRAIN_YEARS, WINDOWS, apply_rules, fit_rules,
                          load_site, paired_error, rolling_all, series_scale)

RESULTS = os.path.join(DATA, 'pyhydroqc_results')
EST = 'trimmed_mean'


def detection_files():
    out = {}
    if not os.path.isdir(RESULTS):
        return out
    for f in sorted(os.listdir(RESULTS)):
        if 'detections' not in f or not f.endswith('.csv'):
            continue
        stem = f[:-4].replace('_detections', '')
        parts = stem.split('_')
        model, site, var = '_'.join(parts[:-2]), parts[-2], parts[-1]
        out.setdefault((site, var), []).append((model, os.path.join(RESULTS, f)))
    return out


def prf(y, p):
    tp = int((y & p).sum())
    fp = int((~y & p).sum())
    fn = int((y & ~p).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1, tp, fp, fn


def main():
    det = detection_files()
    if not det:
        print('no pyhydroqc detection files in %s' % RESULTS)
        return
    rows = []
    for (site, var), models in sorted(det.items()):
        loaded = load_site(site)
        if loaded is None:
            continue
        d, step = loaded
        idx = d.datetime
        raw = d[var].to_numpy(float)
        cor = d['%s_cor' % var].to_numpy(float)
        qual = d['%s_qual' % var].astype(str).str.strip()
        y = (~qual.isin(['NULL', 'nan', 'None', '', '<NA>'])).to_numpy()
        yr = idx.dt.year.to_numpy()
        mtr = (yr >= TRAIN_YEARS[0]) & (yr <= TRAIN_YEARS[1])
        mte = (yr >= TEST_YEARS[0]) & (yr <= TEST_YEARS[1])
        scale = series_scale(cor, mtr)
        w = WINDOWS[0]
        ec = rolling_all(cor, w)[EST]
        te_win = mte[:len(ec)]

        def window_mae(series, mask=None):
            e = rolling_all(series, w)[EST]
            m = np.isfinite(e) & np.isfinite(ec)
            if mask is not None:
                m &= mask
            return float(np.abs(e[m] - ec[m]).mean() / scale) if m.any() else np.nan

        rules = fit_rules(raw, mtr)
        base_all, base_te = window_mae(raw), window_mae(raw, te_win)
        scr = apply_rules(raw, rules)
        rows.append(dict(site=site, var=var, model='raw (no action)', n_flagged=0,
                         precision=np.nan, recall=np.nan, f1=np.nan, f1_test=np.nan,
                         window_mae=base_all, window_mae_test=base_te))
        p_rules = ~np.isfinite(scr) & np.isfinite(raw)
        pr, rc, f1, *_ = prf(y, p_rules)
        _, _, f1t, *_ = prf(y[mte], p_rules[mte])
        rows.append(dict(site=site, var=var, model='rules (range+rate+flatline)',
                         n_flagged=int(p_rules.sum()), precision=pr, recall=rc, f1=f1, f1_test=f1t,
                         window_mae=window_mae(scr), window_mae_test=window_mae(scr, te_win)))
        for model, path in models:
            dd = pd.read_csv(path, parse_dates=['datetime'])
            s = dd.set_index('datetime')['anomaly'].reindex(idx).fillna(False)
            p = s.to_numpy(bool)
            masked = raw.copy()
            masked[p] = np.nan
            pr, rc, f1, tp, fp, fn = prf(y, p)
            _, _, f1t, *_ = prf(y[mte], p[mte])
            rows.append(dict(site=site, var=var, model=model, n_flagged=int(p.sum()),
                             precision=pr, recall=rc, f1=f1, f1_test=f1t,
                             window_mae=window_mae(masked),
                             window_mae_test=window_mae(masked, te_win)))
        oracle = raw.copy()
        oracle[y] = np.nan
        rows.append(dict(site=site, var=var, model='ORACLE mask technician labels',
                         n_flagged=int(y.sum()), precision=1.0, recall=1.0, f1=1.0, f1_test=1.0,
                         window_mae=window_mae(oracle), window_mae_test=window_mae(oracle, te_win)))
        sub = pd.DataFrame([r for r in rows if r['site'] == site and r['var'] == var])
        print('\n%s %s   label rate %.4f   robust scale %.3f' % (site, var, y.mean(), scale))
        print(sub[['model', 'n_flagged', 'precision', 'recall', 'f1', 'f1_test',
                   'window_mae', 'window_mae_test']].to_string(index=False, float_format='%.4f'))
    r = pd.DataFrame(rows)
    r.to_csv(os.path.join(OUT, 'baseline_bench.csv'), index=False)
    print('\nwrote %s' % os.path.join(OUT, 'baseline_bench.csv'))
    best = (r[r.f1.notna() & ~r.model.str.startswith('ORACLE')]
            .sort_values('f1_test', ascending=False).groupby(['site', 'var']).head(1))
    print('\nBEST PUBLISHED DETECTOR PER SERIES (test years), the score to beat:')
    print(best[['site', 'var', 'model', 'f1', 'f1_test', 'window_mae_test']]
          .to_string(index=False, float_format='%.4f'))


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    main()
