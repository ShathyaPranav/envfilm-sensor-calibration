"""Summarise the SARN results into the tables a paper would carry.

Three questions, in the order a reviewer would ask them:

  1. Does training through the statistic beat the identical network trained on point
     labels? This is the decisive comparison; nothing else matters if it fails.
  2. Does either beat doing nothing, and does either make things WORSE than doing
     nothing? A data-quality tool that degrades the statistic it reports is worse than
     useless, and on this data the error is largely irreducible drift, so "act only when
     acting helps" is the behaviour worth measuring.
  3. How does each compare with the published detectors on the series where those exist?

Errors are in units of each series' robust scale, on test years 2018-2020, after the
physical-range prescreen that every arm receives.
"""
import os
import sys

import numpy as np
import pandas as pd

_EXP = os.path.dirname(os.path.abspath(__file__))
if _EXP not in sys.path:
    sys.path.insert(0, _EXP)
from lro_killtest import OUT

ARMS = ['raw', 'rules', 'oracle_labels', 'pointnet', 'pointnet_soft', 'sarn', 'sarn_hard']
NICE = {'raw': 'no action (prescreened)', 'rules': 'rule screen',
        'oracle_labels': 'oracle: technician labels', 'pointnet': 'point-label net (hard mask)',
        'pointnet_soft': 'point-label net (soft weights)', 'sarn': 'SARN (soft weights)',
        'sarn_hard': 'SARN (hard threshold)'}


def main():
    f = os.path.join(OUT, 'sarn_results.csv')
    r = pd.read_csv(f)
    arms = [a for a in ARMS if a in r.columns]
    pub = sorted(c for c in r.columns if c.startswith('pyhydroqc_'))
    per = r.groupby(['var', 'site'])[arms + pub].mean().reset_index()

    print('=== error in the reported statistic, test years, mean over seeds ===')
    print(per.to_string(index=False, float_format='%.4f'))

    print('\n=== mean over series, by variable ===')
    print(r.groupby('var')[arms].mean().to_string(float_format='%.4f'))

    print('\n=== 1. SARN versus the point-label control, per series ===')
    if 'pointnet' in per.columns:
        w = per.copy()
        w['better'] = w.sarn < w.pointnet
        w['rel'] = (w.sarn - w.pointnet) / w.pointnet
        print(w[['var', 'site', 'pointnet', 'sarn', 'rel', 'better']]
              .to_string(index=False, float_format='%.4f'))
        print('SARN lower on %d of %d series; median relative change %+.3f'
              % (int(w.better.sum()), len(w), float(w.rel.median())))
        if 'diff_lo' in r.columns:
            sig = r[(r.diff_hi < 0)]
            bad = r[(r.diff_lo > 0)]
            print('paired bootstrap over test windows: interval entirely below zero (SARN '
                  'better) in %d of %d series-seeds; entirely above (worse) in %d'
                  % (len(sig), len(r), len(bad)))

    print('\n=== 2. does the method ever make things worse than doing nothing ===')
    tol = 1.02      # counted as worse only if it degrades by more than 2%
    for a in arms:
        if a in ('raw', 'oracle_labels'):
            continue
        worse = int((per[a] > per['raw'] * tol).sum())
        better = int((per[a] < per['raw'] / tol).sum())
        print('  %-32s better on %2d series, WORSE on %2d, of %d   (mean ratio to no action %.3f)'
              % (NICE.get(a, a), better, worse, len(per), float((per[a] / per['raw']).mean())))

    if pub:
        print('\n=== 3. versus the published detectors, identical windows ===')
        sub = per[per[pub].notna().any(axis=1)]
        cols = ['var', 'site', 'raw', 'rules'] + pub + ['sarn', 'sarn_hard']
        print(sub[[c for c in cols if c in sub.columns]]
              .to_string(index=False, float_format='%.4f'))

    out = os.path.join(OUT, 'sarn_summary.csv')
    per.to_csv(out, index=False)
    print('\nwrote %s' % out)


if __name__ == '__main__':
    main()
