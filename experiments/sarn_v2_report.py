"""Paper-ready numbers from sarn_eval.py output (full-coverage scoring).

Usage: python sarn_v2_report.py [retention|corrected]

Comparisons are made at the level of series (mean over seeds), because test windows
overlap at stride one and are far from independent. The sign test treats series as the
unit; series of one variable share a model, so even that is optimistic and is reported as
descriptive rather than confirmatory.
"""
import os
import sys
from math import comb

import numpy as np
import pandas as pd

_EXP = os.path.dirname(os.path.abspath(__file__))
if _EXP not in sys.path:
    sys.path.insert(0, _EXP)
from lro_killtest import OUT

ARMS = ['no_action', 'rules', 'pointnet', 'sarn_hard', 'sarn_soft', 'oracle']


def sign_p(k, n):
    """One-sided binomial sign test, P(X >= k) under p = 0.5."""
    return sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else 'retention'
    r = pd.read_csv(os.path.join(OUT, 'sarn_v2_%s.csv' % target))
    print('target %s | rows %d | series %d | seeds %s | test windows (one seed) %d'
          % (target, len(r), r.groupby(['var', 'site']).ngroups, sorted(set(r.seed)),
             int(r[r.seed == r.seed.min()].n_target.sum())))
    for kind in ('hold', 'common', 'cov'):
        print('\n=== %s: mean over sites and seeds ===' % kind)
        print(r.groupby('var')[['%s_%s' % (kind, a) for a in ARMS]].mean().to_string(float_format='%.4f'))
    print('\nshare of valid points kept: SARN %.4f  label-trained %.4f'
          % (r.kept_sarn.mean(), r.kept_pointnet.mean()))
    print(r.groupby('var')[['kept_sarn', 'kept_pointnet']].mean().to_string(float_format='%.4f'))

    per = r.groupby(['var', 'site']).mean(numeric_only=True).reset_index()
    n = len(per)
    for kind in ('hold', 'common'):
        s, p, na, ru = (per['%s_%s' % (kind, a)] for a in ('sarn_hard', 'pointnet', 'no_action', 'rules'))
        k = int((s < p).sum())
        print('\n--- series level, %s ---' % kind)
        print('SARN < label-trained on %d of %d series (one-sided sign p = %.4f)' % (k, n, sign_p(k, n)))
        print('   by variable: %s' % {v: '%d/%d' % (int((g['%s_sarn_hard' % kind] < g['%s_pointnet' % kind]).sum()), len(g))
                                     for v, g in per.groupby('var')})
        print('   per seed: %s' % {int(sd): int((g['%s_sarn_hard' % kind] < g['%s_pointnet' % kind]).sum())
                                  for sd, g in r.groupby('seed')})
        print('SARN < rules on %d of %d; SARN < no action on %d of %d'
              % (int((s < ru).sum()), n, int((s < na).sum()), n))
        for a in ('rules', 'pointnet', 'sarn_hard'):
            x = per['%s_%s' % (kind, a)]
            print('   %-10s mean ratio to no action %.3f | worse by >2%% on %d | better by >2%% on %d'
                  % (a, float((x / na).mean()), int((x > na * 1.02).sum()), int((x < na * 0.98).sum())))
        print('   median relative change, SARN vs label-trained: %+.3f'
              % float(((s - p) / p).median()))

    pub = sorted(c[len('hold_'):] for c in r.columns if c.startswith('hold_pyhydroqc_'))
    if pub:
        print('\n=== published detections, same windows (mean over seeds) ===')
        sub = per[per[['hold_' + c for c in pub]].notna().any(axis=1)]
        for kind in ('hold', 'common', 'cov'):
            cols = ['%s_%s' % (kind, a) for a in ['no_action', 'rules', 'pointnet', 'sarn_hard', 'oracle'] + pub]
            print('--- %s ---' % kind)
            print(sub.set_index(['var', 'site'])[cols].T.to_string(float_format='%.4f'))

    if 'ap_pointnet' in per.columns:
        print('\n=== can the networks separate technician flags? average precision vs base rate ===')
        a = per[['var', 'site', 'base_rate', 'ap_pointnet', 'ap_sarn']].copy()
        a['lift'] = a.ap_pointnet / a.base_rate
        print(a.to_string(index=False, float_format='%.4f'))
        print('label-trained AP: median %.3f | below 0.25 on %d of %d series | lift below 2x on %d'
              % (a.ap_pointnet.median(), int((a.ap_pointnet < 0.25).sum()), len(a),
                 int((a.lift < 2).sum())))
        print(a.groupby('var')[['base_rate', 'ap_pointnet', 'ap_sarn']].mean().to_string(float_format='%.4f'))

    print('\n=== per series, hold ===')
    print(per[['var', 'site', 'cov_rules', 'cov_pointnet', 'cov_sarn_hard', 'hold_no_action',
               'hold_rules', 'hold_pointnet', 'hold_sarn_hard', 'hold_oracle']]
          .to_string(index=False, float_format='%.4f'))
    per.to_csv(os.path.join(OUT, 'sarn_v2_%s_summary.csv' % target), index=False)


if __name__ == '__main__':
    main()
