"""Audit: are drop-based arms being rewarded for abstaining?

Every arm that removes points is scored on the windows where it still yields an estimate.
A window whose points are ALL removed produces no estimate and silently leaves that arm's
average. If the technician-corrected series does have a value there, the arm has failed to
report something that exists, and excluding the window flatters it. Arms would then be
compared on different window sets, which is not a comparison.

For the arms that need no trained model (no action, the rule screen, the label oracle and
the published detections), on the same prescreened series and test years used for the
model comparison, this reports three things per series:

  coverage   share of target-valid windows where the arm produces an estimate
  own        mean error over the arm's own covered windows (what was reported before)
  common     mean error over windows covered by EVERY arm (a like-for-like comparison)
  hold       mean error over all target-valid windows, where a window with no estimate
             carries the arm's last available estimate forward. This is what a deployed
             monitor would actually display, so it is the honest full-coverage score.
"""
import os
import sys

import numpy as np
import pandas as pd

_EXP = os.path.dirname(os.path.abspath(__file__))
if _EXP not in sys.path:
    sys.path.insert(0, _EXP)
from lro_killtest import ALLVARS, OUT, SITES, TEST_YEARS, TRAIN_YEARS, apply_rules, fit_rules, load_site, series_scale
from sarn import build_series, trimmed_mean_np, window_index, windows_for
from sarn_train import published_detections


def hold_forward(pred):
    """Carry the last finite estimate forward (causal); leading gaps stay missing."""
    s = pd.Series(pred).ffill()
    return s.to_numpy()


def main():
    frames = {}
    for s in SITES:
        loaded = load_site(s)
        if loaded is not None:
            frames[s] = loaded[0]
    rows = []
    for var in ALLVARS:
        for site in frames:
            S = build_series(frames, site, var)
            mtr = (S['year'] >= TRAIN_YEARS[0]) & (S['year'] <= TRAIN_YEARS[1])
            scale = series_scale(S['cor'], mtr)
            if not np.isfinite(scale) or scale <= 0:
                continue
            ends = windows_for(S, TEST_YEARS, 1)
            if len(ends) < 100:
                continue
            idx = window_index(ends)
            raw = S['raw'][idx]
            tgt = trimmed_mean_np(S['cor'][idx])
            tv = np.isfinite(tgt)
            scr = apply_rules(S['raw'], fit_rules(S['raw'], mtr))
            masks = {'no_action': np.zeros(len(S['raw']), bool),
                     'rules': ~np.isfinite(scr) & np.isfinite(S['raw']),
                     'oracle': S['point']}
            for name, m in published_detections(site, var, S['datetime']).items():
                masks[name] = m
            preds = {}
            for name, m in masks.items():
                preds[name] = trimmed_mean_np(np.where(m[idx], np.nan, raw))
            core = ['no_action', 'rules', 'oracle']
            common = tv.copy()
            for name in core:
                common &= np.isfinite(preds[name])
            rec = dict(var=var, site=site, n_target=int(tv.sum()), n_common=int(common.sum()))
            for name, p in preds.items():
                own = tv & np.isfinite(p)
                h = hold_forward(p)
                full = tv & np.isfinite(h)
                rec['cov_' + name] = float(own.sum() / max(tv.sum(), 1))
                rec['own_' + name] = float(np.abs(p[own] - tgt[own]).mean() / scale) if own.any() else np.nan
                cm = common & np.isfinite(p)
                rec['common_' + name] = float(np.abs(p[cm] - tgt[cm]).mean() / scale) if cm.any() else np.nan
                rec['hold_' + name] = float(np.abs(h[full] - tgt[full]).mean() / scale) if full.any() else np.nan
            rows.append(rec)
            print('  %-4s %-14s target windows %7d | coverage: none %.4f rules %.4f oracle %.4f | '
                  'own: %.4f %.4f %.4f | common: %.4f %.4f %.4f | hold: %.4f %.4f %.4f'
                  % (var, site, rec['n_target'], rec['cov_no_action'], rec['cov_rules'],
                     rec['cov_oracle'], rec['own_no_action'], rec['own_rules'], rec['own_oracle'],
                     rec['common_no_action'], rec['common_rules'], rec['common_oracle'],
                     rec['hold_no_action'], rec['hold_rules'], rec['hold_oracle']), flush=True)
    r = pd.DataFrame(rows)
    r.to_csv(os.path.join(OUT, 'coverage_audit.csv'), index=False)
    print('\n=== mean over series, by variable ===')
    cols = [c for c in r.columns if c.split('_', 1)[0] in ('cov', 'own', 'common', 'hold')
            and c.split('_', 1)[1] in ('no_action', 'rules', 'oracle')]
    print(r.groupby('var')[cols].mean().T.to_string(float_format='%.4f'))
    print('\nwrote %s' % os.path.join(OUT, 'coverage_audit.csv'))


if __name__ == '__main__':
    main()
