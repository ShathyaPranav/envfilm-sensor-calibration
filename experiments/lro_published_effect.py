"""What does acting on the published detections do to the reported statistic?

For the three series whose model-stage detections ship with the dataset, restricted to
the part of the test years those detections actually cover (they end in late 2019, so
scoring through 2020 would dilute any effect with windows where nothing was flagged).

Each detector is applied two ways: removing exactly the flagged points, and removing the
events pyhydroqc forms from them with its own widening (anomaly_events, their widen
parameter), which is closer to what its correction stage acts on. Event precision, recall
and F1 come from pyhydroqc's own compare_events and metrics on the same span.

Every arm sees the physical-range screen first, as in the model comparison, and is scored
at full coverage (previous estimate carried forward) and on windows all arms cover.
"""
import os
import sys

import numpy as np
import pandas as pd

_EXP = os.path.dirname(os.path.abspath(__file__))
if _EXP not in sys.path:
    sys.path.insert(0, _EXP)
from lro_bench_protocol import AU, PARAMS, SITE_KEY, event_metrics
from lro_killtest import OUT, SITES, TEST_YEARS, TRAIN_YEARS, load_site, series_scale
from sarn import build_series, trimmed_mean_np, window_index, windows_for
from sarn_train import PYHQC_DIR, published_detections


def main():
    frames = {s: load_site(s)[0] for s in SITES}
    rows = []
    for site, var in (('MainStreet', 'cond'), ('TonyGrove', 'cond'), ('MainStreet', 'ph')):
        S = build_series(frames, site, var)
        mtr = (S['year'] >= TRAIN_YEARS[0]) & (S['year'] <= TRAIN_YEARS[1])
        scale = series_scale(S['cor'], mtr)
        wf = PARAMS[SITE_KEY.get(site, site)][var].widen
        dets = published_detections(site, var, S['datetime'])
        # span covered by every detection file for this series
        hi = None
        for f in sorted(os.listdir(PYHQC_DIR)):
            parts = f[:-4].replace('_detections', '').split('_')
            if f.endswith('.csv') and 'detections' in f and parts[-2] == site and parts[-1] == var:
                last = pd.read_csv(os.path.join(PYHQC_DIR, f), usecols=['datetime'],
                                   parse_dates=['datetime']).datetime.max()
                hi = last if hi is None else min(hi, last)
        when = S['datetime'].to_numpy()
        ends = windows_for(S, TEST_YEARS, 1)
        ends = ends[when[ends] <= np.datetime64(hi)]
        idx = window_index(ends)
        raw = S['raw'][idx]
        tgt = trimmed_mean_np(S['cor'][idx])
        keep = np.isfinite(tgt)
        idx, raw, tgt, ends = idx[keep], raw[keep], tgt[keep], ends[keep]
        span = np.zeros(len(when), bool)
        span[ends.min() - 29:ends.max() + 1] = True
        y = S['labeled']
        base = trimmed_mean_np(raw)

        def score(mask):
            p = trimmed_mean_np(np.where(mask[idx], np.nan, raw))
            held = pd.Series(p).ffill().to_numpy()
            lead = ~np.isfinite(held)
            held[lead] = pd.Series(base).ffill().to_numpy()[lead]
            ok = np.isfinite(held)
            cm = np.isfinite(p) & np.isfinite(base)
            return (float(np.abs(held[ok] - tgt[ok]).mean() / scale),
                    float(np.abs(p[cm] - tgt[cm]).mean() / scale), float(np.isfinite(p).mean()))

        h0, _, _ = score(np.zeros(len(when), bool))
        print('\n%s %s  test span %s .. %s  windows %d  labelled points %d  no action %.4f'
              % (site, var, pd.Timestamp(when[ends.min()]).date(), pd.Timestamp(hi).date(),
                 len(ends), int(y[span].sum()), h0), flush=True)
        for name, flags in dets.items():
            ev = event_metrics(y[span], flags[span], wf)
            widened = np.zeros(len(when), bool)
            widened[span] = np.asarray(AU.anomaly_events(pd.Series(flags[span]), wf)) > 0
            hf, cf, cov_f = score(flags)
            hw, cw, cov_w = score(widened)
            rows.append(dict(site=site, var=var, model=name.replace('pyhydroqc_', ''),
                             flags=int(flags[span].sum()), widened_points=int(widened.sum()),
                             labelled=int(y[span].sum()), event_prc=ev['prc'], event_rcl=ev['rcl'],
                             event_f1=ev['f1'], no_action=h0, hold_flags=hf, hold_widened=hw,
                             change_flags=hf / h0 - 1, change_widened=hw / h0 - 1))
            r = rows[-1]
            print('   %-22s flags %4d (widened %4d)  event F1 %.3f (prc %.3f rcl %.3f) | statistic '
                  '%.4f -> flags %.4f (%+.1f%%)  widened %.4f (%+.1f%%)'
                  % (r['model'], r['flags'], r['widened_points'], r['event_f1'], r['event_prc'],
                     r['event_rcl'], h0, hf, 100 * r['change_flags'], hw, 100 * r['change_widened']),
                  flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, 'published_effect.csv'), index=False)
    print('\nwrote %s' % os.path.join(OUT, 'published_effect.csv'))


if __name__ == '__main__':
    main()
