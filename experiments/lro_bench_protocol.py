"""The benchmark, scored three ways, with pyhydroqc's own evaluation code.

An earlier attempt scored pyhydroqc's published detections point by point and got F1
around 0.02, which contradicts their paper. The cause was mine, not theirs: their
'anomaly' column marks detection ONSETS (178 flags against 106 labelled events on
TonyGrove conductivity, 66% of events touched within six hours), and their published
metric runs those flags through anomaly_events + compare_events, which credits an entire
labelled event as detected when any detection overlaps it. So the only fair comparison
runs their code.

THREE SCORINGS, reported side by side because they disagree and the disagreement is the
interesting part:

  EVENT (their protocol)   pyhydroqc anomaly_utilities.anomaly_events, compare_events and
                           metrics, with their per-site widen parameter. Comparable to
                           the numbers in Jones et al. 2022.
  POINT (strict)           precision, recall and F1 per data point, no widening, no event
                           credit. What a per-point classifier is really doing.
  WINDOW STATISTIC         the paper's metric: drop what the detector flags, recompute the
                           rolling statistic, compare against the technician-corrected
                           series. What a data-quality monitor actually reports.

Baselines: pyhydroqc's published detections, pyhydroqc's own rules (their range and
persistence parameters, not ones fitted by us), doing nothing, and an oracle that masks
exactly the technician-labelled points.
"""
import importlib.util
import os
import sys

import numpy as np
import pandas as pd

_EXP = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_EXP)
if _EXP not in sys.path:
    sys.path.insert(0, _EXP)
from lro_killtest import DATA, OUT, TEST_YEARS, TRAIN_YEARS, WINDOWS, load_site, rolling_all, series_scale

PYHQC = os.path.join(_ROOT, 'external', 'pyhydroqc', 'pyhydroqc')
RESULTS = os.path.join(DATA, 'pyhydroqc_results')
EST = 'trimmed_mean'
SITE_KEY = {'WaterLab': 'Waterlab', 'BlackSmithFork': 'BlacksmithFork'}


def _load(name):
    """Import a pyhydroqc module directly, bypassing its __init__ (which needs pmdarima)."""
    spec = importlib.util.spec_from_file_location(name, os.path.join(PYHQC, name + '.py'))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


AU = _load('anomaly_utilities')
PARAMS = _load('parameters').site_params


def event_metrics(labeled, detected, wf):
    """pyhydroqc's published protocol, run through their own functions."""
    df = pd.DataFrame({'labeled_anomaly': np.asarray(labeled, dtype=bool),
                       'detected_anomaly': np.asarray(detected, dtype=bool)})
    df['labeled_event'] = AU.anomaly_events(df['labeled_anomaly'], wf)
    df['detected_event'] = AU.anomaly_events(df['detected_anomaly'], wf)
    df = AU.compare_events(df, wf)
    try:
        m = AU.metrics(df)
    except ZeroDivisionError:
        # their metrics() divides by (tp + fp), which is zero for a detector that flags
        # nothing; that arm is a reference line, not a competitor
        return dict(prc=0.0, rcl=0.0, f1=0.0, tp=0, fp=0, fn=int(np.asarray(labeled, bool).sum()))
    return dict(prc=m.prc, rcl=m.rcl, f1=m.f1, tp=m.true_positives,
                fp=m.false_positives, fn=m.false_negatives)


def point_metrics(y, p):
    y, p = np.asarray(y, bool), np.asarray(p, bool)
    tp, fp, fn = int((y & p).sum()), int((~y & p).sum()), int((y & ~p).sum())
    prc = tp / (tp + fp) if tp + fp else 0.0
    rcl = tp / (tp + fn) if tp + fn else 0.0
    return dict(prc=prc, rcl=rcl, f1=2 * prc * rcl / (prc + rcl) if prc + rcl else 0.0)


def pyhydroqc_rules(raw, site, var):
    """Their rules-based screen, with their own published parameters."""
    p = PARAMS[SITE_KEY.get(site, site)][var]
    x = pd.DataFrame({'raw': raw})
    bad = ~np.isfinite(raw)
    bad |= (raw > p.max_range) | (raw < p.min_range)
    same = np.r_[False, np.diff(raw) == 0] & np.isfinite(raw)
    run, flags = 0, np.zeros(len(raw), bool)
    for i, s in enumerate(same):
        run = run + 1 if s else 0
        if run >= p.persist:
            flags[i - run:i + 1] = True
    return bad | flags, p.widen


def detection_files():
    out = {}
    if not os.path.isdir(RESULTS):
        return out
    for f in sorted(os.listdir(RESULTS)):
        if 'detections' in f and f.endswith('.csv'):
            parts = f[:-4].replace('_detections', '').split('_')
            out.setdefault((parts[-2], parts[-1]), []).append(('_'.join(parts[:-2]),
                                                               os.path.join(RESULTS, f)))
    return out


def main():
    det = detection_files()
    rows = []
    for (site, var), models in sorted(det.items()):
        d, step = load_site(site)
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

        # restrict to the span the published detections actually cover
        spans = []
        for _, path in models:
            dd = pd.read_csv(path, usecols=['datetime'], parse_dates=['datetime'])
            spans.append((dd.datetime.min(), dd.datetime.max()))
        lo, hi = max(s[0] for s in spans), min(s[1] for s in spans)
        cover = ((idx >= lo) & (idx <= hi)).to_numpy()
        cover_win = cover[:len(ec)]
        print('\n%s %s   labels %.4f   scale %.3f   published detections cover %s .. %s'
              % (site, var, y.mean(), scale, lo.date(), hi.date()))

        def window_mae(flags, mask):
            x = raw.copy()
            x[flags] = np.nan
            e = rolling_all(x, w)[EST]
            m = np.isfinite(e) & np.isfinite(ec) & mask
            return float(np.abs(e[m] - ec[m]).mean() / scale) if m.any() else np.nan

        rule_flags, wf = pyhydroqc_rules(raw, site, var)
        cands = [('raw (no action)', np.zeros(len(raw), bool)),
                 ('pyhydroqc rules', rule_flags)]
        for model, path in models:
            dd = pd.read_csv(path, parse_dates=['datetime'])
            s = dd.set_index('datetime')['anomaly'].reindex(idx).fillna(False)
            cands.append((model, s.to_numpy(bool)))
        cands.append(('ORACLE technician labels', y))

        for name, flags in cands:
            ev = event_metrics(y[cover], flags[cover], wf)
            evt = event_metrics(y[cover & mte], flags[cover & mte], wf)
            pt = point_metrics(y[cover], flags[cover])
            rows.append(dict(site=site, var=var, model=name, n_flagged=int(flags[cover].sum()),
                             event_prc=ev['prc'], event_rcl=ev['rcl'], event_f1=ev['f1'],
                             event_f1_test=evt['f1'], point_f1=pt['f1'], point_prc=pt['prc'],
                             point_rcl=pt['rcl'],
                             window_mae=window_mae(flags, cover_win),
                             window_mae_test=window_mae(flags, cover_win & te_win)))
            r = rows[-1]
            print('   %-26s flags %6d | EVENT f1 %.3f (prc %.3f rcl %.3f) test %.3f | POINT f1 %.3f '
                  '| window MAE %.4f test %.4f'
                  % (name, r['n_flagged'], r['event_f1'], r['event_prc'], r['event_rcl'],
                     r['event_f1_test'], r['point_f1'], r['window_mae'], r['window_mae_test']),
                  flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(OUT, 'bench_protocol.csv'), index=False)

    r = pd.DataFrame(rows)
    pub = r[~r.model.isin(['raw (no action)', 'ORACLE technician labels', 'pyhydroqc rules'])]
    print('\n=== BEST PUBLISHED MODEL PER SERIES (their event protocol) ===')
    if len(pub):
        best = pub.sort_values('event_f1', ascending=False).groupby(['site', 'var']).head(1)
        print(best[['site', 'var', 'model', 'event_f1', 'event_f1_test', 'point_f1',
                    'window_mae_test']].to_string(index=False, float_format='%.4f'))
    print('\n=== what the rules and the oracle get, same protocol ===')
    print(r[r.model.isin(['pyhydroqc rules', 'ORACLE technician labels', 'raw (no action)'])]
          [['site', 'var', 'model', 'event_f1', 'point_f1', 'window_mae', 'window_mae_test']]
          .to_string(index=False, float_format='%.4f'))
    print('\nwrote %s' % os.path.join(OUT, 'bench_protocol.csv'))


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    main()
