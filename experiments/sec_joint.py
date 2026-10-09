"""One calibration network for three cities' fleets.

Same protocol as sec_protocol.py (targets' second co-location, odd days), but the network
is trained once on what all three networks legitimately have: every unit's first
co-location and every anchor's deployment labels, from Antwerp, Oslo and Zagreb together.
The units are the same hardware (AirSensEUR) and the references report in the same units
(ppb, ug/m3), so one model can serve all of them.

Two settings:
  joint        all three cities' first co-locations and anchors
  city-blind   the target city's anchors removed (its units' own first co-locations kept):
               a city with no reference-sited sensors borrowing from other cities

The network is sec_net.py v3 (residual on each unit's own first-co-location MLP, unit
embedding, context encoder), UNCHANGED. The simple arms are rerun on the same pooled
data so they get the same extra data.

PASS MARK, fixed 2026-10-03 before the first run. Primary pollutants NO, NO2, PM2.5.
Comparator per city and pollutant: the simple deployable arm with the lowest median RMSE,
chosen among the single-city arms (protocol_simple_*.csv) and the pooled arms here.
  PASS if the joint network beats that comparator by >= 5% (median of per-unit gains) and
  on >= 2/3 of the target units, for at least 2 of the 3 pollutants in at least 2 of the
  3 cities, and loses by more than 5% in no city-pollutant pair.
O3 and the city-blind setting are reported but are not part of the pass mark.
"""
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

_EXP = os.path.dirname(os.path.abspath(__file__))
if _EXP not in sys.path:
    sys.path.insert(0, _EXP)
os.environ.setdefault('SEC_CFG', 'dict(residual=1)')
import sec_net
from sec_protocol import OUT, build, report, score, simple_arms

warnings.filterwarnings('ignore')
CITIES = ('Antwerp', 'Oslo', 'Zagreb')


def merged(task, drop_anchors_of=None):
    D = None
    for c in CITIES:
        d = build(c, task)
        if drop_anchors_of == c:
            d['anchor'] = {}
        if D is None:
            D = d
            D['city'] = 'All' if drop_anchors_of is None else 'All-no-' + drop_anchors_of
            continue
        for k in ('first', 'anchor', 'test', 'test_anchor', 'stream', 'raw', 'stat', 'ctx'):
            D[k].update(d[k])
    return D


def by_city(D, P):
    out = {}
    for c in CITIES:
        units = [u for u in D['test'] if u.startswith(c)]
        if not units:
            continue
        Dc = dict(D, city=c, test={u: D['test'][u] for u in units},
                  anchor={a: v for a, v in D['anchor'].items() if a.startswith(c)})
        out[c] = (Dc, {arm: {u: p[u] for u in units if u in p} for arm, p in P.items()})
    return out


def run_net(D, variants, seeds):
    P = {}
    for name, (c, e) in variants.items():
        runs = []
        for s in range(seeds):
            t = time.time()
            p, v = sec_net.train(D, c, e, seed=s)
            runs.append(p)
            print('  %s %s %-9s seed %d  anchor-holdout RMSE %.3f  (%.0f s)' % (D['city'], D['task'], name, s, v,
                                                                             time.time() - t), flush=True)
        P[name] = {u: np.mean([r[u] for r in runs], axis=0) for u in D['test']}
    return P


def main():
    os.makedirs(OUT, exist_ok=True)
    tasks = [a for a in sys.argv[1:] if a in ('NO', 'NO2', 'O3', 'PM2.5', 'PM10')] or ['NO', 'NO2', 'PM2.5', 'O3']
    seeds = int(os.environ.get('SEC_SEEDS', '3'))
    blind = os.environ.get('SEC_BLIND', '1') == '1'
    rows = []
    for task in tasks:
        D = merged(task)
        P = run_net(D, {'joint net': (True, True), 'joint net -ctx': (False, True)}, seeds)
        for kind in ('mlp', 'boosting'):
            for arm, p in simple_arms(D, kind).items():
                if arm.startswith('[oracle]') and kind != 'mlp':
                    continue
                P['%s own-first' % kind if arm == 'own-first' else 'pooled %s %s' % (kind, arm)] = p
        if blind:
            for c in CITIES:
                Db = merged(task, drop_anchors_of=c)
                tu = {u: Db['test'][u] for u in Db['test'] if u.startswith(c)}
                if not tu:
                    continue
                Db['test'] = tu
                pb = run_net(Db, {'city-blind net': (True, True)}, seeds)['city-blind net']
                P.setdefault('city-blind net', {}).update(pb)
        for c, (Dc, Pc) in by_city(D, P).items():
            S = score(Dc, {a: p for a, p in Pc.items() if p})
            report(Dc, S, base='joint net')
            rows.append(S.assign(city=c, task=task))
        pd.concat(rows).to_csv(os.path.join(OUT, 'joint_v3.csv'), index=False)


if __name__ == '__main__':
    main()
