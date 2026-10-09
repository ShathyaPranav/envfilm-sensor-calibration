"""Development protocol: choose model components WITHOUT touching the official val/test units.

Inside each city's official training units only (the last two of each list are never
used here), 4 folds over units: a fold's units are held out and scored, two other units
validate, the rest train (every 4th window). Per-unit RMSE is compared between models.
"""
import os
import sys
import time

import numpy as np
import pandas as pd

_EXP = os.path.dirname(os.path.abspath(__file__))
if _EXP not in sys.path:
    sys.path.insert(0, _EXP)
import tsl_bench as B
from tsl_data import CITIES, PMS
from tsl_rotate import unit_index

FOLDS, STRIDE = 4, 4


def main():
    names = [a for a in sys.argv[1:] if a in B.MODELS]
    pms = [a for a in sys.argv[1:] if a in PMS] or list(PMS)
    cities = [a for a in sys.argv[1:] if a in CITIES] or ['Oslo', 'Zagreb']
    tag = os.environ.get('TSL_TAG', 'dev')
    out = os.path.join(B.OUT, 'dev_%s.csv' % tag)
    rows = pd.read_csv(out).to_dict('records') if os.path.exists(out) else []
    done = {(r['pm'], r['city'], r['fold'], r['model']) for r in rows}
    for pm in pms:
        for city in cities:
            S = B.Split(city, pm)
            B.CURRENT['pm'] = pm
            runs = unit_index(S)[:-2]                      # official val and test units excluded
            n = len(runs)
            for f in range(FOLDS):
                held = [k for k in range(n) if k % FOLDS == f]
                rest = [k for k in range(n) if k not in held]
                val = rest[-2:]
                train = [k for k in rest if k not in val]
                S.train = np.concatenate([runs[k][::STRIDE] for k in train])
                S.val = np.concatenate([runs[k] for k in val])
                for name in names:
                    if (pm, city, f, name) in done:
                        continue
                    t = time.time()
                    model, v, _ = B.fit(name, S, seed=0)
                    for k in held:
                        rows.append(dict(pm=pm, city=city, fold=f, model=name, unit=S.units[k], val=v,
                                         rmse=B.rmse_on(model, S, runs[k])))
                    print('%-5s %-8s fold %d %-14s val %.3f  %.0f s' % (pm, city, f, name, v, time.time() - t),
                          flush=True)
                    pd.DataFrame(rows).to_csv(out, index=False)
    r = pd.DataFrame(rows)
    w = r.pivot_table(index=['pm', 'city', 'unit'], columns='model', values='rmse')
    base = 'TCN'
    print('\nmedian per-unit RMSE ratio to %s (below 1 = better), and share of units better:' % base)
    for m in w.columns:
        if m == base:
            continue
        ratio = (w[m] / w[base]).groupby(level=[0, 1]).median()
        share = (w[m] < w[base]).groupby(level=[0, 1]).mean()
        print('  %-14s ' % m + '  '.join('%s/%s %.3f (%.0f%%)' % (pm, c[:3], ratio[(pm, c)], 100 * share[(pm, c)])
                                          for pm, c in ratio.index))


if __name__ == '__main__':
    main()
