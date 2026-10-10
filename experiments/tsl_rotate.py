"""Criterion B of results/tesla/PREREG.md: every unit tests once.

Per city and PM size the unit list is cut into 5 folds (round robin over the official
order). For a fold, its units test; the two units preceding the fold's first unit in the
official order (cyclically, skipping test units) validate; all others train. Training
windows are taken at every 4th row, for every model alike.
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

FOLDS = 5
STRIDE = 4


def unit_index(S):
    """Window indices of each unit, rebuilt from the Split's concatenated layout."""
    all_idx = np.concatenate([S.train, S.val, S.test])
    bounds = [np.arange(0)] * len(S.units)
    # Split put units in order; recover each unit's windows from contiguous runs
    breaks = np.where(np.diff(all_idx) != 1)[0] + 1
    runs = np.split(all_idx, breaks)
    assert len(runs) == len(S.units), (len(runs), len(S.units))
    return runs


def main():
    names = [a for a in sys.argv[1:] if a in B.MODELS] or ['TCN', 'GainTCN+ctx+aug']
    pms = [a for a in sys.argv[1:] if a in PMS] or list(PMS)
    cities = [a for a in sys.argv[1:] if a in CITIES] or list(CITIES)
    tag = os.environ.get('TSL_TAG', 'rotate')
    out = os.path.join(B.OUT, 'rotate_%s.csv' % tag)
    rows = pd.read_csv(out).to_dict('records') if os.path.exists(out) else []
    done = {(r['pm'], r['city'], r['fold'], r['model']) for r in rows}
    for pm in pms:
        for city in cities:
            S = B.Split(city, pm)
            runs = unit_index(S)
            n = len(runs)
            for f in range(FOLDS):
                test = [k for k in range(n) if k % FOLDS == f]
                rest = [k for k in range(n) if k not in test]
                first = test[0]
                val = [k for k in reversed(range(first)) if k in rest][:2]
                if len(val) < 2:
                    val += [k for k in reversed(rest) if k not in val][:2 - len(val)]
                train = [k for k in rest if k not in val]
                S.train = np.concatenate([runs[k][::STRIDE] for k in train])
                S.val = np.concatenate([runs[k] for k in val])
                for name in names:
                    if (pm, city, f, name) in done:
                        continue
                    t = time.time()
                    S.test = np.concatenate([runs[k] for k in test])
                    model, v, _ = B.fit(name, S, seed=int(os.environ.get('TSL_ROT_SEED', '0')))
                    for k in test:
                        rows.append(dict(pm=pm, city=city, fold=f, model=name, unit=S.units[k], val=v,
                                         rmse=B.rmse_on(model, S, runs[k])))
                    print('%-5s %-8s fold %d %-16s val %.3f  units %s  %.0f s' % (
                        pm, city, f, name, v, ' '.join('%.2f' % r['rmse'] for r in rows[-len(test):]),
                        time.time() - t), flush=True)
                    pd.DataFrame(rows).to_csv(out, index=False)


if __name__ == '__main__':
    main()
