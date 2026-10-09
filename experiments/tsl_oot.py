"""Out-of-season test: train on the TESLA protocol (first co-location, official train/val
units), test on the SECOND co-location of every unit, 8-10 months later, which the TESLA
benchmark never uses and which no model selection in this project has touched.

Second co-location rows: the unit at its city's reference station (same station as the first
co-location) on or after 2021-01-01; rows with a missing value or a quality flag on the six PM
columns dropped, as the authors did for the first co-location.
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

_EXP = os.path.dirname(os.path.abspath(__file__))
if _EXP not in sys.path:
    sys.path.insert(0, _EXP)
import tsl_bench as B
from tsl_data import COLS, EXTRA, LISTS, OPC, PMS, UNITS

STATION = {'Antwerp': 'ANT_REF_R801', 'Oslo': 'OSL_REF_KVN', 'Zagreb': 'ZAG_REF_IMI'}
CACHE = os.path.join(os.path.dirname(UNITS), 'second_col')


def second_frame(city, ase):
    os.makedirs(CACHE, exist_ok=True)
    dst = os.path.join(CACHE, '%s_%s.parquet' % (city, ase))
    if os.path.exists(dst):
        return pd.read_parquet(dst)
    cols = ['date', 'Location.ID'] + COLS + [c + '_flag' for c in COLS] + EXTRA
    d = pd.read_parquet(os.path.join(UNITS, '%s_%s.parquet' % (city, ase)), columns=cols)
    d = d[(d['Location.ID'].astype(str) == STATION[city]) & (d.date >= pd.Timestamp('2021-01-01'))]
    ok = d[COLS].notna().all(axis=1) & d[[c + '_flag' for c in COLS]].isna().all(axis=1)
    d = d.loc[ok, ['date'] + COLS + EXTRA].reset_index(drop=True)
    d[EXTRA] = d[EXTRA].interpolate(limit_direction='both')
    d.to_parquet(dst, index=False)
    return d


def attach(S, city, pm):
    """Append every unit's second co-location to the Split; return {unit: window indices}."""
    xs, ys, es, idx = [], [], [], {}
    off = len(S.x)
    lo, hi, mu, sd = S.ex_norm
    for ase in LISTS[city]:
        f = second_frame(city, ase)
        if len(f) < B.N + 500:
            continue
        xs.append(f[PMS[pm]].to_numpy(np.float32) / B.SCALE)
        ys.append(f[OPC[pm]].to_numpy(np.float32) / B.SCALE)
        e = np.clip(np.nan_to_num(f[EXTRA].to_numpy(np.float32), nan=0.0), lo, hi)
        es.append(((e - mu) / sd).astype(np.float32))
        idx[ase] = np.arange(off + B.N, off + len(f) + 1)
        off += len(f)
    S.x = torch.cat([S.x, torch.tensor(np.concatenate(xs), device=B.DEV)])
    S.y = torch.cat([S.y, torch.tensor(np.concatenate(ys), device=B.DEV)])
    S.ex = torch.cat([S.ex, torch.tensor(np.concatenate(es), device=B.DEV)])
    return idx


def main():
    names = [a for a in sys.argv[1:] if a in B.MODELS]
    pms = [a for a in sys.argv[1:] if a in PMS] or list(PMS)
    cities = [a for a in sys.argv[1:] if a in ('Antwerp', 'Oslo', 'Zagreb')] or ['Antwerp', 'Oslo', 'Zagreb']
    tag = os.environ.get('TSL_TAG', 'oot')
    out = os.path.join(B.OUT, 'oot_%s.csv' % tag)
    rows = pd.read_csv(out).to_dict('records') if os.path.exists(out) else []
    done = {(r['pm'], r['city'], r['model']) for r in rows}
    for pm in pms:
        for city in cities:
            S = B.Split(city, pm)
            B.CURRENT['pm'] = pm
            idx = attach(S, city, pm)
            for name in names:
                if (pm, city, name) in done:
                    continue
                t = time.time()
                model, v, _ = B.fit(name, S, seed=0)
                for ase, ii in idx.items():
                    rows.append(dict(pm=pm, city=city, model=name, unit=ase, val=v, rmse=B.rmse_on(model, S, ii),
                                     n=len(ii)))
                print('%-5s %-8s %-12s val %.3f  second-co-location units %d  %.0f s'
                      % (pm, city, name, v, len(idx), time.time() - t), flush=True)
                pd.DataFrame(rows).to_csv(out, index=False)


if __name__ == '__main__':
    main()
