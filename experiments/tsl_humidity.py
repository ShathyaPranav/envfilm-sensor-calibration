"""Humidity breakdown of the official-split test error (results/tesla/PREREG_humidity.md).

Usage: python tsl_humidity.py PM10 [Antwerp Oslo ...]; writes results/tesla/humid_<pm>_<city>.parquet with one row
per test window, model and seed: prediction, target, ambient RH at the last row and (EnvFiLM-b) modulation strength.
"""
import os
import sys

os.environ.setdefault('TSL_SOURCE', 'orig')
os.environ.setdefault('TSL_EXTRA', 'SHT31HE,SHT31TE,OPCN3Hum,OPCN3Temp')
import numpy as np
import pandas as pd
import torch

_EXP = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _EXP)
import tsl_bench as B
from tsl_data import CITIES, PMS

MODELS = ('EnvFiLM-b', 'TCN+ENV')
SEEDS = (0, 1, 2)


@torch.no_grad()
def modulation(model, S, idx):
    """Mean |scale - 1| of EnvFiLM-b over layers and channels, per window."""
    out = []
    for j in range(0, len(idx), 8192):
        e = torch.clamp(S.extra(idx[j:j + 8192])[..., :model.E], -3, 3)
        z = torch.cat([e[:, -1], e.mean(1), e[:, -1] - e[:, -60:].mean(1), e.amin(1), e.amax(1)], -1)
        film = torch.tanh(model.env(z)).view(len(e), len(model.dil), 2, model.C) * 0.5
        out.append(film[:, :, 0].abs().mean((1, 2)))
    return torch.cat(out).cpu().numpy()


def main():
    pm = next(a for a in sys.argv[1:] if a in PMS)
    cities = [a for a in sys.argv[1:] if a in CITIES] or list(CITIES)
    for city in cities:
        out = os.path.join(B.OUT, 'humid_%s_%s.parquet' % (pm, city))
        if os.path.exists(out):
            continue
        S = B.Split(city, pm)
        idx = S.test
        lo, hi, mu, sd = S.ex_norm
        rh = (S.ex[torch.as_tensor(idx, device=B.DEV) - 1, 0].cpu().numpy() * sd[0] + mu[0])
        y = S.y[torch.as_tensor(idx, device=B.DEV) - 1].cpu().numpy() * B.SCALE
        parts = []
        for seed in SEEDS:
            for name in MODELS:
                model, v, te = B.fit(name, S, seed=seed)
                p = B.predict(model, S, idx).cpu().numpy() * B.SCALE
                d = pd.DataFrame(dict(pm=pm, city=city, model=name, seed=seed, rh=rh, y=y, pred=p))
                d['mod'] = modulation(model, S, idx) if name == 'EnvFiLM-b' else np.nan
                parts.append(d)
                print('%-5s %-8s %-10s seed %d  val %.3f  test %.3f' % (pm, city, name, seed, v, te), flush=True)
        pd.concat(parts).to_parquet(out, index=False)


if __name__ == '__main__':
    main()
