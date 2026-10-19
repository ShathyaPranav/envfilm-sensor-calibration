"""Strength of EnvFiLM's modulation (mean |scale - 1| over layers and channels) against ambient humidity and against
the target concentration, on the official test units (results/paper/fig_modulation.pdf). Reads the per-window
records written by tsl_humidity.py; each line is one city and size, averaged over three seeds, in deciles."""
import glob
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, 'results', 'tesla')
OUT = os.path.join(ROOT, 'results', 'paper')
COL = {'PM10': '#1f4e79', 'PM2.5': '#c55a11', 'PM1': '#548235'}
LS = {'Antwerp': '-', 'Oslo': '--', 'Zagreb': ':'}


def binned(d, x):
    q = pd.qcut(d[x], 10, labels=False, duplicates='drop')
    g = d.groupby(q).agg(x=(x, 'median'), m=('mod', 'mean'))
    return g.x.to_numpy(), g.m.to_numpy()


def main():
    plt.rcParams['font.family'] = 'serif'
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.5))
    for f in sorted(glob.glob(os.path.join(R, 'humid_*.parquet'))):
        d = pd.read_parquet(f)
        d = d[d.model == 'EnvFiLM-b']
        pm, city = d.pm.iloc[0], d.city.iloc[0]
        kw = dict(color=COL[pm], ls=LS[city], lw=0.9, marker='o', ms=2.5, label='%s %s' % (pm, city))
        ax[0].plot(*binned(d, 'rh'), **kw)
        ax[1].plot(*binned(d.assign(y=d.y.clip(lower=0.05)), 'y'), **kw)
    ax[0].set_xlabel('ambient relative humidity (%)', fontsize=8)
    ax[1].set_xlabel('target concentration ($\\mu$g/m$^3$)', fontsize=8)
    ax[1].set_xscale('log')
    for a in ax:
        a.set_ylabel('mean $|$scale $-$ 1$|$', fontsize=8)
        a.tick_params(labelsize=7)
        a.grid(alpha=0.3, lw=0.5)
    ax[1].legend(fontsize=5.5, ncol=1, frameon=False, loc='center left', bbox_to_anchor=(1.0, 0.5))
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig_modulation.pdf'))
    fig.savefig(os.path.join(OUT, 'fig_modulation.png'), dpi=200)


if __name__ == '__main__':
    main()
