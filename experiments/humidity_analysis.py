"""Pre-registered analysis of results/tesla/humid_*.parquet (results/tesla/PREREG_humidity.md).

Per cell: terciles of the test unit's ambient humidity; per band, model and seed the RMSE, averaged over seeds;
r_band = RMSE(EnvFiLM-b) / RMSE(TCN+ENV). Writes results/paper/humidity.json and results/paper/fig_humidity.pdf.
"""
import glob
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, 'results', 'tesla')
OUT = os.path.join(ROOT, 'results', 'paper')
PMS = ['PM10', 'PM2.5', 'PM1']
CITIES = ['Antwerp', 'Oslo', 'Zagreb']
BANDS = ['low', 'mid', 'high']


def main():
    cells, rows = {}, []
    for f in sorted(glob.glob(os.path.join(R, 'humid_*.parquet'))):
        d = pd.read_parquet(f)
        pm, city = d.pm.iloc[0], d.city.iloc[0]
        q = np.nanquantile(d[(d.seed == 0) & (d.model == 'EnvFiLM-b')].rh, [1 / 3, 2 / 3])   # same windows for all
        d['band'] = np.select([d.rh <= q[0], d.rh <= q[1]], ['low', 'mid'], 'high')
        e = d.assign(se=(d.pred - d.y) ** 2).groupby(['model', 'seed', 'band']).se.mean().pow(0.5)
        e = e.groupby(['model', 'band']).mean().unstack('model')
        r = (e['EnvFiLM-b'] / e['TCN+ENV']).reindex(BANDS)
        m = d[d.model == 'EnvFiLM-b']
        rho = [spearmanr(g['mod'], g.rh).statistic for _, g in m.groupby('seed')]
        cells['%s %s' % (pm, city)] = dict(rh_terciles=[float(x) for x in q],
                                           ratio={b: float(r[b]) for b in BANDS},
                                           rmse={b: {k: float(e.loc[b, k]) for k in e.columns} for b in BANDS},
                                           spearman_mod_rh=[float(x) for x in rho])
        rows.append((pm, city, r))
    lr = np.array([np.log(c['ratio']['high'] / c['ratio']['low']) for c in cells.values()])
    res = dict(cells=cells, n_cells=len(cells), high_lt_low=int((lr < 0).sum()),
               median_log_high_over_low=float(np.median(lr)),
               primary_pass=bool((lr < 0).sum() >= 6 and np.median(lr) < 0),
               mod_rh_positive_cells=int(sum(np.mean(c['spearman_mod_rh']) > 0 for c in cells.values())))
    json.dump(res, open(os.path.join(OUT, 'humidity.json'), 'w'), indent=1)
    plt.rcParams['font.family'] = 'serif'
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    for pm, city, r in rows:
        ax.plot(range(3), np.log2(r.values), marker='o', ms=3, lw=0.8,
                color={'PM10': '#1f4e79', 'PM2.5': '#c55a11', 'PM1': '#548235'}[pm],
                ls={'Antwerp': '-', 'Oslo': '--', 'Zagreb': ':'}[city], label='%s %s' % (pm, city))
    ax.axhline(0, color='k', lw=0.5)
    ax.set_xticks(range(3))
    ax.set_xticklabels(['low RH', 'middle RH', 'high RH'], fontsize=7)
    ax.set_ylabel('log$_2$ RMSE ratio\nEnvFiLM / TCN + env.', fontsize=7)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=5, ncol=3, frameon=False, loc='upper center', bbox_to_anchor=(0.5, 1.32))
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig_humidity.pdf'))
    print(json.dumps({k: v for k, v in res.items() if k != 'cells'}, indent=1))
    for k, c in cells.items():
        print(k, {b: round(v, 3) for b, v in c['ratio'].items()}, 'rho', np.round(c['spearman_mod_rh'], 2), c['rh_terciles'])


if __name__ == '__main__':
    main()
