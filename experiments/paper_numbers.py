"""Every number, table and figure in the EnvFiLM paper, computed from the result files.

Writes results/paper/numbers.json, results/paper/tab_*.tex and results/paper/fig_*.pdf.
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon

_EXP = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_EXP)
R = os.path.join(ROOT, 'results', 'tesla')
OUT = os.path.join(ROOT, 'results', 'paper')
PUB_TESLA = {('PM10', 'Antwerp'): 14.58, ('PM10', 'Oslo'): 15.79, ('PM10', 'Zagreb'): 13.87,
             ('PM2.5', 'Antwerp'): 7.54, ('PM2.5', 'Oslo'): 7.90, ('PM2.5', 'Zagreb'): 3.03,
             ('PM1', 'Antwerp'): 4.78, ('PM1', 'Oslo'): 3.63, ('PM1', 'Zagreb'): 1.14}
PUB_SCARE_MEAN = {'PM10': 14.03, 'PM2.5': 5.67, 'PM1': 2.51}
PUB_TESLA_MEAN_SCARE = {'PM10': 14.21, 'PM2.5': 6.19, 'PM1': 2.82}     # TESLA as listed in SCARE's table
PMS = ['PM10', 'PM2.5', 'PM1']
CITIES = ['Antwerp', 'Oslo', 'Zagreb']
NAMES = {'EnvFiLM-b': 'EnvFiLM (ours)', 'TCN+ENV': 'TCN + env. channels', 'TCN+ENVwide': 'TCN + env., size-matched',
         'TESLA+ENV': 'TESLA + env. channels', 'LogTCN': 'LogTCN (particles only)', 'TESLA-reimpl': 'TESLA (our re-impl.)',
         'TESLA-pub': 'TESLA (published)', 'EnvLate': 'EnvFiLM, modulation off', 'TCN': 'LogTCN (particles only)',
         'TESLA': 'TESLA (particles only)', 'PatchTST+ENV': 'PatchTST + env. channels',
         'PatchTST': 'PatchTST (particles only)'}


def csv(name):
    return pd.read_csv(os.path.join(R, name))


def main_table():
    env = csv('bench_final_env.csv')
    env = env[env.pm == 'PM10']                                   # PM2.5/PM1 of this run came from the second process
    rows = pd.concat([env, csv('bench_final_env_b.csv'), csv('bench_final_filmb.csv'),
                      csv('bench_orig_logtcn.csv').assign(model='LogTCN'),
                      csv('bench_orig_logtcn_rest.csv').assign(model='LogTCN'),
                      csv('bench_orig_tesla.csv').assign(model='TESLA-reimpl')])
    import glob
    pt = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(R, 'bench_patchtst_*.csv')) if 'crashed' not in f])
    for name in ('PatchTST', 'PatchTST+ENV'):
        x = pt[pt.model == name].drop_duplicates(['pm', 'city', 'seed'])
        if len(x) == 27:                                       # all 9 cells x 3 seeds, else left out of the table
            rows = pd.concat([rows, x])
    rows = rows[rows.model != 'EnvFiLM']                          # unbounded variant: reported separately
    m = rows.groupby(['pm', 'city', 'model']).test.agg(['mean', 'std', 'count']).reset_index()
    t = m.pivot_table(index=['pm', 'city'], columns='model', values='mean')
    s = m.pivot_table(index=['pm', 'city'], columns='model', values='std')
    t['TESLA-pub'] = [PUB_TESLA[i] for i in t.index]
    t = t.reindex([(p, c) for p in PMS for c in CITIES])
    s = s.reindex(t.index)
    return t, s, rows


def rotation(name, a, b):
    """name: one rotation CSV, or a list of CSVs from different seeds (per-unit RMSE averaged over seeds)."""
    names = name if isinstance(name, list) else [name]
    r = pd.concat([csv(n).assign(seed=k) for k, n in enumerate(names)])
    r = r.groupby(['pm', 'city', 'unit', 'model']).rmse.mean().reset_index()
    w = r.pivot_table(index=['pm', 'city', 'unit'], columns='model', values='rmse').dropna(subset=[a, b])
    rat = w[a] / w[b]
    rng = np.random.default_rng(0)
    v = rat.to_numpy()
    bs = rng.choice(v, (10000, len(v)), replace=True)
    ci_better = np.percentile((bs < 1).mean(1), [2.5, 97.5])
    ci_median = np.percentile(np.median(bs, 1), [2.5, 97.5])
    out = dict(units=int(len(rat)), better=float((rat < 1).mean()), median_ratio=float(rat.median()),
               ci_better=[float(x) for x in ci_better], ci_median=[float(x) for x in ci_median],
               worst=float(rat.max()), sign_p=float(binomtest(int((rat < 1).sum()), len(rat)).pvalue),
               wilcoxon_p=float(wilcoxon(np.log(rat)).pvalue), groups={})
    for (pm, c), x in w.groupby(level=[0, 1]):
        rr = x[a] / x[b]
        out['groups']['%s %s' % (pm, c)] = dict(n=int(len(rr)), better=float((rr < 1).mean()),
                                                median_ratio=float(rr.median()),
                                                mean_a=float(x[a].mean()), mean_b=float(x[b].mean()))
    return out, rat


SOTON_ON = ['rotate_soton_onboard_s0.csv', 'rotate_soton_onboard_s1.csv']
SOTON_MATCHED = ['rotate_soton_matched_s0.csv', 'rotate_soton_matched_s1.csv']


def southampton():
    """Independent dataset (Bulot et al.): on-board channels (2 seeds), weather-only full-year run (1 seed) and,
    when present, the matched weather-only control on the on-board rows (2 seeds)."""
    runs = {'onboard': SOTON_ON, 'weather_year': ['rotate_soton_rot.csv']}
    if all(os.path.exists(os.path.join(R, f)) for f in SOTON_MATCHED):
        runs['weather_matched'] = SOTON_MATCHED
    out = {}
    for k, files in runs.items():
        out[k] = {}
        for b in ('TCN+ENV', 'TESLA+ENV'):
            _, rat = rotation(files, 'EnvFiLM-b', b)
            for g in ('SotonPMS', 'SotonSPS'):
                x = rat[rat.index.get_level_values(1) == g]
                d = dict(n=int(len(x)), better=float((x < 1).mean()), median_ratio=float(x.median()),
                         wilcoxon_p=float(wilcoxon(np.log(x)).pvalue))
                for pm in PMS:
                    y = x[x.index.get_level_values(0) == pm]
                    d[pm] = dict(better=int((y < 1).sum()), n=int(len(y)), median_ratio=float(y.median()))
                out[k]['%s %s' % (g, b)] = d
    r = pd.concat([csv(f) for f in SOTON_ON]).groupby(['pm', 'city', 'unit', 'model']).rmse.mean().reset_index()
    out['onboard_mean_rmse'] = {'%s %s' % k: {m: float(v) for m, v in x.groupby('model').rmse.mean().items()}
                                for k, x in r.groupby(['city', 'pm'])}
    return out


def write_soton_tex(S):
    def pmn(pm):
        return pm.replace('PM2.5', 'PM$_{2.5}$').replace('PM10', 'PM$_{10}$').replace('PM1', 'PM$_{1}$')
    cols = [('onboard', 'TCN+ENV'), ('onboard', 'TESLA+ENV')] + (
        [('weather_matched', 'TCN+ENV')] if 'weather_matched' in S else []) + [('weather_year', 'TCN+ENV')]
    heads = {('onboard', 'TCN+ENV'): 'On-board, vs TCN+env.', ('onboard', 'TESLA+ENV'): 'On-board, vs TESLA+env.',
             ('weather_matched', 'TCN+ENV'): 'Weather, same rows, vs TCN+env.',
             ('weather_year', 'TCN+ENV'): 'Weather, full year, vs TCN+env.'}
    lines = ['\\begin{tabular}{ll' + 'rr' * len(cols) + '}', '\\toprule',
             ' & & ' + ' & '.join('\\multicolumn{2}{c}{%s}' % heads[c] for c in cols) + ' \\\\',
             'Sensor & PM & ' + ' & '.join(['better & median'] * len(cols)) + ' \\\\', '\\midrule']
    for g, gn in (('SotonPMS', 'PMS5003'), ('SotonSPS', 'SPS30')):
        for pm in PMS:
            cells = []
            for k, b in cols:
                d = S[k]['%s %s' % (g, b)][pm]
                cells += ['%d/%d' % (d['better'], d['n']), '%.3f' % d['median_ratio']]
            lines.append('%s & %s & %s \\\\' % (gn, pmn(pm), ' & '.join(cells)))
        cells = []
        for k, b in cols:
            d = S[k]['%s %s' % (g, b)]
            cells += ['%.0f\\%%' % (100 * d['better']), '%.3f' % d['median_ratio']]
        lines.append('%s & all & %s \\\\' % (gn, ' & '.join(cells)))
        lines.append('\\midrule' if g == 'SotonPMS' else '\\bottomrule')
    lines.append('\\end{tabular}')
    open(os.path.join(OUT, 'tab_soton.tex'), 'w', encoding='utf-8', newline='\n').write('\n'.join(lines) + '\n')


def sensor_level(files, a='EnvFiLM-b', b='TCN+ENV', n_boot=5000):
    """Rotation statistics with the physical sensor as the independent unit: each sensor's mean log ratio over its
    size fractions for the sign and Wilcoxon tests, and a bootstrap that resamples sensors (cluster bootstrap)."""
    r = pd.concat([csv(f) for f in files]).groupby(['pm', 'city', 'unit', 'model']).rmse.mean().unstack('model')
    lr = np.log(r[a] / r[b]).dropna()
    per = lr.groupby(level=[1, 2]).mean()
    groups = [g.to_numpy() for _, g in lr.groupby(level=[1, 2])]
    rng = np.random.default_rng(0)
    share, med = [], []
    for _ in range(n_boot):
        s = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        share.append((s < 0).mean())
        med.append(np.exp(np.median(s)))
    return dict(sensors=int(len(per)), better=int((per < 0).sum()),
                sign_p=float(binomtest(int((per < 0).sum()), len(per)).pvalue),
                wilcoxon_p=float(wilcoxon(per).pvalue),
                ci_share=[float(x) for x in np.percentile(share, [2.5, 97.5])],
                ci_median=[float(x) for x in np.percentile(med, [2.5, 97.5])])


def fmt(v, best):
    return ('\\textbf{%.2f}' % v) if best else '%.2f' % v


def write_main_tex(t, s):
    cols = ['EnvFiLM-b', 'TCN+ENV', 'TCN+ENVwide', 'TESLA+ENV', 'LogTCN', 'TESLA-reimpl', 'TESLA-pub']
    cols = [c for c in cols[:4] + ['PatchTST+ENV'] + cols[4:5] + ['PatchTST'] + cols[5:] if c in t.columns]
    head = ' & '.join(['PM', 'City'] + ['\\rotatebox{90}{%s}' % NAMES[c] for c in cols]) + ' \\\\'
    lines = ['\\begin{tabular}{ll' + 'r' * len(cols) + '}', '\\toprule', head, '\\midrule']
    for (pm, c), row in t[cols].iterrows():
        best = row.min()
        cells = [fmt(row[k], row[k] == best) for k in cols]
        pmn = pm.replace('PM2.5', 'PM$_{2.5}$').replace('PM10', 'PM$_{10}$').replace('PM1', 'PM$_{1}$')
        lines.append('%s & %s & %s \\\\' % (pmn, c, ' & '.join(cells)))
    lines += ['\\bottomrule', '\\end{tabular}']
    open(os.path.join(OUT, 'tab_main.tex'), 'w', encoding='utf-8', newline='\n').write('\n'.join(lines) + '\n')


def write_ref_tex(t):
    cols = ['EnvFiLM-b', 'TCN+ENV', 'TESLA+ENV', 'TCN', 'TESLA']
    head = ' & '.join(['PM', 'City'] + ['\\rotatebox{90}{%s}' % NAMES[c] for c in cols]) + ' \\\\'
    lines = ['\\begin{tabular}{ll' + 'r' * len(cols) + '}', '\\toprule', head, '\\midrule']
    for (pm, c), row in t[cols].iterrows():
        best = row.min()
        pmn = pm.replace('PM2.5', 'PM$_{2.5}$').replace('PM10', 'PM$_{10}$').replace('PM1', 'PM$_{1}$')
        lines.append('%s & %s & %s \\\\' % (pmn, c, ' & '.join(fmt(row[k], row[k] == best) for k in cols)))
    lines += ['\\bottomrule', '\\end{tabular}']
    open(os.path.join(OUT, 'tab_ref.tex'), 'w', encoding='utf-8', newline='\n').write('\n'.join(lines) + '\n')


def write_groups_tex(rot_main, rot_late, rot_ref):
    """Per-group rotation table: share of units better and median ratio, both benchmarks."""
    def pmn(pm):
        return pm.replace('PM2.5', 'PM$_{2.5}$').replace('PM10', 'PM$_{10}$').replace('PM1', 'PM$_{1}$')
    lines = ['\\begin{tabular}{llrrrrrr}', '\\toprule',
             ' & & \\multicolumn{2}{c}{TESLA bench., vs TCN+env.} & \\multicolumn{2}{c}{TESLA bench., vs mod. off}'
             ' & \\multicolumn{2}{c}{True ref., vs TCN+env.} \\\\',
             'PM & City & better & median & better & median & better & median \\\\', '\\midrule']
    for pm in PMS:
        for c in CITIES:
            k = '%s %s' % (pm, c)
            cells = []
            for r in (rot_main, rot_late, rot_ref):
                g = r['groups'].get(k)
                cells += ['%d/%d' % (round(g['better'] * g['n']), g['n']), '%.3f' % g['median_ratio']] if g else ['n/a', 'n/a']
            lines.append('%s & %s & %s \\\\' % (pmn(pm), c, ' & '.join(cells)))
    lines.append('\\midrule')
    cells = []
    for r in (rot_main, rot_late, rot_ref):
        cells += ['%d/%d' % (round(r['better'] * r['units']), r['units']), '%.3f' % r['median_ratio']]
    lines += ['All & & %s \\\\' % ' & '.join(cells), '\\bottomrule', '\\end{tabular}']
    open(os.path.join(OUT, 'tab_groups.tex'), 'w', encoding='utf-8', newline='\n').write('\n'.join(lines) + '\n')


def dev_ablations():
    """Component checks on the development protocol (official training units only)."""
    def ratio(f, a, b, extra=None):
        r = csv(f) if extra is None else pd.concat([csv(f), csv(extra)])
        w = r.pivot_table(index=['pm', 'city', 'unit'], columns='model', values='rmse').dropna(subset=[a, b])
        rr = w[a] / w[b]
        g = rr.groupby(level=[0, 1]).median()
        return dict(units=int(len(rr)), better=float((rr < 1).mean()), median_ratio=float(rr.median()),
                    groups={'%s %s' % k: float(v) for k, v in g.items()})
    return {'RH vs none': ratio('dev_dev_rh.csv', 'TCN+RH', 'TCN'),
            'four channels vs RH': ratio('dev_dev_env.csv', 'TCN+ENV', 'TCN+RH'),
            'EnvFiLM vs TCN+ENV (dev)': ratio('dev_dev_film.csv', 'EnvFiLM', 'TCN+ENV'),
            'EnvFiLM vs size-matched (dev)': ratio('dev_dev_film.csv', 'EnvFiLM', 'TCN+ENVwide', 'dev_dev_cap.csv'),
            'HumidGate vs TCN+RH (dev)': ratio('dev_dev_hg.csv', 'HumidGate', 'TCN+RH'),
            'latent target gain vs TCN (dev)': ratio('dev_dev_gain.csv', 'TCN+gain', 'TCN'),
            'residual TCN vs TCN (dev)': ratio('dev_dev1.csv', 'ResTCN', 'TCN')}


def out_of_season():
    r = csv('oot_oot1.csv')
    w = r.pivot_table(index=['pm', 'city', 'unit'], columns='model', values='rmse')
    o = {}
    for a in ('EnvFiLM', 'EnvFiLM-b'):
        rr = (w[a] / w['TCN+ENV']).dropna()
        o[a] = dict(units=int(len(rr)), better=float((rr < 1).mean()), median_ratio=float(rr.median()),
                    worst=float(rr.max()))
    o['means'] = {'%s %s' % k: {m: float(v) for m, v in row.items()} for k, row in
                  w.groupby(level=[0, 1]).mean().iterrows()}
    ins = csv('bench_final_env.csv')
    o['in_season_TCN+ENV_PM10'] = {c: float(ins[(ins.pm == 'PM10') & (ins.city == c) & (ins.model == 'TCN+ENV')].test.mean())
                                  for c in CITIES}
    return o


def params_and_latency():
    sys.path.insert(0, _EXP)
    os.environ.setdefault('TSL_EXTRA', 'SHT31HE,SHT31TE,OPCN3Hum,OPCN3Temp')
    import torch
    import tsl_bench as B
    res = {}
    for name in ('EnvFiLM-b', 'TCN+ENV', 'TCN+ENVwide', 'TESLA+ENV', 'TCN', 'TESLA'):
        m = B.MODELS[name]().eval()
        n = sum(p.numel() for p in m.parameters())
        x = torch.rand(1, B.N) * 0.1
        e = torch.randn(1, B.N, 4)
        f = (lambda: m(x, e)) if getattr(m, 'uses_extra', False) else (lambda: m(x))
        with torch.no_grad():
            for _ in range(20):
                f()
            t = time.perf_counter()
            for _ in range(300):
                f()
            ms = (time.perf_counter() - t) / 300 * 1000
        res[name] = dict(params=int(n), cpu_ms_per_window=float(ms))
    return res


def figures(rot_main, rot_ref, rot_late):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 8, 'font.family': 'serif'})
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    for k, (rat, lab) in enumerate([(rot_main, 'TESLA benchmark\nvs TCN + env.'), (rot_late, 'TESLA benchmark\nvs modulation off'),
                                    (rot_ref, 'True reference\nvs TCN + env.')]):
        v = np.log2(rat.to_numpy())
        ax.boxplot(v, positions=[k], widths=0.5, showfliers=False)
        ax.scatter(np.full(len(v), k) + np.random.default_rng(k).uniform(-0.18, 0.18, len(v)), v, s=3, alpha=0.35)
    ax.axhline(0, color='k', lw=0.6, ls='--')
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['TESLA bench.\nvs TCN+env.', 'TESLA bench.\nvs mod. off', 'True ref.\nvs TCN+env.'])
    ax.set_ylabel('log$_2$(RMSE ratio)')
    lo = min(np.log2(r.to_numpy()).min() for r in (rot_main, rot_ref, rot_late))
    hi = max(np.log2(r.to_numpy()).max() for r in (rot_main, rot_ref, rot_late))
    ax.set_ylim(lo - 0.1, hi + 0.1)                                # every unit visible
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig_rotation.pdf'))
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    N = {}
    t, s, _ = main_table()
    write_main_tex(t, s)
    N['main_table'] = {'%s %s' % k: {c: float(v) for c, v in row.items() if pd.notna(v)} for k, row in t.iterrows()}
    N['main_std'] = {'%s %s' % k: {c: float(v) for c, v in row.items() if pd.notna(v)} for k, row in s.iterrows()}
    cmp = {}
    for b in [c for c in ('TCN+ENV', 'TCN+ENVwide', 'TESLA+ENV', 'PatchTST+ENV', 'LogTCN', 'PatchTST', 'TESLA-reimpl', 'TESLA-pub') if c in t.columns]:
        cmp[b] = dict(cells_better=int((t['EnvFiLM-b'] < t[b]).sum()), mean_ratio=float((t['EnvFiLM-b'] / t[b]).mean()))
    N['main_compare'] = cmp
    cm = t.groupby(level=0).mean()
    N['city_means'] = {p: {c: float(cm.loc[p, c]) for c in ('EnvFiLM-b', 'TCN+ENV', 'LogTCN', 'TESLA-reimpl', 'TESLA-pub')}
                       for p in PMS}
    N['city_means_published'] = dict(SCARE=PUB_SCARE_MEAN, TESLA_in_SCARE_table=PUB_TESLA_MEAN_SCARE)
    TESLA_SEEDS = ['rotate_rot_filmb.csv', 'rotate_rot_filmb_s1.csv']
    N['rotation_tesla_vs_tcnenv'], rot_main = rotation(TESLA_SEEDS, 'EnvFiLM-b', 'TCN+ENV')
    N['rotation_tesla_per_seed'] = {k: {kk: vv for kk, vv in rotation(f, 'EnvFiLM-b', 'TCN+ENV')[0].items() if kk != 'groups'}
                                    for k, f in enumerate(TESLA_SEEDS)}
    N['rotation_tesla_vs_late'], rot_late = rotation('rotate_rot_filmb.csv', 'EnvFiLM-b', 'EnvLate')
    N['rotation_unbounded_vs_tcnenv'], _ = rotation('rotate_rot_film.csv', 'EnvFiLM', 'TCN+ENV')
    r = csv('bench_ref10_off.csv')
    tr = r.groupby(['pm', 'city', 'model']).test.mean().unstack('model')
    tr = tr.reindex([(p, c) for p in PMS for c in ('Antwerp', 'Oslo')])
    write_ref_tex(tr)
    N['ref_table'] = {'%s %s' % k: {c: float(v) for c, v in row.items()} for k, row in tr.iterrows()}
    N['ref_compare'] = {b: dict(cells_better=int((tr['EnvFiLM-b'] < tr[b]).sum()), mean_ratio=float((tr['EnvFiLM-b'] / tr[b]).mean()))
                        for b in ('TCN+ENV', 'TESLA+ENV', 'TCN', 'TESLA')}
    REF_SEEDS = ['rotate_ref10_rot.csv', 'rotate_ref10_rot_s1.csv', 'rotate_ref10_rot_s2.csv', 'rotate_ref10_rot_s3.csv']
    N['rotation_ref_vs_tcnenv'], rot_ref = rotation(REF_SEEDS, 'EnvFiLM-b', 'TCN+ENV')
    N['rotation_ref_per_seed'] = {k: {kk: vv for kk, vv in rotation(f, 'EnvFiLM-b', 'TCN+ENV')[0].items() if kk != 'groups'}
                                  for k, f in enumerate(REF_SEEDS)}
    write_groups_tex(N['rotation_tesla_vs_tcnenv'], N['rotation_tesla_vs_late'], N['rotation_ref_vs_tcnenv'])
    N['sensor_level'] = dict(tesla=sensor_level(TESLA_SEEDS), ref=sensor_level(REF_SEEDS), soton=sensor_level(SOTON_ON))
    N['southampton'] = southampton()
    write_soton_tex(N['southampton'])
    N['dev'] = dev_ablations()
    N['out_of_season'] = out_of_season()
    o = csv('oracles.csv')
    N['oracle'] = dict(gain_share_median=float(o.gain_share.median()), gain_share_max=float(o.gain_share.max()),
                       O1_over_floor_median=float((o.rmse_O1 / o.floor).median()),
                       daily_gain_spread_median=float(o.O3_daily_spread.median()))
    N['models'] = params_and_latency()
    figures(rot_main, rot_ref, rot_late)
    json.dump(N, open(os.path.join(OUT, 'numbers.json'), 'w'), indent=1)
    print(json.dumps({k: N[k] for k in ('main_compare', 'city_means', 'rotation_tesla_vs_tcnenv', 'rotation_tesla_per_seed', 'rotation_tesla_vs_late',
                                        'ref_compare', 'rotation_ref_vs_tcnenv', 'models')}, indent=1)[:6000])


if __name__ == '__main__':
    main()
