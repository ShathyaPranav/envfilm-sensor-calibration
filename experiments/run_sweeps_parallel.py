"""Finish the window and trim sweeps in parallel, with the consistency-scaled trimmed std.

run_window_sweep.py and run_trim_sweep.py are single-process and take hours end to end.
This driver runs the identical per-cell computation (their own sweep_one_window and
sweep_one_tau, unmodified) split over series in a process pool, skips any per-cell file
already on disk, and then writes the same summary tables those scripts write. It exists
only to make the sweeps finish; it computes nothing they do not.

Always run with DQ_CONSISTENCY=1, which both sweep modules read at import.
"""
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

os.environ.setdefault('DQ_CONSISTENCY', '1')
_EXP = os.path.dirname(os.path.abspath(__file__))
if _EXP not in sys.path:
    sys.path.insert(0, _EXP)

N_PROC = 12
N_CHUNKS = 12


def _work(job):
    """One (kind, parameter, chunk) unit. Imports inside so each worker loads its own data."""
    kind, param, chunk, n_chunks = job
    import run_phase2a_full as r2a
    if kind == 'window':
        import run_window_sweep as mod
        fn = mod.sweep_one_window
    else:
        import run_trim_sweep as mod
        fn = mod.sweep_one_tau
    assert mod.CONSISTENCY, 'DQ_CONSISTENCY must be 1'
    series = r2a.load_series()
    regimes = r2a.build_regime_grid()
    mine = [s for i, s in enumerate(series) if i % n_chunks == chunk]
    df = fn(mine, regimes, param)
    df['_order'] = np.repeat([i for i in range(len(series)) if i % n_chunks == chunk],
                             len(df) // max(len(mine), 1)) if len(mine) else []
    return kind, param, df


def main():
    import run_trim_sweep as ts
    import run_window_sweep as ws
    assert ws.CONSISTENCY and ts.CONSISTENCY
    os.makedirs(ws.OUT_DIR, exist_ok=True)
    os.makedirs(ts.OUT_DIR, exist_ok=True)
    t0 = time.time()
    jobs = []
    for w in ws.WINDOWS:
        if not os.path.exists(os.path.join(ws.OUT_DIR, 'percell_w%d.csv' % w)):
            jobs += [('window', w, c, N_CHUNKS) for c in range(N_CHUNKS)]
    for tau in ts.TAUS:
        if not os.path.exists(os.path.join(ts.OUT_DIR, 'percell_tau%s.csv' % tau)):
            jobs += [('trim', tau, c, N_CHUNKS) for c in range(N_CHUNKS)]
    print('%d jobs on %d processes' % (len(jobs), N_PROC), flush=True)
    parts = {}
    if jobs:
        with Pool(N_PROC) as pool:
            for kind, param, df in pool.imap_unordered(_work, jobs):
                parts.setdefault((kind, param), []).append(df)
                done = sum(len(v) for v in parts.values())
                print('  %3d/%d  %s %s  [%.0fs]' % (done, len(jobs), kind, param, time.time() - t0),
                      flush=True)
    for (kind, param), dfs in parts.items():
        full = pd.concat(dfs, ignore_index=True).sort_values('_order', kind='stable').drop(columns='_order')
        if kind == 'window':
            full.to_csv(os.path.join(ws.OUT_DIR, 'percell_w%d.csv' % param), index=False)
        else:
            full.to_csv(os.path.join(ts.OUT_DIR, 'percell_tau%s.csv' % param), index=False)

    # ---- window summaries, exactly as run_window_sweep.main() writes them ----
    t1s, t2s, perts = [], [], []
    for w in ws.WINDOWS:
        pc = pd.read_csv(os.path.join(ws.OUT_DIR, 'percell_w%d.csv' % w))
        t1s.append(ws.table1_style(pc, w))
        t2s.append(ws.table2_style(pc, w))
        for domain, d in ws.perturbation_fit(pc, w).items():
            perts.append(dict(domain=domain, **d))
    t1, t2, pert = pd.concat(t1s, ignore_index=True), pd.concat(t2s, ignore_index=True), pd.DataFrame(perts)
    t1.to_csv(os.path.join(ws.OUT_DIR, 'table1_by_window.csv'), index=False)
    t2.to_csv(os.path.join(ws.OUT_DIR, 'table2_by_window.csv'), index=False)
    pert.to_csv(os.path.join(ws.OUT_DIR, 'perturbation_fit_by_window.csv'), index=False)
    with open(os.path.join(ws.OUT_DIR, 'window_sweep_summary.json'), 'w') as f:
        json.dump(dict(table1=t1.to_dict(orient='records'), table2=t2.to_dict(orient='records'),
                       perturbation=pert.to_dict(orient='records')), f, indent=2)

    # ---- trim summaries, exactly as run_trim_sweep.main() writes them ----
    a1, a2 = [], []
    for tau in ts.TAUS:
        pc = pd.read_csv(os.path.join(ts.OUT_DIR, 'percell_tau%s.csv' % tau))
        a1.append(ts.table1_style(pc, tau))
        a2.append(ts.table2_style(pc, tau))
    u1, u2 = pd.concat(a1, ignore_index=True), pd.concat(a2, ignore_index=True)
    u1.to_csv(os.path.join(ts.OUT_DIR, 'table1_by_tau.csv'), index=False)
    u2.to_csv(os.path.join(ts.OUT_DIR, 'table2_by_tau.csv'), index=False)
    o1 = u1.loc[u1.groupby(['domain', 'mechanism'])['trimmed_std_over_std'].idxmin()]
    o2 = u2.loc[u2.groupby(['domain', 'magnitude'])['trimmed_std_over_std'].idxmin()]
    with open(os.path.join(ts.OUT_DIR, 'trim_sweep_summary.json'), 'w') as f:
        json.dump(dict(table1_by_tau=u1.to_dict(orient='records'), table2_by_tau=u2.to_dict(orient='records'),
                       optimal_tau_table1=o1.to_dict(orient='records'),
                       optimal_tau_table2=o2.to_dict(orient='records')), f, indent=2)
    print('ALL_DONE in %.0fs' % (time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
