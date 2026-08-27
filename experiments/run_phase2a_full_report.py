"""Phase 2A (full): assemble gate results + diagnostics + figures + report
from an already-computed results/phase2a_full/phase2a_cells.csv (produced
by run_phase2a_full.py).
"""
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import phase2a_full_analysis as an
from run_phase2a_full import ALL_MECH, MAGNITUDES, BURST_LENS, AMPLITUDE_MECH, BURST_MECH, POINTWISE_MECH

OUT_DIR = os.path.join(_REPO_ROOT, 'results', 'phase2a_full')


def macro_average_loo(results, statistic, scope):
    rows = [results[k]['summary'] for k in results
            if k[0] == statistic and k[1] == 'leave_one_mechanism_out' and k[3] == scope]
    if not rows:
        return None
    hf = np.array([r['headroom_fixed_mean'] for r in rows])
    hi = np.array([r['headroom_intermediate_mean'] for r in rows])
    return dict(statistic=statistic, scope=scope, n_folds=len(rows),
                headroom_fixed_macro_mean=float(hf.mean()), headroom_fixed_by_fold=dict(zip([r['held_out'] for r in rows], hf.tolist())),
                headroom_intermediate_macro_mean=float(hi.mean()), headroom_intermediate_by_fold=dict(zip([r['held_out'] for r in rows], hi.tolist())),
                min_headroom_fixed=float(hf.min()), max_headroom_fixed=float(hf.max()),
                min_headroom_intermediate=float(hi.min()), max_headroom_intermediate=float(hi.max()))


def gate_verdict(headroom_fixed_pct, headroom_intermediate_pct):
    if headroom_fixed_pct < 25 or headroom_intermediate_pct < 15:
        if 15 <= headroom_intermediate_pct < 20 and headroom_fixed_pct >= 25:
            return 'GRAY ZONE'
        return 'KILL'
    return 'PROCEED'


def main():
    cells = pd.read_csv(os.path.join(OUT_DIR, 'phase2a_cells.csv'))
    print(f"Loaded {len(cells)} cells.")

    results = an.run_all_splits(cells)

    # ---- serialize per-fold results ----
    all_summaries = [v['summary'] for v in results.values()]
    summary_df = pd.DataFrame(all_summaries)
    summary_df.to_csv(os.path.join(OUT_DIR, 'headroom_by_fold.csv'), index=False)

    unit_rows = []
    for v in results.values():
        unit_rows.append(v['units'])
    units_df = pd.concat(unit_rows, ignore_index=True)
    units_df.to_csv(os.path.join(OUT_DIR, 'headroom_by_unit.csv'), index=False)

    # ---- macro-averages for leave-one-mechanism-out ----
    loo_macro = {}
    for statistic in ('location', 'spread'):
        for scope in ('combined', 'UCI_only'):
            m = macro_average_loo(results, statistic, scope)
            if m:
                loo_macro[(statistic, scope)] = m

    # ---- diagnostics ----
    mixed = an.mixed_configuration_frequency(cells)
    oracle_map = an.oracle_configuration_map(cells)
    oracle_map.to_csv(os.path.join(OUT_DIR, 'oracle_configuration_map.csv'), index=False)
    crossover = an.crossover_localization(cells)
    per_channel = an.per_channel_breakdown(cells)
    per_channel.to_csv(os.path.join(OUT_DIR, 'per_channel_breakdown.csv'), index=False)
    per_run = an.per_run_breakdown(cells)
    per_run.to_csv(os.path.join(OUT_DIR, 'per_run_breakdown.csv'), index=False)

    # ---- gate verdicts ----
    gate_rows = []
    for (statistic, scope), m in loo_macro.items():
        hf_pct = m['headroom_fixed_macro_mean'] * 100
        hi_pct = m['headroom_intermediate_macro_mean'] * 100
        v = gate_verdict(hf_pct, hi_pct)
        gate_rows.append(dict(split='leave_one_mechanism_out (macro)', statistic=statistic, scope=scope,
                               headroom_fixed_pct=hf_pct, headroom_intermediate_pct=hi_pct, verdict=v))
    for k, v in results.items():
        statistic, split, held_out, scope = k
        if split == 'leave_one_mechanism_out':
            continue  # already macro-averaged above
        s = v['summary']
        hf_pct = s['headroom_fixed_mean'] * 100
        hi_pct = s['headroom_intermediate_mean'] * 100
        gv = gate_verdict(hf_pct, hi_pct)
        gate_rows.append(dict(split=f'{split}::{held_out}', statistic=statistic, scope=scope,
                               headroom_fixed_pct=hf_pct, headroom_intermediate_pct=hi_pct, verdict=gv))
    gate_df = pd.DataFrame(gate_rows)
    gate_df.to_csv(os.path.join(OUT_DIR, 'gate_verdicts.csv'), index=False)

    print("\n=== Gate verdicts (all splits) ===")
    print(gate_df.to_string(index=False))

    # ---- overall verdict: combined, leave-one-mechanism-out macro (the most complete/representative split) ----
    overall = {}
    for statistic in ('location', 'spread'):
        m = loo_macro.get((statistic, 'combined'))
        m_uci = loo_macro.get((statistic, 'UCI_only'))
        if m:
            hf_pct, hi_pct = m['headroom_fixed_macro_mean'] * 100, m['headroom_intermediate_macro_mean'] * 100
            overall[statistic] = dict(combined=dict(headroom_fixed_pct=hf_pct, headroom_intermediate_pct=hi_pct,
                                                       verdict=gate_verdict(hf_pct, hi_pct)))
        if m_uci:
            hf_pct_u, hi_pct_u = m_uci['headroom_fixed_macro_mean'] * 100, m_uci['headroom_intermediate_macro_mean'] * 100
            overall[statistic]['UCI_only'] = dict(headroom_fixed_pct=hf_pct_u, headroom_intermediate_pct=hi_pct_u,
                                                    verdict=gate_verdict(hf_pct_u, hi_pct_u))

    result = dict(
        n_cells=len(cells), n_regimes=int(cells['regime'].nunique()),
        n_series=int(cells.groupby(['domain', 'run', 'channel']).ngroups),
        gate_verdicts=gate_df.to_dict(orient='records'),
        leave_one_mechanism_out_macro=[{**{'statistic': k[0], 'scope': k[1]}, **v} for k, v in loo_macro.items()],
        overall=overall,
        mixed_configuration_frequency={k: dict(overall_fraction_mixed=v['overall_fraction_mixed'],
                                                 n_regimes=v['n_regimes'], per_mechanism=v['per_mechanism'])
                                        for k, v in mixed.items()},
        crossover_localization=crossover,
    )
    with open(os.path.join(OUT_DIR, 'phase2a_report.json'), 'w') as f:
        json.dump(result, f, indent=2, default=str)

    print("\n=== Mixed-configuration frequency ===")
    for k, v in mixed.items():
        print(f"  {k}: {v['overall_fraction_mixed']:.1%} of {v['n_regimes']} regimes")
        for mech, frac in v['per_mechanism'].items():
            print(f"      {mech}: {frac:.1%}")

    print("\n=== Crossover localization (amplitude defects, spread estimator) ===")
    for mech in AMPLITUDE_MECH:
        for domain in ('SKAB', 'UCI'):
            flip = crossover[mech].get(f'{domain}_flip_between')
            print(f"  {mech} / {domain}: flip between {flip}")

    print("\n=== Per-channel spread ratio (madn/std geomean) -- Pressure check ===")
    print(per_channel.sort_values(['domain', 'spr_madn_over_std_geomean']).to_string(index=False))

    print("\n=== FINAL PER-STATISTIC VERDICTS (leave-one-mechanism-out macro, the primary split) ===")
    for statistic, scopes in overall.items():
        for scope, d in scopes.items():
            print(f"  {statistic} [{scope}]: headroom_fixed={d['headroom_fixed_pct']:.1f}%  "
                  f"headroom_intermediate={d['headroom_intermediate_pct']:.1f}%  -> {d['verdict']}")

    return result, cells, results, gate_df, mixed, oracle_map, crossover, per_channel, per_run


if __name__ == '__main__':
    main()
