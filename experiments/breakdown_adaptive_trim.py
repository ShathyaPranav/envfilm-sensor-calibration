"""Exploratory post-hoc analysis: per-mechanism breakdown of the leave-one-
mechanism-out results in evaluate_adaptive_trim.py. Re-slices the already-
materialized results/adaptive_trim/adaptive_trim_percell.csv.gz -- no new
harness sweep. Reuses the exact same train/test partitions, ATS proxy, and
LTP fitting code as the primary evaluation (imported, not reimplemented),
so the 7 held-out-mechanism numbers here are byte-identical in setup to
that run, just reported per fixed-tau instead of collapsed to "best fixed".

NOT a pre-specified comparison: run after seeing the aggregate result that
motivated the question (does ATS recover on the mechanisms where heavy
trimming is wrong?).
"""
import os
import sys

import numpy as np
import pandas as pd

_EXP_DIR = os.path.dirname(os.path.abspath(__file__))
if _EXP_DIR not in sys.path:
    sys.path.insert(0, _EXP_DIR)

from evaluate_adaptive_trim import (load_data, select_ats_proxy, fit_ats_edges, apply_ats,
                                     fit_gbt, fit_mlp, snap_to_tau, lookup_err, ERR_COLS,
                                     LTP_FEATURES, TAUS)
from run_phase2a_full import ALL_MECH

OUT_DIR = os.path.join(os.path.dirname(_EXP_DIR), 'results', 'adaptive_trim')


def main():
    df = load_data()
    ats_proxy_col, ats_sign, _ = select_ats_proxy(df)
    all_regimes = sorted(df['regime'].unique().tolist())

    rows = []
    for mech in ALL_MECH:
        held_regimes = df.loc[df['mechanism'] == mech, 'regime'].unique().tolist()
        train_regimes = [r for r in all_regimes if r not in held_regimes]
        train = df[df['regime'].isin(train_regimes)]
        test = df[df['regime'].isin(held_regimes)]
        if len(train) < 50 or len(test) < 20:
            continue

        Xtr = train[LTP_FEATURES].to_numpy()
        Xte = test[LTP_FEATURES].to_numpy()
        ytr = train['oracle_tau'].to_numpy()

        edges, order = fit_ats_edges(train[ats_proxy_col].to_numpy(), ats_sign)
        ats_pred = apply_ats(test[ats_proxy_col].to_numpy(), edges, order)
        ats_err = lookup_err(test, ats_pred).mean()

        gbt = fit_gbt(Xtr, ytr)
        gbt_pred = snap_to_tau(gbt.predict(Xte))
        gbt_err = lookup_err(test, gbt_pred).mean()

        mlp, scaler = fit_mlp(Xtr, ytr)
        mlp_pred = snap_to_tau(mlp.predict(scaler.transform(Xte)))
        mlp_err = lookup_err(test, mlp_pred).mean()

        fixed_scores_train = {tau: train[col].mean() for tau, col in ERR_COLS.items()}
        best_fixed_tau = min(fixed_scores_train, key=fixed_scores_train.get)
        best_fixed_err = test[ERR_COLS[best_fixed_tau]].mean()

        row = dict(mechanism=mech, n_train=len(train), n_test=len(test),
                   best_fixed_tau=best_fixed_tau, best_fixed_err=best_fixed_err)
        for tau, col in ERR_COLS.items():
            row[f'tau{int(tau*100):02d}_err'] = test[col].mean()
            row[f'tau{int(tau*100):02d}_pct_vs_fixed'] = 100.0 * (test[col].mean() - best_fixed_err) / best_fixed_err
        row['ats_err'] = ats_err
        row['ats_pct_vs_fixed'] = 100.0 * (ats_err - best_fixed_err) / best_fixed_err
        row['gbt_err'] = gbt_err
        row['gbt_pct_vs_fixed'] = 100.0 * (gbt_err - best_fixed_err) / best_fixed_err
        row['mlp_err'] = mlp_err
        row['mlp_pct_vs_fixed'] = 100.0 * (mlp_err - best_fixed_err) / best_fixed_err
        row['oracle_err'] = test['oracle_err'].mean()
        row['oracle_pct_vs_fixed'] = 100.0 * (row['oracle_err'] - best_fixed_err) / best_fixed_err
        rows.append(row)

        print(f"[{mech}] n_test={len(test)} best_fixed=tau{best_fixed_tau:.2f} err={best_fixed_err:.5g}")
        for tau in TAUS:
            print(f"    tau={tau:.2f}: {row[f'tau{int(tau*100):02d}_pct_vs_fixed']:+7.1f}%")
        print(f"    ATS: {row['ats_pct_vs_fixed']:+7.1f}%   GBT: {row['gbt_pct_vs_fixed']:+7.1f}%   "
              f"MLP: {row['mlp_pct_vs_fixed']:+7.1f}%   oracle: {row['oracle_pct_vs_fixed']:+7.1f}%")

    out = pd.DataFrame(rows)
    out_path = os.path.join(OUT_DIR, 'adaptive_trim_per_mechanism.csv')
    out.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")


if __name__ == '__main__':
    main()
