"""Part B, second iteration: evaluated on CONSISTENCY-CORRECTED trim errors.

Adds one method to the comparison. The first-iteration ATS mapped a single
proxy monotonically onto tau. That cannot express the structure the corrected
data actually has: amplitude-perturbed windows want heavy trimming, while
structurally-degraded windows (stale, missing, duplicated, reordered records)
want none -- and structural windows ALSO open a trimmed-vs-untrimmed gap, so a
one-signal monotone rule trims hardest exactly where it should not trim at all.

ATS2 therefore gates on two signals, both computable from the dirty window:

    A = trimmed-vs-untrimmed spread gap        (amplitude / tail-excess evidence)
    B = max percentile-rank over
        {nan_frac, flat_run, 1 - uniq_ratio}   (structural-degradation evidence)

    if B > b_threshold:  tau = 0        (structural regime: do not trim)
    else:                tau = monotone_quantile_map(A)

Every threshold, bin edge and the b_threshold grid choice is fit on TRAINING
rows only, by training error. Nothing is selected against a test split.
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats as spstats

_EXP_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_EXP_DIR)
for p in (_EXP_DIR, _REPO_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from evaluate_adaptive_trim import (snap_to_tau, lookup_err, ERR_COLS, PROXY_COLS,
                                     LTP_FEATURES, TAUS, fit_gbt, fit_mlp,
                                     fit_ats_edges, apply_ats, build_splits)

DATA_DIR = os.path.join(_REPO_ROOT, 'results', 'adaptive_trim')
STRUCT_COLS = ['nan_frac', 'flat_run', 'uniq_ratio']
B_GRID = [0.50, 0.60, 0.70, 0.80, 0.90, 1.01]   # 1.01 => gate never fires (reduces to ATS)


def load(path):
    df = pd.read_csv(path)
    df = df.replace([np.inf, -np.inf], np.nan)
    needed = LTP_FEATURES + list(ERR_COLS.values()) + ['oracle_tau', 'err_std', 'err_madn', 'err_iqr']
    before = len(df)
    df = df.dropna(subset=needed)
    print(f"  {path.split(os.sep)[-1]}: {before} rows, {len(df)} usable")
    return df


def select_proxy(df):
    true_pert = df['err_std'] / df['channel_sigma']
    corr = {c: spstats.spearmanr(df[c], true_pert)[0] for c in PROXY_COLS}
    best = max(corr, key=lambda c: abs(corr[c]))
    print("  ATS proxy selection (Spearman vs true normalized perturbation):")
    for c, r in corr.items():
        print(f"    {c}: {r:+.4f}{'  <== selected' if c == best else ''}")
    return best, (1 if corr[best] >= 0 else -1)


def _rank_ref(train):
    """Sorted reference arrays for percentile-rank mapping, from TRAIN only."""
    ref = {}
    for c in STRUCT_COLS:
        v = train[c].to_numpy()
        ref[c] = np.sort(v[np.isfinite(v)])
    return ref


def _struct_score(df, ref):
    """B = max percentile rank across the structural-degradation features."""
    ranks = []
    for c in STRUCT_COLS:
        v = df[c].to_numpy()
        r = np.searchsorted(ref[c], v, side='right') / max(len(ref[c]), 1)
        if c == 'uniq_ratio':
            r = 1.0 - r          # LOW unique-value ratio is the structural signal
        ranks.append(r)
    return np.maximum.reduce(ranks)


def fit_ats2(train, proxy_col, sign):
    ref = _rank_ref(train)
    edges, order = fit_ats_edges(train[proxy_col].to_numpy(), sign)
    B_train = _struct_score(train, ref)
    A_tau = apply_ats(train[proxy_col].to_numpy(), edges, order)

    best_b, best_err = None, np.inf
    for b in B_GRID:
        tau = np.where(B_train > b, 0.0, A_tau)
        err = lookup_err(train, tau).mean()
        if err < best_err:
            best_err, best_b = err, b
    return dict(ref=ref, edges=edges, order=order, b_threshold=best_b, proxy_col=proxy_col)


def apply_ats2(df, p):
    B = _struct_score(df, p['ref'])
    A_tau = apply_ats(df[p['proxy_col']].to_numpy(), p['edges'], p['order'])
    return np.where(B > p['b_threshold'], 0.0, A_tau)


def evaluate(df, tag):
    proxy_col, sign = select_proxy(df)
    splits = build_splits(df)
    rows = []
    for sp in splits:
        train, test = df[sp['train_mask']], df[sp['test_mask']]
        if len(train) < 50 or len(test) < 20:
            continue
        Xtr, Xte, ytr = train[LTP_FEATURES].to_numpy(), test[LTP_FEATURES].to_numpy(), train['oracle_tau'].to_numpy()

        edges, order = fit_ats_edges(train[proxy_col].to_numpy(), sign)
        ats_err = lookup_err(test, apply_ats(test[proxy_col].to_numpy(), edges, order)).mean()

        p2 = fit_ats2(train, proxy_col, sign)
        ats2_err = lookup_err(test, apply_ats2(test, p2)).mean()

        gbt = fit_gbt(Xtr, ytr)
        gbt_err = lookup_err(test, snap_to_tau(gbt.predict(Xte))).mean()
        mlp, sc = fit_mlp(Xtr, ytr)
        mlp_err = lookup_err(test, snap_to_tau(mlp.predict(sc.transform(Xte)))).mean()

        fixed_train = {t: train[c].mean() for t, c in ERR_COLS.items()}
        bf = min(fixed_train, key=fixed_train.get)
        bf_err = test[ERR_COLS[bf]].mean()

        row = dict(setting=tag, split=sp['split'], held_out=sp['held_out'],
                   n_train=len(train), n_test=len(test), best_fixed_tau=bf,
                   best_fixed_err=bf_err, b_threshold=p2['b_threshold'])
        for name, val in [('std', test['err_std'].mean()), ('madn', test['err_madn'].mean()),
                          ('iqr', test['err_iqr'].mean()), ('ats', ats_err), ('ats2', ats2_err),
                          ('gbt', gbt_err), ('mlp', mlp_err), ('oracle', test['oracle_err'].mean())]:
            row[f'{name}_pct_vs_fixed'] = 100.0 * (val - bf_err) / bf_err
        rows.append(row)
        print(f"  [{sp['split']}/{sp['held_out']}] fixed=tau{bf:.2f} b*={p2['b_threshold']:.2f} | "
              f"ats={row['ats_pct_vs_fixed']:+7.1f}% ats2={row['ats2_pct_vs_fixed']:+7.1f}% "
              f"gbt={row['gbt_pct_vs_fixed']:+7.1f}% mlp={row['mlp_pct_vs_fixed']:+7.1f}% "
              f"oracle={row['oracle_pct_vs_fixed']:+7.1f}%")
    return pd.DataFrame(rows)


def main():
    t0 = time.time()
    out = []
    for tag, fname in [('corrected', 'adaptive_trim_percell_corrected.csv.gz'),
                       ('uncorrected', 'adaptive_trim_percell.csv.gz')]:
        path = os.path.join(DATA_DIR, fname)
        if not os.path.exists(path):
            continue
        print(f"\n{'='*78}\n=== {tag} ===")
        df = load(path)
        out.append(evaluate(df, tag))

    res = pd.concat(out, ignore_index=True)
    res.to_csv(os.path.join(DATA_DIR, 'adaptive_trim_results_v2.csv'), index=False)

    print(f"\n{'='*78}\n=== MEAN %% vs best-fixed-tau (negative = better) ===")
    methods = ['std', 'madn', 'iqr', 'ats', 'ats2', 'gbt', 'mlp', 'oracle']
    for tag in res['setting'].unique():
        d = res[res.setting == tag]
        print(f"\n  [{tag}]  ({len(d)} splits)")
        for m in methods:
            col = f'{m}_pct_vs_fixed'
            print(f"    {m:7s} {d[col].mean():+9.2f}%   beats fixed in {int((d[col] < 0).sum())}/{len(d)} splits")
    print(f"\nElapsed {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()
