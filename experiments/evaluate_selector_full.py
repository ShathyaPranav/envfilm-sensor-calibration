"""Path A: per-window estimator selection over the FULL action space.

Every earlier version of this experiment could only choose a trim level. That
excluded IQR_scale, which the corrected results show is the estimator to beat
(it wins 10/10 splits against the best fixed tau). A selector that cannot emit
the winning action cannot beat it, so the earlier comparison was against a weak
baseline drawn from inside the trimmed family.

Action space here is all seven spread estimators available per window:
    tau in {0, 5, 10, 20, 30%}  +  MADn  +  IQR_scale

This comparison is only meaningful because trimmed_std is now consistency-
corrected: all seven actions estimate sigma on a common scale, so choosing
between them is a robustness decision rather than a scale decision.

BASELINE IS THE STRONG ONE: best fixed action chosen on TRAINING rows, over
all seven (in practice IQR_scale). Beating that is a real win; beating only
the best fixed tau is not, and is reported separately so the difference stays
visible.

Methods:
  PAS  -- Perturbation-Adaptive Selection. Bin windows by the observable
          perturbation proxy, use the empirically best action within each bin.
          Bin count and per-bin assignment fit on TRAINING rows only. This is
          the paper's Section V-A law (which estimator wins is governed by
          perturbation magnitude) turned into a policy.
  GBT  -- gradient-boosted trees, 7-way classification over actions.
  MLP  -- shallow MLP, same.
Actions are categorical, not ordinal, so the learned models are classifiers
here rather than the regressors used when the action was a trim level.
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

_EXP_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_EXP_DIR)
for p in (_EXP_DIR, _REPO_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from evaluate_adaptive_trim import LTP_FEATURES, PROXY_COLS, build_splits

DATA_DIR = os.path.join(_REPO_ROOT, 'results', 'adaptive_trim')
IN_PATH = os.path.join(DATA_DIR, 'adaptive_trim_percell_corrected.csv.gz')

ACTIONS = ['tau00', 'tau05', 'tau10', 'tau20', 'tau30', 'madn', 'iqr']
ACTION_ERR = {'tau00': 'err_tau00', 'tau05': 'err_tau05', 'tau10': 'err_tau10',
              'tau20': 'err_tau20', 'tau30': 'err_tau30',
              'madn': 'err_madn', 'iqr': 'err_iqr'}
TAU_ACTIONS = ['tau00', 'tau05', 'tau10', 'tau20', 'tau30']
K_GRID = [2, 3, 4, 5, 8, 12]
PROXY = 'gap_trim_untrim'


def load():
    df = pd.read_csv(IN_PATH)
    df = df.replace([np.inf, -np.inf], np.nan)
    needed = LTP_FEATURES + list(ACTION_ERR.values()) + ['channel_sigma', 'err_std']
    before = len(df)
    df = df.dropna(subset=needed)
    print(f"Loaded {before} rows, {len(df)} usable")
    E = df[[ACTION_ERR[a] for a in ACTIONS]].to_numpy()
    df['_oracle_action'] = [ACTIONS[i] for i in np.argmin(E, axis=1)]
    df['_oracle_err'] = E.min(axis=1)
    return df


def err_of(df, actions):
    """Error actually incurred by the chosen action, per row."""
    out = np.empty(len(df))
    a = np.asarray(actions)
    for name in ACTIONS:
        m = (a == name)
        if m.any():
            out[m] = df.loc[m, ACTION_ERR[name]].to_numpy()
    return out


def fit_pas(train):
    """Bin by perturbation proxy; best training action per bin. Train only."""
    best = None
    v = train[PROXY].to_numpy()
    for K in K_GRID:
        edges = np.quantile(v, np.linspace(0, 1, K + 1)[1:-1])
        b = np.digitize(v, edges)
        assign, tot = {}, 0.0
        for bi in range(K):
            m = (b == bi)
            if not m.any():
                assign[bi] = 'iqr'
                continue
            means = {a: train.loc[m, ACTION_ERR[a]].mean() for a in ACTIONS}
            assign[bi] = min(means, key=means.get)
            tot += means[assign[bi]] * m.sum()
        tot /= max(len(train), 1)
        if best is None or tot < best['train_err']:
            best = dict(K=K, edges=edges, assign=assign, train_err=tot)
    return best


def apply_pas(df, p):
    b = np.digitize(df[PROXY].to_numpy(), p['edges'])
    return np.array([p['assign'].get(int(bi), 'iqr') for bi in b])


def main():
    t0 = time.time()
    df = load()
    print(f"Oracle action mix overall: "
          f"{df['_oracle_action'].value_counts(normalize=True).round(3).to_dict()}")

    rows = []
    for sp in build_splits(df):
        train, test = df[sp['train_mask']], df[sp['test_mask']]
        if len(train) < 50 or len(test) < 20:
            continue

        # --- baselines fit on TRAIN ---
        fixed_all = {a: train[ACTION_ERR[a]].mean() for a in ACTIONS}
        best_all = min(fixed_all, key=fixed_all.get)
        fixed_tau = {a: train[ACTION_ERR[a]].mean() for a in TAU_ACTIONS}
        best_tau = min(fixed_tau, key=fixed_tau.get)
        base_err = test[ACTION_ERR[best_all]].mean()       # STRONG baseline
        tau_err = test[ACTION_ERR[best_tau]].mean()        # weak baseline, for contrast

        # --- PAS ---
        p = fit_pas(train)
        pas_err = err_of(test, apply_pas(test, p)).mean()

        # --- learned selectors (7-way classification) ---
        Xtr, Xte = train[LTP_FEATURES].to_numpy(), test[LTP_FEATURES].to_numpy()
        ytr = train['_oracle_action'].to_numpy()
        gbt = HistGradientBoostingClassifier(max_depth=6, max_iter=150,
                                             learning_rate=0.08, random_state=0).fit(Xtr, ytr)
        gbt_err = err_of(test, gbt.predict(Xte)).mean()
        sc = StandardScaler().fit(Xtr)
        # MLPClassifier's early-stopping scorer calls np.isnan on predictions,
        # which fails on string labels -- encode to ints and map back.
        classes = sorted(set(ytr))
        code = {c: i for i, c in enumerate(classes)}
        mlp = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=300, early_stopping=True,
                            n_iter_no_change=10, random_state=0).fit(
            sc.transform(Xtr), np.array([code[v] for v in ytr]))
        mlp_pred = np.array(classes, dtype=object)[mlp.predict(sc.transform(Xte))]
        mlp_err = err_of(test, mlp_pred).mean()

        row = dict(split=sp['split'], held_out=sp['held_out'], n_train=len(train), n_test=len(test),
                   best_action=best_all, best_tau_action=best_tau, K=p['K'],
                   base_err=base_err)
        for name, val in [('best_tau', tau_err), ('pas', pas_err), ('gbt', gbt_err),
                          ('mlp', mlp_err), ('oracle', test['_oracle_err'].mean())]:
            row[f'{name}_vs_best'] = 100.0 * (val - base_err) / base_err
        rows.append(row)
        print(f"  [{sp['split']}/{sp['held_out']}] baseline={best_all} K={p['K']} | "
              f"best_tau={row['best_tau_vs_best']:+7.1f}% pas={row['pas_vs_best']:+7.1f}% "
              f"gbt={row['gbt_vs_best']:+7.1f}% mlp={row['mlp_vs_best']:+7.1f}% "
              f"oracle={row['oracle_vs_best']:+7.1f}%")

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(DATA_DIR, 'selector_full_results.csv'), index=False)
    print(f"\n=== MEAN vs BEST FIXED ACTION over all 7 (negative = better) ===")
    for m in ['best_tau', 'pas', 'gbt', 'mlp', 'oracle']:
        c = f'{m}_vs_best'
        print(f"  {m:9s} {res[c].mean():+9.2f}%   beats strong baseline in "
              f"{int((res[c] < 0).sum())}/{len(res)} splits")
    print(f"\nBaseline action chosen per split: {res['best_action'].value_counts().to_dict()}")
    print(f"Elapsed {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()
