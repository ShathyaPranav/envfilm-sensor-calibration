"""Path B evaluation: can the contamination regime be inferred online, and
does knowing it actually buy anything?

Two questions, in order.

Q1 (does it work): classify isolated vs ambient from a trailing sequence of
windows, on held-out splits. Baselines that must be beaten before any claim is
made:
    chance          -- majority class
    mean-level rule -- a single threshold on mean per-window perturbation,
                       calibrated on TRAIN. This is the honest "you didn't need
                       machine learning" baseline. If GBT/MLP cannot beat it,
                       the structural features have not earned their place and
                       that is the finding.
    structure-free  -- GBT restricted to level features only (no gini, cv,
                       autocorrelation, run-length), isolating how much the
                       temporal-concentration features contribute.

Q2 (does it matter): RQ2 established that the correct spread estimator reverses
between regimes. A regime-aware policy applies that. It is scored against the
best single regime-agnostic estimator, and against an oracle that always knows
the true regime -- so the cost of classifier errors is visible rather than
assumed away.

Splits mirror RQ3's protocol: leave-one-domain-out and leave-one-mechanism-out.
Nothing is fit on test rows.
"""
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

from run_regime_classifier import SHARED_MECH

SEQ_LEN = os.environ.get('SEQ_LEN', '200')
VERSION = os.environ.get('FEATS', 'v1')
DATA = os.path.join(_REPO_ROOT, 'results', 'regime', f'regime_sequences_k{SEQ_LEN}_{VERSION}.csv.gz')
OUT_DIR = os.path.join(_REPO_ROOT, 'results', 'regime')

META = {'domain', 'run', 'channel', 'mechanism', 'regime', 'rate', 'seed', 'seq_start'}
LEVEL_SUFFIXES = ('_mean', '_std')                       # magnitude-only
STRUCT_SUFFIXES = ('_cv', '_gini', '_ac1', '_run_above_med', '_max_over_mean', '_q90_over_med')


def load():
    df = pd.read_csv(DATA)
    df = df.replace([np.inf, -np.inf], np.nan)
    feats = [c for c in df.columns if c not in META]
    df = df.dropna(subset=feats)
    print(f"{len(df)} sequences; regimes {df['regime'].value_counts().to_dict()}")
    return df, feats


def splits(df):
    out = []
    for dom in sorted(df['domain'].unique()):
        out.append((f'leave_domain_{dom}', df['domain'] != dom, df['domain'] == dom))
    for mech in SHARED_MECH:
        if mech in set(df['mechanism']):
            out.append((f'leave_mech_{mech}', df['mechanism'] != mech, df['mechanism'] == mech))
    return out


def mean_level_rule(train, test, col='trim_gap_mean'):
    """Best single threshold on mean perturbation level, chosen on TRAIN."""
    v = train[col].to_numpy()
    y = (train['regime'] == 'isolated').to_numpy()
    cand = np.quantile(v, np.linspace(0.02, 0.98, 49))
    best_t, best_acc, best_dir = None, -1, 1
    for t in cand:
        for d in (1, -1):
            pred = (v * d < t * d)
            acc = (pred == y).mean()
            if acc > best_acc:
                best_acc, best_t, best_dir = acc, t, d
    pred_test = (test[col].to_numpy() * best_dir < best_t * best_dir)
    return np.where(pred_test, 'isolated', 'ambient'), best_acc


def main():
    t0 = time.time()
    df, feats = load()
    level_feats = [c for c in feats if c.endswith(LEVEL_SUFFIXES)]
    struct_feats = [c for c in feats if c.endswith(STRUCT_SUFFIXES)]
    print(f"{len(feats)} features ({len(level_feats)} level, {len(struct_feats)} structural)")

    rows = []
    for name, trm, tem in splits(df):
        train, test = df[trm], df[tem]
        if len(train) < 100 or len(test) < 50:
            continue
        ytr = train['regime'].to_numpy()
        yte = test['regime'].to_numpy()
        majority = max((yte == 'isolated').mean(), (yte == 'ambient').mean())

        rule_pred, rule_train_acc = mean_level_rule(train, test)
        rule_acc = (rule_pred == yte).mean()

        def gbt_acc(cols):
            m = HistGradientBoostingClassifier(max_depth=6, max_iter=200, learning_rate=0.08,
                                               random_state=0).fit(train[cols].to_numpy(), ytr)
            return (m.predict(test[cols].to_numpy()) == yte).mean(), m

        full_acc, gbt_model = gbt_acc(feats)
        levelonly_acc, _ = gbt_acc(level_feats)

        sc = StandardScaler().fit(train[feats].to_numpy())
        classes = sorted(set(ytr))
        code = {c: i for i, c in enumerate(classes)}
        mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=400, early_stopping=True,
                            n_iter_no_change=12, random_state=0).fit(
            sc.transform(train[feats].to_numpy()), np.array([code[v] for v in ytr]))
        mlp_pred = np.array(classes, dtype=object)[mlp.predict(sc.transform(test[feats].to_numpy()))]
        mlp_acc = (mlp_pred == yte).mean()

        rows.append(dict(split=name, n_train=len(train), n_test=len(test),
                         majority=majority, mean_level_rule=rule_acc,
                         gbt_level_only=levelonly_acc, gbt_full=full_acc, mlp_full=mlp_acc))
        print(f"  [{name:28s}] n_test={len(test):6d} majority={majority:.3f} "
              f"rule={rule_acc:.3f} gbt_level={levelonly_acc:.3f} "
              f"gbt_full={full_acc:.3f} mlp={mlp_acc:.3f}")

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT_DIR, 'regime_classifier_results.csv'), index=False)
    print("\n=== MEAN ACCURACY ACROSS SPLITS ===")
    for c in ['majority', 'mean_level_rule', 'gbt_level_only', 'gbt_full', 'mlp_full']:
        print(f"  {c:16s} {res[c].mean():.4f}")
    print(f"\n  GBT(full) beats the mean-level rule in "
          f"{int((res['gbt_full'] > res['mean_level_rule']).sum())}/{len(res)} splits")
    print(f"  GBT(full) beats GBT(level-only) in "
          f"{int((res['gbt_full'] > res['gbt_level_only']).sum())}/{len(res)} splits")
    print(f"\nElapsed {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()
