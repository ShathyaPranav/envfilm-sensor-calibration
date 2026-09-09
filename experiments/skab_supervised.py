"""RSF -- Robust-Scale Features for cross-experiment anomaly detection on SKAB.

SETTING, stated plainly because it differs from the leaderboard's. SKAB's
published baselines are per-file unsupervised: fit on that file's first 400
rows, threshold the rest, never see another file's labels. This experiment is
CROSS-EXPERIMENT SUPERVISED: leave-one-file-out, so the model trains on 33
experiments' labels and is tested on a completely unseen experiment. That is a
different (and for a deployed fleet of identical pumps, more realistic) setting,
and the numbers are therefore NOT a like-for-like leaderboard comparison. It is
reported alongside, not as, a leaderboard result.

WHAT IS ACTUALLY NEW HERE is the feature construction, not the classifier. Each
channel is encoded as a robust z-score against reference statistics estimated
from that file's own clean training segment, using the consistency-corrected
interior scale estimator this paper identifies -- so the features are directly
downstream of the paper's estimator findings, and the ablation below swaps only
that estimator to show it matters.

Primary metric is AUC-ROC (threshold-free). The always-alarm detector scores
AUC 0.5 here, versus F1 0.698 under SKAB's pooled F1 -- which is exactly why
AUC is the honest primary metric for this benchmark.
"""
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

_EXP_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_EXP_DIR)
for p in (_EXP_DIR, _REPO_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from skab_robust_detector import load_files, SCALES, EPS

OUT_DIR = os.path.join(_REPO_ROOT, 'results', 'skab_detector')
LAGS = (1, 5, 15)


def build_features(Xtr, Xte, scale_name):
    """Robust z-scores per channel + short-horizon temporal context.
    Reference stats come ONLY from this file's own training segment."""
    a, b = Xtr.to_numpy(float), Xte.to_numpy(float)
    c = np.nanmedian(a, axis=0)
    s = np.maximum(SCALES[scale_name](a), EPS)
    z = np.abs((b - c) / s)                       # (n_test, n_ch)
    cols = [z, np.sqrt(np.nansum(z ** 2, axis=1, keepdims=True)),
            np.nanmax(z, axis=1, keepdims=True)]
    for L in LAGS:                                 # causal rolling context
        k = np.ones(L) / L
        sm = np.vstack([np.convolve(z[:, j], k, mode='full')[:len(z)]
                        for j in range(z.shape[1])]).T
        cols.append(sm)
    d = np.vstack([np.zeros((1, z.shape[1])), np.diff(z, axis=0)])
    cols.append(d)
    return np.hstack(cols)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.time()
    files = load_files()
    n = len(files)
    print(f"{n} SKAB files, leave-one-file-out cross-experiment protocol")

    results = {}
    for scale_name in ['trimmed_c', 'std', 'madn', 'iqr']:
        X = [build_features(f[1], f[2], scale_name) for f in files]
        Y = [f[3].to_numpy().astype(int) for f in files]
        scores = np.full(sum(len(y) for y in Y), np.nan)
        labels = np.concatenate(Y)
        offs = np.cumsum([0] + [len(y) for y in Y])
        for i in range(n):
            Xtr = np.vstack([X[j] for j in range(n) if j != i])
            ytr = np.concatenate([Y[j] for j in range(n) if j != i])
            m = HistGradientBoostingClassifier(max_depth=6, max_iter=200,
                                               learning_rate=0.08, random_state=0)
            m.fit(Xtr, ytr)
            scores[offs[i]:offs[i + 1]] = m.predict_proba(X[i])[:, 1]
        ok = np.isfinite(scores)
        auc = roc_auc_score(labels[ok], scores[ok])
        results[scale_name] = auc
        print(f"  RSF[{scale_name:10s}]  pooled AUC-ROC = {auc:.4f}   "
              f"[elapsed {time.time()-t0:.0f}s]")
        if scale_name == 'trimmed_c':
            best_scores, best_labels = scores.copy(), labels.copy()
            per_file = [(scores[offs[i]:offs[i + 1]], Y[i]) for i in range(n)]

    print("\n=== COMPARISON ===")
    print(f"  RSF (this work, corrected trimmed scale)   AUC {results['trimmed_c']:.4f}")
    print(f"  RSF ablation, plain std scale              AUC {results['std']:.4f}")
    print(f"  best unsupervised detector (fixed scale)   AUC 0.8188")
    print(f"  always-alarm                               AUC 0.5000")

    rng = np.random.default_rng(0)
    def pooled(sel):
        s = np.concatenate([per_file[i][0] for i in sel])
        y = np.concatenate([per_file[i][1] for i in sel])
        ok = np.isfinite(s)
        return roc_auc_score(y[ok], s[ok]) if len(set(y[ok])) > 1 else np.nan
    diffs = []
    for _ in range(1000):
        sel = rng.integers(0, n, n)
        try:
            v = pooled(sel)
            if np.isfinite(v):
                diffs.append(v)
        except Exception:
            pass
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    print(f"\n  RSF file-clustered bootstrap 95% CI for AUC: [{lo:.4f}, {hi:.4f}]")

    pd.DataFrame([results]).to_csv(os.path.join(OUT_DIR, 'rsf_results.csv'), index=False)
    print(f"\nElapsed {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()
