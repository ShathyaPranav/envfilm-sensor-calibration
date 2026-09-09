"""Calibration-contamination stress test of SKAB's leaderboard-winning Conv-AE.

THE GAP THIS EXPLOITS. SKAB's protocol fits every detector's reference
statistics on the first 400 rows of each experiment. We measured those rows:
33 of 34 are exactly 0% anomalous (median 0.0000). The benchmark therefore
hands every published method a pristine calibration window, and its reported
scores are conditional on that. A deployed monitor gets no such guarantee --
calibration windows are collected from live streams that may already be
degraded, which is the whole premise of this paper.

THE EXPERIMENT. Contaminate ONLY the calibration segment, at controlled rates,
using this paper's own injection harness. The test segment is untouched, so the
evaluation target is identical across conditions and the scores stay directly
comparable to the published ones. Then compare two input representations inside
an otherwise identical Conv-AE:

    standard : mean + standard deviation      (what published Conv-AE uses)
    robust   : median + consistency-corrected trimmed scale   (this paper's)

Contamination is drawn with the same seed for both representations within each
(file, rate), so the two models see byte-identical corrupted calibration data.
The architecture, optimizer, sequence length, UCL rule and post-processing are
unchanged from the replication, which reproduced the published Conv-AE to
F1 0.7824 / FAR 13.63 / MAR 28.18 against the published 0.78 / 13.55 / 28.02.

PREDICTION, stated before running: at 0% both should match the published score;
as calibration contamination rises the mean/std representation should degrade
faster than the robust one, because an outlier-inflated sigma shrinks every
subsequent z-score and pushes the reconstruction error and its UCL together.
If that does not happen, it is reported as a negative result.
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

_EXP_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_EXP_DIR)
for p in (_EXP_DIR, _REPO_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from skab_robust_detector import load_files, EPS
from skab_conv_ae import fit_predict_file
from harness.harness1c import inject
from sklearn.metrics import roc_auc_score

OUT_DIR = os.path.join(_REPO_ROOT, 'results', 'skab_detector')
RATES = [float(x) for x in os.environ.get('RATES', '0.0,0.05,0.10,0.20').split(',')]
MECH = os.environ.get('MECH', 'outlier')
MAGNITUDE = 4.0


def contaminate(Xtr, rate, seed):
    """Inject into the calibration segment only, per channel, with a fixed seed
    so both representations receive identical corruption."""
    if rate <= 0:
        return Xtr
    a = Xtr.to_numpy(float).copy()
    for j in range(a.shape[1]):
        rng = np.random.default_rng(seed * 1000 + j)
        kw = {'magnitude': MAGNITUDE} if MECH in ('outlier', 'spike_burst') else {}
        a[:, j] = inject(a[:, j], MECH, rate, rng, **kw)
    return pd.DataFrame(a, index=Xtr.index, columns=Xtr.columns)


def run(mode, rate, seed, files):
    TP = TN = FP = FN = 0
    S, Y = [], []
    for fi, (name, Xtr, Xte, yte) in enumerate(files):
        Xc = contaminate(Xtr, rate, seed * 97 + fi)
        out = fit_predict_file(Xc, Xte, mode, seed)
        if out is None:
            continue
        pred, score = out
        t = yte.to_numpy().astype(int)
        TP += int(((pred == 1) & (t == 1)).sum()); TN += int(((pred == 0) & (t == 0)).sum())
        FP += int(((pred == 1) & (t == 0)).sum()); FN += int(((pred == 0) & (t == 1)).sum())
        S.append((score - np.nanmedian(score)) / max(np.nanstd(score), EPS)); Y.append(t)
    f1 = TP / (TP + (FN + FP) / 2) if (TP + FN + FP) else 0.0
    s, y = np.concatenate(S), np.concatenate(Y)
    ok = np.isfinite(s)
    return dict(mode=mode, rate=rate, seed=seed, f1=round(f1, 4),
                auc=round(roc_auc_score(y[ok], s[ok]), 4),
                far=round(FP / (FP + TN) * 100, 2) if (FP + TN) else 0.0,
                mar=round(FN / (FN + TP) * 100, 2) if (FN + TP) else 0.0)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.time()
    files = load_files()
    seeds = [int(x) for x in os.environ.get('SEEDS', '0').split(',')]
    print(f"{len(files)} files | calibration contamination = {MECH} @ {MAGNITUDE} sigma "
          f"| rates {RATES} | seeds {seeds}")
    rows = []
    for seed in seeds:
        for rate in RATES:
            for mode in ('standard', 'robust'):
                r = run(mode, rate, seed, files)
                rows.append(r)
                print(f"  seed={seed} rate={rate:<5} {mode:9s} F1={r['f1']:.4f} "
                      f"AUC={r['auc']:.4f} FAR={r['far']:5.2f} MAR={r['mar']:5.2f} "
                      f"[{time.time()-t0:.0f}s]")
                pd.DataFrame(rows).to_csv(
                    os.path.join(OUT_DIR, 'conv_ae_contaminated.csv'), index=False)

    df = pd.DataFrame(rows)
    print("\n=== F1 vs CALIBRATION CONTAMINATION RATE (mean over seeds) ===")
    piv = df.groupby(['rate', 'mode'])['f1'].mean().unstack()
    piv['robust_minus_standard'] = piv['robust'] - piv['standard']
    print(piv.round(4).to_string())
    print("\n=== AUC ===")
    pa = df.groupby(['rate', 'mode'])['auc'].mean().unstack()
    pa['robust_minus_standard'] = pa['robust'] - pa['standard']
    print(pa.round(4).to_string())
    print("\nPublished Conv-AE (clean calibration): F1 0.78 | our replication 0.7824")
    print("Always-alarm baseline:                 F1 0.6984")
    print(f"\nElapsed {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()
