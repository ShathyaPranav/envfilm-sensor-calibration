"""RPF: a robust point-level pre-filter that restores a windowed neural
detector's performance under contaminated calibration.

THE CHAIN OF REASONING THIS IMPLEMENTS, each step measured rather than assumed:

  1. SKAB's calibration windows are clean -- 33 of 34 files are exactly 0%
     anomalous. Every published score is conditional on that, and deployment
     does not guarantee it.
  2. Contaminating ONLY the calibration window collapses the top-ranked
     Conv-AE: F1 0.782 -> 0.721 -> 0.628 -> 0.499 at 0/5/10/20%. By 10% it is
     below a trivial always-alarm detector (0.698).
  3. AUC is unchanged throughout (0.828 -> 0.827). The encoder is fine; the
     control limit is what breaks.
  4. Robustifying the input scaler does nothing (~0 at every rate) and a robust
     control limit recovers only ~0.005, BECAUSE windowing amplifies
     contamination: a 5% point rate contaminates 95.4% of length-60 sequences,
     99.8% at 10%. That is past the breakdown point of every robust estimator,
     so sequence-level robustness cannot work in principle.
  5. Therefore filter at the POINT level, before windowing, where the
     contaminated fraction is still 5-20% and robust estimation is valid.

RPF winsorizes each calibration channel at median +/- C * s_corrected, using
the consistency-corrected trimmed scale this paper identifies. Contaminated
points are pulled back to the bulk of the distribution rather than dropped, so
sequence structure and length are preserved exactly.

CONTROLS. The autoencoder, optimizer, sequence length, UCL rule, scaler and
post-processing are all unchanged from the replication. Test data is never
touched. C is fixed a priori at 3.0 (a conventional 3-sigma envelope), not
selected against test performance -- a sensitivity sweep over C is reported
separately so the choice is visible rather than hidden.
"""
import os
import sys
import time

import numpy as np
import pandas as pd

_EXP_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_EXP_DIR)
for p in (_EXP_DIR, _REPO_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from skab_robust_detector import load_files, SCALES, EPS
from skab_conv_ae import fit_predict_file
from skab_conv_ae_contaminated import contaminate
from sklearn.metrics import roc_auc_score

OUT_DIR = os.path.join(_REPO_ROOT, 'results', 'skab_detector')
RATES = [float(x) for x in os.environ.get('RATES', '0.0,0.05,0.10,0.20').split(',')]
C_ENV = os.environ.get('CVALS', '3.0')
CVALS = [float(x) for x in C_ENV.split(',')]


def robust_prefilter(Xtr, C):
    """Winsorize each channel at median +/- C * consistency-corrected trimmed
    scale, estimated on the calibration segment itself. Point-level, so the
    contaminated fraction stays within the estimator's breakdown point."""
    a = Xtr.to_numpy(float).copy()
    med = np.nanmedian(a, axis=0)
    s = np.maximum(SCALES['trimmed_c'](a), EPS)
    lo, hi = med - C * s, med + C * s
    a = np.clip(a, lo, hi)
    return pd.DataFrame(a, index=Xtr.index, columns=Xtr.columns)


def run(rate, seed, files, C, use_filter):
    TP = TN = FP = FN = 0
    S, Y = [], []
    for fi, (name, Xtr, Xte, yte) in enumerate(files):
        Xc = contaminate(Xtr, rate, seed * 97 + fi)      # same seed as the stress test
        if use_filter:
            Xc = robust_prefilter(Xc, C)
        out = fit_predict_file(Xc, Xte, 'standard', seed)  # PUBLISHED scaler kept
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
    return dict(rate=rate, seed=seed, C=C if use_filter else None,
                method='RPF' if use_filter else 'published',
                f1=round(f1, 4), auc=round(roc_auc_score(y[ok], s[ok]), 4),
                far=round(FP / (FP + TN) * 100, 2) if (FP + TN) else 0.0,
                mar=round(FN / (FN + TP) * 100, 2) if (FN + TP) else 0.0)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.time()
    files = load_files()
    seeds = [int(x) for x in os.environ.get('SEEDS', '0').split(',')]
    print(f"{len(files)} files | rates {RATES} | C {CVALS} | seeds {seeds}")
    rows = []
    for seed in seeds:
        for rate in RATES:
            for C in CVALS:
                r = run(rate, seed, files, C, use_filter=True)
                rows.append(r)
                print(f"  seed={seed} rate={rate:<5} RPF(C={C}) F1={r['f1']:.4f} "
                      f"AUC={r['auc']:.4f} FAR={r['far']:5.2f} MAR={r['mar']:5.2f} "
                      f"[{time.time()-t0:.0f}s]")
                pd.DataFrame(rows).to_csv(
                    os.path.join(OUT_DIR, 'prefilter_results.csv'), index=False)
    df = pd.DataFrame(rows)
    print("\n=== RPF F1 by contamination rate (C=3.0) ===")
    base = {0.0: 0.7824, 0.05: 0.7208, 0.10: 0.6280, 0.20: 0.4989}
    sub = df[df.C == 3.0].groupby('rate')['f1'].mean()
    for rate, v in sub.items():
        b = base.get(rate, float('nan'))
        print(f"  rate={rate:<5} published={b:.4f}   RPF={v:.4f}   gain={v-b:+.4f}")
    print("\nPublished Conv-AE (clean calibration): 0.78 | replication 0.7824")
    print("Always-alarm baseline:                 0.6984")
    print(f"\nElapsed {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()
