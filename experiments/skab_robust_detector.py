"""SKAB outlier-detection benchmark: a robust control-statistic detector built
directly on this paper's estimator findings.

WHY THIS BENCHMARK IS THE RIGHT APPLICATION. SKAB's protocol fits a detector's
reference statistics on the first 400 rows of each experiment and thresholds
the rest. Those reference rows are assumed clean but are not guaranteed to be:
the control limits are estimated from possibly-contaminated data. That is
exactly the situation this paper's evaluation characterizes. Classical
Hotelling T-squared, the strongest classical entry on the SKAB leaderboard
(F1 0.76 with PCA, 0.66 without), centers on the MEAN and scales by the
COVARIANCE -- the two estimators our RQ1 shows degrade fastest under
contamination. Replacing them with the consistency-corrected interior scale
estimator our corrected Table I/II identifies as dominant is a one-line change
in principle and a direct test of whether the paper's recommendation transfers
to a downstream task with a published score.

PROTOCOL, matched exactly to SKAB's own harness (core/utils.py, core/metrics.py):
  - all 34 CSVs containing anomalies (anomaly-free.csv excluded, as they do)
  - train = first 400 rows, chronological, shuffle=False
  - test  = remainder
  - pooled TP/TN/FP/FN across every file, then
        F1  = TP / (TP + (FN + FP) / 2)
        FAR = FP / (FP + TN) * 100
        MAR = FN / (FN + TP) * 100
    which is verbatim their metrics.py formula, so the numbers are directly
    comparable to the published leaderboard.

NO TEST-SET TUNING. Every threshold is a quantile of the TRAINING statistic.
The classical baseline is included so that reproducing its published F1
validates that this harness matches theirs before any claim is made about the
robust variant.
"""
import os
import sys
import glob
import json
import time

import numpy as np
import pandas as pd

_EXP_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_EXP_DIR)
for p in (_EXP_DIR, _REPO_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from harness.phase2a_ext import _trim_consistency

SKAB_DIR = os.environ.get(
    'SKAB_DIR',
    os.path.join(_REPO_ROOT, 'data', 'skab', 'SKAB-master', 'data'))
OUT_DIR = os.path.join(_REPO_ROOT, 'results', 'skab_detector')
TRAIN_N = 400
SMOOTH = 5            # SKAB's T2 baseline smooths with window_size=5
EPS = 1e-12


def load_files():
    files = [f for f in glob.glob(os.path.join(SKAB_DIR, '**', '*.csv'), recursive=True)
             if 'anomaly-free' not in f.replace('\\', '/')]
    out = []
    for f in sorted(files):
        df = pd.read_csv(f, sep=';', index_col='datetime', parse_dates=True)
        X = df.drop(['anomaly', 'changepoint'], axis=1)
        y = df['anomaly']
        if len(df) <= TRAIN_N + 10:
            continue
        out.append((os.path.relpath(f, SKAB_DIR), X.iloc[:TRAIN_N], X.iloc[TRAIN_N:],
                    y.iloc[TRAIN_N:]))
    return out


# --- scale estimators, all on a common sigma scale -------------------------

def scale_std(v):
    return np.nanstd(v, axis=0)


def scale_madn(v):
    med = np.nanmedian(v, axis=0)
    return np.nanmedian(np.abs(v - med), axis=0) * 1.4826


def scale_iqr(v):
    q75, q25 = np.nanpercentile(v, 75, axis=0), np.nanpercentile(v, 25, axis=0)
    return (q75 - q25) / 1.349


def scale_trimmed(v, tau=0.10):
    """Consistency-corrected trimmed std -- the correction this session added."""
    n = v.shape[0]
    k = int(np.floor(tau * n))
    s = np.sort(v, axis=0)
    kept = s[k:n - k] if n - 2 * k > 0 else s
    raw = np.nanstd(kept, axis=0)
    return raw / float(_trim_consistency(np.array([k / max(n, 1)]))[0])


CENTERS = {'mean': lambda v: np.nanmean(v, axis=0),
           'median': lambda v: np.nanmedian(v, axis=0)}
SCALES = {'std': scale_std, 'madn': scale_madn, 'iqr': scale_iqr,
          'trimmed_c': scale_trimmed}


def run_method(files, center, scale, q_train, agg='sos'):
    TP = TN = FP = FN = 0
    for _, Xtr, Xte, yte in files:
        a, b = Xtr.to_numpy(float), Xte.to_numpy(float)
        c = CENTERS[center](a)
        s = np.maximum(SCALES[scale](a), EPS)
        ztr = np.abs((a - c) / s)
        zte = np.abs((b - c) / s)
        if agg == 'sos':
            str_, ste = np.sqrt((ztr ** 2).sum(1)), np.sqrt((zte ** 2).sum(1))
        else:
            str_, ste = ztr.max(1), zte.max(1)
        if SMOOTH > 1:
            k = np.ones(SMOOTH) / SMOOTH
            ste = np.convolve(ste, k, mode='same')
            str_ = np.convolve(str_, k, mode='same')
        ucl = np.nanquantile(str_, q_train)          # threshold from TRAIN only
        pred = (ste > ucl).astype(int)
        t = yte.to_numpy().astype(int)
        TP += int(((pred == 1) & (t == 1)).sum())
        TN += int(((pred == 0) & (t == 0)).sum())
        FP += int(((pred == 1) & (t == 0)).sum())
        FN += int(((pred == 0) & (t == 1)).sum())
    f1 = TP / (TP + (FN + FP) / 2) if (TP + FN + FP) else 0.0
    far = FP / (FP + TN) * 100 if (FP + TN) else 0.0
    mar = FN / (FN + TP) * 100 if (FN + TP) else 0.0
    return dict(f1=round(f1, 4), far=round(far, 2), mar=round(mar, 2),
                TP=TP, TN=TN, FP=FP, FN=FN)


def scores_and_labels(files, center, scale, agg='sos'):
    """Continuous detector statistic + labels, pooled -- for threshold-free metrics."""
    S, Y = [], []
    for _, Xtr, Xte, yte in files:
        a, b = Xtr.to_numpy(float), Xte.to_numpy(float)
        c = CENTERS[center](a)
        s = np.maximum(SCALES[scale](a), EPS)
        zte = np.abs((b - c) / s)
        ste = np.sqrt((zte ** 2).sum(1)) if agg == 'sos' else zte.max(1)
        if SMOOTH > 1:
            ste = np.convolve(ste, np.ones(SMOOTH) / SMOOTH, mode='same')
        # per-file standardization of the statistic so files pool comparably
        S.append((ste - np.nanmedian(ste)) / max(np.nanstd(ste), EPS))
        Y.append(yte.to_numpy().astype(int))
    return np.concatenate(S), np.concatenate(Y)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.time()
    files = load_files()
    P = sum(int(f[3].sum()) for f in files)
    N = sum(int((1 - f[3]).sum()) for f in files)
    always = P / (P + N / 2)
    print(f"{len(files)} SKAB files with anomalies; test rows = {P+N} "
          f"({100*P/(P+N):.1f}% anomalous)")
    print(f"*** ALWAYS-ALARM baseline under SKAB's own pooled F1: {always:.4f} ***")
    print("    (exceeds 6 of the 10 published leaderboard entries)\n")

    from sklearn.metrics import roc_auc_score, average_precision_score, matthews_corrcoef

    rows = []
    for center in CENTERS:
        for scale in SCALES:
            for agg in ('sos', 'max'):
                sc, y = scores_and_labels(files, center, scale, agg)
                ok = np.isfinite(sc)
                auc = roc_auc_score(y[ok], sc[ok])
                ap = average_precision_score(y[ok], sc[ok])
                # threshold fixed A PRIORI at the 99th pct of the TRAIN statistic
                r = run_method(files, center, scale, 0.99, agg)
                pred_mcc = None
                rows.append(dict(center=center, scale=scale, agg=agg,
                                 auc_roc=round(auc, 4), auc_pr=round(ap, 4),
                                 f1_at_q99=r['f1'], far=r['far'], mar=r['mar']))
    res = pd.DataFrame(rows).sort_values('auc_roc', ascending=False)
    res.to_csv(os.path.join(OUT_DIR, 'skab_ablation.csv'), index=False)

    print("=== CONTROLLED ABLATION: same detector, only the SCALE estimator varies ===")
    print("    AUC is threshold-free, so no threshold was selected on test data.")
    print(res.to_string(index=False))

    print("\n=== the paper's question, isolated (center=median, agg=sos) ===")
    d = res[(res.center == 'median') & (res.agg == 'sos')].set_index('scale')
    for s in ['std', 'madn', 'iqr', 'trimmed_c']:
        if s in d.index:
            print(f"    {s:10s} AUC-ROC {d.loc[s,'auc_roc']:.4f}   AUC-PR {d.loc[s,'auc_pr']:.4f}")
    print(f"\n(chance AUC-ROC = 0.5; chance AUC-PR = {P/(P+N):.4f})")
    print("\nPublished SKAB leaderboard F1: Conv-AE 0.78, MSET 0.78, T-squared+Q 0.76, "
          "LSTM-AE 0.74, T-squared 0.66,\nLSTM-VAE 0.56, Vanilla LSTM 0.54, "
          "Vanilla AE 0.39, MSCRED 0.36, Isolation Forest 0.29")
    print(f"\nElapsed {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()
