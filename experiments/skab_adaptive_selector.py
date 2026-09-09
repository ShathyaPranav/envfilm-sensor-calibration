"""ASE -- Adaptive Scale-Estimator selection for streaming anomaly detection,
evaluated on the public SKAB benchmark with leave-one-file-out generalization.

IDEA, and where it comes from. This paper's Section V-A establishes that which
spread estimator wins is governed by how much a defect perturbs the estimator
relative to the channel's own scale. A detector that fixes ONE scale estimator
for every channel of every experiment ignores that. ASE instead predicts, per
channel, which scale estimator to use, from features of that channel's
TRAINING segment alone -- the segment SKAB's protocol already assumes is
available and clean.

WHY THIS IS A FAIR ML EXPERIMENT:
  - Leave-one-file-out: the meta-model never sees the held-out experiment.
  - Features come only from the training segment (first 400 rows) of the
    held-out file, which every baseline also gets.
  - The label the meta-model learns from is computed on TRAINING files only.
  - Primary metric is AUC-ROC, which is threshold-free, so no threshold is
    tuned anywhere. (SKAB's own pooled F1 is reported too, but see the
    always-alarm caveat: a detector that alarms on every row scores 0.698
    under it, so F1 differences in the 0.7-0.8 band are weak evidence.)

Baselines: every fixed scale estimator (std, MADn, IQR, consistency-corrected
trimmed), plus the oracle per-channel choice as a ceiling.
"""
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats as spstats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

_EXP_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_EXP_DIR)
for p in (_EXP_DIR, _REPO_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from skab_robust_detector import load_files, SCALES, EPS, SMOOTH

OUT_DIR = os.path.join(_REPO_ROOT, 'results', 'skab_detector')
SCALE_NAMES = ['std', 'madn', 'iqr', 'trimmed_c']


def channel_features(a_col):
    """Descriptors of one channel's TRAINING segment. No test data, no labels."""
    v = a_col[np.isfinite(a_col)]
    if v.size < 10:
        return dict(kurt=0., skew=0., trim_gap=0., iqr_over_std=1., madn_over_std=1.,
                    uniq=0., ac1=0., rough=0.)
    sd = max(np.std(v), EPS)
    med = np.median(v)
    madn = np.median(np.abs(v - med)) * 1.4826
    q75, q25 = np.percentile(v, [75, 25])
    iqr = (q75 - q25) / 1.349
    k = int(np.floor(0.10 * len(v)))
    trimmed = np.sort(v)[k:len(v) - k] if len(v) - 2 * k > 0 else v
    d = np.diff(v)
    return dict(
        kurt=float(spstats.kurtosis(v)), skew=float(spstats.skew(v)),
        trim_gap=float((sd - np.std(trimmed)) / sd),
        iqr_over_std=float(iqr / sd), madn_over_std=float(madn / sd),
        uniq=float(len(np.unique(v)) / len(v)),
        ac1=float(np.corrcoef(v[:-1], v[1:])[0, 1]) if len(v) > 2 else 0.,
        rough=float(np.mean(np.abs(d)) / sd) if len(d) else 0.,
    )


def channel_auc(a_col, b_col, y, scale_name):
    """Univariate detection AUC for one channel under one scale estimator."""
    c = np.nanmedian(a_col)
    s = max(float(SCALES[scale_name](a_col[:, None])[0]), EPS)
    z = np.abs((b_col - c) / s)
    z = np.convolve(z, np.ones(SMOOTH) / SMOOTH, mode='same')
    ok = np.isfinite(z)
    if len(set(y[ok])) < 2:
        return np.nan
    return roc_auc_score(y[ok], z[ok])


def multivariate_auc(files, choice_fn):
    """Pooled AUC of the multivariate detector, given a per-(file,channel)
    scale-estimator choice function."""
    S, Y = [], []
    for fi, (name, Xtr, Xte, yte) in enumerate(files):
        a, b = Xtr.to_numpy(float), Xte.to_numpy(float)
        zs = []
        for j in range(a.shape[1]):
            sn = choice_fn(fi, j)
            c = np.nanmedian(a[:, j])
            s = max(float(SCALES[sn](a[:, j][:, None])[0]), EPS)
            zs.append(np.abs((b[:, j] - c) / s))
        st = np.sqrt(np.nansum(np.square(np.array(zs)), axis=0))
        st = np.convolve(st, np.ones(SMOOTH) / SMOOTH, mode='same')
        S.append((st - np.nanmedian(st)) / max(np.nanstd(st), EPS))
        Y.append(yte.to_numpy().astype(int))
    s, y = np.concatenate(S), np.concatenate(Y)
    ok = np.isfinite(s)
    return roc_auc_score(y[ok], s[ok])


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.time()
    files = load_files()
    n_files = len(files)
    n_ch = files[0][1].shape[1]
    print(f"{n_files} files x {n_ch} channels")

    # --- per-(file,channel) features and per-scale univariate AUC ---
    feats, aucs = [], []
    for fi, (name, Xtr, Xte, yte) in enumerate(files):
        a, b, y = Xtr.to_numpy(float), Xte.to_numpy(float), yte.to_numpy().astype(int)
        for j in range(n_ch):
            f = channel_features(a[:, j])
            f.update(file_idx=fi, ch=j)
            feats.append(f)
            aucs.append([channel_auc(a[:, j], b[:, j], y, s) for s in SCALE_NAMES])
    F = pd.DataFrame(feats)
    A = pd.DataFrame(aucs, columns=SCALE_NAMES)
    A = A.fillna(0.5)
    F['best'] = A.values.argmax(axis=1)
    feat_cols = [c for c in F.columns if c not in ('file_idx', 'ch', 'best')]
    print(f"oracle per-channel best-estimator mix: "
          f"{pd.Series([SCALE_NAMES[i] for i in F['best']]).value_counts().to_dict()}")

    # --- leave-one-file-out meta-model ---
    pred_choice = np.zeros((n_files, n_ch), dtype=int)
    for fi in range(n_files):
        tr = F[F.file_idx != fi]
        te = F[F.file_idx == fi]
        m = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=0,
                                   class_weight='balanced')
        m.fit(tr[feat_cols].to_numpy(), tr['best'].to_numpy())
        p = m.predict(te[feat_cols].to_numpy())
        for row, ch in zip(p, te['ch'].to_numpy()):
            pred_choice[fi, ch] = row

    results = {}
    for s in SCALE_NAMES:
        results[f'fixed_{s}'] = multivariate_auc(files, lambda fi, j, s=s: s)
    results['ASE_leave_one_file_out'] = multivariate_auc(
        files, lambda fi, j: SCALE_NAMES[pred_choice[fi, j]])
    oracle_choice = F.set_index(['file_idx', 'ch'])['best'].to_dict()
    results['oracle_per_channel'] = multivariate_auc(
        files, lambda fi, j: SCALE_NAMES[oracle_choice[(fi, j)]])

    print("\n=== POOLED AUC-ROC on SKAB (threshold-free; chance = 0.5) ===")
    for k, v in sorted(results.items(), key=lambda kv: -kv[1]):
        print(f"  {k:28s} {v:.4f}")

    # --- file-clustered bootstrap: ASE vs best fixed ---
    best_fixed = max(SCALE_NAMES, key=lambda s: results[f'fixed_{s}'])
    print(f"\nbest fixed estimator = {best_fixed}")

    def per_file_scores(choice_fn):
        out = []
        for fi, (name, Xtr, Xte, yte) in enumerate(files):
            a, b = Xtr.to_numpy(float), Xte.to_numpy(float)
            zs = []
            for j in range(a.shape[1]):
                sn = choice_fn(fi, j)
                c = np.nanmedian(a[:, j])
                s = max(float(SCALES[sn](a[:, j][:, None])[0]), EPS)
                zs.append(np.abs((b[:, j] - c) / s))
            st = np.sqrt(np.nansum(np.square(np.array(zs)), axis=0))
            st = np.convolve(st, np.ones(SMOOTH) / SMOOTH, mode='same')
            st = (st - np.nanmedian(st)) / max(np.nanstd(st), EPS)
            out.append((st, yte.to_numpy().astype(int)))
        return out

    Aase = per_file_scores(lambda fi, j: SCALE_NAMES[pred_choice[fi, j]])
    Bfix = per_file_scores(lambda fi, j, s=best_fixed: s)

    def pooled(sel, D):
        s = np.concatenate([D[i][0] for i in sel]); y = np.concatenate([D[i][1] for i in sel])
        ok = np.isfinite(s)
        return roc_auc_score(y[ok], s[ok]) if len(set(y[ok])) > 1 else np.nan

    rng = np.random.default_rng(0)
    obs = pooled(range(n_files), Aase) - pooled(range(n_files), Bfix)
    diffs = []
    for _ in range(2000):
        sel = rng.integers(0, n_files, n_files)
        try:
            d = pooled(sel, Aase) - pooled(sel, Bfix)
            if np.isfinite(d):
                diffs.append(d)
        except Exception:
            pass
    diffs = np.array(diffs)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    print(f"ASE - fixed_{best_fixed}:  observed {obs:+.4f}   "
          f"95% CI [{lo:+.4f}, {hi:+.4f}]   favours ASE in {(diffs>0).mean():.3f} of resamples")

    pd.DataFrame([results]).to_csv(os.path.join(OUT_DIR, 'ase_results.csv'), index=False)
    print(f"\nElapsed {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()
