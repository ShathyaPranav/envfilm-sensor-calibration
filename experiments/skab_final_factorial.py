"""RCC (Robust Calibration Correction) for windowed neural anomaly detectors:
full 2x2 factorial on SKAB, multiple seeds.

Two components, each derived from a measured failure of the published pipeline
and each targeting a different point in it:

  RPF  robust point-level pre-filter. Winsorizes the calibration segment at
       median +/- 3 * s_corrected per channel, BEFORE windowing -- because
       windowing amplifies a 5% point-contamination rate into 95.4% of
       length-60 sequences, past the breakdown point of any robust estimator,
       so sequence-level robustness cannot work and point-level must.
  RUCL robust control limit. Replaces quantile(train_residuals, 0.999)*4/3
       with median(r) + K * s_corrected(r), K fixed a priori on CLEAN
       calibration so the two rules coincide at 0% contamination.

s_corrected is the consistency-corrected trimmed scale this paper's harness
adds. The 2x2 shows each component alone and both together, against the
faithfully replicated published Conv-AE (0.7824 vs published 0.78).

Only the calibration segment is ever contaminated. Test data is untouched, so
every number stays comparable to the published leaderboard.
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

_EXP_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_EXP_DIR)
for p in (_EXP_DIR, _REPO_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from skab_robust_detector import load_files, SCALES, EPS
from skab_conv_ae import (ARCHES, ARCH, create_sequences, scale_params,
                          N_STEPS, Q, EPOCHS, BATCH, PATIENCE, VAL_FRAC, DEV)
from skab_conv_ae_contaminated import contaminate
from skab_prefilter import robust_prefilter
from sklearn.metrics import roc_auc_score

OUT_DIR = os.path.join(_REPO_ROOT, 'results', 'skab_detector')
RATES = [float(x) for x in os.environ.get('RATES', '0.0,0.05,0.10,0.20').split(',')]
SEEDS = [int(x) for x in os.environ.get('SEEDS', '0,1,2').split(',')]
C_FILTER = 3.0


def train_and_residuals(Xtr, Xte, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    a, b = Xtr.to_numpy(float), Xte.to_numpy(float)
    c, s = scale_params(a, 'standard')          # published scaler throughout
    Xa, Xb = create_sequences((a - c) / s), create_sequences((b - c) / s)
    if len(Xa) < 20:
        return None
    xt = torch.tensor(Xa, dtype=torch.float32).permute(0, 2, 1)
    n_val = max(1, int(len(xt) * VAL_FRAC))
    xtr, xval = xt[:-n_val], xt[-n_val:]
    model = ARCHES[ARCH](xt.shape[1]).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = nn.MSELoss()
    best, bad, state = np.inf, 0, None
    for _ in range(EPOCHS):
        model.train()
        perm = torch.randperm(len(xtr))
        for i in range(0, len(xtr), BATCH):
            xb_ = xtr[perm[i:i + BATCH]].to(DEV)
            opt.zero_grad(); lossf(model(xb_), xb_).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vl = lossf(model(xval.to(DEV)), xval.to(DEV)).item()
        if vl < best - 1e-6:
            best, bad, state = vl, 0, {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    if state is not None:
        model.load_state_dict(state)

    def resid(seqs):
        t = torch.tensor(seqs, dtype=torch.float32).permute(0, 2, 1)
        outs = []
        model.eval()
        with torch.no_grad():
            for i in range(0, len(t), 256):
                z = t[i:i + 256].to(DEV)
                outs.append(torch.abs(z - model(z)).mean(dim=2).sum(dim=1).cpu().numpy())
        return np.concatenate(outs)
    return resid(Xa), resid(Xb), len(b)


def predict(r_te, ucl, n_rows):
    anom = r_te > ucl
    idx = [i for i in range(N_STEPS - 1, len(r_te) - N_STEPS + 1)
           if np.all(anom[i - N_STEPS + 1:i])]
    p = np.zeros(n_rows, dtype=int); p[idx] = 1
    return p


def s_corr(r):
    return max(float(SCALES['trimmed_c'](np.asarray(r)[:, None])[0]), EPS)


def score(cache, ucl_fn):
    TP = TN = FP = FN = 0
    S, Y = [], []
    for r_tr, r_te, n_rows, t in cache:
        p = predict(r_te, ucl_fn(r_tr), n_rows)
        TP += int(((p == 1) & (t == 1)).sum()); TN += int(((p == 0) & (t == 0)).sum())
        FP += int(((p == 1) & (t == 0)).sum()); FN += int(((p == 0) & (t == 1)).sum())
        sc = np.full(n_rows, np.nan); sc[:len(r_te)] = r_te
        sc = pd.Series(sc).ffill().bfill().to_numpy()
        S.append((sc - np.nanmedian(sc)) / max(np.nanstd(sc), EPS)); Y.append(t)
    f1 = TP / (TP + (FN + FP) / 2) if (TP + FN + FP) else 0.0
    s, y = np.concatenate(S), np.concatenate(Y); ok = np.isfinite(s)
    return (round(f1, 4), round(roc_auc_score(y[ok], s[ok]), 4),
            round(FP / (FP + TN) * 100, 2) if (FP + TN) else 0.0,
            round(FN / (FN + TP) * 100, 2) if (FN + TP) else 0.0)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.time()
    files = load_files()
    print(f"{len(files)} files | rates {RATES} | seeds {SEEDS} | arch {ARCH}")

    # K fixed a priori on CLEAN, unfiltered calibration
    Ks = []
    for name, Xtr, Xte, yte in files:
        out = train_and_residuals(Xtr, Xte, SEEDS[0])
        if out is None:
            continue
        r_tr = out[0]
        Ks.append((float(np.quantile(r_tr, Q) * 4 / 3) - float(np.median(r_tr))) / s_corr(r_tr))
    K = float(np.median(Ks))
    print(f"K (fixed on clean calibration, a priori) = {K:.3f}   [{time.time()-t0:.0f}s]")

    rows = []
    for seed in SEEDS:
        for rate in RATES:
            caches = {}
            for use_rpf in (False, True):
                cache = []
                for fi, (name, Xtr, Xte, yte) in enumerate(files):
                    Xc = contaminate(Xtr, rate, seed * 97 + fi)
                    if use_rpf:
                        Xc = robust_prefilter(Xc, C_FILTER)
                    out = train_and_residuals(Xc, Xte, seed)
                    if out is None:
                        continue
                    cache.append((out[0], out[1], out[2], yte.to_numpy().astype(int)))
                caches[use_rpf] = cache
            for use_rpf in (False, True):
                for ucl_name, fn in (('quantile', lambda r: float(np.quantile(r, Q) * 4 / 3)),
                                      ('robust', lambda r: float(np.median(r)) + K * s_corr(r))):
                    f1, auc, far, mar = score(caches[use_rpf], fn)
                    method = ('RPF+' if use_rpf else '') + ('RUCL' if ucl_name == 'robust' else 'published')
                    if not use_rpf and ucl_name == 'quantile':
                        method = 'published Conv-AE'
                    rows.append(dict(seed=seed, rate=rate, rpf=use_rpf, ucl=ucl_name,
                                     method=method, f1=f1, auc=auc, far=far, mar=mar))
                    print(f"  seed={seed} rate={rate:<5} rpf={int(use_rpf)} ucl={ucl_name:8s} "
                          f"F1={f1:.4f} AUC={auc:.4f} MAR={mar:5.2f}  [{time.time()-t0:.0f}s]")
            pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, 'final_factorial.csv'), index=False)

    df = pd.DataFrame(rows)
    print("\n=== F1: mean over seeds (2x2 factorial) ===")
    piv = df.groupby(['rate', 'rpf', 'ucl'])['f1'].agg(['mean', 'std']).round(4)
    print(piv.to_string())
    print("\n=== headline: published vs both components ===")
    for rate in RATES:
        d = df[df.rate == rate]
        pub = d[(~d.rpf) & (d.ucl == 'quantile')]['f1']
        both = d[(d.rpf) & (d.ucl == 'robust')]['f1']
        print(f"  rate={rate:<5} published {pub.mean():.4f} (sd {pub.std():.4f})   "
              f"RPF+RUCL {both.mean():.4f} (sd {both.std():.4f})   "
              f"gain {both.mean()-pub.mean():+.4f}")
    print("\nPublished Conv-AE (clean): 0.78 | always-alarm: 0.6984")
    print(f"\nElapsed {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()
