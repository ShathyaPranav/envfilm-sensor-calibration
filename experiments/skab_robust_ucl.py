"""RUCL: a robust control limit for reconstruction-error anomaly detectors.

WHERE THE FRAGILITY ACTUALLY IS. The calibration stress test showed that when
the calibration window is contaminated, Conv-AE's AUC barely moves (0.828 ->
0.828 at 5% contamination) but its F1 collapses (0.782 -> 0.721) via a jump in
missed alarms (MAR 28% -> 40%). The learned representation is fine. What breaks
is the threshold: SKAB's published rule sets

    UCL = quantile(train_residuals, 0.999) * 4/3

and a 0.999 quantile of a contaminated residual distribution is set by the
contaminated points themselves. Contamination inflates the control limit, and
an inflated limit makes the detector blind.

This is the paper's own finding relocated to the place it actually bites: a
control limit is a scale estimate on the residual distribution, and RQ1 says
which scale estimators survive contamination. So:

    UCL_robust = median(r) + K * s_corrected(r)

with s_corrected the consistency-corrected trimmed scale this paper identifies.

K IS NOT TUNED ON TEST. It is fixed once, a priori, by requiring the robust
rule to AGREE with the published quantile rule on CLEAN calibration data --
so at 0% contamination the two rules coincide by construction and the published
score is preserved. Any divergence at higher contamination is therefore
attributable to robustness, not to a differently-tuned threshold.
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

from skab_robust_detector import load_files, SCALES, EPS
from skab_conv_ae import (ARCHES, ARCH, create_sequences, scale_params,
                          N_STEPS, Q, EPOCHS, BATCH, PATIENCE, VAL_FRAC, DEV)
from skab_conv_ae_contaminated import contaminate
from sklearn.metrics import roc_auc_score
import torch.nn as nn

OUT_DIR = os.path.join(_REPO_ROOT, 'results', 'skab_detector')
RATES = [float(x) for x in os.environ.get('RATES', '0.0,0.05,0.10,0.20').split(',')]


def train_residuals(Xtr, Xte, mode, seed):
    """Train the AE and return (train residuals, test residuals, n_test_rows)."""
    torch.manual_seed(seed); np.random.seed(seed)
    a, b = Xtr.to_numpy(float), Xte.to_numpy(float)
    c, s = scale_params(a, mode)
    A, B = (a - c) / s, (b - c) / s
    Xa = create_sequences(A)
    if len(Xa) < 20:
        return None
    xt = torch.tensor(Xa, dtype=torch.float32).permute(0, 2, 1)
    n_val = max(1, int(len(xt) * VAL_FRAC))
    xtr, xval = xt[:-n_val], xt[-n_val:]
    model = ARCHES[ARCH](xt.shape[1]).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = nn.MSELoss()
    best, bad, best_state = np.inf, 0, None
    for _ in range(EPOCHS):
        model.train()
        perm = torch.randperm(len(xtr))
        for i in range(0, len(xtr), BATCH):
            xb = xtr[perm[i:i + BATCH]].to(DEV)
            opt.zero_grad(); loss = lossf(model(xb), xb); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vl = lossf(model(xval.to(DEV)), xval.to(DEV)).item()
        if vl < best - 1e-6:
            best, bad = vl, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    if best_state is not None:
        model.load_state_dict(best_state)

    def resid(seqs):
        t = torch.tensor(seqs, dtype=torch.float32).permute(0, 2, 1)
        outs = []
        model.eval()
        with torch.no_grad():
            for i in range(0, len(t), 256):
                xb = t[i:i + 256].to(DEV)
                outs.append(torch.abs(xb - model(xb)).mean(dim=2).sum(dim=1).cpu().numpy())
        return np.concatenate(outs)

    return resid(Xa), resid(create_sequences(B)), len(b)


def ucl_quantile(r):
    return float(np.quantile(r, Q) * 4 / 3)


def ucl_robust(r, K):
    med = float(np.median(r))
    s = float(SCALES['trimmed_c'](np.asarray(r)[:, None])[0])
    return med + K * max(s, EPS)


def predict_from(r_te, ucl, n_rows):
    anom = r_te > ucl
    idxs = [i for i in range(N_STEPS - 1, len(r_te) - N_STEPS + 1)
            if np.all(anom[i - N_STEPS + 1:i])]
    pred = np.zeros(n_rows, dtype=int)
    pred[idxs] = 1
    return pred


def score_all(cache, ucl_fn):
    TP = TN = FP = FN = 0
    for (r_tr, r_te, n_rows, t) in cache:
        pred = predict_from(r_te, ucl_fn(r_tr), n_rows)
        TP += int(((pred == 1) & (t == 1)).sum()); TN += int(((pred == 0) & (t == 0)).sum())
        FP += int(((pred == 1) & (t == 0)).sum()); FN += int(((pred == 0) & (t == 1)).sum())
    f1 = TP / (TP + (FN + FP) / 2) if (TP + FN + FP) else 0.0
    return dict(f1=round(f1, 4),
                far=round(FP / (FP + TN) * 100, 2) if (FP + TN) else 0.0,
                mar=round(FN / (FN + TP) * 100, 2) if (FN + TP) else 0.0)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.time()
    files = load_files()
    seed = int(os.environ.get('SEED', '0'))
    mode = os.environ.get('MODE', 'standard')   # keep the PUBLISHED input scaler
    print(f"arch={ARCH} input_scaler={mode} seed={seed} rates={RATES}")

    # ---- pass 1: clean calibration, to fix K a priori ----
    clean_cache = []
    Ks = []
    for name, Xtr, Xte, yte in files:
        out = train_residuals(Xtr, Xte, mode, seed)
        if out is None:
            continue
        r_tr, r_te, n_rows = out
        clean_cache.append((r_tr, r_te, n_rows, yte.to_numpy().astype(int)))
        med = float(np.median(r_tr))
        s = max(float(SCALES['trimmed_c'](np.asarray(r_tr)[:, None])[0]), EPS)
        Ks.append((ucl_quantile(r_tr) - med) / s)
    K = float(np.median(Ks))
    print(f"K fixed on CLEAN calibration so the two rules agree at 0%: K = {K:.3f} "
          f"(across-file median; IQR {np.percentile(Ks,25):.2f}-{np.percentile(Ks,75):.2f})")
    print(f"  [pass 1 done {time.time()-t0:.0f}s]")

    rows = []
    for rate in RATES:
        if rate == 0.0:
            cache = clean_cache
        else:
            cache = []
            for fi, (name, Xtr, Xte, yte) in enumerate(files):
                Xc = contaminate(Xtr, rate, seed * 97 + fi)
                out = train_residuals(Xc, Xte, mode, seed)
                if out is None:
                    continue
                r_tr, r_te, n_rows = out
                cache.append((r_tr, r_te, n_rows, yte.to_numpy().astype(int)))
        for rule, fn in (('published_quantile', ucl_quantile),
                          ('robust_ucl', lambda r: ucl_robust(r, K))):
            r = score_all(cache, fn)
            r.update(rate=rate, rule=rule, seed=seed, arch=ARCH, scaler=mode)
            rows.append(r)
            print(f"  rate={rate:<5} {rule:19s} F1={r['f1']:.4f} FAR={r['far']:5.2f} "
                  f"MAR={r['mar']:5.2f}  [{time.time()-t0:.0f}s]")
        pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, f'robust_ucl_{ARCH}.csv'), index=False)

    df = pd.DataFrame(rows)
    print("\n=== F1 vs CALIBRATION CONTAMINATION ===")
    piv = df.pivot(index='rate', columns='rule', values='f1')
    piv['gain'] = piv['robust_ucl'] - piv['published_quantile']
    print(piv.round(4).to_string())
    print("\nPublished Conv-AE (clean calibration): F1 0.78 | our replication 0.7824")
    print("Always-alarm baseline:                 F1 0.6984")
    print(f"\nElapsed {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()
