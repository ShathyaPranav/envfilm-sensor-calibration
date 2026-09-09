"""RS-ConvAE: a convolutional autoencoder anomaly detector whose input
representation is built from consistency-corrected robust scale estimates.

THE CLAIM BEING TESTED. SKAB's published Conv-AE (F1 0.78, the top leaderboard
entry) standardizes each channel with sklearn's StandardScaler -- the MEAN and
the STANDARD DEVIATION -- estimated from a 400-row training segment that the
protocol assumes is clean but does not guarantee to be. Those are precisely the
two estimators this paper's RQ1 shows degrade fastest under contamination. We
change ONLY that: the same architecture, the same optimizer, the same sequence
length, the same UCL rule, the same post-processing -- with the standardizer
replaced by a robust one (median centre, consistency-corrected trimmed scale).

Everything else is held fixed so that any difference is attributable to the
input representation and not to architecture, capacity, or tuning.

FAITHFULNESS. The architecture, hyperparameters (N_STEPS=60, Q=0.999, UCL*4/3,
Adam 1e-3, 100 epochs, batch 32, 10% validation split, early stopping patience
5), the residual definition, and the sequence-to-timestamp mapping are all
transcribed from SKAB's own core/Conv_AE.py and notebooks/Conv_AE.ipynb --
including their final Conv1DTranspose emitting a single channel that broadcasts
against the 8-channel input. That quirk is reproduced deliberately: it is part
of the published model that scored 0.78, and "fixing" it would break the
attribution.

Variant `standard` is therefore a REPLICATION and should land near 0.78. If it
does not, the harness is wrong and no claim should be made from variant
`robust`.
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
from sklearn.metrics import roc_auc_score

OUT_DIR = os.path.join(_REPO_ROOT, 'results', 'skab_detector')
N_STEPS = 60
Q = 0.999
EPOCHS = 100
BATCH = 32
PATIENCE = 5
VAL_FRAC = 0.1
DEV = 'cpu'


class ConvAE(nn.Module):
    """Transcription of SKAB core/Conv_AE.py into torch (NCL layout)."""

    def __init__(self, n_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_ch, 32, 7, stride=2, padding=3), nn.ReLU(), nn.Dropout(0.2),
            nn.Conv1d(32, 16, 7, stride=2, padding=3), nn.ReLU(),
            nn.ConvTranspose1d(16, 16, 7, stride=2, padding=3, output_padding=1), nn.ReLU(),
            nn.Dropout(0.2),
            nn.ConvTranspose1d(16, 32, 7, stride=2, padding=3, output_padding=1), nn.ReLU(),
            nn.ConvTranspose1d(32, 1, 7, stride=1, padding=3),
        )

    def forward(self, x):
        return self.net(x)


class LSTMAE(nn.Module):
    """A sequence-to-sequence LSTM autoencoder, used to test whether any
    calibration-contamination effect is a property of the STANDARDIZATION
    rather than of the convolutional architecture. This is our own
    implementation, not a transcription of SKAB's LSTM-AE, and is therefore
    not offered as a replication of their 0.74."""

    def __init__(self, n_ch, hidden=32):
        super().__init__()
        self.enc = nn.LSTM(n_ch, hidden, batch_first=True)
        self.dec = nn.LSTM(hidden, hidden, batch_first=True)
        self.out = nn.Linear(hidden, n_ch)

    def forward(self, x):                      # x: (N, C, L)
        z = x.permute(0, 2, 1)                 # (N, L, C)
        _, (h, _) = self.enc(z)
        rep = h[-1].unsqueeze(1).repeat(1, z.shape[1], 1)
        y, _ = self.dec(rep)
        return self.out(y).permute(0, 2, 1)    # (N, C, L)


ARCHES = {'conv': ConvAE, 'lstm': LSTMAE}
ARCH = os.environ.get('ARCH', 'conv')


def create_sequences(values, time_steps=N_STEPS):
    return np.stack([values[i:i + time_steps] for i in range(len(values) - time_steps + 1)])


def scale_params(a, mode):
    if mode == 'standard':
        return np.nanmean(a, axis=0), np.maximum(np.nanstd(a, axis=0), EPS)
    if mode == 'robust':
        return np.nanmedian(a, axis=0), np.maximum(SCALES['trimmed_c'](a), EPS)
    raise ValueError(mode)


def fit_predict_file(Xtr, Xte, mode, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    a, b = Xtr.to_numpy(float), Xte.to_numpy(float)
    c, s = scale_params(a, mode)
    A, B = (a - c) / s, (b - c) / s

    Xa = create_sequences(A)                       # (n, 60, ch)
    if len(Xa) < 20:
        return None
    xt = torch.tensor(Xa, dtype=torch.float32).permute(0, 2, 1)

    n_val = max(1, int(len(xt) * VAL_FRAC))
    xtr, xval = xt[:-n_val], xt[-n_val:]

    model = ARCHES[ARCH](xt.shape[1]).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = nn.MSELoss()
    best, bad, best_state = np.inf, 0, None
    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(len(xtr))
        for i in range(0, len(xtr), BATCH):
            xb = xtr[perm[i:i + BATCH]].to(DEV)
            opt.zero_grad()
            loss = lossf(model(xb), xb)            # 1-ch output broadcasts vs n-ch input
            loss.backward()
            opt.step()
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

    def residuals(seqs):
        t = torch.tensor(seqs, dtype=torch.float32).permute(0, 2, 1)
        outs = []
        model.eval()
        with torch.no_grad():
            for i in range(0, len(t), 256):
                xb = t[i:i + 256].to(DEV)
                outs.append(torch.abs(xb - model(xb)).mean(dim=2).sum(dim=1).cpu().numpy())
        return np.concatenate(outs)

    r_tr = residuals(Xa)
    ucl = np.quantile(r_tr, Q) * 4 / 3
    Xb = create_sequences(B)
    r_te = residuals(Xb)

    # SKAB's sequence -> timestamp mapping, transcribed verbatim
    anom = r_te > ucl
    idxs = [i for i in range(N_STEPS - 1, len(Xb) - N_STEPS + 1) if np.all(anom[i - N_STEPS + 1:i])]
    pred = np.zeros(len(b), dtype=int)
    pred[idxs] = 1

    # continuous score aligned to timestamps, for threshold-free AUC
    score = np.full(len(b), np.nan)
    score[:len(r_te)] = r_te
    score = pd.Series(score).ffill().bfill().to_numpy()
    return pred, score


def run(mode, seed, files):
    TP = TN = FP = FN = 0
    S, Y = [], []
    for name, Xtr, Xte, yte in files:
        out = fit_predict_file(Xtr, Xte, mode, seed)
        if out is None:
            continue
        pred, score = out
        t = yte.to_numpy().astype(int)
        TP += int(((pred == 1) & (t == 1)).sum()); TN += int(((pred == 0) & (t == 0)).sum())
        FP += int(((pred == 1) & (t == 0)).sum()); FN += int(((pred == 0) & (t == 1)).sum())
        sc = (score - np.nanmedian(score)) / max(np.nanstd(score), EPS)
        S.append(sc); Y.append(t)
    f1 = TP / (TP + (FN + FP) / 2) if (TP + FN + FP) else 0.0
    far = FP / (FP + TN) * 100 if (FP + TN) else 0.0
    mar = FN / (FN + TP) * 100 if (FN + TP) else 0.0
    s, y = np.concatenate(S), np.concatenate(Y)
    ok = np.isfinite(s)
    auc = roc_auc_score(y[ok], s[ok])
    return dict(mode=mode, seed=seed, f1=round(f1, 4), far=round(far, 2),
                mar=round(mar, 2), auc=round(auc, 4))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.time()
    files = load_files()
    seeds = [int(x) for x in os.environ.get('SEEDS', '0,1,2').split(',')]
    print(f"{len(files)} files, seeds={seeds}, device={DEV}")
    rows = []
    for seed in seeds:
        for mode in ('standard', 'robust'):
            r = run(mode, seed, files)
            rows.append(r)
            print(f"  seed={seed} {mode:9s} F1={r['f1']:.4f} AUC={r['auc']:.4f} "
                  f"FAR={r['far']:.2f} MAR={r['mar']:.2f}  [{time.time()-t0:.0f}s]")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, 'conv_ae_results.csv'), index=False)
    print("\n=== MEAN OVER SEEDS ===")
    print(df.groupby('mode')[['f1', 'auc', 'far', 'mar']].agg(['mean', 'std']).round(4).to_string())
    print("\nPublished Conv-AE (same architecture, StandardScaler input): F1 0.78")
    print("Always-alarm baseline under the same pooled F1:                0.6984")
    print(f"\nElapsed {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()
