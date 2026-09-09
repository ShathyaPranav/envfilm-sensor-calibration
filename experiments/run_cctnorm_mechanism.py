"""The falsification test for CCT-Norm's central claim.

CLAIM: the in-layer consistency correction is what makes a trim fraction
learnable. Without it, alpha is a free activation-scale knob (measured: it moves
the emitted scale by 3.4x across its range) and gradient descent will move it to
serve the loss's scale preference, carrying no information about contamination.
With it, alpha has essentially no scale effect (measured: 0.3% across the same
range) and can only move for genuine robustness reasons.

FALSIFIABLE PREDICTION, stated before running: train the layer jointly with an
autoencoder on calibration data contaminated at rate eps.
  - CORRECTED arm:   learned alpha* should INCREASE with eps.
  - UNCORRECTED arm: learned alpha* should be driven by the loss's scale
                     preference and be roughly BLIND to eps.
If the corrected arm's alpha* does not track eps, the claim is false and this is
reported as a negative result.
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

from skab_robust_detector import load_files
from skab_conv_ae_contaminated import contaminate
from cct_norm import CCTNorm, winsorize, ALPHA_MAX

OUT_DIR = os.path.join(_REPO_ROOT, 'results', 'cctnorm')
W = 60
CDF_SUB = 2          # use every 2nd calibration point for the O(n^2) soft CDF
EPOCHS = int(os.environ.get('EPOCHS', '40'))
RATES = [float(x) for x in os.environ.get('RATES', '0.0,0.05,0.10,0.20').split(',')]
SEEDS = [int(x) for x in os.environ.get('SEEDS', '0,1,2').split(',')]
LOSS = os.environ.get('LOSS', 'mse')


class TinyAE(nn.Module):
    def __init__(self, n_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_ch, 16, 7, stride=2, padding=3), nn.ReLU(),
            nn.Conv1d(16, 8, 7, stride=2, padding=3), nn.ReLU(),
            nn.ConvTranspose1d(8, 16, 7, stride=2, padding=3, output_padding=1), nn.ReLU(),
            nn.ConvTranspose1d(16, n_ch, 7, stride=2, padding=3, output_padding=1),
        )

    def forward(self, x):
        return self.net(x)


def build(files, rate, seed):
    """Contaminated calibration slices, as tensors, one per file."""
    out = []
    for fi, (name, Xtr, Xte, yte) in enumerate(files):
        Xc = contaminate(Xtr, rate, seed * 97 + fi)
        out.append(torch.tensor(Xc.to_numpy(float), dtype=torch.float32))
    return out


def recon_loss(pred, target, mode):
    """mode='mse'      -- plain MSE. SCALE-SENSITIVE: shrinking the target
                          reduces it, so alpha can be exploited as a scale knob.
                          This is the arm that exposes the degeneracy.
       mode='relative' -- ||x-xhat||^2 / ||x||^2. SCALE-INVARIANT: shrinking the
                          representation gains nothing, so alpha can only move
                          for structural reasons. This is the fair arm for
                          asking whether alpha tracks contamination."""
    if mode == 'mse':
        return ((pred - target) ** 2).mean()
    num = ((pred - target) ** 2).flatten(1).sum(1)
    den = (target ** 2).flatten(1).sum(1) + 1e-6
    return (num / den).mean()


def run_arm(cals, correct, seed, n_ch, loss_mode):
    torch.manual_seed(seed)
    norm = CCTNorm(n_ch, correct=correct, per_channel=False, init_alpha=0.10)
    ae = TinyAE(n_ch)
    opt = torch.optim.Adam(list(norm.parameters()) + list(ae.parameters()), lr=3e-3)
    hist = []
    tot, nb = 0.0, 0
    for ep in range(EPOCHS):
        tot, nb = 0.0, 0
        for C in cals:
            m, s = norm(C[::CDF_SUB])
            r = (C - m) / s
            r = winsorize(r, torch.zeros(1) + np.log(4.0))
            if len(r) < W + 2:
                continue
            win = r.unfold(0, W, 1)                                     # (n, ch, W)
            opt.zero_grad()
            loss = recon_loss(ae(win), win, loss_mode)
            loss.backward()
            opt.step()
            tot += float(loss); nb += 1
        a_loc, a_sc = norm.alphas()
        hist.append(float(a_sc.mean()))
    a_loc, a_sc = norm.alphas()
    return float(a_sc.mean()), tot / max(nb, 1), hist


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.time()
    files = load_files()
    n_ch = files[0][1].shape[1]
    print(f"{len(files)} files, {n_ch} channels, epochs={EPOCHS}, rates={RATES}, seeds={SEEDS}, loss={LOSS}")
    print("PREDICTION: corrected alpha* rises with contamination; uncorrected alpha* does not.\n")

    rows = []
    for seed in SEEDS:
        for rate in RATES:
            cals = build(files, rate, seed)
            for correct in (True, False):
                a_star, loss, hist = run_arm(cals, correct, seed, n_ch, LOSS)
                rows.append(dict(seed=seed, rate=rate, corrected=correct, loss_mode=LOSS,
                                 alpha_star=round(a_star, 5), final_loss=round(loss, 6),
                                 alpha_traj=';'.join(f'{h:.4f}' for h in hist)))
                print(f"  seed={seed} rate={rate:<5} corrected={int(correct)}  "
                      f"alpha*={a_star:.4f}  loss={loss:.5f}  [{time.time()-t0:.0f}s]")
                pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, f'mechanism_{LOSS}.csv'), index=False)

    df = pd.DataFrame(rows)
    print("\n=== learned alpha* by contamination rate (mean +/- sd over seeds) ===")
    piv = df.groupby(['rate', 'corrected'])['alpha_star'].agg(['mean', 'std']).round(4)
    print(piv.to_string())
    for corr in (True, False):
        d = df[df.corrected == corr].groupby('rate')['alpha_star'].mean()
        rng = d.max() - d.min()
        slope = np.polyfit(d.index.values, d.values, 1)[0] if len(d) > 1 else float('nan')
        print(f"  corrected={int(corr)}: alpha* range across eps = {rng:.4f}, "
              f"slope d(alpha*)/d(eps) = {slope:+.4f}")
    print(f"\n(alpha is capped at {ALPHA_MAX}; alpha*->0 or ->cap indicates a degenerate corner)")
    print(f"\nElapsed {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()
