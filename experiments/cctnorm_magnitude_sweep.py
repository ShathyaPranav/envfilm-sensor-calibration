"""Where does CCT-Norm's alpha-tracking hold? A controlled magnitude sweep.

WHY THIS EXISTS. On INJECTED 4-sigma contamination, the corrected trim fraction
tracked the contamination rate (alpha* = 1.32 eps + 0.013, MAE 0.0035). On Intel
Lab's REAL faults it did not move at all (alpha* 0.095-0.105 from a 0.10 init).
The real faults are saturation at ~100 sigma.

HYPOTHESIS UNDER TEST. Tracking comes from an interaction: the reconstruction loss is
scale-invariant, but the point-level winsorizer (kappa = 4) is not. With outliers
near the knee, changing alpha changes what gets clipped, so the loss can feel alpha.
Far past the knee, tanh is flat, the winsorizer's output no longer depends on alpha,
and alpha becomes unidentifiable. If so, tracking should be strong at small
magnitudes and vanish as magnitude grows -- and removing the winsorizer should change
that pattern.

DESIGN. Real clean Intel Lab backgrounds (healthy voltage, time-ordered), with
contamination injected at a CONTROLLED magnitude and rate through the paper's own
harness. This is deliberately a controlled experiment, not a claim about organic
faults: it characterizes the operating range of the mechanism, the way the paper's
severity grids characterize estimators. Every combination of {consistency correction
on/off} x {winsorizer on/off} is run so the explanation can be confirmed or refuted
rather than asserted.
"""
import math
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
import torch

DQ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (DQ, os.path.join(DQ, 'experiments')):
    if p not in sys.path:
        sys.path.insert(0, p)
from cct_norm import CCTNorm, winsorize
from run_cctnorm_mechanism import TinyAE, recon_loss
from harness.harness1c import inject

torch.set_num_threads(max(1, (os.cpu_count() or 4) - 2))
W = 60
MAGS = [2, 4, 8, 16, 32, 100]
RATES = [0.0, 0.05, 0.10, 0.20]
EPOCHS = int(os.environ.get('EPOCHS', '60'))
OUT = os.path.join(DQ, 'results', 'cctnorm')


def backgrounds(n=12):
    d = pickle.load(open(os.path.join(DQ, 'data', 'intel_lab', 'intel_parsed_ts.pkl'), 'rb'))
    segs = []
    for m, g in d['df'][d['df'].volt >= 2.5].sort_values(['moteid', 'epoch_c']).groupby('moteid'):
        v = g.temp.to_numpy().astype(float)
        if len(v) >= 1200:
            segs.append(v[400:800])
        if len(segs) >= n:
            break
    return segs


def train_condition(segs, mag, rate, wins, correct, seed=0):
    torch.manual_seed(seed)
    norm = CCTNorm(1, correct=correct, per_channel=False, init_alpha=0.10)
    ae = TinyAE(1)
    opt = torch.optim.Adam(list(norm.parameters()) + list(ae.parameters()), lr=3e-3)
    log4 = torch.zeros(1) + math.log(4.0)
    data = []
    for i, x in enumerate(segs):
        # inject() forces at least one corrupted point even at rate 0, so skip it for a truly clean arm
        xb = inject(x, 'outlier', rate, np.random.default_rng(1000 + i), magnitude=mag) if rate > 0 else x.copy()
        data.append(torch.tensor(xb, dtype=torch.float32).unsqueeze(1))
    for _ in range(EPOCHS):
        for C in data:
            mu, s = norm(C[::2])
            r = (C - mu) / s
            if wins:
                r = winsorize(r, log4)
            win = r.squeeze(1).unfold(0, W, 1).unsqueeze(1)
            loss = recon_loss(ae(win), win, 'relative')
            opt.zero_grad(); loss.backward(); opt.step()
    return float(norm.alphas()[1].mean())


def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    segs = backgrounds()
    print(f"{len(segs)} real clean backgrounds; magnitudes {MAGS}; rates {RATES}; epochs {EPOCHS}")
    rows = []
    for mag in MAGS:
        for wins in (True, False):
            for correct in (True, False):
                for rate in RATES:
                    a = train_condition(segs, mag, rate, wins, correct)
                    rows.append(dict(magnitude=mag, winsor=wins, corrected=correct, rate=rate, alpha=a))
                pd.DataFrame(rows).to_csv(os.path.join(OUT, 'magnitude_sweep.csv'), index=False)
                print(f"  mag={mag:>3} winsor={int(wins)} corrected={int(correct)}  "
                      f"alpha* at rates {RATES}: "
                      + ' '.join(f"{r['alpha']:.4f}" for r in rows[-len(RATES):])
                      + f"   [{time.time()-t0:.0f}s]")
    df = pd.DataFrame(rows)
    print("\n=== slope of alpha* against true contamination rate (tracking strength) ===")
    print(f"{'magnitude':>10s}{'winsor+corr':>14s}{'winsor+uncorr':>15s}{'NOwins+corr':>14s}{'NOwins+uncorr':>15s}")
    for mag in MAGS:
        cells = []
        for wins, correct in ((True, True), (True, False), (False, True), (False, False)):
            g = df[(df.magnitude == mag) & (df.winsor == wins) & (df.corrected == correct)].sort_values('rate')
            cells.append(np.polyfit(g.rate, g.alpha, 1)[0])
        print(f"{mag:>10}" + ''.join(f"{c:>14.3f}" for c in cells))
    print("\ninjected-4-sigma reference from the original experiment: slope 1.319")
    print(f"elapsed {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()
