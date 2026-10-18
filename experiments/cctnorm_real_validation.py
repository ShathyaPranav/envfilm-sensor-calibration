"""Does CCT-Norm's learned trim fraction track REAL contamination?

CCT-Norm's central claim (cct_norm.py, run_cctnorm_mechanism.py): with the in-layer
consistency correction, the learned trim fraction alpha* stops being a scale knob
and becomes an unsupervised estimate of how contaminated the data is. It was shown
on INJECTED contamination (alpha* ~= 1.32 eps, MAE 0.0035; uncorrected alpha pinned
at a corner). This tests the same claim on GENUINE errors.

WHY INTEL LAB SUITS THIS, EVEN THOUGH IT IS TOO EASY FOR A DETECTOR. Its faults are
gross (a zero-parameter baseline rule reaches F1 0.986), so it cannot distinguish a
sophisticated fault DETECTOR from a threshold. But this experiment is not detection.
The 51 sensors carry real battery-driven contamination ranging 0% to 34.5%, which is
exactly the spread of true contamination levels needed to test whether an
unsupervised robustness parameter tracks them.

NON-CIRCULAR BY CONSTRUCTION:
  - alpha is learned from each sensor's temperature VALUES only -- no labels, no
    voltage, no neighbours -- through the same scale-invariant reconstruction
    objective used in the injected-contamination experiment;
  - the true contamination rate comes from the unambiguous range-violation labels,
    which the layer never sees.

PREDICTION, stated before running: corrected alpha* correlates positively with each
sensor's true fault share; uncorrected alpha* does not (it degenerates regardless).

KNOWN RISK, stated before running: real faults are almost entirely one-sided
(saturation at 122.153 C, the upper tail) while CCT-Norm trims symmetrically. Removing
an upper-tail fraction eps also discards eps of good lower-tail data, so tracking may
be imperfect. If it is, that motivates an asymmetric variant -- a real-data finding,
not a failure to hide.
"""
import math
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
import torch
from scipy import stats as spst

_EXP = os.path.dirname(os.path.abspath(__file__))
if _EXP not in sys.path:
    sys.path.insert(0, _EXP)
from cct_norm import CCTNorm, winsorize, ALPHA_MAX
from run_cctnorm_mechanism import TinyAE, recon_loss

torch.set_num_threads(max(1, (os.cpu_count() or 4) - 2))
DQ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W = 60                 # TinyAE halves twice and doubles twice, so W must be divisible by 4
N_CAL = int(os.environ.get('N_CAL', '400'))   # stride subsample for the O(n^2) soft CDF; preserves the fault mix
N_WIN = int(os.environ.get('N_WIN', '500'))   # contiguous reconstruction windows sampled across the whole stream
# Each epoch is ONE full-batch optimizer step. The first run used 40 steps, which caps how
# far Adam (lr 3e-3, ~lr per step) can move the trim logit at ~0.12 -- so alpha could shift
# by at most ~0.009 from its 0.10 init (d alpha / d logit = 0.078 there). Observed shifts
# were +/-0.005, inside that ceiling: the run never gave alpha room to move, so it was not
# a test of the mechanism. A gradient diagnostic separately ruled out the alternative
# explanation (winsorizer absorbing gross outliers): |d loss / d alpha| RISES with outlier
# magnitude and is unaffected by the winsorizer. The injected-contamination experiment that
# showed tracking took ~3,400 steps; the default here is now 1,000.
EPOCHS = int(os.environ.get('EPOCHS', '1000'))
SEEDS = [int(s) for s in os.environ.get('SEEDS', '0').split(',')]


def run_sensor(x, correct, seed):
    torch.manual_seed(seed)
    n = len(x)
    stride = max(1, n // N_CAL)
    cal = torch.tensor(x[::stride][:N_CAL], dtype=torch.float32).unsqueeze(1)
    rng = np.random.default_rng(seed)
    starts = rng.choice(n - W, size=min(N_WIN, n - W), replace=False)
    xw = torch.tensor(x[starts[:, None] + np.arange(W)[None, :]], dtype=torch.float32)
    norm = CCTNorm(1, correct=correct, per_channel=False, init_alpha=0.10)
    ae = TinyAE(1)
    opt = torch.optim.Adam(list(norm.parameters()) + list(ae.parameters()), lr=3e-3)
    log4 = torch.zeros(1) + math.log(4.0)
    for _ in range(EPOCHS):
        m, s = norm(cal)
        win = winsorize((xw - m) / s, log4).unsqueeze(1)
        loss = recon_loss(ae(win), win, 'relative')
        opt.zero_grad(); loss.backward(); opt.step()
    return float(norm.alphas()[1].mean()), x[::stride][:N_CAL]


def main():
    t0 = time.time()
    seqs = pickle.load(open(os.path.join(DQ, 'data', 'intel_lab', 'organic_sequences_satimpl.pkl'), 'rb'))
    rows = []
    for m, sq in seqs.items():
        x = sq['temp'].astype(np.float64)
        lab = sq['label'].astype(bool)
        if len(x) < 5000:
            continue
        stride = max(1, len(x) // N_CAL)
        cal_eps = lab[::stride][:N_CAL].mean()
        for seed in SEEDS:
            a_c, calx = run_sensor(x, True, seed)
            a_u, _ = run_sensor(x, False, seed)
            upper = (calx > 50).mean()
            lower = (calx < 0).mean()
            rows.append(dict(mote=m, seed=seed, true_eps=lab.mean(), cal_eps=cal_eps,
                             upper_tail=upper, lower_tail=lower,
                             alpha_corrected=a_c, alpha_uncorrected=a_u))
        print(f"  sensor {m:>2}: true eps {lab.mean():.3f}  alpha* corrected {a_c:.4f}  "
              f"uncorrected {a_u:.4f}   [{time.time()-t0:.0f}s]")
    r = pd.DataFrame(rows).groupby('mote').mean(numeric_only=True).reset_index()
    out = os.path.join(DQ, 'results', 'intel', 'cctnorm_real_alpha.csv')
    r.to_csv(out, index=False)

    print(f"\n=== {len(r)} real sensors ===")
    print(f"true contamination: min {r.cal_eps.min():.3f}  median {r.cal_eps.median():.3f}  "
          f"max {r.cal_eps.max():.3f}   (upper-tail share of faults: "
          f"{r.upper_tail.sum() / max((r.upper_tail + r.lower_tail).sum(), 1e-9):.3f})")
    for col, name in (('alpha_corrected', 'CORRECTED'), ('alpha_uncorrected', 'UNCORRECTED')):
        rho, p = spst.spearmanr(r.cal_eps, r[col])
        pr, pp = spst.pearsonr(r.cal_eps, r[col])
        slope, icpt = np.polyfit(r.cal_eps, r[col], 1)
        pinned = ((r[col] < 0.01) | (r[col] > ALPHA_MAX - 0.01)).mean()
        print(f"  {name:12s} alpha*: range {r[col].min():.4f}-{r[col].max():.4f}  "
              f"Spearman {rho:+.3f} (p={p:.2g})  Pearson {pr:+.3f} (p={pp:.2g})  "
              f"fit alpha*={slope:.3f}*eps+{icpt:.4f}  pinned at a corner {pinned:.0%}")
    print("\ninjected-contamination reference: alpha* = 1.319*eps + 0.013 (MAE 0.0035)")
    print(f"saved {out}   elapsed {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()
