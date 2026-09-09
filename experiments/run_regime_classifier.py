"""Path B: online contamination-regime classification.

MOTIVATION (from this paper's own RQ2): the sign of the robust-versus-naive
spread comparison reverses between the isolated-incident and ambient
contamination models -- 1.89 vs 0.80 on identical data with an identical
metric. A monitor therefore needs to know which regime it is operating in
before it can choose an estimator. No streaming DQ evaluation we surveyed even
states its contamination model, so there is no existing method for inferring it
online. That is the gap this task addresses.

TASK: given a trailing sequence of K windows of a live stream, and using only
quantities computable from the dirty windows themselves (no clean reference),
classify the contamination regime as isolated-incident or ambient.

THE EXPERIMENT IS DESIGNED SO THE CLASSIFIER CANNOT CHEAT. The naive shortcut
is to measure how MUCH contamination is present: if ambient always carried more
total corruption than isolated, total perturbation alone would separate the
classes and nothing would have been learned about temporal structure. The
corruption budgets are therefore deliberately overlapped --

    ambient  total corrupted fraction in {2%, 5%, 10%}, scattered
    isolated total corrupted fraction in {1%, 2%, 5%}, delivered in a few
             temporally localized incidents against a clean background

-- so 2% and 5% occur under BOTH regimes. A classifier that only reads the
overall perturbation level cannot separate those, and a "mean perturbation
level" heuristic is included as an explicit baseline to demonstrate this.
The discriminative signal has to be temporal concentration: under isolated
injection most windows are clean and a few are badly hit, so the per-window
perturbation series is bursty and high-Gini; under ambient injection nearly
every window carries some contamination, so it is comparatively uniform.

Restricted to the four mechanisms Phase 1B and Phase 2A both tested (missing,
duplicate, outlier, out_of_order), matching the restriction RQ2 already uses
for its cross-model comparison.
"""
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

warnings.filterwarnings('ignore', category=RuntimeWarning)

_EXP_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_EXP_DIR)
for p in (_EXP_DIR, _REPO_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from run_phase2a_full import load_series
from harness.harness1c import W, inject
from harness import phase2a_ext as p2x

OUT_DIR = os.path.join(_REPO_ROOT, 'results', 'regime')
SHARED_MECH = ['missing', 'duplicate', 'outlier', 'out_of_order']
SEEDS = [0, 1]
SEQ_LEN = int(os.environ.get('SEQ_LEN', '200'))   # windows per classified sequence
AMBIENT_RATES = [0.02, 0.05, 0.10]
ISOLATED_RATES = [0.01, 0.02, 0.05]  # overlaps ambient at 2% and 5% by design
INCIDENT_LEN = 60
LOCAL_RATE = 0.5
EPS = 1e-8


def inject_isolated(x, kind, total_rate, rng, magnitude=None):
    """Rare, temporally localized incidents against an otherwise clean
    background. Reuses inject() unchanged on each incident slice, so the
    corruption semantics are identical to the ambient case -- only the
    temporal arrangement differs, which is exactly the contrast RQ2 draws.
    Does not touch pipeline.py."""
    n = len(x)
    y = x.copy()
    budget = int(round(total_rate * n))
    per_incident = max(1, int(round(INCIDENT_LEN * LOCAL_RATE)))
    n_inc = max(1, budget // per_incident)
    starts = rng.choice(max(1, n - INCIDENT_LEN), size=min(n_inc, max(1, (n - INCIDENT_LEN) // INCIDENT_LEN)),
                        replace=False)
    for s in starts:
        seg = y[s:s + INCIDENT_LEN]
        if len(seg) < 5:
            continue
        y[s:s + INCIDENT_LEN] = inject(seg, kind, LOCAL_RATE, rng, magnitude=magnitude)
    return y


def per_window_signals(x_bad):
    """Online-computable per-window signals (dirty window only).

    The first four are spread-based. They are blind to defects that barely
    move a spread estimate -- which is exactly why the v1 feature set sat at
    chance on duplicate (0.525) and out_of_order (0.506): those mechanisms
    perturb WHICH records occupy the window, not their magnitudes.

    The last four target those signatures directly, each motivated by the
    defect's own mechanics rather than by inspection of test scores:
      repeat_frac  -- duplicated/retransmitted records produce exact adjacent
                      repeats, which no spread statistic registers.
      rough1/rough2 -- swapping two adjacent samples of a smooth,
                      autocorrelated series injects a local zigzag, inflating
                      the first/second absolute differences while leaving the
                      window's spread almost unchanged. This is the only
                      signature out_of_order leaves.
      extreme_frac -- direct count of points beyond 3 MADn, a magnitude
                      channel independent of the trimmed/untrimmed gap.
    """
    _, med, std_w, madn = p2x.rolling_stats(x_bad, W)
    _, _, ts10, iqr = p2x.rolling_extended(x_bad, W, trim_frac=0.10, consistency=True)
    sw = sliding_window_view(x_bad, W)
    n_valid = np.isfinite(sw).sum(axis=1)
    same = (sw[:, 1:] == sw[:, :-1])
    scale = np.maximum(madn, EPS)

    with np.errstate(invalid='ignore'):
        d1 = np.abs(np.diff(sw, axis=1))
        d2 = np.abs(np.diff(sw, n=2, axis=1))
        rough1 = np.nanmean(d1, axis=1) / scale
        rough2 = np.nanmean(d2, axis=1) / scale
        z = np.abs(sw - med[:, None]) / scale[:, None]
        extreme_frac = np.nanmean((z > 3.0).astype(float), axis=1)

    return dict(
        trim_gap=np.abs(std_w - ts10) / np.maximum(std_w, EPS),
        iqr_gap=np.abs(std_w - iqr) / np.maximum(std_w, EPS),
        nan_frac=1.0 - n_valid / W,
        flat_frac=same.sum(axis=1) / (W - 1),
        repeat_frac=np.nansum(same.astype(float), axis=1) / np.maximum(n_valid - 1, 1),
        rough1=rough1,
        rough2=rough2,
        extreme_frac=extreme_frac,
    )


def _gini(v):
    v = np.sort(np.maximum(v, 0.0))
    n = len(v)
    s = v.sum()
    if n == 0 or s <= 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2.0 * (idx * v).sum()) / (n * s) - (n + 1.0) / n)


def _longest_run_above(v, thresh):
    above = v > thresh
    best = cur = 0
    for a in above:
        cur = cur + 1 if a else 0
        best = max(best, cur)
    return float(best) / max(len(v), 1)


def seq_features(sig_slice):
    """Aggregate one sequence of windows into concentration/burstiness
    descriptors. These are the features that can distinguish 'a few windows
    badly hit' from 'most windows mildly hit' at matched total corruption."""
    out = {}
    for name, v in sig_slice.items():
        v = np.asarray(v, dtype=float)
        v = v[np.isfinite(v)]
        if v.size == 0:
            v = np.zeros(1)
        m, sd = float(v.mean()), float(v.std())
        med = float(np.median(v))
        out[f'{name}_mean'] = m
        out[f'{name}_std'] = sd
        out[f'{name}_cv'] = sd / max(m, EPS)
        out[f'{name}_max_over_mean'] = float(v.max()) / max(m, EPS)
        out[f'{name}_q90_over_med'] = float(np.quantile(v, 0.90)) / max(med, EPS)
        out[f'{name}_gini'] = _gini(v)
        out[f'{name}_run_above_med'] = _longest_run_above(v, med)
        d = v - m
        denom = float((d * d).sum())
        out[f'{name}_ac1'] = float((d[:-1] * d[1:]).sum() / denom) if denom > EPS else 0.0
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.time()
    series_list = load_series()
    print(f"{len(series_list)} series x {len(SHARED_MECH)} mechanisms x 2 regimes "
          f"x 3 rates x {len(SEEDS)} seeds, sequences of {SEQ_LEN} windows")

    rows = []
    for si, s in enumerate(series_list):
        x = s['x']
        for mech in SHARED_MECH:
            for regime, rates in (('ambient', AMBIENT_RATES), ('isolated', ISOLATED_RATES)):
                for rate in rates:
                    for seed in SEEDS:
                        rng = np.random.default_rng(seed)
                        mag = 4.0 if mech in ('outlier',) else None
                        if regime == 'ambient':
                            xb = inject(x, mech, rate, rng, magnitude=mag)
                        else:
                            xb = inject_isolated(x, mech, rate, rng, magnitude=mag)
                        sig = per_window_signals(xb)
                        n_win = len(sig['trim_gap'])
                        for start in range(0, n_win - SEQ_LEN + 1, SEQ_LEN):
                            sl = {k: v[start:start + SEQ_LEN] for k, v in sig.items()}
                            f = seq_features(sl)
                            f.update(domain=s['domain'], run=s['run'], channel=s['channel'],
                                     mechanism=mech, regime=regime, rate=rate, seed=seed,
                                     seq_start=start)
                            rows.append(f)
        if (si + 1) % 10 == 0:
            print(f"  series {si+1}/{len(series_list)}  sequences: {len(rows)}  "
                  f"elapsed {time.time()-t0:.0f}s")

    df = pd.DataFrame(rows)
    VERSION = os.environ.get('FEATS', 'v1')
    out = os.path.join(OUT_DIR, f'regime_sequences_k{SEQ_LEN}_{VERSION}.csv.gz')
    df.to_csv(out, index=False, compression='gzip')
    print(f"\n{len(df)} sequences  ({df['regime'].value_counts().to_dict()})")
    print(f"rate overlap check -- sequences per (regime, rate):")
    print(df.groupby(['regime', 'rate']).size().to_string())
    print(f"Wrote {out}   elapsed {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()
