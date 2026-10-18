"""Intel Berkeley Lab: characterise the REAL faults and measure the reference
noise floor before designing anything.

Three questions, each of which can kill the plan:

  1. Where do faults begin as a function of battery voltage? Healthy must be
     defined by the PHYSICAL cause (voltage), not by the measured value --
     defining 'healthy' as 'temperature in a plausible range' would make any
     temperature-estimator evaluation circular.

  2. What do real faults actually look like? Sharma et al. define stuck faults
     as 'anomalous constant offset readings'. Are they bit-identical repeats
     (which the exact-repeat repair would catch) or noisy near-constants (which
     it would miss)?

  3. NOISE FLOOR. The reference for a faulty mote is built from its healthy
     spatial neighbours. On points where everything is healthy, pretend the
     point is faulty and impute it from neighbours: how large is that error
     relative to the within-window spread the estimators operate on? If the
     imputation error is comparable to the spread, the reference cannot
     separate estimators and the plan fails here.
"""
import os
import sys
import gzip
import pickle
import numpy as np
import pandas as pd

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'intel_lab')
W = 30


def load():
    cols = ['date', 'time', 'epoch', 'moteid', 'temp', 'hum', 'light', 'volt']
    df = pd.read_csv(os.path.join(DATA, 'data.txt.gz'), sep=r'\s+', names=cols,
                     on_bad_lines='skip', engine='c', compression='gzip')
    n0 = len(df)
    for c in ['epoch', 'moteid', 'temp', 'hum', 'light', 'volt']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=['epoch', 'moteid', 'temp', 'volt'])
    df = df[(df.moteid >= 1) & (df.moteid <= 54)]
    df['epoch'] = df['epoch'].astype(int)
    df['moteid'] = df['moteid'].astype(int)
    print(f"rows parsed {n0:,}, usable {len(df):,}, motes {df.moteid.nunique()}, "
          f"epochs {df.epoch.min()}..{df.epoch.max()}")
    return df


def main():
    df = load()

    # ---------- 1. fault onset vs voltage ----------
    print("\n=== 1. temperature plausibility by voltage bin ===")
    print("    (indoor lab: a reading outside 0-50 C is physically implausible)")
    bins = [0, 2.0, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 3.5]
    df['vbin'] = pd.cut(df.volt, bins)
    df['implausible'] = (df.temp < 0) | (df.temp > 50)
    t = df.groupby('vbin', observed=True).agg(n=('temp', 'size'),
                                              implausible=('implausible', 'mean'),
                                              temp_median=('temp', 'median'),
                                              temp_iqr=('temp', lambda v: v.quantile(.75) - v.quantile(.25)))
    print(t.round(4).to_string())

    # ---------- 2. what do real faults look like? ----------
    print("\n=== 2. shape of real faults ===")
    df = df.sort_values(['moteid', 'epoch'])
    df['dtemp'] = df.groupby('moteid')['temp'].diff()
    exact = (df['dtemp'] == 0)
    for label, m in [('healthy (volt>=2.5)', df.volt >= 2.5), ('low-volt (volt<2.4)', df.volt < 2.4)]:
        sub = df[m]
        print(f"  {label:22s} n={len(sub):>9,}  exact repeats={exact[m].mean():.4f}  "
              f"median|dtemp|={sub['dtemp'].abs().median():.4f}  "
              f"temp p1/p50/p99={sub.temp.quantile(.01):.1f}/{sub.temp.median():.1f}/{sub.temp.quantile(.99):.1f}")

    # ---------- 3. noise floor of the spatial reference ----------
    print("\n=== 3. reference noise floor (healthy points only) ===")
    locs = pd.read_csv(os.path.join(DATA, 'mote_locs.txt'), sep=r'\s+', names=['moteid', 'x', 'y'])
    locs = locs.set_index('moteid')
    healthy = df[df.volt >= 2.5]
    P = healthy.pivot_table(index='epoch', columns='moteid', values='temp', aggfunc='mean')
    motes = [m for m in P.columns if m in locs.index]
    P = P[motes]
    xy = locs.loc[motes, ['x', 'y']].to_numpy()
    D = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1))
    print(f"  healthy epoch x mote matrix: {P.shape}, fill {P.notna().mean().mean():.3f}")

    A = P.to_numpy()
    for k in (3, 5, 8):
        errs_med, errs_off, spreads = [], [], []
        for j, m in enumerate(motes):
            nbr = np.argsort(D[j])[1:k + 1]
            own = A[:, j]
            nb = A[:, nbr]
            med = np.nanmedian(nb, axis=1)
            ok = np.isfinite(own) & np.isfinite(med) & (np.isfinite(nb).sum(1) >= max(2, k // 2))
            if ok.sum() < 500:
                continue
            # offset-corrected: learn this mote's level offset on the FIRST half only
            idx = np.where(ok)[0]
            half = idx[: len(idx) // 2]
            off = np.nanmedian(own[half] - med[half])
            test = idx[len(idx) // 2:]
            errs_med.append(np.nanmedian(np.abs(own[test] - med[test])))
            errs_off.append(np.nanmedian(np.abs(own[test] - (med[test] + off))))
            s = pd.Series(own).rolling(W, min_periods=W // 2).std().to_numpy()
            spreads.append(np.nanmedian(s))
        em, eo, sp = map(np.array, (errs_med, errs_off, spreads))
        print(f"  k={k}: motes={len(em)}  median imputation error  raw-median={np.median(em):.3f} C  "
              f"offset-corrected={np.median(eo):.3f} C   | mote within-window std={np.median(sp):.3f} C"
              f"   ratio(offset-err / spread)={np.median(eo) / np.median(sp):.2f}")

    with open(os.path.join(DATA, 'intel_parsed.pkl'), 'wb') as f:
        pickle.dump({'df': df[['epoch', 'moteid', 'temp', 'hum', 'light', 'volt']], 'locs': locs}, f)
    print("\nsaved intel_parsed.pkl")


if __name__ == '__main__':
    main()
