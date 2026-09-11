"""Organic evaluation on REAL dying trajectories: baselines and headroom.

No transplantation. Each mote's genuine time-ordered stream is used as-is, from
healthy operation through battery failure, with its real voltage.

GROUND TRUTH is built WITHOUT voltage, so any voltage-using method is being asked
to predict something it cannot read off directly. A reading is labelled faulty if:
  - it is saturated (|temp - 122.153| < 0.01), or
  - physically implausible (< 0 C or > 50 C), or
  - it deviates from the median of its 5 nearest HEALTHY neighbours (V >= 2.5 at
    the same corrected epoch), after removing the mote's own healthy-period offset,
    by more than DEV_THR.
The spatial reference is valid for location (a 1.5 C deviation is ~5x its 0.28 C
imputation error) even though it was measured unusable for spread (3.9x).

Why this is not circular for voltage: the best single global voltage threshold
reaches only F1 0.92 against these labels and 0.08 on the worst mote, so voltage
is an imperfect, heterogeneous predictor -- not a separator.

METRIC. There is no clean twin, so each estimator is compared with the SAME
estimator computed over only the label-clean readings in that window -- the
organic analogue of the paper's paired error. Windows need >= W/2 clean readings
to have a reference.

Reported on all windows and on DYING-PHASE windows (window median voltage < 2.5),
where faults actually occur and methods can differ.
"""
import os
import sys
import pickle
import warnings
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

warnings.filterwarnings('ignore', category=RuntimeWarning)
DQ = r"C:\Users\luckf\OneDrive\Desktop\dq-estimator-eval"
OUT = os.path.join(DQ, 'results', 'intel')
W = 30
SAT = 122.153
GLOBAL_T = 2.31
DEV_THRS = [1.0, 1.5, 2.0]
PRIMARY_DEV = 1.5
T_GRID = np.round(np.arange(2.25, 2.46, 0.01), 2)


def build_sequences(dev_thr):
    d = pickle.load(open(os.path.join(DQ, 'data', 'intel_lab', 'intel_parsed_ts.pkl'), 'rb'))
    df, locs = d['df'], d['locs']
    Pall = df.pivot_table(index='epoch_c', columns='moteid', values='temp', aggfunc='mean')
    Vall = df.pivot_table(index='epoch_c', columns='moteid', values='volt', aggfunc='mean')
    pl = df[(df.temp > 0) & (df.temp < 50)]
    Ppl = pl.pivot_table(index='epoch_c', columns='moteid', values='temp', aggfunc='mean')
    motes = [m for m in Pall.columns if m in locs.index]
    Pall = Pall[motes]
    Vall = Vall.reindex(index=Pall.index, columns=motes)
    Ppl = Ppl.reindex(index=Pall.index, columns=motes)
    xy = locs.loc[motes, ['x', 'y']].to_numpy()
    D = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1))
    A, V, APL = Pall.to_numpy(), Vall.to_numpy(), Ppl.to_numpy()

    seqs = {}
    comp = {'sat': 0, 'implausible': 0, 'deviation_only': 0, 'total': 0}
    for j, m in enumerate(motes):
        own, v = A[:, j], V[:, j]
        nbr = np.argsort(D[j])[1:6]
        nb_h = np.where(V[:, nbr] >= 2.5, APL[:, nbr], np.nan)
        med = np.nanmedian(nb_h, axis=1)
        ref_ok = np.isfinite(med) & (np.isfinite(nb_h).sum(1) >= 3)
        base_m = ref_ok & np.isfinite(APL[:, j]) & (v >= 2.5)
        if base_m.sum() < 200:
            continue
        offset = np.nanmedian(APL[base_m, j] - med[base_m])
        keep = np.isfinite(own) & np.isfinite(v)
        own, v, med, ref_ok = own[keep], v[keep], med[keep], ref_ok[keep]
        sat = np.abs(own - SAT) < 0.01
        impl = (own < 0) | (own > 50)
        dev = ref_ok & ~sat & ~impl & (np.abs(own - med - offset) > dev_thr)
        lab = sat | impl | dev
        seqs[m] = dict(temp=own, volt=v, label=lab)
        comp['sat'] += int(sat.sum()); comp['implausible'] += int((impl & ~sat).sum())
        comp['deviation_only'] += int(dev.sum()); comp['total'] += int(len(own))
    return seqs, comp


def best_mote_threshold(v, lab):
    best, bt = -1.0, GLOBAL_T
    for t in T_GRID:
        pred = v < t
        tp = (pred & lab).sum(); fp = (pred & ~lab).sum(); fn = (~pred & lab).sum()
        p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
        f = 2 * p * r / max(p + r, 1e-12)
        if f > best:
            best, bt = f, t
    return bt


def window_stats(sw):
    med = np.nanmedian(sw, axis=1)
    return dict(mean=np.nanmean(sw, axis=1), median=med, std=np.nanstd(sw, axis=1),
                madn=np.nanmedian(np.abs(sw - med[:, None]), axis=1) * 1.4826,
                iqr=(np.nanpercentile(sw, 75, axis=1) - np.nanpercentile(sw, 25, axis=1)) / 1.349)


def evaluate(seqs):
    rows = []
    for m, s in seqs.items():
        x, v, lab = s['temp'], s['volt'], s['label']
        if len(x) < 5 * W:
            continue
        sw = sliding_window_view(x, W)
        lw = sliding_window_view(lab, W)
        vw = sliding_window_view(v, W)
        clean = np.where(~lw, sw, np.nan)
        nclean = (~lw).sum(1)
        valid = nclean >= W // 2
        ref = window_stats(clean)
        obs = window_stats(sw)
        t_or = best_mote_threshold(v, lab)
        thr_g = window_stats(np.where(vw >= GLOBAL_T, sw, np.nan))
        thr_o = window_stats(np.where(vw >= t_or, sw, np.nan))
        dying = np.nanmedian(vw, axis=1) < 2.5
        nfault = lw.sum(1)
        for i in np.where(valid)[0]:
            r = dict(mote=m, dying=bool(dying[i]), has_fault=bool(nfault[i] > 0),
                     e_std=abs(obs['std'][i] - ref['std'][i]),
                     e_madn=abs(obs['madn'][i] - ref['madn'][i]),
                     e_iqr=abs(obs['iqr'][i] - ref['iqr'][i]),
                     e_thrG_std=abs(thr_g['std'][i] - ref['std'][i]),
                     e_thrO_std=abs(thr_o['std'][i] - ref['std'][i]),
                     e_mean=abs(obs['mean'][i] - ref['mean'][i]),
                     e_median=abs(obs['median'][i] - ref['median'][i]),
                     e_thrG_mean=abs(thr_g['mean'][i] - ref['mean'][i]),
                     e_thrO_mean=abs(thr_o['mean'][i] - ref['mean'][i]))
            rows.append(r)
    return pd.DataFrame(rows)


def summarize(df, label):
    print(f"\n--- {label}: {len(df):,} windows, {df.mote.nunique()} motes, "
          f"{100 * df.has_fault.mean():.1f}% contain a fault ---")
    eps = 1e-9
    print(f"  {'SPREAD':10s}{'median |err| C':>16s}{'ratio to std':>14s}")
    base = np.median(df.e_std) + eps
    for c, name in [('e_std', 'std'), ('e_madn', 'MADn'), ('e_iqr', 'IQR'),
                    ('e_thrG_std', 'V>=2.31 std'), ('e_thrO_std', 'oracle-t std')]:
        print(f"  {name:10s}{np.median(df[c]):16.4f}{np.median(df[c]) / base:14.3f}"
              f"   mean {df[c].mean():.4f}")
    print(f"  {'LOCATION':10s}")
    base = np.median(df.e_mean) + eps
    for c, name in [('e_mean', 'mean'), ('e_median', 'median'),
                    ('e_thrG_mean', 'V>=2.31 mean'), ('e_thrO_mean', 'oracle-t mean')]:
        print(f"  {name:10s}{np.median(df[c]):16.4f}{np.median(df[c]) / base:14.3f}"
              f"   mean {df[c].mean():.4f}")


def main():
    os.makedirs(OUT, exist_ok=True)
    print("=== label composition by deviation threshold (is the plausible-but-wrong category real?) ===")
    primary = None
    for dt in DEV_THRS:
        seqs, comp = build_sequences(dt)
        tot = comp['total']
        print(f"  DEV_THR={dt:.1f} C: saturated {comp['sat']/tot:.4f}  implausible-only "
              f"{comp['implausible']/tot:.4f}  PLAUSIBLE-BUT-WRONG {comp['deviation_only']/tot:.4f}  "
              f"({comp['deviation_only']:,} readings)  motes {len(seqs)}")
        if dt == PRIMARY_DEV:
            primary = seqs
    pickle.dump(primary, open(os.path.join(DQ, 'data', 'intel_lab', 'organic_sequences.pkl'), 'wb'))

    res = evaluate(primary)
    res.to_csv(os.path.join(OUT, 'organic_baselines.csv'), index=False)
    summarize(res, 'ALL windows')
    summarize(res[res.dying], 'DYING-PHASE windows (median V < 2.5)')
    summarize(res[res.has_fault], 'windows CONTAINING >=1 fault')
    print("\noracle-t uses each mote's own labels to pick its threshold: an UPPER BOUND on thresholding,")
    print("not a deployable method. The gap between V>=2.31 and oracle-t is what per-mote learning could claim.")


if __name__ == '__main__':
    main()
