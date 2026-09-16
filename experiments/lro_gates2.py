"""Kill-test gates G2 and G4 for the LRO plan. Thresholds fixed in lro_killtest.py.

G2 asks WHICH CHANNEL carries the error between the raw and the technician-corrected
rolling statistic, by running three oracles and recomputing the statistic each time:

  mask-only    the technician's deletions, applied to otherwise raw values
  affine-only  one (gain, offset) per CALIBRATION PERIOD, robustly fitted; period
               boundaries come from the documented calibration dates where they exist
               and from RAW-SIDE change points (long gaps, abrupt jumps) elsewhere, so
               the segmentation never peeks at the corrected series
  both         deletions and the period affine together

The fitted (gain, offset) is an oracle: it is what a perfect period-level calibration
channel could achieve, not what a causal model would achieve. That is the point. If the
two channels together do not explain most of the error, the (delete, affine) architecture
is wrong for this data regardless of how well it is trained.

G4 asks whether the technician's DELETIONS are identifiable at all from causal raw-only
features, and compares that against rule screening on the same series, scored by the
window-statistic error that results from acting on each.
"""
import os
import sys

import numpy as np
import pandas as pd

_EXP = os.path.dirname(os.path.abspath(__file__))
if _EXP not in sys.path:
    sys.path.insert(0, _EXP)
from lro_killtest import (ALLVARS, DATA, G2_MIN_AFFINE_FOR_HEADB, G2_MIN_BOTH, G2_MIN_MASK,
                          OUT, PRIMARY, SITES, TEST_YEARS, TRAIN_YEARS, WINDOWS, apply_rules,
                          fit_rules, load_site, paired_error, resolution, rolling_all,
                          run_lengths, series_scale)

MIN_REGIME = 4 * 24 * 7             # one week at a 15-minute timestep
EST_G2 = 'trimmed_mean'


def calib_periods(site, var):
    """Documented drift periods, where the technicians recorded them (MainStreet only)."""
    f = os.path.join(DATA, '%s_%s_calib_dates.csv' % (site, var))
    if not os.path.exists(f):
        return None
    d = pd.read_csv(f, skiprows=1)
    d.columns = [c.strip().lower() for c in d.columns]
    if 'start' not in d.columns:
        return None
    d['start'] = pd.to_datetime(d['start'], errors='coerce')
    d['end'] = pd.to_datetime(d['end'], errors='coerce')
    return d.dropna(subset=['start', 'end'])


def raw_side_boundaries(dt, raw, step):
    """Regime boundaries from the RAW series only: long outages and abrupt level jumps."""
    n = len(raw)
    b = [0]
    finite = np.isfinite(raw)
    # a gap of more than 12 hours ends a regime
    gap_len = max(4, int(round(12 * 3600.0 / step)))
    runs = 0
    for i in range(n):
        if not finite[i]:
            runs += 1
        else:
            if runs >= gap_len:
                b.append(i)
            runs = 0
    d = np.abs(np.diff(raw, prepend=raw[0]))
    fd = d[np.isfinite(d) & (d > 0)]
    if fd.size:
        jump = np.quantile(fd, 0.99995)
        b += list(np.where(np.isfinite(d) & (d > jump))[0])
    b = sorted(set(b + [n]))
    out = [b[0]]
    for x in b[1:]:
        if x - out[-1] >= MIN_REGIME:
            out.append(x)
    if out[-1] != n:
        out[-1] = n
    return out


def robust_affine(raw, cor, idx):
    """Least-absolute-deviation-ish affine fit of cor on raw over one regime."""
    m = np.isfinite(raw[idx]) & np.isfinite(cor[idx])
    x, y = raw[idx][m], cor[idx][m]
    if x.size < 50:
        return 1.0, 0.0
    if np.ptp(x) < 1e-9:
        return 1.0, float(np.median(y - x))
    # Theil-Sen on a capped random subsample keeps this cheap and outlier-resistant
    rng = np.random.default_rng(0)
    k = min(4000, x.size)
    i1 = rng.integers(0, x.size, k)
    i2 = rng.integers(0, x.size, k)
    ok = np.abs(x[i2] - x[i1]) > 1e-9
    if ok.sum() < 20:
        return 1.0, float(np.median(y - x))
    g = np.median((y[i2][ok] - y[i1][ok]) / (x[i2][ok] - x[i1][ok]))
    if not np.isfinite(g) or abs(g) > 10 or abs(g) < 0.1:
        g = 1.0
    b = float(np.median(y - g * x))
    return float(g), b


def gate2():
    rows = []
    for site in SITES:
        loaded = load_site(site)
        if loaded is None:
            continue
        d, step = loaded
        dt = d.datetime
        yr = dt.dt.year.to_numpy()
        mtr = (yr >= TRAIN_YEARS[0]) & (yr <= TRAIN_YEARS[1])
        for var in ALLVARS:
            raw = d[var].to_numpy(float)
            cor = d['%s_cor' % var].to_numpy(float)
            scale = series_scale(cor, mtr)
            if not np.isfinite(scale) or scale <= 0:
                continue
            deleted = np.isfinite(raw) & ~np.isfinite(cor)

            masked = raw.copy()
            masked[deleted] = np.nan

            per = calib_periods(site, var)
            if per is not None and len(per):
                bounds = [0]
                for _, p in per.iterrows():
                    j = int(np.searchsorted(dt.to_numpy(), np.datetime64(p['end'])))
                    if j > bounds[-1]:
                        bounds.append(min(j, len(raw)))
                bounds.append(len(raw))
                bounds = sorted(set(bounds))
                src = 'documented'
            else:
                bounds = raw_side_boundaries(dt, raw, step)
                src = 'raw-side'

            affine = raw.copy()
            for a, b in zip(bounds[:-1], bounds[1:]):
                idx = np.arange(a, b)
                g, off = robust_affine(raw, cor, idx)
                affine[a:b] = g * raw[a:b] + off
            both = affine.copy()
            both[deleted] = np.nan

            for w in WINDOWS:
                ec = rolling_all(cor, w)[EST_G2]
                base = paired_error(rolling_all(raw, w)[EST_G2], ec)[0]
                if not np.isfinite(base) or base <= 0:
                    continue
                e_mask = paired_error(rolling_all(masked, w)[EST_G2], ec)[0]
                e_aff = paired_error(rolling_all(affine, w)[EST_G2], ec)[0]
                e_both = paired_error(rolling_all(both, w)[EST_G2], ec)[0]
                rows.append(dict(site=site, var=var, w=w, regimes=len(bounds) - 1, regime_src=src,
                                 scale=scale, identity_mae=base / scale,
                                 expl_mask=1 - e_mask / base, expl_affine=1 - e_aff / base,
                                 expl_both=1 - e_both / base))
                r = rows[-1]
                print('  %-14s %-4s W%-3d %-10s regimes %3d  identity %.4f  explained: mask %+.3f  '
                      'affine %+.3f  both %+.3f'
                      % (site, var, w, src, r['regimes'], r['identity_mae'], r['expl_mask'],
                         r['expl_affine'], r['expl_both']), flush=True)
    r = pd.DataFrame(rows)
    r.to_csv(os.path.join(OUT, 'g2_channels.csv'), index=False)
    prim = r[(r['var'].isin(PRIMARY)) & (r.w == WINDOWS[0])]
    both_m, mask_m, aff_m = prim.expl_both.mean(), prim.expl_mask.mean(), prim.expl_affine.mean()
    ok = (both_m >= G2_MIN_BOTH) and (mask_m >= G2_MIN_MASK)
    print('\nG2 (primary vars, %s, W=%d, %d series): explained by mask %.3f (need >= %.2f), '
          'affine %.3f, both %.3f (need >= %.2f)  ->  %s'
          % (EST_G2, WINDOWS[0], len(prim), mask_m, G2_MIN_MASK, aff_m, both_m, G2_MIN_BOTH,
             'PASS' if ok else 'FAIL'))
    byvar = r[r.w == WINDOWS[0]].groupby('var')[['expl_mask', 'expl_affine', 'expl_both']].mean()
    print(byvar.round(3).to_string())
    n_aff = int((byvar.expl_affine >= G2_MIN_AFFINE_FOR_HEADB).sum())
    print('   variables where the affine channel alone explains >= %.2f: %d (Head B is built '
          'only if >= 2)' % (G2_MIN_AFFINE_FOR_HEADB, n_aff))
    if not ok:
        print('   FAIL ACTION: real validation is not (delete + period affine). The architecture '
              'is wrong; the honest negative becomes a paper section.')
    return ok, r


# ------------------------------------------------------------ G4 identifiability
def causal_features(raw, neigh, other_vars, step):
    """Raw-only, strictly causal features. No corrected value, no future value."""
    n = len(raw)
    x = pd.Series(raw)
    win = max(8, int(round(6 * 3600.0 / step)))       # 6-hour trailing window
    med = x.rolling(win, min_periods=4).median()
    madn = (x - med).abs().rolling(win, min_periods=4).median() * 1.4826
    sc = madn.replace(0, np.nan).ffill().fillna(1.0).to_numpy()
    d1 = np.abs(np.diff(raw, prepend=raw[0]))
    d2 = np.abs(np.diff(raw, n=2, prepend=[raw[0], raw[0]]))
    feats = {
        'z': np.nan_to_num((raw - med.to_numpy()) / np.maximum(sc, 1e-9), nan=0.0, posinf=0, neginf=0),
        'd1': np.nan_to_num(d1 / np.maximum(sc, 1e-9), nan=0.0, posinf=0, neginf=0),
        'd2': np.nan_to_num(d2 / np.maximum(sc, 1e-9), nan=0.0, posinf=0, neginf=0),
        'flat': np.minimum(run_lengths(raw), 500).astype(float),
        'miss_near': pd.Series(~np.isfinite(raw)).rolling(win, min_periods=1).mean().to_numpy(),
        'spatial': np.nan_to_num((raw - neigh) / np.maximum(sc, 1e-9), nan=0.0, posinf=0, neginf=0)
        if neigh is not None else np.zeros(n),
    }
    for i, ov in enumerate(other_vars):
        zo = pd.Series(ov)
        mo = zo.rolling(win, min_periods=4).median()
        so = (zo - mo).abs().rolling(win, min_periods=4).median() * 1.4826
        feats['cross%d' % i] = np.nan_to_num((ov - mo.to_numpy()) / np.maximum(so.to_numpy(), 1e-9),
                                             nan=0.0, posinf=0, neginf=0)
    return pd.DataFrame(feats).to_numpy(float)


def gate4():
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score

    site_data = {}
    for site in SITES:
        loaded = load_site(site)
        if loaded is not None:
            site_data[site] = loaded
    rows = []
    for site, (d, step) in site_data.items():
        yr = d.datetime.dt.year.to_numpy()
        mtr = (yr >= TRAIN_YEARS[0]) & (yr <= TRAIN_YEARS[1])
        mte = (yr >= TEST_YEARS[0]) & (yr <= TEST_YEARS[1])
        for var in PRIMARY:
            raw = d[var].to_numpy(float)
            cor = d['%s_cor' % var].to_numpy(float)
            scale = series_scale(cor, mtr)
            if not np.isfinite(scale) or scale <= 0:
                continue
            y = (np.isfinite(raw) & ~np.isfinite(cor)).astype(int)
            if y[mtr].sum() < 50 or y[mte].sum() < 50:
                print('  %-14s %-4s skipped: too few deletions (train %d, test %d)'
                      % (site, var, y[mtr].sum(), y[mte].sum()))
                continue
            # neighbour reference: median of the same variable at the other sites
            others = []
            for s2, (d2, _) in site_data.items():
                if s2 == site:
                    continue
                v = d2.set_index('datetime')[var].reindex(d.datetime).to_numpy(float)
                z = (v - np.nanmedian(v[mtr])) / max(np.nanstd(v[mtr]), 1e-9)
                others.append(z)
            neigh = None
            if others:
                zn = np.nanmedian(np.vstack(others), axis=0)
                neigh = zn * max(np.nanstd(raw[mtr]), 1e-9) + np.nanmedian(raw[mtr])
            ov = [d[v2].to_numpy(float) for v2 in ALLVARS if v2 != var]
            X = causal_features(raw, neigh, ov, step)
            ok = np.isfinite(X).all(axis=1)
            clf = LogisticRegression(max_iter=1000, class_weight='balanced')
            clf.fit(X[mtr & ok], y[mtr & ok])
            p = np.zeros(len(raw))
            p[ok] = clf.predict_proba(X[ok])[:, 1]
            ap = average_precision_score(y[mte & ok], p[mte & ok])

            rules = fit_rules(raw, mtr)
            screened = apply_rules(raw, rules)
            learned = raw.copy()
            learned[p >= 0.5] = np.nan
            w = WINDOWS[0]
            ec = rolling_all(cor, w)[EST_G2]
            te_win = mte[:len(ec)]
            def mae_on_test(series):
                e = rolling_all(series, w)[EST_G2]
                m = np.isfinite(e) & np.isfinite(ec) & te_win
                return float(np.abs(e[m] - ec[m]).mean() / scale) if m.any() else np.nan
            rows.append(dict(site=site, var=var, base_rate=float(y[mte].mean()), auprc=float(ap),
                             mae_raw=mae_on_test(raw), mae_rules=mae_on_test(screened),
                             mae_learned=mae_on_test(learned)))
            r = rows[-1]
            print('  %-14s %-4s deletions %.3f  AUPRC %.3f (base %.3f)  window MAE: raw %.4f  '
                  'rules %.4f  learned %.4f'
                  % (site, var, r['base_rate'], r['auprc'], r['base_rate'], r['mae_raw'],
                     r['mae_rules'], r['mae_learned']), flush=True)
    r = pd.DataFrame(rows)
    if len(r):
        r.to_csv(os.path.join(OUT, 'g4_identifiability.csv'), index=False)
        better = int((r.mae_learned < r.mae_rules).sum())
        print('\nG4: learned retention beats rule screening on %d of %d series '
              '(median MAE learned %.4f vs rules %.4f vs raw %.4f)'
              % (better, len(r), r.mae_learned.median(), r.mae_rules.median(), r.mae_raw.median()))
    return True, r


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    which = sys.argv[1:] or ['g2', 'g4']
    if 'g2' in which:
        print('=== G2 CHANNEL DECOMPOSITION ===')
        gate2()
    if 'g4' in which:
        print('\n=== G4 IDENTIFIABILITY ===')
        gate4()
