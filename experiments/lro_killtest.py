"""Pre-registered kill test for the DICE-W/R plan, on REAL technician-corrected data.

Logan River Observatory (HydroShare a6ea89ae20354e39b3c9f1228997e27a): six aquatic
monitoring sites, 2014-2020, 15-minute sampling. Every row carries, per variable, the
RAW sensor value, the TECHNICIAN-CORRECTED value (_cor) and the technician's label
(_qual). That makes it the first dataset in this project where the "clean" signal is a
human correction rather than something we manufactured.

EVERY THRESHOLD BELOW WAS FIXED BEFORE ANY NUMBER WAS COMPUTED (workflow synthesis,
run wf_62b2d25b-3a7). Gates are evaluated in order; a FAIL is a stop, not a nudge to
retune. The gates exist to kill the plan cheaply if the data cannot support it:

  G0  Is the technician's edit expressible as (delete, affine drift)? If instead they
      substituted values from field notes, no causal model can reproduce it.
  G1  Is there headroom at all, and does it survive simple rule screening? This is the
      gate the failed ATS/LTP work never had: if cheap rules already remove the error,
      a learned head has nothing to win.
  G2  Which channel carries the error: deletions, or period-level affine drift?
  G4  Is retention identifiable from causal raw-only features, versus rule screening?

Estimators come from the paper's own harness, unmodified.
"""
import os
import sys

import numpy as np
import pandas as pd

_EXP = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_EXP)
for p in (_ROOT, _EXP):
    if p not in sys.path:
        sys.path.insert(0, p)
from harness.harness1c import rolling_stats
from harness.phase2a_ext import rolling_extended

DATA = os.path.join(_ROOT, 'data', 'lro')
OUT = os.path.join(_ROOT, 'results', 'lro')
SITES = ['FranklinBasin', 'TonyGrove', 'WaterLab', 'MainStreet', 'Mendon', 'BlackSmithFork']
YEARS = range(2014, 2021)
PRIMARY = ['cond', 'do']            # pyhydroqc's own hardest variables
ALLVARS = ['temp', 'cond', 'ph', 'do']
TRAIN_YEARS = (2014, 2016)          # rule thresholds and scales are fitted here only
TEST_YEARS = (2018, 2020)
WINDOWS = [30, 96]                  # harness default, and 24 h at a 15-minute timestep
CHUNK = 150000

# ---- pre-registered pass thresholds -----------------------------------------
G0_MAX_REPLACEMENT = 0.25
G0_MAX_RESTORED = 0.10
G1_MIN_MEDIAN_MAE = 0.05            # normalised by the variable's robust scale
G1_MIN_PEAK_SERIES = 4              # series with normalised PEAK >= 0.5
G1_MIN_PEAK = 0.50
G1_MIN_NONZERO_FRAC = 0.02
G1_MIN_RESIDUAL_MAE = 0.03          # after rule screening
G2_MIN_BOTH = 0.60
G2_MIN_MASK = 0.20
G2_MIN_AFFINE_FOR_HEADB = 0.15


def load_site(site):
    """Concatenate a site's years onto a regular timestep grid (gaps become NaN)."""
    frames = []
    for y in YEARS:
        f = os.path.join(DATA, '%s%d.csv' % (site, y))
        if os.path.exists(f):
            frames.append(pd.read_csv(f, parse_dates=['datetime'], low_memory=False))
    if not frames:
        return None
    d = pd.concat(frames, ignore_index=True).drop_duplicates('datetime').sort_values('datetime')
    for c in d.columns:
        if c != 'datetime' and not c.endswith('_qual'):
            d[c] = pd.to_numeric(d[c], errors='coerce')
    step = d.datetime.diff().dt.total_seconds().median()
    grid = pd.date_range(d.datetime.iloc[0], d.datetime.iloc[-1], freq='%ds' % int(step))
    d = d.set_index('datetime').reindex(grid)
    d.index.name = 'datetime'
    return d.reset_index(), step


def resolution(x):
    """Sensor quantisation step: the small end of the positive gaps between values."""
    u = np.unique(x[np.isfinite(x)])
    if u.size < 100:
        return 1e-9
    dif = np.diff(u)
    dif = dif[dif > 0]
    return float(np.quantile(dif, 0.05)) if dif.size else 1e-9


def rolling_all(x, w, consistency=True):
    """Every estimator the paper uses, over rolling windows, in memory-safe chunks."""
    n = len(x)
    keys = ['mean', 'median', 'std', 'madn', 'trimmed_mean', 'huber_M', 'trimmed_std', 'iqr_scale']
    out = {k: np.empty(n - w + 1, dtype=float) for k in keys}
    start = 0
    while start < n - w + 1:
        stop = min(start + CHUNK, n - w + 1)
        seg = x[start:stop + w - 1]
        mean, med, std, mad = rolling_stats(seg, w)
        tm, hm, ts, iqr = rolling_extended(seg, w=w, consistency=consistency)
        for k, v in zip(keys, (mean, med, std, mad, tm, hm, ts, iqr)):
            out[k][start:stop] = v
        start = stop
    return out


def series_scale(cor, mask_train):
    """Robust scale of the corrected variable over TRAINING years only."""
    v = cor[mask_train]
    v = v[np.isfinite(v)]
    if v.size < 100:
        return np.nan
    return float(np.median(np.abs(v - np.median(v))) * 1.4826)


def paired_error(a, b):
    """MAE, PEAK and non-zero fraction over windows where both estimates exist."""
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() == 0:
        return np.nan, np.nan, np.nan, 0
    e = np.abs(a[m] - b[m])
    return float(e.mean()), float(e.max()), float((e > 0).mean()), int(m.sum())


def run_lengths(x):
    """Length of the current run of identical consecutive values, per position."""
    same = np.r_[False, np.diff(x) == 0]
    out = np.zeros(len(x), dtype=int)
    run = 0
    for i in range(len(x)):
        run = run + 1 if same[i] else 0
        out[i] = run
    return out


# ---------------------------------------------------------------- G0 inventory
def gate0():
    """Is the technician edit (delete, affine drift), or did they substitute values?"""
    rows = []
    for site in SITES:
        loaded = load_site(site)
        if loaded is None:
            print('  %s: no files' % site)
            continue
        d, step = loaded
        day = max(4, int(round(86400.0 / step)))
        for var in ALLVARS:
            raw = d[var].to_numpy(float)
            cor = d['%s_cor' % var].to_numpy(float)
            qual = d['%s_qual' % var].astype(str).str.strip()
            flagged = ~qual.isin(['NULL', 'nan', 'None', '', '<NA>']).to_numpy()
            fr, fc = np.isfinite(raw), np.isfinite(cor)
            n = len(raw)
            tol = max(resolution(raw), 1e-9)
            both = fr & fc
            delta = np.where(both, cor - raw, np.nan)
            # split the edit into a slow drift part and a point-substitution part
            ds = pd.Series(delta).rolling(day, center=True, min_periods=day // 4).median().to_numpy()
            resid = np.abs(delta - ds)
            modified = both & (np.abs(delta) > tol)
            replaced = modified & np.isfinite(resid) & (resid > 3 * tol)
            rows.append(dict(
                site=site, var=var, n=n, timestep_s=step, day_pts=day,
                raw_present=float(fr.mean()), cor_present=float(fc.mean()),
                deleted=float((fr & ~fc).mean()), restored=float((~fr & fc).mean()),
                modified=float(modified.mean()), replaced=float(replaced.mean()),
                drift_only=float((modified & ~replaced).mean()),
                flagged=float(flagged.mean()), resolution=tol,
                med_abs_delta=float(np.nanmedian(np.abs(delta[modified]))) if modified.any() else 0.0))
            r = rows[-1]
            print('  %-14s %-4s n=%7d step=%5.0fs  deleted %.3f  restored %.3f  modified %.3f '
                  '(drift %.3f / replaced %.3f)  flagged %.3f'
                  % (site, var, n, step, r['deleted'], r['restored'], r['modified'],
                     r['drift_only'], r['replaced'], r['flagged']), flush=True)
    r = pd.DataFrame(rows)
    r.to_csv(os.path.join(OUT, 'g0_inventory.csv'), index=False)
    prim = r[r['var'].isin(PRIMARY)]
    rep, res = prim.replaced.mean(), prim.restored.mean()
    ok = (rep < G0_MAX_REPLACEMENT) and (res < G0_MAX_RESTORED)
    print('\nG0: pooled over %s: replacement %.3f (limit %.2f), restored %.3f (limit %.2f)  ->  %s'
          % (PRIMARY, rep, G0_MAX_REPLACEMENT, res, G0_MAX_RESTORED, 'PASS' if ok else 'FAIL'))
    if not ok:
        print('   FAIL ACTION: the (delete, affine) factorisation is wrong for this data. '
              'Report as a finding and re-scope; do not build DICE-W/R.')
    return ok, r


# ----------------------------------------------------------------- G1 headroom
def fit_rules(raw, mask_train):
    """B1 rule screening. Thresholds are fitted on TRAINING years only."""
    tr = raw[mask_train]
    tr = tr[np.isfinite(tr)]
    if tr.size < 500:
        return None
    lo, hi = np.quantile(tr, 0.0005), np.quantile(tr, 0.9995)
    dtr = np.abs(np.diff(tr))
    rate = float(np.quantile(dtr[dtr > 0], 0.9995)) if (dtr > 0).any() else np.inf
    runs = run_lengths(tr)
    flat = max(int(np.quantile(runs, 0.999)), 8)
    return dict(lo=float(lo), hi=float(hi), rate=rate, flat=flat)


def apply_rules(raw, rules):
    """Return raw with rule-flagged points set to NaN (causal: uses only raw)."""
    if rules is None:
        return raw.copy()
    x = raw.copy()
    bad = ~np.isfinite(x)
    with np.errstate(invalid='ignore'):
        bad |= (x < rules['lo']) | (x > rules['hi'])
        d = np.abs(np.diff(x, prepend=x[0]))
        bad |= np.isfinite(d) & (d > rules['rate'])
    bad |= run_lengths(x) >= rules['flat']
    x[bad] = np.nan
    return x


def gate1():
    """Headroom: identity error, and what survives rule screening."""
    rows = []
    for site in SITES:
        loaded = load_site(site)
        if loaded is None:
            continue
        d, step = loaded
        yr = d.datetime.dt.year.to_numpy()
        mtr = (yr >= TRAIN_YEARS[0]) & (yr <= TRAIN_YEARS[1])
        for var in ALLVARS:
            raw = d[var].to_numpy(float)
            cor = d['%s_cor' % var].to_numpy(float)
            scale = series_scale(cor, mtr)
            if not np.isfinite(scale) or scale <= 0:
                continue
            rules = fit_rules(raw, mtr)
            screened = apply_rules(raw, rules)
            for w in WINDOWS:
                er, ec, es = rolling_all(raw, w), rolling_all(cor, w), rolling_all(screened, w)
                for est in ['median', 'madn', 'trimmed_mean', 'trimmed_std', 'mean', 'std']:
                    mae, peak, nz, n = paired_error(er[est], ec[est])
                    rmae, rpeak, rnz, _ = paired_error(es[est], ec[est])
                    rows.append(dict(site=site, var=var, w=w, est=est, n_windows=n, scale=scale,
                                     mae=mae / scale, peak=peak / scale, nonzero=nz,
                                     res_mae=rmae / scale, res_peak=rpeak / scale, res_nonzero=rnz))
            shown = [r for r in rows[-12:] if r['est'] in ('median', 'trimmed_mean')]
            print('  %-14s %-4s scale=%8.3f  ' % (site, var, scale)
                  + '  '.join('W%d/%s mae %.3f->%.3f' % (r['w'], r['est'], r['mae'], r['res_mae'])
                              for r in shown), flush=True)
    r = pd.DataFrame(rows)
    r.to_csv(os.path.join(OUT, 'g1_headroom.csv'), index=False)
    prim = r[(r['var'].isin(PRIMARY)) & (r.est == 'trimmed_mean') & (r.w == WINDOWS[0])]
    med_mae = prim.mae.median()
    n_peak = int((prim.peak >= G1_MIN_PEAK).sum())
    nz = prim.nonzero.median()
    res_mae = prim.res_mae.median()
    ok = (med_mae >= G1_MIN_MEDIAN_MAE and n_peak >= G1_MIN_PEAK_SERIES
          and nz >= G1_MIN_NONZERO_FRAC and res_mae >= G1_MIN_RESIDUAL_MAE)
    print('\nG1 (primary vars, trimmed_mean, W=%d, %d series):' % (WINDOWS[0], len(prim)))
    print('   median normalised MAE %.4f (need >= %.2f)' % (med_mae, G1_MIN_MEDIAN_MAE))
    print('   series with PEAK >= %.1f: %d (need >= %d)' % (G1_MIN_PEAK, n_peak, G1_MIN_PEAK_SERIES))
    print('   median non-zero window fraction %.4f (need >= %.2f)' % (nz, G1_MIN_NONZERO_FRAC))
    print('   median MAE AFTER rule screening %.4f (need >= %.2f)' % (res_mae, G1_MIN_RESIDUAL_MAE))
    print('   ->  %s' % ('PASS' if ok else 'FAIL'))
    if not ok:
        print('   FAIL ACTION: this is the ATS/LTP small-oracle-gain trap. Stop on this dataset.')
    return ok, r


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    which = sys.argv[1:] or ['g0', 'g1']
    results = {}
    if 'g0' in which:
        print('=== G0 INVENTORY ===')
        results['g0'] = gate0()[0]
    if 'g1' in which:
        print('\n=== G1 HEADROOM ===')
        results['g1'] = gate1()[0]
    print('\nGATES: %s' % {k: ('PASS' if v else 'FAIL') for k, v in results.items()})
