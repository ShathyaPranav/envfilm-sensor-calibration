"""What exactly happens in the 2.3-2.4 V transition band, and how smoothly does
reliability decay with voltage?

The saturation faults below 2.3 V are trivial (122.153 C, caught by any range
check). The hard, realistic faults are in the transition band, where readings are
still physically plausible but the spread widens ~3x. Any proposed algorithm has
to earn its keep there, so the band's behaviour needs characterising before
designing one:

  - is it elevated NOISE (variance up, level unchanged),
  - an OFFSET (level shifted),
  - intermittent SATURATION spikes mixed into healthy readings,
  - or a drift?

And the reliability curve: P(implausible | voltage) at fine resolution. If
reliability decays smoothly rather than as a step, a learned per-reading
reliability function of voltage has something continuous to fit; if it is a hard
step, a single threshold does the whole job and no learning is warranted.
"""
import os
import pickle
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore', category=RuntimeWarning)
DQ = r"C:\Users\luckf\OneDrive\Desktop\dq-estimator-eval"
W = 30
SAT = 122.153


def main():
    d = pickle.load(open(os.path.join(DQ, 'data', 'intel_lab', 'intel_parsed.pkl'), 'rb'))
    df = d['df'].sort_values(['moteid', 'epoch']).copy()
    df['is_sat'] = (df.temp - SAT).abs() < 0.01
    df['implausible'] = (df.temp < 0) | (df.temp > 50)
    healthy_med = df[df.volt >= 2.5].groupby('moteid')['temp'].median()
    df['offset'] = df.temp - df.moteid.map(healthy_med)
    df['roll_std'] = df.groupby('moteid')['temp'].transform(
        lambda s: s.rolling(W, min_periods=W // 2).std())
    df['absd'] = df.groupby('moteid')['temp'].diff().abs()

    print("=== behaviour by voltage band ===")
    bands = [('healthy >=2.5', df.volt >= 2.5), ('2.4-2.5', (df.volt >= 2.4) & (df.volt < 2.5)),
             ('2.3-2.4', (df.volt >= 2.3) & (df.volt < 2.4)), ('<2.3', df.volt < 2.3)]
    rows = []
    for name, m in bands:
        g = df[m]
        pl = g[~g.implausible]
        rows.append(dict(band=name, n=len(g), frac_saturated=g.is_sat.mean(),
                         frac_implausible=g.implausible.mean(),
                         plausible_roll_std_med=pl.roll_std.median(),
                         plausible_absd_med=pl.absd.median(),
                         plausible_offset_med=pl.offset.median(),
                         plausible_offset_iqr=pl.offset.quantile(.75) - pl.offset.quantile(.25)))
    print(pd.DataFrame(rows).round(4).to_string(index=False))

    print("\n=== inside the transition band: are saturations interleaved with healthy readings? ===")
    g = df[(df.volt >= 2.3) & (df.volt < 2.4)].copy()
    g['prev_sat'] = g.groupby('moteid')['is_sat'].shift(1)
    g['next_sat'] = g.groupby('moteid')['is_sat'].shift(-1)
    sat_in_band = g[g.is_sat]
    iso = sat_in_band[(sat_in_band.prev_sat == False) & (sat_in_band.next_sat == False)]
    print(f"  saturated readings in band: {len(sat_in_band):,}; isolated (both neighbours "
          f"non-saturated): {len(iso):,} ({100*len(iso)/max(len(sat_in_band),1):.1f}%)")

    print("\n=== reliability curve P(implausible | voltage), 0.01 V bins ===")
    fine = df[(df.volt >= 2.20) & (df.volt < 2.55)].copy()
    fine['vb'] = (fine.volt // 0.01) * 0.01
    curve = fine.groupby('vb').agg(n=('temp', 'size'), p_implausible=('implausible', 'mean'),
                                   p_sat=('is_sat', 'mean'))
    curve = curve[curve.n >= 500]
    print(curve.round(4).to_string())

    print("\n=== per-mote heterogeneity of the onset voltage (first V where p_implausible > 0.5) ===")
    onsets = []
    for mote, gm_ in df.groupby('moteid'):
        gm_ = gm_.copy()
        gm_['vb'] = (gm_.volt // 0.01) * 0.01
        c = gm_.groupby('vb')['implausible'].agg(['mean', 'size'])
        c = c[c['size'] >= 50].sort_index(ascending=False)
        hit = c[c['mean'] > 0.5]
        if len(hit):
            onsets.append(hit.index.max())
    onsets = np.array(onsets)
    if len(onsets):
        print(f"  motes with an observed onset: {len(onsets)}; onset voltage "
              f"min {onsets.min():.2f}  median {np.median(onsets):.2f}  max {onsets.max():.2f}  "
              f"IQR {np.percentile(onsets,25):.2f}-{np.percentile(onsets,75):.2f}")


if __name__ == '__main__':
    main()
