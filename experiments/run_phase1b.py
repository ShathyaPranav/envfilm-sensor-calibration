"""Phase 1B orchestration: run the incident experiment + clean-stream
baseline across 10 real-data segments, aggregate, compare against Phase 0.
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import phase1b_real as p1b
from harness.pipeline import W_S

OUT_DIR = os.path.join(_REPO_ROOT, 'results', 'phase1b')
T_CRIT_95 = stats.t.ppf(0.975, df=p1b.N_SEGMENTS - 1)  # df=9


def ci95(values):
    values = np.asarray(values, dtype=float)
    sem = values.std(ddof=1) / np.sqrt(len(values))
    mean = values.mean()
    return mean - T_CRIT_95 * sem, mean + T_CRIT_95 * sem


def summarize(values, label):
    values = np.asarray(values, dtype=float)
    lo, hi = ci95(values)
    return {f'{label}_mean': float(values.mean()), f'{label}_median': float(np.median(values)),
            f'{label}_sd': float(values.std(ddof=1)), f'{label}_ci95_lo': float(lo), f'{label}_ci95_hi': float(hi)}


def cohens_d(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    na, nb = len(a), len(b)
    pooled_sd = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return float((a.mean() - b.mean()) / pooled_sd) if pooled_sd > 0 else float('nan')


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.time()

    print("=== Phase 1B: loading & selecting real-data segments ===")
    segments = p1b.select_segments()
    seg_meta = pd.DataFrame([{k: v for k, v in s.items() if k != 'x'} for s in segments])
    seg_meta.to_csv(os.path.join(OUT_DIR, 'segments.csv'), index=False)
    print(seg_meta[['label', 'start_time', 'end_time', 'n_rows']].to_string(index=False))

    print("\n=== Phase 1B part 1: incident experiment ===")
    all_rows = []
    max_resid = 0.0
    for k, seg in enumerate(segments):
        for fm in ('meanvol', 'robust'):
            df, resid = p1b.run_incident_real(seg, fm, N=p1b.N_INCIDENTS, seed_for_rng=k + 1)
            df['estimator'] = 'naive' if fm == 'meanvol' else 'robust'
            all_rows.append(df)
            max_resid = max(max_resid, resid)
    combined = pd.concat(all_rows, ignore_index=True)
    print(f"  post-full-exit residual (should be ~0): {max_resid:.3e}")
    combined.to_csv(os.path.join(OUT_DIR, 'incident_level.csv'), index=False)

    metrics = ['mae_loc', 'peak_loc', 'mae_spr', 'peak_spr']
    naive = combined[combined.estimator == 'naive'].set_index(['segment', 'incident_id'])
    robust = combined[combined.estimator == 'robust'].set_index(['segment', 'incident_id'])
    assert naive.index.equals(robust.index)

    # per-segment x type
    seg_type_rows = []
    for (seg, typ), g_n in naive.groupby(['segment', naive['type']]):
        g_r = robust.loc[g_n.index]
        row = dict(segment=seg, type=typ, n=len(g_n))
        for met in metrics:
            nm, rm = g_n[met].mean(), g_r[met].mean()
            row[f'naive_{met}'], row[f'robust_{met}'] = nm, rm
            row[f'ratio_{met}'] = rm / nm if nm > 0 else float('nan')
        seg_type_rows.append(row)
    seg_type_df = pd.DataFrame(seg_type_rows)
    seg_type_df.to_csv(os.path.join(OUT_DIR, 'segment_type_summary.csv'), index=False)

    by_type_rows = []
    for typ, g in seg_type_df.groupby('type'):
        row = dict(type=typ, n_segments=len(g))
        for met in metrics:
            row.update({f'naive_{k}': v for k, v in summarize(g[f'naive_{met}'], met).items()})
            row.update({f'robust_{k}': v for k, v in summarize(g[f'robust_{met}'], met).items()})
            row.update({f'ratio_{k}': v for k, v in summarize(g[f'ratio_{met}'], met).items()})
        by_type_rows.append(row)
    by_type_df = pd.DataFrame(by_type_rows)
    by_type_df.to_csv(os.path.join(OUT_DIR, 'across_segment_by_type.csv'), index=False)

    # pooled across all primary types, per segment
    pooled_seg_rows = []
    for seg, g_n in naive.groupby('segment'):
        g_r = robust.loc[g_n.index]
        row = dict(segment=seg, n=len(g_n))
        for met in metrics:
            nm, rm = g_n[met].mean(), g_r[met].mean()
            row[f'naive_{met}'], row[f'robust_{met}'] = nm, rm
            row[f'ratio_{met}'] = rm / nm if nm > 0 else float('nan')
        pooled_seg_rows.append(row)
    pooled_seg_df = pd.DataFrame(pooled_seg_rows)
    pooled_seg_df.to_csv(os.path.join(OUT_DIR, 'pooled_per_segment.csv'), index=False)

    pooled_summary = dict(n_segments=len(pooled_seg_df))
    for met in metrics:
        pooled_summary.update({f'naive_{k}': v for k, v in summarize(pooled_seg_df[f'naive_{met}'], met).items()})
        pooled_summary.update({f'robust_{k}': v for k, v in summarize(pooled_seg_df[f'robust_{met}'], met).items()})
        pooled_summary.update({f'ratio_{k}': v for k, v in summarize(pooled_seg_df[f'ratio_{met}'], met).items()})

    print(f"  elapsed so far: {time.time()-t0:.1f}s")

    print("\n=== Phase 1B part 2: clean-stream baseline ===")
    clean_rows = []
    for k, seg in enumerate(segments):
        loc_n, spr_n = p1b.run_clean_baseline_real(seg, 'meanvol')
        loc_r, spr_r = p1b.run_clean_baseline_real(seg, 'robust')
        clean_rows.append(dict(
            segment=seg['label'],
            loc_err_mean_naive=float(loc_n.mean()), loc_err_mean_robust=float(loc_r.mean()),
            loc_err_median_naive=float(np.median(loc_n)), loc_err_median_robust=float(np.median(loc_r)),
            spr_err_mean_naive=float(spr_n.mean()), spr_err_mean_robust=float(spr_r.mean()),
            spr_err_median_naive=float(np.median(spr_n)), spr_err_median_robust=float(np.median(spr_r)),
        ))
        # keep raw arrays for pooled effect sizes
        clean_rows[-1]['_loc_n'] = loc_n
        clean_rows[-1]['_loc_r'] = loc_r
        clean_rows[-1]['_spr_n'] = spr_n
        clean_rows[-1]['_spr_r'] = spr_r
    clean_df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith('_')} for r in clean_rows])
    clean_df['loc_ratio_mean'] = clean_df['loc_err_mean_robust'] / clean_df['loc_err_mean_naive']
    clean_df['spr_ratio_mean'] = clean_df['spr_err_mean_robust'] / clean_df['spr_err_mean_naive']
    clean_df.to_csv(os.path.join(OUT_DIR, 'clean_baseline_per_segment.csv'), index=False)

    clean_summary = {}
    for comp in ('loc', 'spr'):
        clean_summary.update({f'{comp}_naive_{k}': v for k, v in
                               summarize(clean_df[f'{comp}_err_mean_naive'], 'errmean').items()})
        clean_summary.update({f'{comp}_robust_{k}': v for k, v in
                               summarize(clean_df[f'{comp}_err_mean_robust'], 'errmean').items()})
        clean_summary.update({f'{comp}_ratio_{k}': v for k, v in
                               summarize(clean_df[f'{comp}_ratio_mean'], 'ratio').items()})
        pooled_naive = np.concatenate([r[f'_{comp}_n'] for r in clean_rows])
        pooled_robust = np.concatenate([r[f'_{comp}_r'] for r in clean_rows])
        clean_summary[f'{comp}_cohens_d'] = cohens_d(pooled_robust, pooled_naive)
        clean_summary[f'{comp}_mse_naive'] = float((pooled_naive ** 2).mean())
        clean_summary[f'{comp}_mse_robust'] = float((pooled_robust ** 2).mean())
        clean_summary[f'{comp}_mse_ratio'] = clean_summary[f'{comp}_mse_robust'] / clean_summary[f'{comp}_mse_naive']

    print(f"  elapsed so far: {time.time()-t0:.1f}s")

    result = dict(
        dataset='UCI Individual Household Electric Power Consumption (id 235)',
        n_segments=p1b.N_SEGMENTS, seg_len_minutes=p1b.SEG_LEN, n_incidents_per_segment=p1b.N_INCIDENTS,
        w_s=W_S,
        max_post_exit_residual=max_resid,
        pooled_incident_summary=pooled_summary,
        clean_baseline_summary=clean_summary,
        by_type=by_type_df.to_dict(orient='records'),
        segments=seg_meta.drop(columns=['raw_mean', 'raw_std']).to_dict(orient='records'),
    )
    with open(os.path.join(OUT_DIR, 'phase1b_result.json'), 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\nTotal elapsed: {time.time()-t0:.1f}s")
    print("Wrote outputs to", OUT_DIR)
    return combined, seg_type_df, by_type_df, pooled_seg_df, pooled_summary, clean_df, clean_summary


if __name__ == '__main__':
    main()
