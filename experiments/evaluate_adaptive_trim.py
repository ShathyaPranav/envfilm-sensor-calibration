"""Part B evaluation: ATS (rule) and LTP (learned) adaptive-trim methods,
scored on the same 3 held-out splits as RQ3 (unseen severity,
leave-one-mechanism-out x7, leave-one-domain-out x2), against fixed-tau,
plain std/madn/IQR_scale baselines, and the per-window oracle ceiling.

Strict constraints enforced by construction:
  - ATS's bin edges and LTP's model weights are fit on TRAIN rows only,
    never on TEST rows, for every split.
  - All 9 LTP features and all 3 candidate ATS proxies are computed from
    the dirty window alone (see run_adaptive_trim.py) -- no clean-side
    reference reaches either method at evaluation time.
  - No hyperparameter search against test-split performance: GBT/MLP use
    one fixed hyperparameter setting across every split.
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats as spstats
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

_EXP_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_EXP_DIR)
for p in (_EXP_DIR, _REPO_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from run_phase2a_full import ALL_MECH
from phase2a_full_analysis import severity_split_regimes

DATA_DIR = os.path.join(_REPO_ROOT, 'results', 'adaptive_trim')
IN_PATH = os.path.join(DATA_DIR, 'adaptive_trim_percell.csv.gz')

TAUS = np.array([0.0, 0.05, 0.10, 0.20, 0.30])
ERR_COLS = {0.0: 'err_tau00', 0.05: 'err_tau05', 0.10: 'err_tau10', 0.20: 'err_tau20', 0.30: 'err_tau30'}
PROXY_COLS = ['gap_trim_untrim', 'gap_mean_median', 'gap_iqr_std']
LTP_FEATURES = ['gap_trim_untrim', 'gap_mean_median', 'gap_iqr_std', 'skew', 'kurt',
                 'flat_run', 'nan_frac', 'uniq_ratio', 'ac1', 'ts_viol']


def load_data():
    df = pd.read_csv(IN_PATH)
    for c in ERR_COLS.values():
        df[c] = df[c].astype(float)
    df = df.replace([np.inf, -np.inf], np.nan)
    # drop rows with any missing feature/error (unusable for both methods)
    needed = LTP_FEATURES + list(ERR_COLS.values()) + ['oracle_tau', 'err_std', 'err_madn', 'err_iqr']
    before = len(df)
    df = df.dropna(subset=needed)
    print(f"Loaded {before} rows, {len(df)} after dropping incomplete rows ({before-len(df)} dropped).")
    return df


def snap_to_tau(pred):
    pred = np.clip(pred, TAUS.min(), TAUS.max())
    idx = np.argmin(np.abs(pred[:, None] - TAUS[None, :]), axis=1)
    return TAUS[idx]


def lookup_err(df, tau_choice):
    """tau_choice: array of tau values (one of TAUS) aligned to df rows. Returns the
    precomputed error at that tau for each row."""
    out = np.empty(len(df))
    for tau, col in ERR_COLS.items():
        mask = np.isclose(tau_choice, tau)
        out[mask] = df.loc[mask, col].to_numpy()
    return out


# ---------------------------------------------------------------------------
# Step 1: pick the ATS proxy by correlation with true normalized perturbation
# ---------------------------------------------------------------------------

def select_ats_proxy(df):
    true_pert = df['err_std'] / df['channel_sigma']
    correlations = {}
    for col in PROXY_COLS:
        rho, p = spstats.spearmanr(df[col], true_pert)
        correlations[col] = dict(spearman_rho=float(rho), p=float(p))
    best = max(correlations, key=lambda c: abs(correlations[c]['spearman_rho']))
    sign = 1 if correlations[best]['spearman_rho'] >= 0 else -1
    print("ATS candidate proxies (Spearman rho vs. true normalized perturbation MAE_std/sigma):")
    for col, r in correlations.items():
        marker = ' <== selected' if col == best else ''
        print(f"  {col}: rho={r['spearman_rho']:.4f} (p={r['p']:.2e}){marker}")
    return best, sign, correlations


# ---------------------------------------------------------------------------
# Step 2: split builders (reusing RQ3's regime-partition logic)
# ---------------------------------------------------------------------------

def build_splits(df):
    all_regimes = sorted(df['regime'].unique().tolist())
    splits = []

    train_r, test_r = severity_split_regimes()
    train_r = [r for r in train_r if r in all_regimes]
    test_r = [r for r in test_r if r in all_regimes]
    splits.append(dict(split='unseen_severity', held_out='unseen_severity',
                        train_mask=df['regime'].isin(train_r), test_mask=df['regime'].isin(test_r)))

    for mech in ALL_MECH:
        held_regimes = df.loc[df['mechanism'] == mech, 'regime'].unique().tolist()
        train_regimes = [r for r in all_regimes if r not in held_regimes]
        splits.append(dict(split='leave_one_mechanism_out', held_out=mech,
                            train_mask=df['regime'].isin(train_regimes), test_mask=df['regime'].isin(held_regimes)))

    for train_dom, test_dom, label in [('SKAB', 'UCI', 'train_SKAB_test_UCI'),
                                         ('UCI', 'SKAB', 'train_UCI_test_SKAB')]:
        splits.append(dict(split='leave_one_domain_out', held_out=label,
                            train_mask=df['domain'] == train_dom, test_mask=df['domain'] == test_dom))
    return splits


# ---------------------------------------------------------------------------
# Step 3: ATS (quantile-binned monotone rule, fit on TRAIN proxy values only,
# never on oracle labels -- "no training" in the ML-fitting sense)
# ---------------------------------------------------------------------------

def fit_ats_edges(train_proxy, sign):
    edges = np.quantile(train_proxy, [0.2, 0.4, 0.6, 0.8])
    order = TAUS if sign > 0 else TAUS[::-1]
    return edges, order


def apply_ats(proxy_values, edges, order):
    bins = np.digitize(proxy_values, edges)  # 0..4
    return order[bins]


# ---------------------------------------------------------------------------
# Step 4: LTP models
# ---------------------------------------------------------------------------

def fit_gbt(X_train, y_train):
    m = HistGradientBoostingRegressor(max_depth=6, max_iter=150, learning_rate=0.08, random_state=0)
    m.fit(X_train, y_train)
    return m


def fit_mlp(X_train, y_train):
    scaler = StandardScaler().fit(X_train)
    Xs = scaler.transform(X_train)
    m = MLPRegressor(hidden_layer_sizes=(32, 16), activation='relu', max_iter=300,
                      early_stopping=True, n_iter_no_change=10, random_state=0)
    m.fit(Xs, y_train)
    return m, scaler


def main():
    t0 = time.time()
    df = load_data()

    ats_proxy_col, ats_sign, proxy_corr = select_ats_proxy(df)

    splits = build_splits(df)
    results = []
    timing = dict(ats_us_per_window=[], gbt_us_per_window=[], mlp_us_per_window=[])

    for sp in splits:
        train = df[sp['train_mask']]
        test = df[sp['test_mask']]
        if len(train) < 50 or len(test) < 20:
            print(f"SKIP {sp['split']}/{sp['held_out']}: too few rows (train={len(train)}, test={len(test)})")
            continue

        Xtr = train[LTP_FEATURES].to_numpy()
        Xte = test[LTP_FEATURES].to_numpy()
        ytr = train['oracle_tau'].to_numpy()

        # --- ATS ---
        edges, order = fit_ats_edges(train[ats_proxy_col].to_numpy(), ats_sign)
        t_ats0 = time.time()
        ats_pred = apply_ats(test[ats_proxy_col].to_numpy(), edges, order)
        t_ats1 = time.time()
        ats_err = lookup_err(test, ats_pred)

        # --- LTP: GBT ---
        gbt = fit_gbt(Xtr, ytr)
        t_g0 = time.time()
        gbt_raw = gbt.predict(Xte)
        t_g1 = time.time()
        gbt_pred = snap_to_tau(gbt_raw)
        gbt_err = lookup_err(test, gbt_pred)

        # --- LTP: MLP ---
        mlp, scaler = fit_mlp(Xtr, ytr)
        Xte_s = scaler.transform(Xte)
        t_m0 = time.time()
        mlp_raw = mlp.predict(Xte_s)
        t_m1 = time.time()
        mlp_pred = snap_to_tau(mlp_raw)
        mlp_err = lookup_err(test, mlp_pred)

        # --- baselines ---
        fixed_scores = {tau: train[col].mean() for tau, col in ERR_COLS.items()}
        best_fixed_tau = min(fixed_scores, key=fixed_scores.get)
        best_fixed_test_err = test[ERR_COLS[best_fixed_tau]].mean()

        std_err = test['err_std'].mean()
        madn_err = test['err_madn'].mean()
        iqr_err = test['err_iqr'].mean()
        oracle_err = test['oracle_err'].mean()

        row = dict(
            split=sp['split'], held_out=sp['held_out'], n_train=len(train), n_test=len(test),
            best_fixed_tau=best_fixed_tau, best_fixed_err=float(best_fixed_test_err),
            std_err=float(std_err), madn_err=float(madn_err), iqr_err=float(iqr_err),
            oracle_err=float(oracle_err),
            ats_err=float(ats_err.mean()), gbt_err=float(gbt_err.mean()), mlp_err=float(mlp_err.mean()),
        )
        for name in ['std', 'madn', 'iqr', 'ats', 'gbt', 'mlp', 'oracle']:
            key = f'{name}_err' if name != 'oracle' else 'oracle_err'
            row[f'{name}_pct_vs_fixed'] = 100.0 * (row[key] - best_fixed_test_err) / best_fixed_test_err
        results.append(row)

        timing['ats_us_per_window'].append(1e6 * (t_ats1 - t_ats0) / len(test))
        timing['gbt_us_per_window'].append(1e6 * (t_g1 - t_g0) / len(test))
        timing['mlp_us_per_window'].append(1e6 * (t_m1 - t_m0) / len(test))

        print(f"[{sp['split']}/{sp['held_out']}] n_train={len(train)} n_test={len(test)} "
              f"best_fixed=tau{best_fixed_tau:.2f} err={best_fixed_test_err:.5g} | "
              f"std={row['std_pct_vs_fixed']:+.1f}% madn={row['madn_pct_vs_fixed']:+.1f}% iqr={row['iqr_pct_vs_fixed']:+.1f}% "
              f"ats={row['ats_pct_vs_fixed']:+.1f}% gbt={row['gbt_pct_vs_fixed']:+.1f}% mlp={row['mlp_pct_vs_fixed']:+.1f}% "
              f"oracle={row['oracle_pct_vs_fixed']:+.1f}%")

    res_df = pd.DataFrame(results)
    res_df.to_csv(os.path.join(DATA_DIR, 'adaptive_trim_results.csv'), index=False)

    summary = dict(
        ats_proxy_selected=ats_proxy_col, ats_proxy_correlations=proxy_corr, ats_sign=ats_sign,
        n_splits=len(res_df),
        timing_us_per_window={k: float(np.mean(v)) for k, v in timing.items()},
        mean_pct_vs_fixed={name: float(res_df[f'{name}_pct_vs_fixed'].mean())
                            for name in ['std', 'madn', 'iqr', 'ats', 'gbt', 'mlp', 'oracle']},
        n_splits_ats_beats_fixed=int((res_df['ats_pct_vs_fixed'] < 0).sum()),
        n_splits_gbt_beats_fixed=int((res_df['gbt_pct_vs_fixed'] < 0).sum()),
        n_splits_mlp_beats_fixed=int((res_df['mlp_pct_vs_fixed'] < 0).sum()),
        n_splits_ltp_beats_ats=int(((res_df['gbt_pct_vs_fixed'] < res_df['ats_pct_vs_fixed']) |
                                     (res_df['mlp_pct_vs_fixed'] < res_df['ats_pct_vs_fixed'])).sum()),
    )
    with open(os.path.join(DATA_DIR, 'adaptive_trim_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print("\n=== Summary (mean %% vs. best-fixed-tau baseline across all", len(res_df), "splits; negative = better) ===")
    for name in ['std', 'madn', 'iqr', 'ats', 'gbt', 'mlp', 'oracle']:
        print(f"  {name}: {summary['mean_pct_vs_fixed'][name]:+.2f}%")
    print("Timing (microseconds per window, averaged over splits):", summary['timing_us_per_window'])
    print(f"\nTotal elapsed: {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
