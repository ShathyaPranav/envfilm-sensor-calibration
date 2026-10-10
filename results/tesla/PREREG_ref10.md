# Q1 feasibility test 1: second benchmark with the TRUE reference (written 2026-10-09, before any run)

Data: same units, periods and train/val/test lists as the TESLA protocol, but target = the station's reference
instrument (FIDAS in Antwerp, TEOM/FIDAS in Oslo; Zagreb excluded: its reference is not sub-hourly), all
signals averaged to 10 minutes, window 36 steps (6 h, same span as TESLA's 360 minutes) (TSL_SOURCE=ref10, TSL_N=36).
Models and hyperparameters unchanged from the main study: EnvFiLM-b, TCN+ENV, LogTCN (TCN, particle only),
TESLA (particle only), TESLA+ENV. No tuning on this benchmark. (EnvFiLM-b's "1-hour change" feature becomes a
change over the whole window at this resolution; left as is.)
Runs: official split, 3 seeds, 2 cities x 3 PM sizes; unit rotation (tsl_rotate.py, stride 4 -> stride kept), seed 0.
EnvFiLM-b generalises to the second benchmark if:
 1. rotation vs TCN+ENV: better on >= 55% of units pooled and pooled median ratio < 1;
 2. official split: lower mean RMSE than TESLA and TESLA+ENV in at least 5 of 6 cells.

## Outcome (2026-10-10)
Rotation (153 units): EnvFiLM-b better than TCN+ENV on 88.9% of units, pooled median ratio 0.893, mean RMSE lower in
6 of 6 groups, worst unit 1.14; sign test p = 3e-24, Wilcoxon p = 2e-23 -> criterion 1 PASSES.
Official split, 3 seeds: EnvFiLM-b lower than TESLA and TESLA+ENV in 6 of 6 cells -> criterion 2 PASSES.
EnvFiLM-b generalises to the true-reference benchmark: SATISFACTORY under the reporting contract.
