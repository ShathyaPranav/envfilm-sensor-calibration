# Pre-registration 2: LogTCN on the TESLA/SCARE benchmark (written 2026-10-04, before the run)

Candidate (frozen): class TCN in experiments/tsl_bench.py ("LogTCN"): log1p-scaled input, seven dilated
causal convolutions (32 channels, dilations 1..64), head on [last state, mean state, last log value]. 23k params.
Recipe for every model: Adam 1e-3, batch 512, 10 epochs, MSE, best epoch on the validation unit.
Data: TESLA protocol rebuilt from SensEURCity v03 (tsl_data.py); numbers are close to, not identical with, theirs.
Already seen before freezing: Oslo PM2.5 test (4.47 vs TESLA re-impl 7.04).

WIN, all required:
 1. Per cell (3 cities x 3 PM sizes), LogTCN mean over 3 seeds is below published TESLA in >= 8 of 9 cells
    and below our TESLA re-implementation (1 seed) in >= 8 of 9 cells.
 2. City-averaged RMSE per PM size below published SCARE (14.03 / 5.67 / 2.51) and published TESLA
    (14.21 / 6.19 / 2.82) for all three PM sizes, by >= 10% for at least two of them.
 3. Robustness: in the unit rotation of tsl_rotate.py (PM2.5, all cities), LogTCN beats the TESLA
    re-implementation on >= 75% of units.
Reported regardless: ablation rows (raw-scale TCN, small TCN), parameter counts, top-5% sudden-change RMSE.
