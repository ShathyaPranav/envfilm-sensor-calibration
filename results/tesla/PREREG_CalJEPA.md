# Pre-registration 3: CalJEPA pretraining (written 2026-10-04, before any fine-tuning result)

Candidate: CalJEPA-TCN = LogTCN (PREREG_LogTCN.md) whose convolution trunk is initialised from
joint-embedding predictive pretraining on unlabelled minute data of all units except every city's
validation and test units (experiments/tsl_jepa.py, CFG steps 20000, batch 512, lr 1e-3, EMA 0.996,
horizons 5/15/60/180 min). Fine-tuning recipe identical to LogTCN from scratch.
Control: LogTCN from scratch, same seeds (logtcn_official run).

WIN (on top of PREREG_LogTCN passing):
 1. Official protocol, 3 seeds: CalJEPA-TCN mean RMSE below LogTCN-from-scratch in >= 6 of 9 cells,
    city-averaged gain >= 3% for at least two PM sizes, and no cell worse by more than 5%.
 2. Unit rotation (PM2.5, all cities, stride 4, seed 0): better than LogTCN on >= 60% of units with a
    median per-unit gain >= 3%.
Pretraining settings will not be changed after seeing any fine-tuning test number; only a collapse
(embedding spread near zero) may trigger a documented fix before fine-tuning.

Collapse fix (2026-10-04, before any fine-tuning): smoke pretraining (2000 steps) gave targets whose
spread across samples was 0.06 (partial collapse). Targets are now standardised per dimension over the
batch, and a VICReg variance term (hinge at std 1) is added on the context embedding. Nothing else changed.
Second collapse fix (same day, before any fine-tuning): with the variance term the effective rank of the
64-d embedding was 3; the VICReg covariance term (weight 0.04, off-diagonal squared covariance / dim) is added.

## Outcome (2026-10-04)
Antwerp PM10, seeds 0 and 1: 11.20 and 11.30 vs LogTCN from scratch 10.18 / 10.39 / 10.53 (mean 10.37). The 3-seed mean cannot come within 5% of scratch unless seed 2 is below 10.17, so the no-cell-worse-than-5% clause fails. Stopped early; CalJEPA as specified FAILED. Validation was also worse (9.19 / 8.65 vs 8.65 / 8.21 / 8.82).
