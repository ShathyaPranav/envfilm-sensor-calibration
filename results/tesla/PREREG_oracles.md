# Triage oracles (written 2026-10-06, before running them)

Diagnostics only; oracle fits use labels of held-out TRAINING units (dev folds of tsl_dev.py), never the
official test units. LogTCN = plain TCN, trained per city x PM on the dev fold's training units.
Per held-out unit u with LogTCN prediction p and target y:
 O1 scalar gain: g = sum(y p)/sum(p^2); RMSE_O1 = RMSE(y, g p); gain share = 1 - RMSE_O1^2 / RMSE^2
 O2 state dependence: oracle affine (a p + b) and oracle tiny MLP on (p, log p) fit on even days, scored on odd
    days; compare with the scalar gain fitted the same way
 O3 drift: scalar gain fitted per day; spread (IQR/median) of daily gains
 O4 floor: RMSE_O1 vs the target noise floor sqrt(var(diff y)/2)
Decision rules:
 - Idea 1 (Wiener-Hammerstein) proceeds only if median RMSE_O1 >= 2 x noise floor.
 - Idea 5 (amortised few-shot unit calibrator) proceeds as a neural model only if the oracle MLP beats the
   oracle scalar gain by >= 5% (median over units); otherwise the unit effect is scalar and no new model is
   justified.
 - Idea 6 (drifting gain) proceeds only if the daily-gain IQR/median >= 0.10 (median over units).
Gas drift (idea 8): with the batch-1+2 classifier (source z-scoring, logistic regression) fixed, fit with
target labels (oracle) a per-SENSOR gain (16 numbers, applied to all 8 features of a sensor) on each target
batch. Idea 8 proceeds only if this oracle's weighted accuracy over batches 3..10 exceeds 83.21 (AMDS-PFFA).

## Gas-drift oracle outcome (2026-10-06)
Pre-registered per-sensor gain (16 numbers): weighted 79.09 < 83.21 -> idea 8 FAILS its pre-registered triage.
Also computed (not pre-registered; uses target labels, so cannot be used to choose a design): per-sensor gain + steady-state offset (32) 87.15; per-feature affine (256) 98.08.

## TESLA oracle outcome (2026-10-06)
Median over held-out training units: RMSE_O1 / noise floor 4.86 (idea 1 proceeds); oracle MLP vs scalar gain -3.7% (idea 5 not justified as a neural model); daily-gain IQR/median 0.447 (idea 6 proceeds). Scalar unit gain explains only 0-38% of error variance: earlier claim that residual error is dominated by target-unit gain was too strong; most remaining error varies day to day.
