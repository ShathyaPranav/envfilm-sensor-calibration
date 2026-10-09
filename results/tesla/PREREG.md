# Pre-registration: gain-calibration model on the TESLA benchmark (written 2026-10-03, before the decisive runs)

Candidate (frozen, no changes after this line): GainTCN+ctx+aug in experiments/tsl_bench.py
  - dilated causal TCN trunk over the 360-minute window, log-scaled input
  - output = learned short average of the latest readings x exp(predicted log-gain) + small offset
  - label-free context: 10/50/90th percentile of the unit's own input channel over the past 3 days
  - training-time unit-gain augmentation (log-gain sd 0.3) applied to window and context, not to the target
Probes seen before freezing: Oslo PM2.5 only (test unit 653257).

Comparators: TESLA (PyTorch re-implementation of the official code, same sizes), Linear,
and a strong generic TCN (same trunk, additive head, no context, no augmentation).
All get the same training recipe (Adam 1e-3, batch 512, 10 epochs, MSE, best epoch on the validation unit).

WIN, both required:
 A. Official protocol (fixed unit lists; last unit tests, second-to-last validates), mean of 3 seeds:
    candidate RMSE below the TESLA re-implementation AND below published TESLA in at least 7 of 9 city x PM cells.
 B. Unit rotation (every unit tests once; 5 folds per city; validation = 2 other units; training windows
    taken every 4th row for all models alike; 1 seed): against the generic TCN, over all test units of all
    9 cells, the candidate is better on at least 60% of units with a median per-unit gain of at least 3%,
    and its median RMSE is not worse by more than 3% in any city x PM cell.
If A holds and B fails, the claim is only that a generic TCN beats TESLA, which is not our model: not a win.

## Outcome (2026-10-04)
B stopped early after Antwerp PM10 folds 0-3: candidate better than the generic TCN on about 5 of 24 units. Criterion B cannot plausibly pass. Candidate FAILED; A not completed.
