# Pre-registration: summary-feature control (written 2026-10-19, before any run)

Question raised in external review: EnvFiLM-b's head receives the state summary z (last, mean, 1-hour change, min, max
of the four clipped state channels), while TCN+ENV's head receives only the last state values. The gain over TCN+ENV
could therefore come from z rather than from modulation.

## Control
TCN+ENV+z (class TCNxZ in experiments/tsl_bench.py): TCN+ENV with exactly EnvFiLM-b's z in its head. Conv path,
recipe and everything else unchanged. 24,705 parameters.

## Runs
1. Unit rotation on the TESLA benchmark, TSL_SOURCE=orig, the four on-board channels, seeds 0 and 1 (tags
   rot_ctrlz and rot_ctrlz_s1), compared with the existing EnvFiLM-b rotation rows of the same seeds and folds
   (rotate_rot_filmb.csv, rotate_rot_filmb_s1.csv); per-unit RMSE averaged over the two seeds.
2. Official split, three seeds, all nine cells (tag final_ctrlz).

## Criterion (same as the pre-registered architecture rule)
EnvFiLM-b beats TCN+ENV+z if it is better on at least 55% of the 195 unit cases with a pooled median ratio below one.
Also reported: the sensor-level count and sign test, and the official-split cells and city means.

## Reporting
Same benchmark and protocol as the main results, so reported whatever it shows. If the criterion fails, the paper's
claim that modulation beats concatenation is withdrawn and the paper is reframed around state-aware calibration.
