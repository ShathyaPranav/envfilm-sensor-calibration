# Seed replication of the two rotation tests (written 2026-10-11, before running)

Same protocol and models as rot_filmb (TESLA benchmark, 195 units) and ref10_rot (true reference, 153 units), with
training seed 1 instead of 0 (TSL_ROT_SEED=1). Not a new regime: the outcome is reported in the paper whatever it is.
Replicates if, on each benchmark, EnvFiLM-b is better than TCN+ENV on >= 55% of units with pooled median ratio < 1.

## True-reference seed-1 outcome (2026-10-11)
153 units: EnvFiLM-b better than TCN+ENV on 51.6%, median ratio 0.997, Wilcoxon p = 0.16 -> does NOT replicate seed 0
(88.9%, 0.893). EnvFiLM-b seed1/seed0 median 1.085; TCN+ENV 0.981: EnvFiLM-b is more seed-sensitive. Two-seed average:
better on 79.1%, median 0.950. Reported in the paper.

## Follow-up (written before running): seeds 2 and 3 on the true-reference rotation
The paper will report, for the true-reference rotation, per-seed results and the comparison of the per-unit RMSE averaged
over seeds 0 to 3 for each model. The architectural claim on this benchmark holds if the 4-seed average is better on >= 55%
of units with median ratio < 1; the paper reports the outcome either way.

## True-reference seeds 2 and 3 outcome (2026-10-11)
Per seed (EnvFiLM-b better than TCN+ENV / median ratio): s0 88.9% / 0.893, s1 51.6% / 0.997, s2 49.0% / 1.002,
s3 88.2% / 0.896. Pre-registered 4-seed average of per-unit RMSE: better on 83.7% of 153 units, median ratio 0.946,
Wilcoxon p = 1.8e-17 -> HOLDS. Bimodal: two trainings clearly better, two tied, none worse.
Post hoc (not pre-registered): keeping, per fold, the seed with the lowest validation RMSE for each model: 73.2%, 0.951.
