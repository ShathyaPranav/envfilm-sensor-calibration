# Seed replication of the two rotation tests (written 2026-10-11, before running)

Same protocol and models as rot_filmb (TESLA benchmark, 195 units) and ref10_rot (true reference, 153 units), with
training seed 1 instead of 0 (TSL_ROT_SEED=1). Not a new regime: the outcome is reported in the paper whatever it is.
Replicates if, on each benchmark, EnvFiLM-b is better than TCN+ENV on >= 55% of units with pooled median ratio < 1.
