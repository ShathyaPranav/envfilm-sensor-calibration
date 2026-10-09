# Pre-registration: gas-drift model (written 2026-10-06, before building the model)

Benchmark: UCI Gas Sensor Array Drift, protocol of AMDS-PFFA (arXiv 2409.13167): labelled sources batches 1+2,
each target batch 3..10 with features available and labels hidden; accuracy per target batch; headline =
sample-weighted mean over targets (reproduces their 83.21 from their own per-batch numbers).
Simple baseline measured: per-batch z-score + MLP = 73.95 weighted.

Model selection rule: ALL design and hyperparameter choices are made on a development task that never
touches batches 3..10: source batch 1 (labelled) -> target batch 2 (labels used ONLY to score the dev
task). Batches 3..10 are scored once, with the frozen configuration, 5 seeds.

WIN (all required):
 1. weighted mean accuracy over batches 3..10 (mean of 5 seeds) above AMDS-PFFA's 83.21;
 2. better than AMDS-PFFA on at least 5 of the 8 target batches;
 3. the novel component matters: the same network without it loses at least 2 points (weighted).
