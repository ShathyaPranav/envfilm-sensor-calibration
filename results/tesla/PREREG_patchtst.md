# Pre-registration: PatchTST baseline on the official TESLA split

Written 2026-10-11, before any PatchTST result exists.

## Why
A reviewer may ask how EnvFiLM-b compares with a modern patch Transformer. PatchTST
(Nie et al., ICLR 2023) is cited in the paper and trains from scratch (no downloads).

## Models (experiments/tsl_bench.py)
- PatchTST: particle channel only (log1p), patch 16, stride 8, end padding, d_model 64,
  3 encoder layers, 4 heads, feed-forward 128, dropout 0.2, instance normalisation per
  window; window mean, spread and last log value go to the output MLP.
- PatchTST+ENV: the same, channel-independent over the particle channel plus the four
  on-board channels (same inputs as EnvFiLM-b and TCN+ENV).
- Sizes fixed here, not tuned. Same recipe as every other model (Adam 1e-3, batch 512,
  10 epochs, best validation epoch), 3 seeds, all 9 official cells, TSL_SOURCE=orig.

## Reporting
Same benchmark and regime as the main table, so the result is reported whatever it is:
both rows go into Table main with 3-seed means. If PatchTST+ENV beats EnvFiLM-b on the
city-mean of a PM size, the paper says so and the claims are revised accordingly.

## Outcome (2026-10-12)
PatchTST (particles only): worse than EnvFiLM-b and than LogTCN in 9/9 cells (EnvFiLM-b / PatchTST mean ratio 0.65).
PatchTST+ENV: EnvFiLM-b lower in 6/9 cells (mean ratio 0.942); PatchTST+ENV lower in PM2.5 Zagreb (2.40 vs 2.72),
PM1 Oslo (1.29 vs 1.32), PM1 Zagreb (0.85 vs 1.07). City means EnvFiLM-b 8.04/2.71/1.36 vs PatchTST+ENV 8.61/3.11/1.53,
so the pre-registered revision trigger (PatchTST+ENV better on a size's city mean) did not fire. Both rows in tab_main.
