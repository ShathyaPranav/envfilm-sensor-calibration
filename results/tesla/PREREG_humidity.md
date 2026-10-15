# Pre-registration: where does EnvFiLM's advantage come from? (written 2026-10-15, before any run)

Question raised in internal review: the paper motivates EnvFiLM by the multiplicative effect of humidity on
optical sensors but shows no result linking the gain to humidity.

## Design (experiments/tsl_humidity.py)
Official TESLA split, authors' data (TSL_SOURCE=orig), the four on-board channels, EnvFiLM-b and TCN+ENV, seeds 0-2,
unchanged recipe. Models are retrained (GPU training is not bit-reproducible, so RMSEs can differ slightly from the
main table; the main table is not changed). For every test window we store the prediction, the target, and the
ambient relative humidity (SHT31HE) at the last row; for EnvFiLM-b also the modulation strength, the mean over
layers and channels of |scale - 1|.

Humidity bands: terciles of the test unit's ambient humidity, per cell. Per band and model, RMSE per seed, averaged
over the three seeds. r_band = RMSE(EnvFiLM-b) / RMSE(TCN+ENV).

## Primary criterion (humidity link supported)
r_high < r_low in at least 6 of the 9 cells, and the median over cells of log(r_high / r_low) < 0.

## Secondary (descriptive)
- Spearman correlation between the modulation strength and ambient humidity over test windows, per cell and seed;
  number of cells with a positive 3-seed mean correlation.
- r per band per cell, reported in a table or figure.

## Reporting
Same benchmark and test units as the main table, so reported whatever it shows. If the primary criterion fails,
the paper says the gain is not concentrated in humid conditions and softens the physical motivation accordingly.

## Outcome (2026-10-15)
Primary criterion FAILS: r_high < r_low in 3 of 9 cells; median log(r_high/r_low) = +0.040. The advantage is larger in
the driest tercile in 6 of 9 cells. Modulation strength vs ambient RH: 3-seed mean Spearman negative in 8 of 9 cells
(positive only in PM1 Oslo). Reported in the paper's limitations ("What the modulation responds to").
