# Southampton with ON-BOARD humidity/temperature (written 2026-10-11, before any run)

New-regime test under the reporting contract. Data: Bulot et al. Zenodo 7198378 (2-minute low-cost sensors, FIDAS
reference) plus Zenodo 6637781 (each monitor's own SHT35 humidity and temperature, 10 s, averaged to 2 min). Period
limited by the on-board data: 2020-07-01 to 2020-12-30. Each sensor gets four state channels: its own monitor's SHT35
relative humidity and temperature, and the site weather station's relative humidity and temperature. All models get the
same channels. Window 180 steps (6 h). Unit rotation within sensor type (8 PMS5003, 8 SPS30), 3 PM sizes, stride 4.
Because EnvFiLM-b is seed-sensitive, every model is trained with seeds 0 and 1 and per-unit RMSE is averaged over the two.
EnvFiLM-b replicates if, for the PMS5003 group (primary):
 1. vs TCN+ENV: better on >= 55% of unit cases with pooled median ratio < 1;
 2. vs TESLA+ENV: better on >= 55% of unit cases with pooled median ratio < 1.
If it fails it is out of scope and not reported. If it passes, the paper also states the earlier result with weather-station
channels only (which did not meet criterion 1), because the contrast is part of the evidence.

## Outcome (2026-10-11)
PMS5003 group (primary), 24 unit cases, per-unit RMSE averaged over seeds 0 and 1:
vs TCN+ENV better on 91.7%, median ratio 0.959 -> criterion 1 passes (seed 0: 75.0%, 0.960; seed 1: 87.5%, 0.957);
vs TESLA+ENV better on 100.0%, median 0.859 -> criterion 2 passes.
SPS30 (secondary): vs TCN+ENV 58.3%, 0.988; vs TESLA+ENV 91.7%, 0.821.
SATISFACTORY: reported in the paper together with the weather-station-only run (PREREG_southampton.md).
Caveat for the write-up: the two Southampton runs differ in period (Jul-Dec 2020 vs Jul 2020-Jul 2021), channel set
and number of seeds, so the contrast between them is not a controlled ablation.
