# Q1 feasibility test 3: independent dataset, Southampton UK (written 2026-10-10, before any run)

Data: Bulot et al., Zenodo 7198378 (CC-BY-4.0): 8 Plantower PMS5003 and 8 Sensirion SPS30 sensors in four monitors at one
site, 2-minute medians, Jul 2020 - Jul 2021; target = co-located FIDAS 200 (PM1, PM2.5, PM10); environment = the site
weather station's relative humidity, temperature, dew point and pressure (the only environmental channels in the record;
given identically to every model). Window 180 steps (6 h). Rows with any missing input, target or environment value dropped.
Protocol: unit rotation within each sensor type (tsl_rotate.py: every unit tests once, 5 folds, 2 validation units, stride 4,
seed 0), 3 PM sizes. Models and hyperparameters unchanged: EnvFiLM-b, TCN+ENV, TESLA+ENV.
EnvFiLM-b replicates on the independent dataset if, for the PMS5003 group (the sensor of the main study):
 1. vs TCN+ENV: better on >= 55% of unit cases pooled over PM sizes, pooled median ratio < 1;
 2. vs TESLA+ENV: better on >= 55% of unit cases, pooled median ratio < 1.
The SPS30 group is reported as secondary (same criteria), not required.

## Outcome (2026-10-10)
PMS5003 group (primary), 24 unit cases: vs TCN+ENV better on 45.8%, median ratio 1.002 -> criterion 1 FAILS;
vs TESLA+ENV better on 95.8%, median 0.925 -> criterion 2 passes. Replication rule (both required) FAILS.
SPS30 (secondary): vs TCN+ENV 58.3%, 0.983; vs TESLA+ENV 95.8%, 0.853.
UNSATISFACTORY under the reporting contract: out of scope, not reported. Consequence: paper claims must stay scoped
to SensEURCity (no claim that the mechanism generalises to other datasets).
