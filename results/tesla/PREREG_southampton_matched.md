# Southampton matched control (written 2026-10-11, after the on-board run, before this run)

Purpose: the on-board run (PREREG_southampton_onboard.md) passed and the earlier weather-station-only run
(PREREG_southampton.md) did not, but the two differ in period, channel set and seeds. This control removes the
confound: identical rows (Jul-Dec 2020, the on-board run's rows), window 180, folds, stride and seeds 0 and 1, models
EnvFiLM-b, TCN+ENV, TESLA+ENV; the four state channels are the weather station's humidity, temperature, dew point and
pressure (experiments/soton_matched.py).

Reading, fixed in advance: if EnvFiLM-b vs TCN+ENV in the PMS5003 group is clearly weaker here than with on-board
channels (91.7%, 0.959), the paper states that the gain on Southampton depends on the monitor's own state sensors.
If it is similar, the paper states that the difference between the two earlier runs came from period and seeds,
not from the channels. Reported either way, since the on-board result it qualifies is in the paper.
