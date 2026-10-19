# Pre-registration: summary-feature control on the true-reference and Southampton benchmarks
(written 2026-10-20, before any run)

Requested in external review: the control TCN+ENV+z (PREREG_summary_control.md) was run only on the TESLA benchmark.

## Runs
1. True-reference benchmark (TSL_SOURCE=ref10, TSL_N=36), unit rotation, seeds 0 to 3 (tags ref10_ctrlz_s0..s3),
   compared with the existing EnvFiLM-b rows of the same seeds and folds (rotate_ref10_rot*.csv); per-unit RMSE
   averaged over the four seeds.
2. Southampton with the monitors' sensors (soton_onboard.py, TSL_N=180), seeds 0 and 1 (tags soton_ctrlz_s0, _s1),
   compared with rotate_soton_onboard_s0/_s1.csv; per-unit RMSE averaged over the two seeds.

## Criterion (same rule as before, per benchmark)
EnvFiLM-b beats TCN+ENV+z if it is better on at least 55% of unit cases with a pooled median ratio below one
(Southampton: the PMS5003 group, as in its original pre-registration). Sensor-level counts reported as well.

## Reporting
Reported whatever it shows. If a benchmark fails, the paper says so for that benchmark.
