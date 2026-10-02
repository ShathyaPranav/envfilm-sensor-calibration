# Logan River Observatory: what expert cleaning is, and what a model can do about it

Data: HydroShare `10.4211/hs.a6ea89ae20354e39b3c9f1228997e27a` (Jones, Jones and Horsburgh
2022). Six sites, 2014-2020, 15-minute sampling; every record carries the raw value, the
technician-corrected value and the technician's qualifier, for temperature, specific
conductance, pH and dissolved oxygen. Fit 2014-2016, select 2017, report 2018-2020.

This note supersedes earlier versions. Section 7 lists what earlier versions got wrong.

## 1. Three data-handling points that each first produced a wrong result

- **Sentinel.** The corrected column encodes "removed" as -9999, not as missing.
- **Error code.** Raw columns contain 7999. At MainStreet, 99 such temperature values in one
  block put 99.8% of that series' test-period error into 0.1% of its windows. Every arm
  therefore gets the same physical-range screen, using pyhydroqc's published limits.
- **Abstention.** An arm that removes every point in a window yields no estimate there.
  Scoring each arm only on the windows it covers rewards abstaining: the rule screen
  appeared to cut MainStreet pH error from 0.772 to 0.454 while covering 85% of windows; on
  windows every arm covers the figures are 0.475 and 0.450. All model comparisons are now at
  full coverage (previous estimate carried forward), with common-window figures alongside.
  Script: `lro_coverage_audit.py`; scoring: `sarn_eval.py`.

## 2. What the technicians change (`lro_killtest.py g0`, records as distributed)

| variable | deleted | edited | point-wise edits |
|---|---|---|---|
| temperature | 0.3-21.1% | 0.0% | 0.0% |
| conductance | 1.5-7.4% | 45.6-64.6% | 0.1-1.2% |
| pH | 4.5-7.5% | 48.8-61.7% | 0.0-0.3% |
| dissolved oxygen | 0.3-7.1% | 12.3-34.0% | 0.0-0.3% |

Temperature is only deleted. The other three are drift-corrected: a slowly varying offset.

## 3. Channel decomposition at MainStreet (`lro_gate2b_ramp.py`)

Share of the raw-versus-corrected error in the rolling trimmed mean (W=30), full record,
before the range screen. MainStreet is the only site with calibration records.

| reconstruction | cond | pH | DO |
|---|---|---|---|
| deletions only | 21.9% | 2.7% | 7.8% |
| deletions + documented linear ramp | 83.0% | 53.2% | 20.0% |
| deletions + realized correction | 100% | 100% | 100% |

## 4. The drift was not predicted (`lro_gate5_drift.py`, `lro_gate5b_drift_lagged.py`)

Thresholds fixed before running.

- Huber linear model on neighbour differences, co-located variables and season: negative
  out-of-sample R^2 on 17 of 18 series (pass mark: R^2 >= 0.25 on 2 of 12 cond/DO series,
  or a 25% median reduction; the median reduction was -0.5%).
- Gradient boosting, same inputs: negative R^2 on all 12 cond/DO series.
- Gradient boosting given the true correction up to 30 days earlier: beat the best trivial
  predictor by the required 20% on 0 of 18 series. On all 12 cond/DO series the best
  predictor was "no drift"; the model's error exceeded it by 6% to 202%.

Context: site medians of conductance span 320-459 uS/cm; the median correction at each site
is under 3 uS/cm. This shows these predictors fail here, not that drift is unpredictable in
principle.

## 5. Estimator choice under real corrections (`lro_estimator_ranking.py`)

Error x100, in robust-scale units, mean over six sites, 2014-2020, after the range screen.

| | mean | median | trimmed | Huber | std | MADn | trim. std | IQR |
|---|---|---|---|---|---|---|---|---|
| cond | 14.54 | 14.45 | 14.50 | 14.46 | 0.89 | 0.24 | 0.95 | 0.79 |
| pH | 31.49 | 31.47 | 31.48 | 31.46 | 0.36 | 0.36 | 0.38 | 0.39 |
| DO | 5.43 | 5.42 | 5.43 | 5.43 | 0.16 | 0.16 | 0.20 | 0.20 |
| temp | 0.057 | 0.064 | 0.054 | 0.059 | 0.053 | 0.031 | 0.051 | 0.048 |

Predicted before running, from translation equivariance: on the drift-corrected variables
the location estimators should be nearly identical. They are. Worst over best is 1.002 at the
median and 1.034 at most over 18 series; median-to-mean ratio 0.991-0.999. Under injected
defects that ratio spans 0.14 to 1.52.

## 6. Published detectors, and SARN

**Published detections** (`lro_published_effect.py`), on the test span they cover, scored
with pyhydroqc's own event protocol, and the change in the statistic's error when the
flagged points / the widened events are removed:

| series | detector | flags | event F1 | points | events |
|---|---|---|---|---|---|
| MainStreet cond | ARIMA | 127 | 0.887 | +5.5% | +19.2% |
| MainStreet cond | LSTM | 97 | 0.153 | +7.0% | +26.6% |
| TonyGrove cond | ARIMA | 90 | 0.966 | -0.4% | -1.2% |
| TonyGrove cond | LSTM (best of 4) | 36 | 0.518 | -0.3% | -0.7% |
| MainStreet pH | LSTM | 14 | 0.083 | 0.0% | 0.0% |

These are model-stage detections; pyhydroqc's full workflow also has a rule stage, and its
correction stage replaces values where this removes them.

**SARN** (`sarn.py`, `sarn_eval.py`, `sarn_v2_report.py`): a 12,129-parameter retention
network trained through a weight-aware trimmed mean, against the identical network trained on
the technician's flags. 23 series (TonyGrove temperature has no usable selection year), 3
seeds, 1.5 million test windows. Error x100, full coverage:

| | no action | rule screen | label-trained | SARN | label oracle |
|---|---|---|---|---|---|
| cond | 29.58 | 29.78 | 29.90 | 29.57 | 30.78 |
| DO | 14.27 | 10.14 | 14.80 | 14.27 | 12.33 |
| pH | 41.97 | 37.94 | 42.69 | 41.94 | 41.18 |
| temp | 0.103 | 0.090 | 0.102 | 0.082 | 0.007 |

Windows every arm covers:

| | no action | rule screen | label-trained | SARN | label oracle |
|---|---|---|---|---|---|
| cond | 13.11 | 13.08 | 13.42 | 13.06 | 12.82 |
| DO | 5.38 | 5.15 | 5.93 | 5.38 | 5.30 |
| pH | 35.99 | 35.08 | 36.74 | 35.96 | 35.88 |
| temp | 0.103 | 0.090 | 0.102 | 0.082 | 0.007 |

- SARN is below the label-trained control on 22 of 23 series (23, 21, 19 per seed), median
  3.2% lower. Series sharing a model are not independent; read as direction, not significance.
- It wins mainly by not acting: it keeps 99.5% of points (control 96.1%) and is within 2% of
  no action on 18 of 23 series. The control is worse than no action by more than 2% on 12.
- It does not beat the rule screen (lower on 8 of 23). The rule screen is the best method on
  DO and pH, though most of its advantage is discarding stretches and holding.
- The label oracle is not an upper bound: the rule screen beats it on DO and pH even on
  common windows. Reproducing expert flags exactly gains 2.3%, 1.4%, 0.3% on cond, DO, pH.
- Flags are hard to recover: the label-trained network's average precision exceeds 0.25 on 2
  of 23 series (median 0.07) and is under twice the base rate on 7.
- Temperature gain is small in absolute terms: about 0.001 C in the window mean.
- Target ablation: trained on the corrected statistic (no point labels needed) SARN keeps
  every point on cond/DO/pH and lowers temperature error by 32% (20% with the retention
  target). The retention target was fixed in advance and stays the headline.

## 7. What earlier versions of this note got wrong

1. "The rule screen cuts MainStreet pH error by 41%" and similar: an abstention artifact
   (Section 1). Like for like it is 5%.
2. "83% of conductance error is drift": deletions explain 21.9%, deletions plus the
   documented ramp 83.0%.
3. "Sites differ by hundreds of uS/cm": site medians span 320-459.
4. "SARN beats the control on 62 of 69 bootstrap intervals": a window-level bootstrap over
   overlapping windows, computed under the abstention-rewarding scoring. Replaced by the
   series-level comparison above.
5. "The label oracle bounds what removal can achieve": false, see Section 6.
6. "The trimmed-mean layer reduces exactly to drop-and-recompute": only when the trim count
   is a whole number; otherwise it interpolates where the harness rounds down.
7. "Deletions are at or below chance on 6 of 10 series": that used a linear model on local
   features only. The stronger test is the label-trained network's average precision above.
8. SARN has 12,129 parameters, not about 25,000.
