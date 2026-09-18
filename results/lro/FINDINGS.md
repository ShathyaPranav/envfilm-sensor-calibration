# What expert data cleaning actually looks like, measured on Logan River Observatory

All numbers below come from `experiments/lro_killtest.py`, `lro_gates2.py`,
`lro_gate2b_ramp.py`, `lro_gate5_drift.py` and `lro_gate5b_drift_lagged.py`, run on the
HydroShare resource `a6ea89ae20354e39b3c9f1228997e27a` (six sites, 2014-2020, 15-minute
sampling, raw value plus technician-corrected value plus technician label in every row).
Every pass threshold was fixed before the corresponding gate was run.

## One correction to an earlier reading of this data

The corrected column encodes "the technician removed this value" as **-9999**, not as an
empty cell (pyhydroqc writes it in `anomaly_utilities.add_labels`). A first pass treated
-9999 as a measurement, which put a -9999 inside rolling windows and produced pH errors
of 1575 times the variable's scale. Every number in this note is from the rerun with the
sentinel mapped to missing.

## G0: what the technicians do

| variable | deleted | value-edited | of which slow drift | point replacement |
|---|---|---|---|---|
| temp | 0.3-21% | 0.0% | n/a | 0.0% |
| cond | 1.5-7.4% | 46-65% | ~99% | 0.1-1.2% |
| ph   | 4.5-7.5% | 49-62% | ~99% | ~0.1% |
| do   | 0.3-7.1% | 12-34% | ~99% | ~0.1% |

Pooled point replacement is 0.2% against a 25% limit, and nothing is restored from field
notes, so the edit is expressible as (delete, drift). **PASS.**

Temperature is never value-edited, only deleted. That matches the instruments: thermistors
do not drift, while conductivity, pH and oxygen sensors need field calibration.

## G1: is there anything for a model to do

Normalised by each variable's robust scale, on the primary variables at W=30:

| criterion | measured | required |
|---|---|---|
| median error, raw statistic vs corrected statistic | 0.0872 | >= 0.05 |
| series with peak error >= 0.5 | 12 of 12 | >= 4 |
| windows with non-zero error | 45.9% | >= 2% |
| error remaining after rule screening | 0.0697 | >= 0.03 |

**PASS.** Roughly 80% of the error survives range, rate-of-change and flatline screening,
which is the check the earlier adaptive-trim work never ran.

## G2 and G2b: which channel carries the error

G2 fitted one constant offset per calibration period and failed its 60% bar (11% explained).
That failure was a mis-specification on our side: the technicians' published procedure
(`pyhydroqc.calibration.lin_drift_cor`) applies the measured gap as a **linear ramp**
across each drift period. G2b tests that form, at MainStreet, the only site with
calibration records (45 documented periods for cond and ph, 13 for do):

| reconstruction | cond | ph | do | mean |
|---|---|---|---|---|
| deletions only | 22% | 3% | 8% | 10% |
| deletions + documented ramp | **83%** | 53% | 20% | 52% |
| deletions + realised delta (ceiling) | 100% | 100% | 100% | 100% |

The ceiling at exactly 100% confirms the factorisation is complete: raw to validated is
removals plus drift, with no third channel.

Knowing *when* calibrations happened is worth a great deal. With documented dates the
decomposition explains 46% of the error; with boundaries guessed from the raw series at
the other five sites, 7-16%.

## G5 and G5b: the drift cannot be predicted

The drift the technicians remove is a level offset applied to a whole period. The paper's
own result says such an offset is invisible to every within-window estimator, since it
moves every point in the window equally. So it can only be recovered from outside the
window. It cannot:

| predictor of today's drift | median absolute error (cond, do) |
|---|---|
| **assume no drift** | **0.727** |
| carry the last known drift forward (known to 30 days ago) | 0.835 |
| extend the recent drift slope | 1.609 |
| gradient boosting on neighbours, co-located variables, season, lagged drift | 1.903 |

Out-of-sample R^2 was negative on 17 of 18 series with a linear model and on all 12
primary series with gradient boosting. **FAIL, confirmed twice.**

The reason is physical. Conductivity differs between river sites by hundreds of uS/cm
while the drift being removed is a few uS/cm, so the signal sits under real geography.
More fundamentally, the correction size is a number a person measures during a field
visit, and it leaves no trace in the data beforehand.

## What this means

1. On conductivity, **83% of the error a data-quality monitor would report is drift**, and
   drift is not predictable in advance by any method, learned or otherwise. That is a
   ceiling on what any causal model can achieve, not a property of a particular model.
2. On temperature, **60-100% of the error is deletions**, which are detectable. The
   learnable channel is real but variable-dependent.
3. Real corrections are dominated by slowly varying offsets, not the sparse amplitude
   outliers that manufactured contamination studies (including this paper's own injection
   mechanisms) assume. A ranking of estimators derived from injected spikes is a ranking
   for the wrong contamination model.

Point 3 is the finding that matters most for the paper, and it is measurable: rank the
estimator configurations under injected mechanisms and under the real validation deltas on
the same series, and report the disagreement.

## The benchmark: detection score and downstream usefulness are decoupled

pyhydroqc (Jones, Jones and Horsburgh 2022) is the published method for this dataset, and
the HydroShare resource ships its per-point detections. Scored with **their own** event
protocol (`anomaly_events` + `compare_events` + `metrics`, their widen parameter), against
what each method does to the statistic a monitor would report:

| series | method | event F1 | window error, test years |
|---|---|---|---|
| MainStreet cond | no action | 0.000 | 0.1385 |
| | published ARIMA | 0.887 | 0.1437 |
| | published LSTM multivar bidir | 0.153 | 0.1416 |
| | rule screen | 0.937 | **0.0441** |
| | oracle: mask technician labels | 1.000 | 0.0443 |
| TonyGrove cond | published ARIMA | 0.966 | 0.0812 |
| | rule screen | 0.684 | 0.0634 |
| | oracle | 1.000 | 0.0516 |

An earlier scoring of these files point by point gave F1 near 0.02 and looked like a
refutation of their paper. It was our error: their `anomaly` column marks detection
onsets (178 flags against 106 labelled events on TonyGrove conductivity, 66% of events
touched within six hours), and their published metric credits a whole labelled event when
any detection overlaps it.

Two things follow, and they matter more than the ranking:

1. **A detector can score 0.89-0.97 on the benchmark and leave the reported statistic
   unchanged or slightly worse**, because flagging an event's onset removes a handful of
   points from a 200,000-point record. Meanwhile a rule screen scoring lower in some cases
   cuts the statistic error by 68%.
2. **Masking exactly the technician's labels is not optimal for the statistic.** On
   MainStreet conductivity the rule screen reaches 0.0441 against the label oracle's
   0.0443. Any method fitted to reproduce expert labels inherits that ceiling; only
   optimising the statistic directly can pass it.

## A measurement artefact worth recording

Raw conductivity and temperature contain the sensor error code **7999.0**. At MainStreet
these are 99 points in one contiguous block, and when a whole window is 7999 no trimming
or reweighting can help, so the statistic becomes 7999. Measured, those windows are 0.1%
of the record and carried **99.8% of the total error**, which turned the benchmark into a
test of a single event. All model comparisons therefore apply the same physical-range
prescreen first, using pyhydroqc's own published limits. That is preprocessing every
deployed pipeline performs, and it is applied identically to every arm including the
baselines.

## G4: the technicians' deletions are mostly not inferable either

Logistic regression on causal raw-only features, fitted on training years and scored on
test years, against the base rate a skill-less classifier would achieve:

| series | AUPRC | base rate | signal |
|---|---|---|---|
| Mendon cond | 0.726 | 0.118 | strong |
| FranklinBasin cond | 0.133 | 0.025 | present |
| WaterLab cond | 0.170 | 0.114 | weak |
| MainStreet cond | 0.129 | 0.117 | none |
| MainStreet do | 0.073 | 0.074 | below chance |
| WaterLab do | 0.067 | 0.080 | below chance |
| Mendon do | 0.088 | 0.117 | below chance |
| BlackSmithFork do | 0.109 | 0.114 | below chance |

On six of ten series the removals are at or below chance. Technicians delete on field
knowledge, fouling seen during a visit, a sonde found out of the water, maintenance
records, and none of that is in the time series.

## SARN: results, and what they mean

Error in the reported statistic, test years 2018-2020, 3 seeds, mean over six sites, after
the physical prescreen every arm receives. Training target is the retention statistic;
evaluation is always against the technician-corrected statistic.

| variable | no action | rule screen | label oracle | point-label net | SARN |
|---|---|---|---|---|---|
| temp | 0.0010 | 0.0009 | 0.0001 | 0.0010 | **0.0008** |
| cond | 0.1407 | **0.1366** | 0.1358 | 0.1391 | 0.1387 |
| do | 0.1181 | **0.0810** | 0.0754 | 0.1240 | 0.1181 |
| ph | 0.4179 | **0.3546** | 0.3665 | 0.4256 | 0.4159 |

Applied on top of the rule screen rather than instead of it:

| variable | rule screen alone | + point-label net | + SARN |
|---|---|---|---|
| temp | 0.0009 | 0.0009 | **0.0007** |
| cond | 0.1366 | 0.1351 | 0.1368 |
| do | 0.0810 | 0.0851 | 0.0809 |
| ph | 0.3546 | 0.3557 | 0.3544 |

What holds up:

1. **Training through the statistic beats training on point labels**, at identical capacity
   and features, with both arms selected on the same validation criterion and the control
   additionally allowed to tune its threshold. 62 of 69 series-seeds have a paired
   bootstrap interval entirely below zero; 2 are above.
2. **SARN is the only method that essentially never degrades what it reports.** Mean ratio
   to doing nothing: SARN 0.98, rule screen 1.03, point-label net 1.64.
3. **It wins where the channel is learnable and abstains where it is not.** On temperature,
   which is only ever deleted and never drift-corrected, it is 22% better than the rule
   screen and 30% better than the control. On conductivity and pH it converges to keeping
   every point, which G4 and G5 show is correct.

What does not hold up, stated plainly:

4. **SARN does not beat a simple rule screen** on conductivity, oxygen or pH, and adds
   nothing on top of one except on temperature.
5. **It does not beat the published detectors by any meaningful margin**: 5% better on
   MainStreet conductivity, tied elsewhere.
6. The absolute differences are small. On temperature, 0.0008 against 0.0010 of a robust
   scale of about 5 C is a difference of roughly 0.001 C in the reported window mean.

The honest summary is that on this dataset the ceiling, not the method, is the finding. The
gap between every causal method and the label oracle is information that does not exist in
the sensor stream.
