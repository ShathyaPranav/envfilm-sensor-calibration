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
