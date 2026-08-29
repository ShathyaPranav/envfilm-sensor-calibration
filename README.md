# dq-estimator-eval

Reproduction artifact for *Beyond Robust vs. Naive: Contamination-Model
Sensitivity in Streaming Data-Quality Estimator Evaluation*.

The paper tests a widespread default in streaming data-quality monitoring:
that robust estimators (median, MAD) are a safer choice than naive ones
(mean, standard deviation) under contamination. It finds that (1) at every severity tested, a fixed interior estimator
dominates both extremes, though which one wins shifts with severity
(a 10% trimmed estimator at or below 2-sigma, an IQR-based estimator
above), making "robust vs. naive" the wrong axis, and (2) the direction of the
robust-vs-naive comparison itself reverses depending on the contamination
model (isolated rare incidents vs. ambient scattered corruption) used to
evaluate it, on identical data with an identical error metric.

This repository contains everything needed to regenerate Tables I-III and
both figures from already-computed results, plus the experiment scripts
that produced those results in the first place. It does not contain the
raw datasets (see "Datasets" below) or the full paper draft.

## What's here

```
harness/        Defect injection, rolling estimators, and the error metric.
experiments/    Scripts that produced Tables I-III and both figures.
results/        Per-cell CSVs and report JSONs, already computed.
figures/        Plotting scripts + the generated PDFs.
```

`results/` is included so the figures and tables can be reproduced
immediately without re-running anything expensive. `experiments/` is
included so the full pipeline (data → per-cell results → tables/figures)
can be re-run from scratch if you want to verify the results end to end,
not just trust the checked-in CSVs.

## Which script produces what

| Output | Script | Reads |
|---|---|---|
| `results/phase1b/pooled_per_segment.csv` (isolated-incident UCI ratio, "1.89" in the paper, geometric mean) | `experiments/run_phase1b.py` | UCI raw data |
| `results/phase2a_full/phase2a_cells.csv` (the 80-series × 23-regime × 10-seed sweep) | `experiments/run_phase2a_full.py` | SKAB + UCI raw data |
| `results/phase2a_full/table1.csv`, `table2.csv` (Tables I, II) | `experiments/reproduce_tables.py` | `phase2a_cells.csv` only |
| `results/phase2a_full/gate_verdicts.csv` (Table III), `headroom_by_fold.csv`, `oracle_configuration_map.csv`, `per_channel_breakdown.csv`, `phase2a_report.json` | `experiments/run_phase2a_full_report.py` | `phase2a_cells.csv` only |
| `figures/fig_contamination_flip.pdf` (Fig. 1, the central figure: "1.89" vs. "0.80", both geometric mean) | `figures/fig_contamination_flip.py` | `pooled_per_segment.csv` + `phase2a_cells.csv` |
| `figures/fig_mixed_config.pdf` (Fig. 2: mixed-configuration frequency by mechanism) | `figures/fig_mixed_config.py` | `phase2a_report.json` |
| `results/perturbation/perturbation_percell.csv`, `perturbation_valid.csv`, `perturbation_summary.json` (per-seed perturbation-magnitude regression) | `experiments/run_perturbation_analysis.py` | SKAB + UCI raw data + `results/phase1b/incident_level.csv` |
| `figures/fig_perturbation_magnitude.pdf` (Fig. 3: log-ratio vs. log-perturbation, both domains) | `figures/fig_perturbation_magnitude.py` | `perturbation_valid.csv` |
| `results/window_sweep/percell_w{15,30,60,120}.csv`, `table1_by_window.csv`, `table2_by_window.csv`, `perturbation_fit_by_window.csv`, `window_sweep_summary.json` (Table I/II + perturbation regression at each window size) | `experiments/run_window_sweep.py` | SKAB + UCI raw data |
| `figures/fig_window_sweep.pdf` (Fig. 4: trimmed/IQR crossover collapsing as W grows) | `figures/fig_window_sweep.py` | `table2_by_window.csv` |
| `results/trim_sweep/percell_tau{0.0,0.05,0.1,0.2,0.3}.csv`, `table1_by_tau.csv`, `table2_by_tau.csv`, `trim_sweep_summary.json` (trim-level sensitivity, both datasets) | `experiments/run_trim_sweep.py` | SKAB + UCI raw data |

Everything in the right two columns except `run_perturbation_analysis.py`'s
outputs is already checked into `results/` and `figures/`, so the table/
figure scripts run in seconds without touching any raw data. `run_phase1b.py`,
`run_phase2a_full.py`, and `run_perturbation_analysis.py` are the scripts
that actually consume raw datasets and are slower (`run_perturbation_analysis.py`
recomputes every (series, regime, seed) cell individually rather than reading
the seed-pooled `phase2a_cells.csv`, so it takes on the order of 20 minutes).

## What's deliberately not here

Phase 0 (synthetic AR(1) mechanism-isolation) and Phase 1A (the controlled
membership/value mechanism probe) are exploratory work from earlier in the
project. No number in Tables I-III or either figure traces back to them, so
they're excluded to keep this artifact scoped to what the paper actually
cites. The original project's earlier, abandoned direction (a downstream-
harm/impact-propagation diagnostic pipeline, superseded before this paper's
research question was settled on) is excluded entirely.

## Datasets

Not committed here (148 MB combined, and both are public and directly
downloadable). Download into `data/`, which is gitignored:

**UCI Individual Household Electric Power Consumption** (archive.ics.uci.edu,
dataset id 235):

```bash
curl -L -o data/household_power_consumption.zip \
  "https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip"
cd data && unzip household_power_consumption.zip && cd ..
```

This should leave `data/household_power_consumption.txt` (127 MB) in place.

**SKAB (Skoltech Anomaly Benchmark)** (github.com/waico/SKAB), 5 CSVs, ~1 MB
total:

```bash
mkdir -p data/skab_data/anomaly-free data/skab_data/other data/skab_data/valve1 data/skab_data/valve2
BASE="https://raw.githubusercontent.com/waico/SKAB/master/data"
curl -L -o data/skab_data/anomaly-free/anomaly-free.csv "$BASE/anomaly-free/anomaly-free.csv"
curl -L -o data/skab_data/other/5.csv                   "$BASE/other/5.csv"
curl -L -o data/skab_data/other/10.csv                  "$BASE/other/10.csv"
curl -L -o data/skab_data/valve1/1.csv                  "$BASE/valve1/1.csv"
curl -L -o data/skab_data/valve2/1.csv                  "$BASE/valve2/1.csv"
```

## Setup

```bash
python -m venv venv
source venv/bin/activate        # venv\Scripts\activate on Windows
pip install -r requirements.txt
```

Python 3.10+ (uses `numpy.lib.stride_tricks.sliding_window_view` and
`include_groups=` on `DataFrame.groupby.apply`, which need modern
numpy/pandas; see `requirements.txt` for tested minimum versions).

## Rerunning everything

All commands below assume your working directory is the repository root.
Every script resolves its own paths relative to its own file location, so
this also works if you invoke them with a full path from elsewhere.

**Fast path (tables + figures only, no raw data needed):** the four
commands under "Which script produces what" whose "Reads" column doesn't
say "raw data" already have their inputs checked in:

```bash
python experiments/reproduce_tables.py
python experiments/run_phase2a_full_report.py
python figures/fig_contamination_flip.py
python figures/fig_mixed_config.py
```

**Full path (from raw data):** after downloading both datasets into `data/`
as above:

```bash
python experiments/run_phase1b.py          # ~1-2 min; writes results/phase1b/
python experiments/run_phase2a_full.py      # ~3-5 min; writes results/phase2a_full/phase2a_cells.csv
python experiments/run_phase2a_full_report.py
python experiments/reproduce_tables.py
python figures/fig_contamination_flip.py
python figures/fig_mixed_config.py
```

`run_phase2a_full.py` fixes all 10 injection seeds internally
(`SEEDS = list(range(10))`), so a full rerun from raw data reproduces the
checked-in `results/` numbers exactly, not just approximately.

## Path/import fixes made when this artifact was assembled

Copied unchanged from the source repository, with two categories of
mechanical fix so every script runs from a clean checkout of this
directory alone:

1. **Import paths.** The source repository used a single flat package;
   this artifact splits `harness/` and `experiments/` into separate
   directories, so imports crossing that boundary were changed from
   relative (`from . import X`) to absolute (`from harness import X`),
   with each affected script computing its own repository root via
   `os.path.dirname(os.path.dirname(os.path.abspath(__file__)))` and
   inserting it on `sys.path`. Imports between siblings within the same
   directory were left as plain bare imports (Python already puts a
   running script's own directory on `sys.path`). No logic changed.
2. **Hardcoded paths.** Five hardcoded paths referencing the source
   repository's own layout (`pilot/data/...`, `pilot/out/...`) were
   replaced with paths computed relative to this repository's root:
   `phase1b_real.py`'s `RAW_PATH`, `run_phase1b.py`'s `OUT_DIR`,
   `run_phase2a_full.py`'s `OUT_DIR` and `SKAB_DIR`, and
   `run_phase2a_full_report.py`'s `OUT_DIR`.

`experiments/reproduce_tables.py` is new (Tables I and II were originally
produced by one-off interactive commands during the analysis session, never
saved as a script). `figures/fig_contamination_flip.py` and
`figures/fig_mixed_config.py` are also new for the same reason: both
figures existed only as already-rendered PDFs, generated by inline code
that was never persisted as a runnable script. All three were written for
this artifact so every figure and table has a real, checked-in script
behind it, and verified to reproduce the checked-in numbers exactly (see
"Verification" below).

## Verification

`experiments/reproduce_tables.py`, `figures/fig_contamination_flip.py`, and
`figures/fig_mixed_config.py` were each run end to end from a clean copy of
this directory (using the checked-in `results/`, no raw data needed) and
confirmed to reproduce the paper's numbers exactly:

- Table I and Table II: every value matches the paper's tables to 3 decimal
  places.
- `fig_contamination_flip.py`: isolated-incident geometric-mean ratio
  1.8873 (95% CI [1.630, 2.189], segment-clustered bootstrap), ambient
  geometric-mean ratio 0.7968 (95% CI [0.760, 0.834]), matching the
  paper's "1.89" and "0.80". Both panels use the geometric mean, per
  Section III-B's stated pooling rule; Phase 1B's original arithmetic-mean
  report (1.9408) is footnoted in the paper, not used as the primary
  number.
- `fig_mixed_config.py`: SKAB 52.2% / UCI 13.0% overall, and every
  per-mechanism value, matching the paper's §4 diagnostic exactly.

`results/phase2a_full/gate_verdicts.csv` (already checked in, no script
needed to inspect it) was also confirmed to match Table III exactly: 8.0% /
21.4% / KILL (location) and 45.2% / 45.2% / PROCEED (spread), combined
scope.

`experiments/run_phase1b.py` and `experiments/run_phase2a_full.py` (the two
scripts that consume raw datasets) were verified to import cleanly and
resolve all paths correctly against this repository's layout, but were not
run end to end here since that requires the ~148 MB of raw data described
above.

`experiments/run_perturbation_analysis.py` was run end to end from raw data
(full 18,400-cell per-seed sweep, ~22 min) and confirmed to produce: SKAB
naive-pooled slope $-0.6119$ ($r^2=0.5265$), (run,channel)-demeaned slope
$-0.6904$ ($r^2=0.8182$); UCI naive-pooled slope $-0.6032$ ($r^2=0.7044$),
demeaned slope $-0.6081$ ($r^2=0.7746$); matching the paper's Section V-B
to the reported precision. An earlier, informal version of this analysis
that fit 11 condition-level geometric means instead of per-seed cells
reported $r^2=0.795$ and a parity-crossing perturbation of $0.026$ -- both
were averaging artifacts (fitting on already-averaged conditions
suppresses within-condition variance) and are not used anywhere in the
paper or reproducible from this script.

`experiments/run_window_sweep.py` was run end to end from raw data at
$W\in\{15,30,60,120\}$ (~113 min total) and confirms Section VIII's
window-invariant perturbation exponent ($-0.52$ to $-0.61$ SKAB, $-0.53$
to $-0.60$ UCI) and the Table II severity crossover collapsing to
"trimmed wins everywhere" at $W\geq60$ -- see `window_sweep_summary.json`
for every per-window number cited in the paper.

`experiments/run_trim_sweep.py` was run end to end from raw data at
$\tau\in\{0\%,5\%,10\%,20\%,30\%\}$, 5 seeds (~41 min total) and confirms
Section VIII's trim-level findings: monotonic improvement with more
trimming for amplitude mechanisms and out\_of\_order (no interior optimum
within the tested range), a non-monotonic peak for missing at
$\tau=10\%$ (worse than $\tau=0\%$ or $\tau=30\%$), and stale\_run never
beating the untrimmed ratio at any tested $\tau$ -- see
`trim_sweep_summary.json` for every number cited in the paper.
