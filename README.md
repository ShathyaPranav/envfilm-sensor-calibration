# EnvFiLM

Code, results and paper source for **EnvFiLM: Environment-Conditioned Calibration of Low-Cost Particulate Matter
Sensors**.

Low-cost optical particle sensors read high or low depending on humidity and temperature, and many sensor boxes
measure both. EnvFiLM is a dilated convolutional calibrator for the particle signal. A small encoder turns the box's
humidity and temperature into a bounded scale and shift for every convolution layer (feature-wise linear modulation),
so the environment changes how the particle signal is read instead of being one more input channel.

## Results

On the SensEURCity benchmark of TESLA (AAAI 2025), with the TESLA authors' own preprocessed files and split:

| | PM10 | PM2.5 | PM1 |
|---|---|---|---|
| EnvFiLM, city-averaged RMSE (µg/m³) | **8.04** | **2.71** | **1.36** |
| SCARE, published | 14.03 | 5.67 | 2.51 |
| TESLA, published | 14.75 | 6.16 | 3.18 |

Against the same convolutional network given the same four humidity and temperature channels as ordinary inputs,
tested by holding out every sensor once (unit rotation):

| Benchmark | Sensors where EnvFiLM is better | Sign test |
|---|---|---|
| TESLA benchmark (target: OPC-N3) | 48 of 65 | p = 1.5e-4 |
| Same sensors, target: station reference instrument | 47 of 51 | p = 2.4e-10 |
| Southampton, UK (independent dataset, FIDAS reference) | 14 of 16 | p = 4.2e-3 |

Switching the modulation off makes EnvFiLM worse on 81% of sensor and size cases. The full tables, including PatchTST,
TESLA with the same channels and a size-matched network, are in `paper/`.

## Layout

```
experiments/   all code (data loading, models, training, evaluation, paper tables and figures)
results/tesla/ every run's CSV and log, and the pre-registration notes (PREREG_*.md)
results/paper/ numbers.json and the generated tables and figures used in the paper
paper/         LaTeX source of the paper (IEEEtran), with the tables and figures it includes
```

The main files in `experiments/`:

| File | What it does |
|---|---|
| `tsl_data.py` | Loads the TESLA benchmark files, attaches the on-board channels, builds the true-reference variant |
| `tsl_bench.py` | All models (EnvFiLM is class `EnvFiLMb`, name `EnvFiLM-b`), the shared training recipe, the official split |
| `tsl_rotate.py` | Unit rotation: every sensor is a test sensor once |
| `soton_onboard.py`, `soton_matched.py` | Southampton rotation with the monitors' sensors, and with station weather only |
| `tsl_oot.py` | Train in spring 2020, test at the second co-location months later |
| `tsl_humidity.py`, `humidity_analysis.py` | Error by humidity tercile and the strength of the modulation |
| `tsl_dev.py`, `tsl_oracle.py` | Development protocol on training sensors only, and diagnostic oracles |
| `paper_numbers.py`, `paper_fig_arch.py` | Every number, table and figure in the paper, computed from `results/tesla/` |
| `sec_extract.py`, `prepare_soton.py` | Data preparation (see below) |

## Data

None of the data is in this repository. Everything is public. Put it under one folder and point `DQ_DATA` at it
(the default is `data/` in this repository):

```
$DQ_DATA/
  senseurcity/senseurcity_data_v03.zip        SensEURCity, Zenodo record 17858205 (5.7 GB)
  tesla_orig/finedust_concentration_alldrop_firstcol_dataset/*.csv
                                               TESLA authors' preprocessed files (link in their GitHub README)
  southampton/lcs.rds, df_pm_2min.csv, df_weather_2min.csv
                                               Bulot et al., Zenodo record 7198378
  southampton_onboard/rh_temperature.csv      Bulot, Zenodo record 6637781
```

Then:

```bash
python experiments/sec_extract.py     # SensEURCity archive -> one parquet file per unit (needs remotezip)
python experiments/prepare_soton.py   # Southampton -> lcs.parquet and onboard_2min.parquet (needs pyreadr)
```

## Running

Install the packages in `requirements.txt` (PyTorch with CUDA for the GPU runs). Every run is configured through
environment variables and writes `results/tesla/<kind>_<TSL_TAG>.csv`.

```bash
export DQ_DATA=/path/to/data TSL_SOURCE=orig TSL_EXTRA=SHT31HE,SHT31TE,OPCN3Hum,OPCN3Temp

# official split, three seeds per model and cell
TSL_SEEDS=3 TSL_TAG=mine python experiments/tsl_bench.py EnvFiLM-b TCN+ENV

# unit rotation, one training seed
TSL_ROT_SEED=0 TSL_TAG=rot_mine python experiments/tsl_rotate.py EnvFiLM-b TCN+ENV

# the same on the true-reference benchmark (10-minute data, 36-step window)
TSL_SOURCE=ref10 TSL_N=36 TSL_ROT_SEED=0 TSL_TAG=ref_mine python experiments/tsl_rotate.py EnvFiLM-b TCN+ENV

# Southampton (2-minute data, 180-step window)
TSL_N=180 TSL_ROT_SEED=0 TSL_TAG=soton_mine python experiments/soton_onboard.py EnvFiLM-b TCN+ENV TESLA+ENV

# rebuild every number, table and figure of the paper from the stored results
python experiments/paper_numbers.py
```

Model names, PM sizes and cities can be given as arguments to restrict a run (for example `PM2.5 Oslo`). On a laptop
RTX 4080 one EnvFiLM training run takes 4 to 17 minutes per city and size. GPU training is not bit-reproducible, so a
rerun with the same seed can differ by a few percent on a single test sensor; this is why the paper rests its claims
on the unit rotation.

## Pre-registration

Before each confirmatory run we wrote its pass criteria into a `results/tesla/PREREG_*.md` file and committed it, then
recorded the outcome in the same file. The git history shows the order.

## Earlier work in this repository

The repository began as a study of robust and naive estimators in streaming data-quality monitoring. Its code is in
`harness/`, `figures/` and the `results/` folders other than `results/tesla/` and `results/paper/`, and the
corresponding scripts are in `experiments/` (`run_phase*.py`, `run_*_sweep.py`, `reproduce_tables.py` and the
`intel_*`, `cctnorm_*` and `lro_*` scripts). EnvFiLM does not depend on any of it.
