# Fuzzy Linguistic Summaries for Electrical Load Forecasting Error Diagnostics

Reproducibility kit for the paper "Fuzzy Linguistic Summaries for Electrical Load Forecasting Error Diagnostics."

## Repository structure

```
data/                          Pre-computed prediction files (parquet)
  predictions_baseline.parquet     Baseline model (14 features)
  predictions_+gated_lags.parquet  +Gated Lags model (35 features)
  predictions_+transitions.parquet +Transitions model (37 features)
plots/                         Generated figures (PDF + PNG)
reproduce.py                   Reproduces all tables from the paper
reproduce_plots.py             Reproduces all figures from the paper
```

## Requirements

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

## Reproducing the results

Using uv (installs dependencies automatically):

```bash
uv run python reproduce.py        # Reproduces all tables
uv run python reproduce_plots.py  # Reproduces all figures (saves to plots/)
```

Alternatively, using pip:

```bash
pip install pandas numpy pyarrow matplotlib
python reproduce.py
python reproduce_plots.py
```

## Data

Each prediction file contains 8,735 deduplicated hourly observations from the evaluation period (March 2025 to February 2026) with the following columns:

| Column | Description |
|--------|-------------|
| `ds` | Timestamp (UTC) |
| `y` | Actual consumption (kWh) |
| `y_pred` | Predicted consumption (kWh) |
| `residual` | y - y_pred |
| `abs_residual` | Absolute residual |
| `temperature_2m_day2` | Day-2 temperature forecast (degrees Celsius) |
| `is_holiday` | Binary holiday indicator |

The fuzzy membership degrees (`mu_cold` through `mu_hot`) are **not precomputed** in the data files. They are computed at runtime from `temperature_2m_day2` using the SIA 380/1 heating standard breakpoints with Ruspini normalisation. The computation is fully transparent in `reproduce.py` (see `compute_fuzzy_memberships` and `trapmf` functions).

## Data sources

**Electricity consumption**

CKW (Centralschweizerische Kraftwerke) open smart meter dataset, Dataset B. Published at [https://www.ckw.ch/ueber-ckw/engagement/open-data.html](https://www.ckw.ch/ueber-ckw/engagement/open-data.html). The raw 15-minute measurements from 116 area codes in central Switzerland are aggregated to hourly resolution and summed across all area codes to produce a single load time series.

**Temperature forecasts**

ECMWF IFS model, provided by the Open-Meteo Previous Model Runs API at [https://open-meteo.com](https://open-meteo.com). Day-2 ahead hourly forecasts (`temperature_2m`) for the Lucerne region (47.05°N, 8.31°E). See Zippenfenig (2023), DOI [10.5281/zenodo.7970649](https://doi.org/10.5281/zenodo.7970649).

**Calendar features**

Public holidays for the canton of Lucerne, manually collected from official cantonal sources. The `is_holiday` flag marks all public holidays. The `nationwide` flag (used in the forecasting model but not included in the reproduction data) distinguishes nationwide from canton-specific holidays.

## Forecasting model

The predictions in `data/` are produced by a [LightGBM](https://github.com/microsoft/LightGBM) gradient boosting regressor (Ke et al., NeurIPS 2017) configured for recursive 48-hour multi-step prediction. A single 1-step model is trained to predict `y_{t+1}` and applied iteratively, feeding predictions back as lag features. Three configurations are compared, differing only in feature sets while sharing identical hyperparameters (63 leaves, learning rate 0.05, 500 boosting rounds). Evaluation uses expanding-window walk-forward validation over 363 daily windows spanning March 2025 to February 2026. See the paper for full details.

The reproduction scripts in this repository operate on the **output** of the forecasting model (the prediction files). They do not retrain the model. The full training pipeline is available in the research repository at `[link to fuzzy-temperature-coupling repo]`.

## Citation

```
Pending
```

## License

Code is released under the MIT License. Data files are derived from third-party sources with their own terms. See [LICENSE](LICENSE) for details.
