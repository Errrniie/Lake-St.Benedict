# Project Map and File Responsibilities

This document is a comprehensive map of the repository as it exists now, with practical descriptions of what each folder/file does.

## Scope notes
- Includes source code, data, models, generated outputs, and cache artifacts currently present.
- `.git/` and `.venv/` are intentionally excluded (tooling/environment internals).
- `__pycache__/` and `*.pyc` are generated Python bytecode caches (safe to delete; recreated automatically).

---

## Repository Root

- `.DS_Store`  
  macOS/Finder metadata file; not part of project logic.
- `DO_lakedata_parsed.csv`  
  Parsed lake data table at root (legacy/reference location used by some scripts as fallback).
- `Main.py`  
  Main launcher entrypoint for the parsing/training GUI/controller pipeline.
- `README.md`  
  Minimal top-level readme.
- `WeatherData_parsed.csv`  
  Parsed weather table at root (reference dataset).
- `requirements.txt`  
  Python dependency list for this project.

---

## `CSV/`

### `CSV/train/ParsingFiles/`

- `Core.py`  
  Main controller/CLI entry for parsing pipeline and GUI mode; calls module connector and weather/model actions.
- `InitSetup.py`  
  Environment/setup helper script for project initialization flow.
- `gui_app.py`  
  PySide GUI: open/process CSVs, run weather analytics, train models, run predictions, model manager.

### `CSV/train/ParsingFiles/Module/`

- `Connector.py`  
  Orchestrates pipeline stages (parse -> clean -> lag -> DO deltas -> model actions).
- `__init__.py`  
  Declares module package.

#### `Module/Clean/`
- `__init__.py`  
  Package marker for cleaning utilities.
- `clean_module.py`  
  Cleaning helpers: fill `hour` from `DATE`, drop zero-DO rows.

#### `Module/DORate/`
- `__init__.py`  
  Package marker for DO rate/delta utilities.
- `dodeltas_module.py`  
  Builds forward DO targets/features (`DO_delta_*h`, `DO_*h`) and helper predicates.

#### `Module/Lag/`
- `__init__.py`  
  Package marker for lag features.
- `lag_module.py`  
  Creates time-lag columns (DO, air temp, aerator rolling lags) and related lag filtering.

#### `Module/Model/`
- `__init__.py`  
  Package marker for model subsystem.
- `model_io.py`  
  Save/load model bundles, model-path helpers.
- `predict_module.py`  
  Runs prediction from saved bundle and writes output CSV + figure.
- `prediction_plot.py`  
  Plotting helpers for prediction outputs (standard and delta-style plots).
- `train_high_module.py`  
  Quantile training wrapper (high quantile).
- `train_low_module.py`  
  Quantile training wrapper (low quantile).
- `train_mid_module.py`  
  Quantile training wrapper (median/mid quantile).
- `train_module.py`  
  Legacy/common training wrapper entry.
- `train_quantile_core.py`  
  Shared core for weather-feature quantile training and split/eval logic.

##### `Module/Model/Models/`
- `model_DO_delta_1h_quantile_high.pkl`  
  Serialized trained bundle for high-quantile `DO_delta_1h`.
- `model_DO_delta_1h_quantile_low.pkl`  
  Serialized trained bundle for low-quantile `DO_delta_1h`.
- `model_DO_delta_1h_quantile_mid.pkl`  
  Serialized trained bundle for mid-quantile `DO_delta_1h`.

#### `Module/Parse/`
- `__init__.py`  
  Package marker for parse helpers.
- `parse_module.py`  
  CSV read/write and column-resolution helpers (`DATE`, DO column detection, datetime coercion).

#### `Module/Weather/`
- `__init__.py`  
  Package marker for weather processing.
- `weather_init_module.py`  
  Raw weather-to-standard transform pipeline (`DATE`, temp, RH, pressure, `hour`).
- `weather_yearly_analytics.py`  
  Produces yearly weather scenario CSVs (largest temp/humidity summer, average year).

#### `Module/*/__pycache__/` and `ParsingFiles/__pycache__/`
- Compiled `*.pyc` caches for Python runtime performance; generated artifacts.

---

## `Original Gathered Data/`

- `DO_lakedata.csv`  
  Original/raw lake DO dataset.
- `WeatherData.csv`  
  Original/raw weather dataset.

### `Original Gathered Data/Parsed Data/`
- (folder currently present; no files listed in current tree snapshot).

---

## `PURE WEATHER/` (experiment/workbench area)

- `README.md`  
  Notes for weather-only experiment paths/usage.
- `add_air_temp_lags.py`  
  Adds/rebuilds `AT_TLAG_*` columns for ParsedWeatherData CSVs.
- `aerator_schedule_simulation.py`  
  Simulates DO under aerator schedule + delta logic; writes sim CSV and figure.
- `build_aerator_window_dataset.py`  
  Builds `DO_lakedata_parsed_aerator_12h.csv` (rows near aerator transitions).
- `predict_weather_only.py`  
  Standalone prediction script for weather-only bundles.
- `run_aerator_sims_for_parsed_weather.py`  
  Batch runner: generate predictions + aerator sims for scenario weather CSVs.
- `train_weather_only.py`  
  Standalone weather-only training script.

### `PURE WEATHER/Aerator/`
- `DO_lakedata.csv`  
  Aerator-focused subset (rows where aerator is on).
- `daily_DO_linear_fit.py`  
  Fits daily linear DO-vs-time slopes and writes analysis outputs.

### `PURE WEATHER/Models/`
Trained model bundles used by GUI/newer flow:
- `model_DO_1h_quantile_high.pkl`
- `model_DO_1h_quantile_mid.pkl`
- `model_DO_3h_quantile_mid.pkl`
- `model_DO_6h_quantile_mid.pkl`
- `model_DO_delta_1h_quantile_mid.pkl`
- `model_DO_delta_6h_quantile_mid.pkl`
- `model_DO_quantile_high.pkl`
- `model_DO_quantile_mid.pkl`

### `PURE WEATHER/ParsedWeatherData/`
Scenario weather tables (with lag features):
- `Average_Year.csv`
- `Largest_Humidity_Summer.csv`
- `Largest_Temp_Summer.csv`

#### `PURE WEATHER/ParsedWeatherData/predictions/`
Predictions generated from scenario weather inputs:
- `.gitkeep` (keeps folder in git)
- `Average_Year_predictions_1hr.csv`
- `Average_Year_predictions_1hrdelta.csv`
- `Largest_Humidity_Summer_predictions.csv`
- `Largest_Humidity_Summer_predictions_1hr.csv`
- `Largest_Humidity_Summer_predictions_1hrdelta.csv`
- `Largest_Humidity_Summer_predictions_6hr.csv`
- `Largest_Temp_Summer_predictions.csv`
- `Largest_Temp_Summer_predictions_1hr.csv`
- `Largest_Temp_Summer_predictions_1hrdelta.csv`

### `PURE WEATHER/data/lake/`
Centralized lake CSV data for PURE WEATHER workflows:
- `DO_lakedata_parsed.csv`  
  Main parsed lake table used by weather-only scripts.
- `DO_lakedata_parsed_DO_ge_1.csv`  
  Filtered variant keeping `DO >= 1`.
- `DO_lakedata_parsed_aerator_12h.csv`  
  Transition-window subset around aerator on/off events.

### `PURE WEATHER/figures/`
Centralized figure output area.

#### `PURE WEATHER/figures/aerator_analysis/`
- `DO_vs_time_by_day_linear_fit.png`  
  Per-day linear-fit visualization from aerator dataset.

#### `PURE WEATHER/figures/aerator_sim/`
- `Average_Year_aerator_sim.png`
- `Largest_Humidity_Summer_aerator_sim.png`
- `Largest_Temp_Summer_aerator_sim.png`  
  DO schedule simulation plots for each scenario.

#### `PURE WEATHER/figures/predictions/`
- `Average_Year_predictions_1hr.png`
- `Average_Year_predictions_1hrdelta.png`
- `Largest_Humidity_Summer_predictions.png`
- `Largest_Humidity_Summer_predictions_1hr.png`
- `Largest_Humidity_Summer_predictions_1hrdelta.png`
- `Largest_Humidity_Summer_predictions_6hr.png`
- `Largest_Temp_Summer_predictions.png`
- `Largest_Temp_Summer_predictions_1hr.png`
- `Largest_Temp_Summer_predictions_1hrdelta.png`  
  Prediction plots corresponding to `ParsedWeatherData/predictions/*.csv`.

### `PURE WEATHER/outputs/`
Centralized non-figure generated output area.

#### `PURE WEATHER/outputs/aerator_analysis/`
- `daily_DO_linear_slopes.csv`  
  Daily slope/intercept/r² summary from aerator linear-fit analysis.

#### `PURE WEATHER/outputs/aerator_simulations/`
- `Average_Year_aerator_sim.csv`
- `Largest_Humidity_Summer_aerator_sim.csv`
- `Largest_Temp_Summer_aerator_sim.csv`  
  Simulated DO time-series outputs used to generate aerator sim figures.

---

## `ParsedWeatherData/` (root-level legacy copy)

- `Average_Year.csv`
- `Largest_Humidity_Summer.csv`
- `Largest_Temp_Summer.csv`

These mirror weather scenario tables outside PURE WEATHER (historical/legacy location).

---

## `temp/` (analysis scratch space)

### `temp/DataCoverageCSVs/`
- `consecutive_day_blocks.csv`  
  Output summary of contiguous day blocks in date coverage analysis.
- `daily_time_coverage.csv`  
  Output table of per-day time coverage/holes.

### `temp/date_coverage_analysis/`
- `analyze_time_blocks.py`  
  Script that computes date/time coverage diagnostics.
- `test_time_blocks.py`  
  Tests for date coverage analysis functions.

---

## Practical dependency flow (high-level)

1. **Main app**: `Main.py` -> `CSV/train/ParsingFiles/Core.py` -> `Module/Connector.py`.
2. Connector calls: **Parse** -> **Clean** -> **Lag** -> **DORate** for lake data.
3. Model train/predict uses `Module/Model/*` and stores bundles in `PURE WEATHER/Models` (and some legacy in lowercase `models`).
4. PURE WEATHER scripts consume `data/lake`, `ParsedWeatherData`, write structured outputs to `outputs/` and figures to `figures/`.

---

If you want, I can also produce a second doc with a **function-level API map** (public functions + signatures + where each is called).
