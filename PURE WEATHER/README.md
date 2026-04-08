# PURE WEATHER — weather-only feature experiment

## Folders

- **`Models/`** — copy of `CSV/train/ParsingFiles/Module/Model/Models` plus new bundles from the **GUI** (Train / Predict / Model Manager all use this path).
- **`models/`** (lowercase) — legacy outputs from `train_weather_only.py` only; prefer **`Models/`** for new work.

## `DO_lakedata_parsed.csv`

Working copy of the lake table: **Water Temp removed**. Columns include weather, `hour`, `AT_TLAG_*`, `DO`, and `DO_delta_*`.

## GUI training (main app)

Training uses **PureWeather** features only (see `train_quantile_core.py`):

- `Air temp C`, `Relative Humidity (%)`, `Atmospheric Pressure (mb)`, `hour`
- Calendar from `DATE`: `year`, `month`, `day`, `dayofyear`
- `AT_TLAG_1hr` / `AT_TLAG_3hr` / `AT_TLAG_6hrs` (or legacy `AT_TLAG_15` / `30` / `60`)

**Target:** pick **either** `DO` **or** a **single** `DO_delta_*` per run (not both in one model).

## Standalone script (`train_weather_only.py`)

From the **repository root**:

```bash
python "PURE WEATHER/train_weather_only.py"
```

Default input is **`PURE WEATHER/DO_lakedata_parsed.csv`**.

Metrics JSON default: **`PURE WEATHER/models/metrics_weather_only.json`** (`--out-dir` to change).

## Predict (standalone)

```bash
python "PURE WEATHER/predict_weather_only.py" --model "PURE WEATHER/Models/model_DO_delta_12h_quantile_mid.pkl" --csv "PURE WEATHER/DO_lakedata_parsed.csv" -o predictions_test.csv
```

Dependencies: same as the main project (`pandas`, `lightgbm`, `scikit-learn`, `joblib`).
