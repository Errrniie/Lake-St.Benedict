#!/usr/bin/env python3
"""Run inference with a weather-only bundle from PURE WEATHER/models (same features as training)."""
from __future__ import annotations

import argparse
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PARSING = os.path.join(_ROOT, "CSV", "train", "ParsingFiles")
_DEFAULT_PREDICTIONS_DIR = os.path.join(
    _ROOT, "PURE WEATHER", "ParsedWeatherData", "predictions"
)
if _PARSING not in sys.path:
    sys.path.insert(0, _PARSING)

import joblib
import pandas as pd

from Module.Model.prediction_plot import save_prediction_plot, save_prediction_plot_time_only

# Reuse feature construction (import would require package path; duplicate minimal list)
WEATHER_COLS = [
    "Air temp C",
    "Relative Humidity (%)",
    "Atmospheric Pressure (mb)",
    "hour",
]
DATE_PARTS = ["year", "month", "day", "dayofyear"]


def add_date_features(df: pd.DataFrame, date_col: str = "DATE") -> pd.DataFrame:
    out = df.copy()
    dt = pd.to_datetime(out[date_col], errors="coerce")
    out["year"] = dt.dt.year
    out["month"] = dt.dt.month
    out["day"] = dt.dt.day
    out["dayofyear"] = dt.dt.dayofyear
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict with a weather-only model bundle.")
    parser.add_argument("--model", required=True, help="Path to model_weather_only_*.pkl")
    parser.add_argument("--csv", required=True, help="Input CSV with DATE + weather columns")
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help=(
            "Output CSV path (default: PURE WEATHER/ParsedWeatherData/predictions/"
            "<input_stem>_predictions.csv)"
        ),
    )
    args = parser.parse_args()

    bundle = joblib.load(args.model)
    model = bundle["model"]
    feature_cols = bundle["feature_cols"]
    medians = bundle.get("feature_medians", {})
    target_col = bundle.get("target_col", "target")

    df = pd.read_csv(args.csv)
    df.columns = df.columns.str.strip()
    df = add_date_features(df)

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        print(f"Missing columns: {missing}", file=sys.stderr)
        sys.exit(1)

    X = df[feature_cols].copy()
    for c in feature_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.fillna(medians)

    pred_col = f"pred_{target_col}_weather_only"
    df[pred_col] = model.predict(X)

    if args.output:
        out_path = args.output
    else:
        os.makedirs(_DEFAULT_PREDICTIONS_DIR, exist_ok=True)
        stem = os.path.splitext(os.path.basename(os.path.abspath(args.csv)))[0]
        out_path = os.path.join(_DEFAULT_PREDICTIONS_DIR, f"{stem}_predictions.csv")
    if not out_path.lower().endswith(".csv"):
        out_path = f"{out_path}.csv"
    df.to_csv(out_path, index=False)
    if "delta" in pred_col.lower():
        plot_path = save_prediction_plot_time_only(
            out_path,
            df,
            pred_col,
            date_col="DATE",
            ylabel="Predicted Delta DO",
            title="Predicted Delta DO",
        )
    else:
        plot_path = save_prediction_plot(
            out_path,
            df,
            pred_col,
            date_col="DATE",
            title=os.path.basename(out_path),
        )
    print(f"Wrote {out_path} ({len(df)} rows), column {pred_col}")
    print(f"Wrote plot {plot_path}")


if __name__ == "__main__":
    main()
