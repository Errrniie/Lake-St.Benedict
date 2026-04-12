from __future__ import annotations

import os

import pandas as pd

from Module.Model.model_io import load_bundle
from Module.Model.prediction_plot import save_prediction_plot, save_prediction_plot_time_only


def predict_from_model_bundle(
    model_path: str,
    input_csv_path: str,
    output_csv_path: str,
) -> dict:
    if not os.path.isfile(input_csv_path):
        raise FileNotFoundError(f"Input CSV not found: {input_csv_path}")

    bundle = load_bundle(model_path)
    model = bundle.get("model")
    target_col = str(bundle.get("target_col", "target"))
    feature_cols = bundle.get("feature_cols")
    medians = bundle.get("feature_medians", {})
    if model is None or not isinstance(feature_cols, list):
        raise ValueError("Invalid model bundle format.")

    df = pd.read_csv(input_csv_path)
    df.columns = df.columns.str.strip()

    feature_mode = bundle.get("feature_mode")
    calendar = ("year", "month", "day", "dayofyear")
    if feature_mode == "pure_weather" or any(c in feature_cols for c in calendar):
        if "DATE" not in df.columns:
            raise ValueError("Input CSV must contain DATE for calendar features.")
        dt = pd.to_datetime(df["DATE"], errors="coerce")
        if "year" in feature_cols:
            df["year"] = dt.dt.year
        if "month" in feature_cols:
            df["month"] = dt.dt.month
        if "day" in feature_cols:
            df["day"] = dt.dt.day
        if "dayofyear" in feature_cols:
            df["dayofyear"] = dt.dt.dayofyear

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Input CSV missing required feature columns: {missing}")

    X = df[feature_cols].copy()
    for c in feature_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.fillna(medians)

    preds = model.predict(X)
    q_label = bundle.get("quantile_label")
    if isinstance(q_label, str) and q_label:
        pred_col = f"pred_{target_col}_quantile_{q_label}"
    else:
        pred_col = f"pred_{target_col}"
    df[pred_col] = preds

    if not output_csv_path.lower().endswith(".csv"):
        output_csv_path = f"{output_csv_path}.csv"
    df.to_csv(output_csv_path, index=False)

    if "delta" in pred_col.lower():
        plot_path = save_prediction_plot_time_only(
            output_csv_path,
            df,
            pred_col,
            date_col="DATE",
            ylabel="Predicted Delta DO",
            title="Predicted Delta DO",
        )
    else:
        plot_path = save_prediction_plot(
            output_csv_path,
            df,
            pred_col,
            date_col="DATE",
            title=os.path.basename(output_csv_path),
        )

    return {
        "output_path": output_csv_path,
        "prediction_column": pred_col,
        "rows": int(len(df)),
        "plot_path": plot_path,
    }

