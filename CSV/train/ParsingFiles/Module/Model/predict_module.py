from __future__ import annotations

import os

import pandas as pd

from Module.Model.model_io import load_bundle


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

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Input CSV missing required feature columns: {missing}")

    X = df[feature_cols].copy()
    for c in feature_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.fillna(medians)

    preds = model.predict(X)
    pred_col = f"pred_{target_col}"
    df[pred_col] = preds

    if not output_csv_path.lower().endswith(".csv"):
        output_csv_path = f"{output_csv_path}.csv"
    df.to_csv(output_csv_path, index=False)

    return {
        "output_path": output_csv_path,
        "prediction_column": pred_col,
        "rows": int(len(df)),
    }

