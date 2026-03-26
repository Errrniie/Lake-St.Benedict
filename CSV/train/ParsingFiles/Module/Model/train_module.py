from __future__ import annotations

import math
import os

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error

from Module.Model.model_io import ensure_models_dir, save_bundle


def _resolve_date_col(df: pd.DataFrame) -> str:
    if "DATE" in df.columns:
        return "DATE"
    return df.columns[0]


def _candidate_targets(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("DO_delta_")]


def _to_numeric_frame(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _split_by_week(df: pd.DataFrame, date_col: str, test_ratio: float = 0.30) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out = out.dropna(subset=[date_col]).sort_values(date_col)
    iso = out[date_col].dt.isocalendar()
    out["_week_key"] = iso["year"].astype(str) + "-W" + iso["week"].astype(str)
    weeks = out["_week_key"].drop_duplicates().tolist()
    if len(weeks) >= 3:
        n_test = max(1, math.ceil(len(weeks) * test_ratio))
        test_weeks = set(weeks[-n_test:])
        test = out[out["_week_key"].isin(test_weeks)].copy()
        train = out[~out["_week_key"].isin(test_weeks)].copy()
    else:
        split_idx = int(len(out) * (1.0 - test_ratio))
        train = out.iloc[:split_idx].copy()
        test = out.iloc[split_idx:].copy()
    return train.drop(columns=["_week_key"], errors="ignore"), test.drop(columns=["_week_key"], errors="ignore")


def train_model_on_csv(
    csv_path: str,
    target_col: str,
    model_output_path: str | None = None,
    include_other_delta_features: bool = True,
) -> dict:
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    date_col = _resolve_date_col(df)

    targets = _candidate_targets(df)
    if target_col not in targets:
        raise ValueError(f"Target column {target_col!r} not found.")

    drop_cols = {target_col}
    if not include_other_delta_features:
        drop_cols.update(c for c in df.columns if c.startswith("DO_delta_"))
    drop_cols.add(date_col)

    feature_cols = [c for c in df.columns if c not in drop_cols]
    if not feature_cols:
        raise ValueError("No feature columns available after exclusions.")

    numeric_cols = feature_cols + [target_col]
    df = _to_numeric_frame(df, numeric_cols)
    df = df.dropna(subset=[target_col]).copy()

    train_df, test_df = _split_by_week(df, date_col=date_col if date_col in df.columns else df.columns[0])
    if len(train_df) < 20 or len(test_df) < 10:
        raise ValueError("Not enough data after split to train/test.")

    X_train = train_df[feature_cols].copy()
    X_test = test_df[feature_cols].copy()
    y_train = train_df[target_col].astype(float)
    y_test = test_df[target_col].astype(float)

    medians = X_train.median(numeric_only=True).to_dict()
    X_train = X_train.fillna(medians)
    X_test = X_test.fillna(medians)

    model = LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
    )
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    mae = float(mean_absolute_error(y_test, pred))

    if not model_output_path:
        models_dir = ensure_models_dir()
        model_output_path = os.path.join(models_dir, f"model_{target_col}.pkl")

    bundle = {
        "model": model,
        "target_col": target_col,
        "feature_cols": feature_cols,
        "feature_medians": medians,
        "mae": mae,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
    }
    saved_path = save_bundle(bundle, model_output_path)
    bundle["model_path"] = saved_path
    return bundle

