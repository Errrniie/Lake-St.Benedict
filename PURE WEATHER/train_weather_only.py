#!/usr/bin/env python3
"""
Train LightGBM quantile (median) models using ONLY weather + calendar features.
Compare MAE against full-feature models trained via ParsingFiles/Module/Model/.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from typing import Any

import joblib
import lightgbm as lgb
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error

# --- Same time-based splits as Module/Model/train_quantile_core.py ---


def _split_by_week(
    df: pd.DataFrame, date_col: str, test_ratio: float = 0.30
) -> tuple[pd.DataFrame, pd.DataFrame]:
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
    return train.drop(columns=["_week_key"], errors="ignore"), test.drop(
        columns=["_week_key"], errors="ignore"
    )


def _safe_filename(name: str) -> str:
    return re.sub(r"[^\w.\-]+", "_", str(name))


WEATHER_COLS = [
    "Air temp C",
    "Relative Humidity (%)",
    "Atmospheric Pressure (mb)",
    "hour",
]

DATE_PARTS = ["year", "month", "day", "dayofyear"]


def _add_date_features(df: pd.DataFrame, date_col: str = "DATE") -> pd.DataFrame:
    out = df.copy()
    dt = pd.to_datetime(out[date_col], errors="coerce")
    out["year"] = dt.dt.year
    out["month"] = dt.dt.month
    out["day"] = dt.dt.day
    out["dayofyear"] = dt.dt.dayofyear
    return out


def _default_targets(df: pd.DataFrame) -> list[str]:
    deltas = sorted(
        [c for c in df.columns if c.startswith("DO_delta_")],
        key=lambda x: (len(x), x),
    )
    futures = sorted(
        [c for c in df.columns if re.fullmatch(r"DO_\d+h", str(c).strip())],
        key=lambda x: (len(x), x),
    )
    out = ["DO"] + deltas + futures
    return [c for c in out if c in df.columns]


def build_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    missing = [c for c in WEATHER_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")
    work = _add_date_features(df)
    at_groups = [
        ("AT_TLAG_1hr", "AT_TLAG_3hr", "AT_TLAG_6hrs"),
        ("AT_TLAG_15", "AT_TLAG_30", "AT_TLAG_60"),
    ]
    at_lags: list[str] = []
    for group in at_groups:
        if all(c in work.columns for c in group):
            at_lags = list(group)
            break
    if not at_lags:
        at_lags = [c for c in work.columns if str(c).startswith("AT_TLAG_")]
    feature_cols = WEATHER_COLS + DATE_PARTS + at_lags
    for c in feature_cols:
        if c not in work.columns:
            raise ValueError(f"Missing feature column after date parsing: {c}")
    return work, feature_cols


def train_one_target(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    date_col: str,
    quantile_alpha: float = 0.5,
) -> dict[str, Any]:
    work = df.copy()
    work[target_col] = pd.to_numeric(work[target_col], errors="coerce")
    work = work.dropna(subset=[target_col]).copy()
    if len(work) < 30:
        raise ValueError(f"Not enough rows with non-null {target_col!r}.")

    numeric = feature_cols + [target_col]
    for c in feature_cols:
        work[c] = pd.to_numeric(work[c], errors="coerce")

    work = work.dropna(subset=feature_cols + [target_col]).copy()

    train_df, test_df = _split_by_week(work, date_col=date_col, test_ratio=0.30)
    if len(train_df) < 20 or len(test_df) < 10:
        raise ValueError("Not enough data after train/test split.")

    train_fit_df, val_df = _split_by_week(train_df, date_col=date_col, test_ratio=0.20)
    if len(train_fit_df) < 10 or len(val_df) < 5:
        raise ValueError("Not enough data after train/validation split.")

    X_train = train_fit_df[feature_cols].copy()
    X_val = val_df[feature_cols].copy()
    X_test = test_df[feature_cols].copy()
    y_train = train_fit_df[target_col].astype(float)
    y_val = val_df[target_col].astype(float)
    y_test = test_df[target_col].astype(float)

    medians = X_train.median(numeric_only=True).to_dict()
    X_train = X_train.fillna(medians)
    X_val = X_val.fillna(medians)
    X_test = X_test.fillna(medians)

    model = LGBMRegressor(
        objective="quantile",
        alpha=quantile_alpha,
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="mae",
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
    )

    pred_val = model.predict(X_val)
    pred_test = model.predict(X_test)
    mae_val = float(mean_absolute_error(y_val, pred_val))
    mae_test = float(mean_absolute_error(y_test, pred_test))

    bundle = {
        "model": model,
        "target_col": target_col,
        "feature_cols": feature_cols,
        "feature_medians": medians,
        "mae": mae_test,
        "mae_validation": mae_val,
        "train_rows": int(len(train_fit_df)),
        "validation_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "quantile_alpha": float(quantile_alpha),
        "quantile_label": "mid",
        "experiment": "pure_weather",
    }
    return bundle


def main() -> None:
    _here = os.path.dirname(os.path.abspath(__file__))
    default_csv = os.path.join(_here, "DO_lakedata_parsed.csv")
    if not os.path.isfile(default_csv):
        default_csv = os.path.join(os.path.dirname(_here), "DO_lakedata_parsed.csv")

    parser = argparse.ArgumentParser(
        description="Train weather-only models for DO / DO_delta_* / DO_*h targets (PURE WEATHER experiment)."
    )
    parser.add_argument(
        "--csv",
        default=default_csv,
        help=f"Parsed lake CSV (default: {default_csv})",
    )
    parser.add_argument(
        "--out-dir",
        default=os.path.join(os.path.dirname(__file__), "models"),
        help="Directory for .pkl bundles and metrics JSON",
    )
    parser.add_argument(
        "--targets",
        nargs="*",
        default=None,
        help="Target columns (default: DO + all DO_delta_* + DO_*h future levels)",
    )
    args = parser.parse_args()

    csv_path = os.path.abspath(args.csv)
    if not os.path.isfile(csv_path):
        print(f"File not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    raw = pd.read_csv(csv_path)
    raw.columns = raw.columns.str.strip()
    if "DATE" not in raw.columns:
        raise SystemExit("CSV must contain a DATE column.")

    work, feature_cols = build_feature_matrix(raw)
    date_col = "DATE"

    targets = args.targets if args.targets else _default_targets(work)
    targets = [t for t in targets if t in work.columns]
    if not targets:
        raise SystemExit("No valid targets found in CSV.")

    all_metrics: list[dict[str, Any]] = []

    for target_col in targets:
        try:
            bundle = train_one_target(work, feature_cols, target_col, date_col)
        except Exception as exc:
            print(f"[skip] {target_col}: {exc}", file=sys.stderr)
            continue

        safe_t = _safe_filename(target_col)
        path = os.path.join(out_dir, f"model_weather_only_{safe_t}_mid.pkl")
        joblib.dump(bundle, path)
        bundle["model_path"] = path
        print(
            f"{target_col}: MAE_test={bundle['mae']:.6f} MAE_val={bundle['mae_validation']:.6f} -> {path}"
        )
        all_metrics.append(
            {
                "target": target_col,
                "mae_test": bundle["mae"],
                "mae_validation": bundle["mae_validation"],
                "train_rows": bundle["train_rows"],
                "validation_rows": bundle["validation_rows"],
                "test_rows": bundle["test_rows"],
                "feature_cols": feature_cols,
                "model_path": path,
            }
        )

    summary_path = os.path.join(out_dir, "metrics_weather_only.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "csv": csv_path,
                "features": feature_cols,
                "targets_trained": [m["target"] for m in all_metrics],
                "results": all_metrics,
            },
            f,
            indent=2,
        )
    print(f"\nWrote summary: {summary_path}")


if __name__ == "__main__":
    main()
