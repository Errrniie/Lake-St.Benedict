"""
Shared quantile LightGBM training — PureWeather feature set only.

Features: Air temp, RH, pressure, hour, calendar (year, month, day, dayofyear from DATE),
and AT_TLAG columns (1hr/3hr/6hrs or legacy 15/30/60). Never uses DO, Water Temp, or
DO_delta / future DO columns as inputs. Target is exactly one of: DO, a single DO_delta_*,
or a single future level column DO_*h (e.g. DO_1h).
"""
from __future__ import annotations

import math
import os
import re

import lightgbm as lgb
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error

from Module.DORate.dodeltas_module import is_future_do_column
from Module.Model.model_io import ensure_models_dir, save_bundle


def _resolve_date_col(df: pd.DataFrame) -> str:
    if "DATE" in df.columns:
        return "DATE"
    return df.columns[0]


def _candidate_targets(df: pd.DataFrame) -> list[str]:
    out: list[str] = []
    if "DO" in df.columns:
        out.append("DO")
    out.extend(sorted(c for c in df.columns if c.startswith("DO_delta_")))
    out.extend(sorted(c for c in df.columns if is_future_do_column(c)))
    return out


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


def _safe_filename_component(name: str) -> str:
    return re.sub(r"[^\w.\-]+", "_", str(name))


def _add_calendar_features(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    out = df.copy()
    dt = pd.to_datetime(out[date_col], errors="coerce")
    out["year"] = dt.dt.year
    out["month"] = dt.dt.month
    out["day"] = dt.dt.day
    out["dayofyear"] = dt.dt.dayofyear
    return out


def _pure_weather_feature_columns(df: pd.DataFrame) -> list[str]:
    base = [
        "Air temp C",
        "Relative Humidity (%)",
        "Atmospheric Pressure (mb)",
        "hour",
    ]
    calendar = ["year", "month", "day", "dayofyear"]
    at_groups = [
        ("AT_TLAG_1hr", "AT_TLAG_3hr", "AT_TLAG_6hrs"),
        ("AT_TLAG_15", "AT_TLAG_30", "AT_TLAG_60"),
    ]
    at_lags: list[str] = []
    for group in at_groups:
        if all(c in df.columns for c in group):
            at_lags = list(group)
            break
    if not at_lags:
        at_lags = [c for c in df.columns if str(c).startswith("AT_TLAG_")]
    if not at_lags:
        raise ValueError(
            "No AT_TLAG_* columns found. Expected AT_TLAG_1hr/3hr/6hrs or AT_TLAG_15/30/60."
        )
    cols = base + calendar + at_lags
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"PURE WEATHER training missing columns: {missing}")
    return cols


def _default_models_dir_for_csv(csv_path: str) -> str:
    """Save new bundles under PURE WEATHER/Models when possible."""
    base = os.path.dirname(os.path.abspath(csv_path))
    if os.path.basename(base) == "PURE WEATHER":
        return ensure_models_dir(os.path.join(base, "Models"))
    return ensure_models_dir(os.path.join(base, "PURE WEATHER", "Models"))


def train_quantile_model_on_csv(
    csv_path: str,
    target_col: str,
    quantile_alpha: float,
    quantile_label: str,
    model_output_path: str | None = None,
    include_other_delta_features: bool = True,
) -> dict:
    """
    Train a LightGBM quantile regressor using the PureWeather feature set only.

    ``include_other_delta_features`` is kept for API compatibility and ignored.

    ``quantile_label`` is one of low / mid / high (used in filenames and bundle metadata).
    """
    del include_other_delta_features  # not used; inputs are fixed

    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    date_col = _resolve_date_col(df)

    targets = _candidate_targets(df)
    if target_col not in targets:
        raise ValueError(
            f"Target column {target_col!r} not found. Expected one of: DO, DO_delta_*, or DO_*h (future level)."
        )

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = _add_calendar_features(df, date_col)
    feature_cols = _pure_weather_feature_columns(df)

    numeric_cols = feature_cols + [target_col]
    df = _to_numeric_frame(df, numeric_cols)
    df = df.dropna(subset=[target_col]).copy()

    date_key = date_col
    train_df, test_df = _split_by_week(df, date_col=date_key, test_ratio=0.30)
    if len(train_df) < 20 or len(test_df) < 10:
        raise ValueError("Not enough data after split to train/test.")

    train_fit_df, val_df = _split_by_week(train_df, date_col=date_key, test_ratio=0.20)
    if len(train_fit_df) < 10 or len(val_df) < 5:
        raise ValueError(
            "Not enough data after train/validation split; need at least 10 fit rows and 5 validation rows."
        )

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
    mae = float(mean_absolute_error(y_test, pred_test))

    if not model_output_path:
        models_dir = _default_models_dir_for_csv(csv_path)
        safe_t = _safe_filename_component(target_col)
        model_output_path = os.path.join(
            models_dir,
            f"model_{safe_t}_quantile_{quantile_label}.pkl",
        )

    bundle = {
        "model": model,
        "target_col": target_col,
        "feature_cols": feature_cols,
        "feature_medians": medians,
        "mae": mae,
        "mae_validation": mae_val,
        "train_rows": int(len(train_fit_df)),
        "validation_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "quantile_alpha": float(quantile_alpha),
        "quantile_label": quantile_label,
        "feature_mode": "pure_weather",
    }
    saved_path = save_bundle(bundle, model_output_path)
    bundle["model_path"] = saved_path
    return bundle
