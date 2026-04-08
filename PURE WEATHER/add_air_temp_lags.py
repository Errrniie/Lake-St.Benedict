#!/usr/bin/env python3
"""Add AT_TLAG_1hr, AT_TLAG_3hr, AT_TLAG_6hrs to ParsedWeatherData CSVs (same logic as Module/Lag/lag_module)."""
from __future__ import annotations

import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PARSED_DIR = os.path.join(HERE, "ParsedWeatherData")

DATE_COL = "DATE"
SOURCE = "Air temp C"
LAGS = (
    (1, "AT_TLAG_1hr"),
    (3, "AT_TLAG_3hr"),
    (6, "AT_TLAG_6hrs"),
)


def add_time_lag(
    df: pd.DataFrame,
    date_col: str,
    hours: int,
    source_col: str,
    new_col: str,
) -> pd.DataFrame:
    lag_df = df[[date_col, source_col]].copy()
    lag_df[date_col] = lag_df[date_col] + pd.Timedelta(hours=hours)
    lag_df = lag_df.rename(columns={source_col: new_col})
    return pd.merge_asof(df, lag_df, on=date_col, direction="backward")


def process_csv(path: str) -> None:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    if DATE_COL not in df.columns or SOURCE not in df.columns:
        raise ValueError(f"{path}: need {DATE_COL} and {SOURCE}")

    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df = df.dropna(subset=[DATE_COL]).sort_values(DATE_COL)

    for _, col in LAGS:
        if col in df.columns:
            df = df.drop(columns=[col])

    out = df
    for hours, col in LAGS:
        out = add_time_lag(out, DATE_COL, hours, SOURCE, col)

    for _, col in LAGS:
        out[col] = pd.to_numeric(out[col], errors="coerce").ffill().bfill()

    base = [DATE_COL, SOURCE, "Relative Humidity (%)", "Atmospheric Pressure (mb)", "hour"]
    tail = [c for _, c in LAGS]
    ordered = [c for c in base if c in out.columns] + [c for c in tail if c in out.columns]
    extra = [c for c in out.columns if c not in ordered]
    out = out[ordered + extra]

    out.to_csv(path, index=False)
    print(f"Updated {path} ({len(out)} rows)")


def main() -> None:
    if not os.path.isdir(PARSED_DIR):
        raise SystemExit(f"Missing folder: {PARSED_DIR}")
    for name in sorted(os.listdir(PARSED_DIR)):
        if not name.lower().endswith(".csv"):
            continue
        process_csv(os.path.join(PARSED_DIR, name))


if __name__ == "__main__":
    main()
