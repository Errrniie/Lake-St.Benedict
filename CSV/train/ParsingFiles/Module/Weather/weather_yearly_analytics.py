#!/usr/bin/env python3
"""
Yearly weather analytics on parsed weather CSVs (DATE, temp, RH, pressure [, hour]).

Produces:
  Largest_Temp_Summer.csv — full year of data for the year with highest JJA mean temperature
  Largest_Humidity_Summer.csv — same for JJA mean relative humidity
  Average_Year.csv — mean temp/RH/pressure at each calendar slot (month, day, time) across years
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

from Module.Parse.parse_module import read_csv
from Module.Weather.weather_init_module import COL_DATE, COL_PRESSURE

COL_TEMP = "Air temp C"
COL_RH = "Relative Humidity (%)"
COL_HOUR = "hour"

SUMMER_MONTHS = (6, 7, 8)
REFERENCE_YEAR = 2000  # leap year for synthetic DATE in Average_Year

OUT_LARGEST_TEMP = "Largest_Temp_Summer.csv"
OUT_LARGEST_RH = "Largest_Humidity_Summer.csv"
OUT_AVERAGE_YEAR = "Average_Year.csv"


def _require_columns(df: pd.DataFrame) -> None:
    need = [COL_DATE, COL_TEMP, COL_RH, COL_PRESSURE]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found: {list(df.columns)}")


def _prepare_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Parse dates and coerce numeric weather columns."""
    out = df.copy()
    out.columns = out.columns.str.strip()
    _require_columns(out)

    out[COL_DATE] = pd.to_datetime(out[COL_DATE], errors="coerce")
    out = out.dropna(subset=[COL_DATE]).reset_index(drop=True)

    for c in (COL_TEMP, COL_RH, COL_PRESSURE):
        out[c] = pd.to_numeric(out[c], errors="coerce")

    if COL_HOUR in out.columns:
        out[COL_HOUR] = pd.to_numeric(out[COL_HOUR], errors="coerce")

    return out


def _year_with_highest_jja_mean(df: pd.DataFrame, value_col: str) -> int:
    dt = df[COL_DATE]
    summer = df[dt.dt.month.isin(SUMMER_MONTHS)]
    if summer.empty:
        raise ValueError("No rows in June–August; cannot compute summer means.")
    summer = summer.assign(_year=dt.loc[summer.index].dt.year)
    by_year = summer.groupby("_year", sort=True)[value_col].mean().dropna()
    if by_year.empty:
        raise ValueError(f"No numeric data for {value_col} in summer months.")
    best = by_year.idxmax()
    return int(best)


def largest_temp_summer_year_frame(df: pd.DataFrame) -> pd.DataFrame:
    """All rows for the year whose June–August mean temperature is highest."""
    work = _prepare_frame(df)
    best_year = _year_with_highest_jja_mean(work, COL_TEMP)
    dt = work[COL_DATE]
    sub = work[dt.dt.year == best_year].copy()
    sub[COL_DATE] = sub[COL_DATE].dt.strftime("%Y-%m-%d %H:%M:%S")
    return _order_output_columns(sub)


def largest_humidity_summer_year_frame(df: pd.DataFrame) -> pd.DataFrame:
    """All rows for the year whose June–August mean relative humidity is highest."""
    work = _prepare_frame(df)
    best_year = _year_with_highest_jja_mean(work, COL_RH)
    dt = work[COL_DATE]
    sub = work[dt.dt.year == best_year].copy()
    sub[COL_DATE] = sub[COL_DATE].dt.strftime("%Y-%m-%d %H:%M:%S")
    return _order_output_columns(sub)


def _order_output_columns(df: pd.DataFrame) -> pd.DataFrame:
    base = [COL_DATE, COL_TEMP, COL_RH, COL_PRESSURE]
    cols = base + ([COL_HOUR] if COL_HOUR in df.columns else [])
    return df[[c for c in cols if c in df.columns]]


def average_year_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each calendar (month, day, hour, minute, second), mean temp, RH, and pressure across years.

    DATE uses a reference leap year (2000) so month-day-times align; hour is the slot hour as float.
    """
    work = _prepare_frame(df)
    dt = work[COL_DATE]
    work = work.assign(
        _m=dt.dt.month,
        _d=dt.dt.day,
        _H=dt.dt.hour,
        _Mi=dt.dt.minute,
        _S=dt.dt.second,
    )

    grouped = (
        work.groupby(["_m", "_d", "_H", "_Mi", "_S"], sort=True)
        .agg({COL_TEMP: "mean", COL_RH: "mean", COL_PRESSURE: "mean"})
        .reset_index()
    )

    ts = pd.to_datetime(
        {
            "year": REFERENCE_YEAR,
            "month": grouped["_m"],
            "day": grouped["_d"],
            "hour": grouped["_H"],
            "minute": grouped["_Mi"],
            "second": grouped["_S"],
        }
    )
    grouped[COL_DATE] = ts.dt.strftime("%Y-%m-%d %H:%M:%S")
    grouped[COL_HOUR] = grouped["_H"].astype(float)
    out = grouped.drop(columns=["_m", "_d", "_H", "_Mi", "_S"])
    out = out[[COL_DATE, COL_TEMP, COL_RH, COL_PRESSURE, COL_HOUR]]
    return out.sort_values(COL_DATE).reset_index(drop=True)


def process_weather_yearly_analytics(
    input_path: str,
    output_dir: str | None = None,
) -> dict[str, str]:
    """
    Read parsed weather CSV, write three analytics CSVs into ``output_dir``.

    Returns a dict of logical name -> absolute path written.
    """
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if not output_dir:
        output_dir = os.path.dirname(os.path.abspath(input_path)) or "."
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    df = read_csv(input_path)

    temp_df = largest_temp_summer_year_frame(df)
    rh_df = largest_humidity_summer_year_frame(df)
    avg_df = average_year_frame(df)

    paths: dict[str, str] = {}
    mapping = [
        ("largest_temp_summer", OUT_LARGEST_TEMP, temp_df),
        ("largest_humidity_summer", OUT_LARGEST_RH, rh_df),
        ("average_year", OUT_AVERAGE_YEAR, avg_df),
    ]

    for key, fname, frame in mapping:
        out_path = os.path.join(output_dir, fname)
        if os.path.abspath(out_path) == os.path.abspath(input_path):
            raise ValueError(f"Refuse to overwrite input file: {input_path}")
        frame.to_csv(out_path, index=False)
        paths[key] = out_path

    return paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Weather yearly analytics: largest JJA year (temp/RH) + average-year profile"
    )
    parser.add_argument("--input", "-i", required=False, help="Parsed weather CSV path")
    parser.add_argument(
        "--output-dir",
        "-d",
        required=False,
        help="Directory for output CSVs (default: same folder as input)",
    )

    args = parser.parse_args()
    input_path = args.input
    if not input_path:
        try:
            input_path = input("Enter path to parsed weather CSV: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("No input provided. Exiting.", file=sys.stderr)
            sys.exit(2)
        if not input_path:
            print("No input provided. Exiting.", file=sys.stderr)
            sys.exit(2)

    try:
        paths = process_weather_yearly_analytics(input_path, output_dir=args.output_dir)
        for k, p in paths.items():
            print(f"{k}: {p}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
