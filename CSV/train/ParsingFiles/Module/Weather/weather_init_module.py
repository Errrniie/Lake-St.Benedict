#!/usr/bin/env python3
"""
Weather init: strip a raw weather CSV to four columns and rename them for lake-style use.

Input columns matched (case-insensitive): valid/date, tmpc, relh, mslp — or already-renamed
headers from a prior export.

Output columns: DATE, Air temp C, Relative Humidity (%), Atmospheric Pressure (mb), hour.

Post-steps: drop rows where pressure contains ``m``; parse DATE; keep only May 1–September 30
each year; ceil to next hour (:00); add ``hour`` as float (e.g. 2.0).
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

from Module.Parse.parse_module import make_default_output_path, read_csv, write_csv

# Core weather columns (same order as source slots below).
OUTPUT_COLUMNS = (
    "DATE",
    "Air temp C",
    "Relative Humidity (%)",
    "Atmospheric Pressure (mb)",
)

COL_DATE = "DATE"
COL_PRESSURE = "Atmospheric Pressure (mb)"
COL_HOUR = "hour"

# For each output name, acceptable source header names (lowercase comparison).
def _mask_may1_through_sept30(dt: pd.Series) -> pd.Series:
    """True where calendar date is between May 1 and September 30 (inclusive) for any year."""
    m = dt.dt.month
    d = dt.dt.day
    return (m == 5) | m.isin((6, 7, 8)) | ((m == 9) & (d <= 30))


_SOURCE_ALIASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("DATE", ("valid", "date")),
    ("Air temp C", ("tmpc", "air temp c")),
    ("Relative Humidity (%)", ("relh", "relative humidity (%)")),
    ("Atmospheric Pressure (mb)", ("mslp", "atmospheric pressure (mb)")),
)


def _find_source_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    """Pick the first column whose stripped lower name matches one of candidates."""
    lowered = {str(c).strip().lower(): str(c) for c in df.columns}
    want = tuple(c.strip().lower() for c in candidates)
    for w in want:
        if w in lowered:
            return lowered[w]
    return None


def filter_weather_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only the four weather columns; rename to OUTPUT_COLUMNS.

    Raises ValueError if any required column is missing.
    """
    work = df.copy()
    work.columns = work.columns.str.strip()

    picked: list[str] = []
    for out_name, candidates in _SOURCE_ALIASES:
        found = _find_source_column(work, candidates)
        if not found:
            raise ValueError(
                f"Missing required column for {out_name!r} "
                f"(expected one of {candidates}). "
                f"Columns present: {list(work.columns)}"
            )
        picked.append(found)

    out = work[picked].copy()
    out.columns = list(OUTPUT_COLUMNS)
    return out


def apply_weather_transforms(df: pd.DataFrame) -> pd.DataFrame:
    """
    After column selection: drop bad pressure rows, normalize DATE, ceil to hour, add hour.

    - Removes any row where ``Atmospheric Pressure (mb)`` contains ``m`` (e.g. missing flags).
    - Parses DATE, drops unparseable rows, keeps only May 1–September 30 (inclusive) each year.
    - Rounds each timestamp up to the next hour boundary so minutes/seconds are zero.
    - Writes DATE as ``YYYY-MM-DD HH:MM:SS``.
    - Adds ``hour`` as the hour from DATE (float, e.g. 2.0).
    """
    out = df.copy()

    bad_pressure = (
        out[COL_PRESSURE]
        .astype(str)
        .str.contains("m", case=False, na=False, regex=False)
    )
    out = out.loc[~bad_pressure].reset_index(drop=True)

    out[COL_DATE] = pd.to_datetime(out[COL_DATE], errors="coerce")
    out = out.dropna(subset=[COL_DATE]).reset_index(drop=True)

    in_season = _mask_may1_through_sept30(out[COL_DATE])
    out = out.loc[in_season].reset_index(drop=True)

    ceiled = out[COL_DATE].dt.ceil("h")
    out[COL_HOUR] = ceiled.dt.hour.astype(float)
    out[COL_DATE] = ceiled.dt.strftime("%Y-%m-%d %H:%M:%S")

    ordered = list(OUTPUT_COLUMNS) + [COL_HOUR]
    return out[ordered]


def process_weather_file(
    input_path: str,
    output_path: str | None = None,
) -> str:
    """Read CSV, filter columns, apply pressure/date/hour rules, write CSV (never overwrites input)."""
    df = read_csv(input_path)
    out_df = apply_weather_transforms(filter_weather_columns(df))

    if not output_path:
        output_path = make_default_output_path(input_path, suffix="_parsed")

    return write_csv(out_df, input_path=input_path, output_path=output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Weather init: four weather columns renamed to DATE, Air temp C, RH %, pressure"
    )
    parser.add_argument("--input", "-i", required=False, help="Path to input CSV")
    parser.add_argument("--output", "-o", required=False, help="Output CSV path")

    args = parser.parse_args()
    input_path = args.input
    if not input_path:
        try:
            input_path = input("Enter path to CSV file: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("No input provided. Exiting.", file=sys.stderr)
            sys.exit(2)
        if not input_path:
            print("No input provided. Exiting.", file=sys.stderr)
            sys.exit(2)

    try:
        out = process_weather_file(input_path, output_path=args.output)
        print(f"Weather file saved to: {out}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
