from __future__ import annotations

import os
import pandas as pd


def read_csv(input_path: str) -> pd.DataFrame:
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    df = pd.read_csv(input_path)
    df.columns = df.columns.str.strip()
    return df


def resolve_date_column(df: pd.DataFrame, date_col_name: str | None = None) -> str:
    if date_col_name and date_col_name in df.columns:
        return date_col_name
    if "DATE" in df.columns:
        return "DATE"
    return df.columns[0]


def parse_date_column(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    return out


def find_do_column(df: pd.DataFrame, do_col_name: str | None = None) -> str | None:
    if do_col_name and do_col_name in df.columns:
        return do_col_name
    if "DO" in df.columns:
        return "DO"
    for col in df.columns:
        if "do" in col.lower():
            return col
    return None


def make_default_output_path(input_path: str, suffix: str = "_parsed_DOdeltas") -> str:
    base, ext = os.path.splitext(input_path)
    return f"{base}{suffix}{ext}"


def write_csv(df: pd.DataFrame, input_path: str, output_path: str) -> str:
    if os.path.abspath(output_path) == os.path.abspath(input_path):
        raise ValueError(
            f"Refuse to overwrite input file. Choose a different output path. Input: {input_path}"
        )
    df.to_csv(output_path, index=False)
    return output_path

