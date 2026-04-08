from __future__ import annotations

import re

import numpy as np
import pandas as pd


DEFAULT_HORIZONS = (1, 3, 6, 12, 24)


def delta_column(hours: int) -> str:
    return f"DO_delta_{hours}h"


def future_do_column(hours: int) -> str:
    """Absolute DO at the first observation at or after (t + hours)."""
    return f"DO_{hours}h"


def add_do_deltas(
    df: pd.DataFrame,
    date_col: str,
    do_col: str,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
) -> pd.DataFrame:
    """
    Forward-looking DO change and future DO levels.

    For each row at time t, we take the first observation at or after t+h (merge_asof,
    direction=\"forward\"). Then:

    - ``DO_delta_{h}h`` = future_DO - DO(t)   (change over the next h hours)
    - ``DO_{h}h``       = future_DO           (absolute DO at that future sample)

    Rows with no future observation at or after t+h get NaN for both columns.
    """
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out = out.dropna(subset=[date_col]).sort_values(date_col)
    do_num = pd.to_numeric(out[do_col], errors="coerce")
    n = len(out)

    right = pd.DataFrame({"_rt": out[date_col].to_numpy(), "_do": do_num.to_numpy()}).sort_values(
        "_rt"
    )

    for hours in horizons:
        col_delta = delta_column(hours)
        col_future = future_do_column(hours)
        left = pd.DataFrame(
            {
                "_ord": np.arange(n, dtype=np.int64),
                "_t": out[date_col].to_numpy(),
                "_current_do": do_num.to_numpy(),
            }
        )
        left["_key"] = left["_t"] + pd.Timedelta(hours=hours)
        merged = pd.merge_asof(
            left.sort_values("_key"),
            right,
            left_on="_key",
            right_on="_rt",
            direction="forward",
        ).sort_values("_ord")

        future = merged["_do"].to_numpy()
        cur = merged["_current_do"].to_numpy()
        valid = ~np.isnan(future) & ~np.isnan(cur)
        delta = np.where(valid, np.round(future - cur, 10), np.nan)
        out[col_delta] = delta
        out[col_future] = np.where(~np.isnan(future), np.round(future, 10), np.nan)

    return out


def apply_nan_string(
    df: pd.DataFrame,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    na_string: str = "NaN",
) -> pd.DataFrame:
    # Keep real NaN values for ML-ready numeric columns.
    # `na_string` is intentionally ignored for this modular pipeline path.
    out = df.copy()
    for col in [delta_column(h) for h in horizons] + [future_do_column(h) for h in horizons]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def is_future_do_column(name: str) -> bool:
    """True for columns like ``DO_1h``, ``DO_24h`` (future DO levels)."""
    return re.fullmatch(r"DO_\d+h", str(name).strip()) is not None
