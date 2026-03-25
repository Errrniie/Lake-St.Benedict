from __future__ import annotations

import numpy as np
import pandas as pd


DEFAULT_HORIZONS = (1, 3, 6, 12, 24)


def add_do_deltas(
    df: pd.DataFrame,
    date_col: str,
    do_col: str,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
) -> pd.DataFrame:
    out = df.copy()
    out = out.dropna(subset=[date_col]).sort_values(date_col)
    do_num = pd.to_numeric(out[do_col], errors="coerce")
    n = len(out)

    right = pd.DataFrame({"_rt": out[date_col].to_numpy(), "_past_do": do_num.to_numpy()}).sort_values(
        "_rt"
    )

    for hours in horizons:
        col_name = f"DO_delta_{hours}h"
        left = pd.DataFrame(
            {
                "_ord": np.arange(n, dtype=np.int64),
                "_t": out[date_col].to_numpy(),
                "_current_do": do_num.to_numpy(),
            }
        )
        left["_key"] = left["_t"] - pd.Timedelta(hours=hours)
        merged = pd.merge_asof(
            left.sort_values("_key"),
            right,
            left_on="_key",
            right_on="_rt",
            direction="backward",
        ).sort_values("_ord")

        past = merged["_past_do"].to_numpy()
        cur = merged["_current_do"].to_numpy()
        valid = ~np.isnan(past)
        delta = np.where(valid, cur - past, np.nan)
        out[col_name] = np.round(delta, 10)

    return out


def apply_nan_string(
    df: pd.DataFrame,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    na_string: str = "NaN",
) -> pd.DataFrame:
    out = df.copy()
    for col in [f"DO_delta_{h}h" for h in horizons]:
        if col in out.columns:
            out[col] = out[col].apply(lambda v: na_string if pd.isna(v) else v)
    return out

