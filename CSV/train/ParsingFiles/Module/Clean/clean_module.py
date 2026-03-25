from __future__ import annotations

import pandas as pd


def fill_hour_column(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    out = df.copy()
    computed_hour = out[date_col].dt.hour
    if "hour" in out.columns:
        mask_missing = out["hour"].isna() | (out["hour"].astype(str).str.strip() == "")
        out.loc[mask_missing, "hour"] = computed_hour.loc[mask_missing]
    else:
        out["hour"] = computed_hour
    return out


def drop_zero_do_rows(df: pd.DataFrame, do_col: str | None) -> pd.DataFrame:
    if not do_col:
        return df
    out = df.copy()
    coerced = pd.to_numeric(out[do_col], errors="coerce")
    zero_mask = (coerced == 0) | (out[do_col].astype(str).str.strip() == "0")
    return out[~zero_mask.fillna(False)]

