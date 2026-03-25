from __future__ import annotations

import pandas as pd


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


def add_aerator_cumulative_lag(
    df: pd.DataFrame,
    date_col: str,
    hours: int,
    aerated_col: str,
    new_col: str,
) -> pd.DataFrame:
    out = df.sort_values(date_col).copy()
    median_interval = out[date_col].diff().median() / pd.Timedelta(minutes=1)
    series = out.set_index(date_col)[aerated_col].rolling(f"{hours}h").sum()
    out[new_col] = series.values * median_interval
    return out


def apply_lag_features(df: pd.DataFrame, date_col: str, do_col: str | None) -> pd.DataFrame:
    out = df.sort_values(date_col).copy()
    out = out.dropna(subset=[date_col])

    if not (do_col and "Air temp C" in out.columns):
        return out

    out = add_time_lag(out, date_col, 1, do_col, "DO_TLAG_15")
    out = add_time_lag(out, date_col, 3, do_col, "DO_TLAG_30")
    out = add_time_lag(out, date_col, 6, do_col, "DO_TLAG_60")
    out = add_time_lag(out, date_col, 1, "Air temp C", "AT_TLAG_15")
    out = add_time_lag(out, date_col, 3, "Air temp C", "AT_TLAG_30")
    out = add_time_lag(out, date_col, 6, "Air temp C", "AT_TLAG_60")

    aerated_col = None
    for col in out.columns:
        if "aerat" in col.lower():
            aerated_col = col
            break

    if aerated_col:
        out = add_aerator_cumulative_lag(out, date_col, 1, aerated_col, "Aerator_TLAG_15")
        out = add_aerator_cumulative_lag(out, date_col, 3, aerated_col, "Aerator_TLAG_30")
        out = add_aerator_cumulative_lag(out, date_col, 6, aerated_col, "Aerator_TLAG_60")

    return out


def drop_rows_missing_lags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    lag_prefixes = ("DO_TLAG_", "AT_TLAG_", "Aerator_TLAG_")
    lag_cols = [col for col in out.columns if col.startswith(lag_prefixes)]

    if lag_cols:
        mask = pd.DataFrame(False, index=out.index, columns=lag_cols)
        for col in lag_cols:
            mask[col] = out[col].isna()
        out = out[~mask.any(axis=1)]

    if "DO_TLAG_15" in out.columns:
        out = out[~out["DO_TLAG_15"].isna()]

    return out

