#!/usr/bin/env python3
"""
Build DO_lakedata_parsed_aerator_12h.csv: copy of PURE WEATHER/DO_lakedata_parsed.csv
keeping rows within ±12 hours of each aerator on/off transition (from DO_lakedata_parsed.csv).
Adds Aerated and Aerator_TLAG_* from the full parsed lake file.

Strips all DO_delta_* columns and keeps only DO_1h among future-DO columns.
"""
from __future__ import annotations

import os
import re

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)


def _build() -> str:
    path_pw = os.path.join(_HERE, "DO_lakedata_parsed.csv")
    path_full = os.path.join(_REPO, "DO_lakedata_parsed.csv")
    out_path = os.path.join(_HERE, "DO_lakedata_parsed_aerator_12h.csv")

    df_pw = pd.read_csv(path_pw)
    df_full = pd.read_csv(path_full)
    df_pw.columns = df_pw.columns.str.strip()
    df_full.columns = df_full.columns.str.strip()

    df_pw["DATE"] = pd.to_datetime(df_pw["DATE"], errors="coerce")
    df_full["DATE"] = pd.to_datetime(df_full["DATE"], errors="coerce")

    aer = pd.to_numeric(df_full["Aerated (1/0)"], errors="coerce").fillna(0)
    prev = aer.shift(1)
    transition_mask = (aer != prev) & prev.notna()
    transition_times = df_full.loc[transition_mask, "DATE"].tolist()

    keep = np.zeros(len(df_full), dtype=bool)
    for t in transition_times:
        t = pd.Timestamp(t)
        keep |= (df_full["DATE"] >= t - pd.Timedelta(hours=12)) & (
            df_full["DATE"] <= t + pd.Timedelta(hours=12)
        )

    df_keep = df_full.loc[keep, ["DATE"]].drop_duplicates()

    extra_cols = [
        c
        for c in ["Aerated (1/0)", "Aerator_TLAG_15", "Aerator_TLAG_30", "Aerator_TLAG_60"]
        if c in df_full.columns
    ]
    aer_part = df_full[["DATE"] + extra_cols].drop_duplicates(subset=["DATE"])

    out = df_pw.merge(df_keep, on="DATE", how="inner")
    out = out.merge(aer_part, on="DATE", how="left")

    cols = list(out.columns)
    for c in extra_cols:
        if c in cols:
            cols.remove(c)
    insert_at = 2 if "DO" in cols else 1
    for i, c in enumerate(extra_cols):
        if c in out.columns:
            cols.insert(insert_at + i, c)
    out = out[cols]

    drop_do_extra = [c for c in out.columns if str(c).startswith("DO_delta_")]
    drop_do_extra += [
        c
        for c in out.columns
        if re.fullmatch(r"DO_\d+h", str(c)) and c != "DO_1h"
    ]
    out = out.drop(columns=drop_do_extra, errors="ignore")

    out.to_csv(out_path, index=False)
    return out_path


if __name__ == "__main__":
    p = _build()
    print(f"Wrote {p}")
