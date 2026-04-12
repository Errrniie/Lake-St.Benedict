#!/usr/bin/env python3
"""Per-calendar-day DO vs time-of-day with linear fit; CSV + figure."""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress

_HERE = os.path.dirname(os.path.abspath(__file__))
INPUT_CSV = os.path.join(_HERE, "DO_lakedata.csv")
OUT_CSV = os.path.join(_HERE, "daily_DO_linear_slopes.csv")
OUT_PNG = os.path.join(_HERE, "DO_vs_time_by_day_linear_fit.png")


def time_of_day_hours(ts: pd.Series) -> np.ndarray:
    """Hours since midnight (float)."""
    t = pd.to_datetime(ts, errors="coerce")
    return (
        t.dt.hour.astype(float)
        + t.dt.minute.astype(float) / 60.0
        + t.dt.second.astype(float) / 3600.0
    )


def main() -> None:
    df = pd.read_csv(INPUT_CSV)
    df.columns = df.columns.str.strip()
    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
    df = df.dropna(subset=["DATE"])
    df["DO"] = pd.to_numeric(df["DO"], errors="coerce")
    df["day"] = df["DATE"].dt.normalize()
    df["time_h"] = time_of_day_hours(df["DATE"])

    days = sorted(df["day"].unique())
    rows: list[dict] = []

    n_days = len(days)
    ncols = 3
    nrows = int(np.ceil(n_days / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3.2 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    for i, day in enumerate(days):
        ax = axes_flat[i]
        sub = df[df["day"] == day].sort_values("DATE")
        x = sub["time_h"].to_numpy()
        y = sub["DO"].to_numpy()
        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]
        n = len(x)

        if n < 2:
            slope = intercept = r2 = np.nan
            ax.text(0.5, 0.5, f"{day.date()}\n(n={n})", ha="center", va="center", transform=ax.transAxes)
        else:
            lr = linregress(x, y)
            slope, intercept, r2 = lr.slope, lr.intercept, lr.rvalue**2
            xx = np.linspace(x.min(), x.max(), 50)
            ax.plot(x, y, "o", color="tab:blue", markersize=3, alpha=0.7)
            ax.plot(xx, intercept + slope * xx, "-", color="tab:red", linewidth=1.2, label=f"slope={slope:.4f}")
            ax.legend(loc="best", fontsize=7)
        ax.set_xlabel("Time of day (h)")
        ax.set_ylabel("DO")
        ax.set_title(str(day.date()))
        ax.grid(True, alpha=0.3)

        rows.append(
            {
                "date": str(day.date()),
                "n_points": n,
                "slope_DO_per_hour": slope,
                "intercept": intercept,
                "r_squared": r2,
                "time_start_h": float(x.min()) if n else np.nan,
                "time_end_h": float(x.max()) if n else np.nan,
            }
        )

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle("DO vs time of day (aerator-on rows) with linear fit", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=130)
    plt.close(fig)

    slopes = [r["slope_DO_per_hour"] for r in rows if np.isfinite(r["slope_DO_per_hour"])]
    mean_slope = float(np.mean(slopes)) if slopes else float("nan")

    summary = {
        "date": "MEAN_ACROSS_DAYS",
        "n_points": int(sum(r["n_points"] for r in rows)),
        "slope_DO_per_hour": mean_slope,
        "intercept": np.nan,
        "r_squared": np.nan,
        "time_start_h": np.nan,
        "time_end_h": np.nan,
    }
    out_df = pd.DataFrame(rows + [summary])
    out_df.to_csv(OUT_CSV, index=False)

    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {OUT_PNG}")
    print(f"Mean slope (DO per hour) across {len(slopes)} days: {mean_slope:.6f}")


if __name__ == "__main__":
    main()
