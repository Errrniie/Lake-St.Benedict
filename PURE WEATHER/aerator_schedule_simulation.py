#!/usr/bin/env python3
"""
Simulate DO with aerator (+0.5/h when low) vs model deltas from prediction CSVs.

Uses:
  - *1hrdelta.csv: pred_DO_1h_quantile_mid (initial DO; optional trigger if < 3)
  - *1hr.csv: pred_DO_delta_1h_quantile_mid (hourly add when aerator off)

Outputs CSV + figure under PURE WEATHER/.
"""
from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_DELTA_CSV = os.path.join(
    _HERE,
    "ParsedWeatherData",
    "predictions",
    "Largest_Humidity_Summer_predictions_1hr.csv",
)
_DEFAULT_FUTURE_CSV = os.path.join(
    _HERE,
    "ParsedWeatherData",
    "predictions",
    "Largest_Humidity_Summer_predictions_1hrdelta.csv",
)


def simulate(
    df_delta: pd.DataFrame,
    df_future: pd.DataFrame,
    *,
    col_delta: str = "pred_DO_delta_1h_quantile_mid",
    col_future_1h: str = "pred_DO_1h_quantile_mid",
    threshold: float = 3.0,
    aerator_rate: float = 0.5,
    min_aerator_hours: int = 3,
    max_aerator_hours_before_force_continue: int = 5,
    use_future_below_threshold_trigger: bool = False,
) -> pd.DataFrame:
    """
    Hourly update:
    - If aerator on: S += aerator_rate; stop when S > threshold and elapsed >= min_aerator_hours.
    - If aerator off: S += delta[t]; if S < threshold (or optional: pred_1h[t] < threshold), start aerator next hour.

    Note: pred_1h from the model is often < 3 for every hour — enabling use_future_below_threshold_trigger
    tends to keep the aerator on almost always. Default is simulated S < 3 only.
    """
    n = len(df_delta)
    if len(df_future) != n:
        raise ValueError("Delta and future CSVs must have the same row count.")

    delta = pd.to_numeric(df_delta[col_delta], errors="coerce").fillna(0.0).to_numpy()
    pred_1h = pd.to_numeric(df_future[col_future_1h], errors="coerce").to_numpy()

    S = float(pred_1h[0])
    aerator = False
    elapsed = 0
    states: list[float] = []
    flags: list[bool] = []

    if S < threshold or (
        use_future_below_threshold_trigger and pred_1h[0] < threshold
    ):
        aerator = True
        elapsed = 0

    for t in range(n):
        if aerator:
            S += aerator_rate
            elapsed += 1
            stop_ok = S > threshold and elapsed >= min_aerator_hours
            if stop_ok:
                aerator = False
                elapsed = 0
        else:
            S += delta[t]
            if S < threshold or (
                use_future_below_threshold_trigger and pred_1h[t] < threshold
            ):
                aerator = True
                elapsed = 0

        states.append(S)
        flags.append(aerator)

    out = df_delta[["DATE"]].copy() if "DATE" in df_delta.columns else pd.DataFrame()
    out["simulated_DO"] = states
    out["aerator_on"] = flags
    out["pred_DO_delta_applied"] = np.where(
        np.array(flags), np.nan, delta
    )
    return out


def plot_result(df: pd.DataFrame, title: str, out_png: str) -> None:
    dt = pd.to_datetime(df["DATE"], errors="coerce")
    y = pd.to_numeric(df["simulated_DO"], errors="coerce").to_numpy()
    aer = df["aerator_on"].astype(bool).to_numpy()
    idx = np.arange(len(df))

    fig, axes = plt.subplots(2, 1, figsize=(11, 7))

    # Top: index, color by aerator
    ax = axes[0]
    if np.any(~aer):
        ax.plot(idx[~aer], y[~aer], color="tab:blue", linewidth=0.9, label="Aerator off (delta)")
    if np.any(aer):
        ax.plot(idx[aer], y[aer], color="tab:orange", linewidth=0.9, label="Aerator on (+0.5/h)")
    ax.axhline(3.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7, label="DO = 3")
    ax.set_xlabel("Row index")
    ax.set_ylabel("Simulated DO")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    if np.any(~aer):
        ax.plot(dt[~aer], y[~aer], color="tab:blue", linewidth=0.9, label="Aerator off")
    if np.any(aer):
        ax.plot(dt[aer], y[aer], color="tab:orange", linewidth=0.9, label="Aerator on")
    ax.axhline(3.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_xlabel("DATE")
    ax.set_ylabel("Simulated DO")
    ax.set_title("Simulated DO vs time (orange = aerator)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=22, ha="right")

    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Aerator + delta DO simulation (Pure Weather).")
    p.add_argument(
        "--delta-csv",
        default=_DEFAULT_DELTA_CSV,
        help="CSV with pred_DO_delta_1h_* (hourly delta when aerator off)",
    )
    p.add_argument(
        "--future-csv",
        default=_DEFAULT_FUTURE_CSV,
        help="CSV with pred_DO_1h_* (initial DO; trigger if < 3)",
    )
    p.add_argument(
        "--out-prefix",
        default=os.path.join(_HERE, "Largest_Humidity_Summer_aerator_sim"),
        help="Output prefix for .csv and .png (no extension)",
    )
    p.add_argument("--threshold", type=float, default=3.0)
    p.add_argument("--rate", type=float, default=0.5, help="DO gain per hour when aerator on")
    p.add_argument("--min-hours", type=int, default=3)
    p.add_argument(
        "--future-trigger",
        action="store_true",
        help="Also turn aerator on when pred_DO_1h (1hrdelta file) < threshold (often too aggressive)",
    )
    args = p.parse_args()

    df_d = pd.read_csv(args.delta_csv)
    df_f = pd.read_csv(args.future_csv)
    df_d.columns = df_d.columns.str.strip()
    df_f.columns = df_f.columns.str.strip()

    col_d = "pred_DO_delta_1h_quantile_mid"
    col_f = "pred_DO_1h_quantile_mid"
    if col_d not in df_d.columns:
        raise SystemExit(f"Missing {col_d} in {args.delta_csv}")
    if col_f not in df_f.columns:
        raise SystemExit(f"Missing {col_f} in {args.future_csv}")

    sim = simulate(
        df_d,
        df_f,
        col_delta=col_d,
        col_future_1h=col_f,
        threshold=args.threshold,
        aerator_rate=args.rate,
        min_aerator_hours=args.min_hours,
        use_future_below_threshold_trigger=args.future_trigger,
    )

    out_csv = f"{args.out_prefix}.csv"
    out_png = f"{args.out_prefix}.png"
    sim.to_csv(out_csv, index=False)
    plot_result(
        sim,
        os.path.basename(args.delta_csv) + " + aerator schedule",
        out_png,
    )
    print(f"Wrote {out_csv}")
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
