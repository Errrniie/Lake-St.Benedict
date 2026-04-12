#!/usr/bin/env python3
"""
Simulate DO with aerator vs model deltas from prediction CSVs.

Delta handling (1hr file): positive pred deltas are halved so the run trends more negative;
negative deltas unchanged.

When aerator is on: each hour S += delta_effective + aerator_rate (default 0.286), so
natural ΔDO still modulates the rise. When off: S += delta_effective only.

Uses:
  - *1hrdelta.csv: pred_DO_1h_quantile_mid (initial DO; optional trigger if < 3)
  - *1hr.csv: pred_DO_delta_1h_quantile_mid

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


def _effective_delta(raw: np.ndarray, *, halve_positive: bool) -> np.ndarray:
    d = np.asarray(raw, dtype=float)
    if not halve_positive:
        return d
    out = d.copy()
    pos = out > 0
    out[pos] = out[pos] * 0.5
    return out


def simulate(
    df_delta: pd.DataFrame,
    df_future: pd.DataFrame,
    *,
    col_delta: str = "pred_DO_delta_1h_quantile_mid",
    col_future_1h: str = "pred_DO_1h_quantile_mid",
    threshold: float = 3.0,
    aerator_rate: float = 0.286,
    min_aerator_hours: int = 3,
    halve_positive_deltas: bool = True,
    use_future_below_threshold_trigger: bool = False,
) -> pd.DataFrame:
    """
    Hourly update (delta_effective = half positive deltas if enabled):
    - If aerator on: S += delta_effective[t] + aerator_rate; stop when S > threshold and elapsed >= min_aerator_hours.
    - If aerator off: S += delta_effective[t]; if S < threshold (or optional pred_1h), start aerator next hour.

    Note: pred_1h from the model is often < 3 for every hour — enabling use_future_below_threshold_trigger
    tends to keep the aerator on almost always. Default is simulated S < 3 only.
    """
    n = len(df_delta)
    if len(df_future) != n:
        raise ValueError("Delta and future CSVs must have the same row count.")

    delta_raw = pd.to_numeric(df_delta[col_delta], errors="coerce").fillna(0.0).to_numpy()
    delta_eff = _effective_delta(delta_raw, halve_positive=halve_positive_deltas)
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
            S += delta_eff[t] + aerator_rate
            elapsed += 1
            stop_ok = S > threshold and elapsed >= min_aerator_hours
            if stop_ok:
                aerator = False
                elapsed = 0
        else:
            S += delta_eff[t]
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
    out["pred_DO_delta_raw"] = delta_raw
    out["pred_DO_delta_effective"] = delta_eff
    out["aerator_boost_h"] = np.where(np.array(flags, dtype=bool), aerator_rate, 0.0)
    out["pred_DO_delta_applied"] = np.where(
        np.array(flags, dtype=bool), np.nan, delta_eff
    )
    return out


def _contiguous_true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Half-open [start, end) index ranges where mask is True consecutively."""
    mask = np.asarray(mask, dtype=bool)
    runs: list[tuple[int, int]] = []
    i, n = 0, len(mask)
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i + 1
        while j < n and mask[j]:
            j += 1
        runs.append((i, j))
        i = j
    return runs


def _plot_segments(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    *,
    color: str,
    label: str,
    linewidth: float = 1.15,
    zorder: int = 2,
) -> None:
    """Plot only within contiguous True runs so lines do not bridge gaps."""
    for k, (s, e) in enumerate(_contiguous_true_runs(mask)):
        lbl = label if k == 0 else None
        ax.plot(
            x[s:e],
            y[s:e],
            color=color,
            linewidth=linewidth,
            label=lbl,
            zorder=zorder,
            solid_capstyle="round",
            solid_joinstyle="round",
        )


def plot_result(
    df: pd.DataFrame,
    title: str,
    out_png: str,
    *,
    aerator_rate: float = 0.286,
) -> None:
    dt = pd.to_datetime(df["DATE"], errors="coerce")
    y = pd.to_numeric(df["simulated_DO"], errors="coerce").to_numpy()
    aer = df["aerator_on"].astype(bool).to_numpy()
    idx = np.arange(len(df), dtype=float)

    # Aerator: deep coral (not yellow/orange tab); reads clearly vs blue
    color_aer = "#C73E1D"
    color_off = "#1f77b4"

    fig, axes = plt.subplots(2, 1, figsize=(11, 7))

    ax = axes[0]
    _plot_segments(
        ax,
        idx,
        y,
        ~aer,
        color=color_off,
        label="Aerator off (ΔDO eff.)",
        zorder=1,
    )
    _plot_segments(
        ax,
        idx,
        y,
        aer,
        color=color_aer,
        label=f"Aerator on (ΔDO eff. + {aerator_rate:g}/h)",
        linewidth=1.35,
        zorder=3,
    )
    ax.axhline(3.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.65, label="DO = 3", zorder=0)
    ax.set_xlabel("Row index")
    ax.set_ylabel("Simulated DO")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.28)

    ax = axes[1]
    xdt = dt.to_numpy()
    _plot_segments(ax, xdt, y, ~aer, color=color_off, label="Aerator off", zorder=1)
    _plot_segments(
        ax,
        xdt,
        y,
        aer,
        color=color_aer,
        label="Aerator on",
        linewidth=1.35,
        zorder=3,
    )
    ax.axhline(3.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.65, zorder=0)
    ax.set_xlabel("DATE")
    ax.set_ylabel("Simulated DO")
    ax.set_title("Simulated DO vs time (red = aerator on)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.28)
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
    p.add_argument(
        "--rate",
        type=float,
        default=0.286,
        help="Aerator DO addition per hour (added on top of effective ΔDO when aerator on)",
    )
    p.add_argument("--min-hours", type=int, default=3)
    p.add_argument(
        "--no-halve-positive",
        action="store_true",
        help="Use raw predicted deltas (do not halve positive ΔDO values)",
    )
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
        halve_positive_deltas=not args.no_halve_positive,
        use_future_below_threshold_trigger=args.future_trigger,
    )

    out_csv = f"{args.out_prefix}.csv"
    out_png = f"{args.out_prefix}.png"
    sim.to_csv(out_csv, index=False)
    plot_result(
        sim,
        os.path.basename(args.delta_csv) + " + aerator schedule",
        out_png,
        aerator_rate=args.rate,
    )
    print(f"Wrote {out_csv}")
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
