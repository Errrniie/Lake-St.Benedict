"""Save a PNG next to prediction CSVs: predicted DO vs row index and vs DATE."""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save_prediction_plot(
    output_csv_path: str,
    df: pd.DataFrame,
    pred_col: str,
    date_col: str = "DATE",
    title: str | None = None,
) -> str:
    """
    Write ``<stem>.png`` beside the CSV path.

    Two panels: (1) row index vs predicted DO, (2) DATE vs predicted DO.
    """
    if pred_col not in df.columns:
        raise ValueError(f"Column {pred_col!r} not in dataframe.")

    y = pd.to_numeric(df[pred_col], errors="coerce").to_numpy()
    n = len(df)
    idx = np.arange(n, dtype=float)

    stem, _ = os.path.splitext(output_csv_path)
    plot_path = f"{stem}.png"

    fig, axes = plt.subplots(2, 1, figsize=(10, 7))
    if title:
        fig.suptitle(title)

    axes[0].plot(idx, y, color="tab:blue", linewidth=0.8)
    axes[0].set_xlabel("Row index")
    axes[0].set_ylabel("Predicted DO")
    axes[0].set_title("Predicted DO vs index")
    axes[0].grid(True, alpha=0.3)

    if date_col in df.columns:
        dt = pd.to_datetime(df[date_col], errors="coerce")
        axes[1].plot(dt, y, color="tab:green", linewidth=0.8)
        axes[1].set_xlabel("DATE")
        axes[1].set_ylabel("Predicted DO")
        axes[1].set_title("Predicted DO vs time")
        axes[1].grid(True, alpha=0.3)
        plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=25, ha="right")
    else:
        axes[1].plot(idx, y, color="tab:green", linewidth=0.8)
        axes[1].set_xlabel("Row index")
        axes[1].set_ylabel("Predicted DO")
        axes[1].set_title("Predicted DO (no DATE column)")
        axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(plot_path, dpi=120)
    plt.close(fig)
    return plot_path
