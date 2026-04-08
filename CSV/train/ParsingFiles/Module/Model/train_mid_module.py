"""Quantile regression training — mid band (median)."""
from __future__ import annotations

from Module.Model.train_quantile_core import train_quantile_model_on_csv

# Median (50th percentile).
QUANTILE_ALPHA = 0.5
QUANTILE_LABEL = "mid"


def train_model_on_csv(
    csv_path: str,
    target_col: str,
    model_output_path: str | None = None,
    include_other_delta_features: bool = True,
) -> dict:
    return train_quantile_model_on_csv(
        csv_path=csv_path,
        target_col=target_col,
        quantile_alpha=QUANTILE_ALPHA,
        quantile_label=QUANTILE_LABEL,
        model_output_path=model_output_path,
        include_other_delta_features=include_other_delta_features,
    )
