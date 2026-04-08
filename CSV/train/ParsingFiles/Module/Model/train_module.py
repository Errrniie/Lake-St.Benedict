"""
Backward compatibility: ``train_model_on_csv`` is the median quantile trainer (same as ``train_mid_module``).
Prefer importing ``train_low_module``, ``train_mid_module``, or ``train_high_module`` explicitly.
"""
from __future__ import annotations

from Module.Model.train_mid_module import train_model_on_csv

__all__ = ["train_model_on_csv"]
