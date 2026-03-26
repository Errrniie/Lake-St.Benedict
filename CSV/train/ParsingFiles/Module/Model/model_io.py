from __future__ import annotations

import os
from typing import Any

import joblib


def ensure_models_dir(base_dir: str | None = None) -> str:
    if base_dir:
        path = base_dir
    else:
        # Default centralized model location in this module tree.
        path = os.path.join(os.path.dirname(__file__), "Models")
    os.makedirs(path, exist_ok=True)
    return path


def save_bundle(bundle: dict[str, Any], path: str) -> str:
    if not path.lower().endswith(".pkl"):
        path = f"{path}.pkl"
    joblib.dump(bundle, path)
    return path


def load_bundle(path: str) -> dict[str, Any]:
    obj = joblib.load(path)
    if not isinstance(obj, dict):
        raise ValueError("Model file does not contain a valid model bundle.")
    return obj

