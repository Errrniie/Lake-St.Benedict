#!/usr/bin/env python3
"""
Regenerate prediction CSVs/PNGs (1hr delta + 1h future DO) for each ParsedWeatherData
weather table, then run aerator_schedule_simulation for each — same workflow as
Largest Humidity Summer.

Models (PURE WEATHER/Models): model_DO_delta_1h_quantile_mid.pkl, model_DO_1h_quantile_mid.pkl
"""
from __future__ import annotations

import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
_PARSING = os.path.join(_REPO, "CSV", "train", "ParsingFiles")
_PW = _HERE
_PD = os.path.join(_PW, "ParsedWeatherData")
_PR = os.path.join(_PD, "predictions")
_M = os.path.join(_PW, "Models")
_VENV_PY = os.path.join(_REPO, ".venv", "bin", "python")

SCENARIOS = [
    ("Largest_Humidity_Summer", "Largest_Humidity_Summer_aerator_sim"),
    ("Largest_Temp_Summer", "Largest_Temp_Summer_aerator_sim"),
    ("Average_Year", "Average_Year_aerator_sim"),
]


def main() -> None:
    sys.path.insert(0, _PARSING)
    from Module.Model.predict_module import predict_from_model_bundle

    os.makedirs(_PR, exist_ok=True)

    for stem, out_prefix in SCENARIOS:
        src = os.path.join(_PD, f"{stem}.csv")
        if not os.path.isfile(src):
            print(f"Skip (missing): {src}", file=sys.stderr)
            continue
        p1 = predict_from_model_bundle(
            os.path.join(_M, "model_DO_delta_1h_quantile_mid.pkl"),
            src,
            os.path.join(_PR, f"{stem}_predictions_1hr.csv"),
        )
        p2 = predict_from_model_bundle(
            os.path.join(_M, "model_DO_1h_quantile_mid.pkl"),
            src,
            os.path.join(_PR, f"{stem}_predictions_1hrdelta.csv"),
        )

        sim_py = os.path.join(_PW, "aerator_schedule_simulation.py")
        out_base = os.path.join(_PW, out_prefix)
        subprocess.run(
            [
                _VENV_PY,
                sim_py,
                "--delta-csv",
                os.path.join(_PR, f"{stem}_predictions_1hr.csv"),
                "--future-csv",
                os.path.join(_PR, f"{stem}_predictions_1hrdelta.csv"),
                "--out-prefix",
                out_base,
            ],
            check=True,
        )
        print(f"OK {out_base}.csv / .png")


if __name__ == "__main__":
    main()
