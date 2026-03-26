#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys

from Module.Connector import run_connected_modules, run_model_prediction, run_model_training
from Module.DORate.dodeltas_module import DEFAULT_HORIZONS
from Module.Parse.parse_module import make_default_output_path, read_csv, write_csv
from gui_app import run_gui


def run_controller(
    input_path: str,
    output_path: str | None = None,
    date_col: str | None = None,
    do_col: str | None = None,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    na_string: str = "NaN",
) -> str:
    df = read_csv(input_path)
    out_df, _date_col, _do_col = run_connected_modules(
        df,
        date_col_name=date_col,
        do_col_name=do_col,
        horizons=horizons,
        na_string=na_string,
    )

    if not output_path:
        output_path = make_default_output_path(input_path, suffix="_parsed_DOdeltas")
    return write_csv(out_df, input_path=input_path, output_path=output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Core controller: parse, clean, lag features, and DO deltas in one run."
    )
    parser.add_argument(
        "--cli",
        action="store_true",
        help="Run command-line mode instead of opening GUI",
    )
    parser.add_argument("--input", "-i", required=False, help="Path to input CSV (prompt if omitted)")
    parser.add_argument("--output", "-o", required=False, help="Output CSV path")
    parser.add_argument("--date-col", "-c", required=False, help="Date column name (default: DATE/first)")
    parser.add_argument("--do-col", "-d", required=False, help="DO column name (default: auto-detect)")
    parser.add_argument(
        "--horizons",
        required=False,
        help="Comma-separated hours for DO deltas, e.g. 1,3,6,12,24",
    )
    parser.add_argument(
        "--na-string",
        default="NaN",
        help='String written for missing DO deltas (default: "NaN")',
    )

    args = parser.parse_args()

    # Default behavior: open GUI menu unless CLI mode is explicitly requested.
    if not args.cli and not args.input:
        try:
            rc = run_gui(
                run_controller_fn=run_controller,
                default_output_fn=lambda p: make_default_output_path(p, suffix="_parsed_DOdeltas"),
                run_model_training_fn=run_model_training,
                run_model_prediction_fn=run_model_prediction,
            )
            if rc:
                sys.exit(rc)
            return
        except ModuleNotFoundError as exc:
            if "PySide6" in str(exc):
                print(
                    "PySide6 is not installed. Install it with:\n"
                    ".venv/bin/pip install PySide6\n"
                    "Or run in CLI mode with: --cli",
                    file=sys.stderr,
                )
                sys.exit(1)
            raise

    input_path = args.input
    if not input_path:
        try:
            input_path = input("Enter path to CSV file: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("No input provided. Exiting.", file=sys.stderr)
            sys.exit(2)
        if not input_path:
            print("No input provided. Exiting.", file=sys.stderr)
            sys.exit(2)

    horizons = DEFAULT_HORIZONS
    if args.horizons:
        horizons = tuple(int(x.strip()) for x in args.horizons.split(",") if x.strip())

    try:
        out = run_controller(
            input_path=input_path,
            output_path=args.output,
            date_col=args.date_col,
            do_col=args.do_col,
            horizons=horizons,
            na_string=args.na_string,
        )
        print(f"Processed file saved to: {out}")
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

