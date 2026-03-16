#!/usr/bin/env python3
"""Split a CSV file into 80% train and 20% test portions.

The script reads a CSV path from standard input, loads it with pandas,
creates two new files in the current working directory named
`train_data.csv` and `test_data.csv`, and reports the row counts.
"""

import os
import sys

import pandas as pd


def split_csv(path: str) -> None:
    """Read `path`, split with top 20% as test and last 80% as train."""
    df = pd.read_csv(path)
    n = len(df)
    if n == 0:
        print("WARNING: input CSV is empty", file=sys.stderr)
    cut = int(n * 0.2)
    test = df.iloc[:cut]
    train = df.iloc[cut:]

    test.to_csv("test_data.csv", index=False)
    train.to_csv("train_data.csv", index=False)

    print(f"read {n} rows from {path}")
    print(f"wrote {len(test)} rows to test_data.csv (top 20%)")
    print(f"wrote {len(train)} rows to train_data.csv (last 80%)")


if __name__ == "__main__":
    csv_path = input("Enter path to CSV file: ").strip()
    if not os.path.isfile(csv_path):
        print(f"ERROR: file not found: {csv_path}", file=sys.stderr)
        sys.exit(1)
    split_csv(csv_path)
