#!/usr/bin/env python3
"""Remove date column and drop rows with NaN in DO column.

This helper script reads a CSV file from a path entered by the user.  It
assumes the first column is a date/year column and the second column
contains dissolved oxygen (DO) values.  The processing steps are:

* drop the first column entirely (so it will no longer appear in output)
* for the DO column (now the first column in memory) drop any row whose
  value is exactly the string "NaN"

The resulting rows are written to a new CSV file alongside the input
file, with ``_cleaned`` appended before the file extension.

Example::

    $ python Drop_NaNData.py
    Enter path to input csv: DO_lakedata.csv
    output written to: DO_lakedata_cleaned.csv
"""

import csv
import os
import sys


def process_file(input_path: str) -> str:
    """Read `input_path` and write cleaned data to new path.

    Returns the path of the cleaned output file.
    """

    base, ext = os.path.splitext(input_path)
    if not ext:
        ext = ".csv"
    output_path = f"{base}_cleaned{ext}"

    with open(input_path, newline="", encoding="utf-8") as infile, \
         open(output_path, "w", newline="", encoding="utf-8") as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        for row in reader:
            # require at least two columns for date and DO
            if len(row) < 2:
                continue
            # drop the date/year (first) column completely
            do_value = row[1]
            if do_value == "NaN":
                # skip rows with NaN in DO column
                continue
            # write row containing only DO (and any subsequent cols)
            writer.writerow(row[1:])

    return output_path


def main() -> None:
    try:
        path = input("Enter path to input csv: ").strip()
        if not path:
            print("No path provided, exiting.")
            sys.exit(1)
        if not os.path.isfile(path):
            print(f"File does not exist: {path}")
            sys.exit(1)

        out = process_file(path)
        print(f"output written to: {out}")
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(1)


if __name__ == "__main__":
    main()
