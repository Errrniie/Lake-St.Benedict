#!/usr/bin/env python3
"""
InitSetup.py

Unified data processing pipeline (updated behavior):
- Parse the DATE (or specified) column into datetimes
- Add or fill `hour`, `minute`, `day_of_month`, `day_of_year`, `month`, `year` columns with actual values
- Drop any rows where the DO column is 0 (instead of converting to the string "NaN")
- Create time-lagged columns for DO, Air temp, and Aerator (1h, 3h, 6h)

Saves a new processed CSV (does not overwrite input by default), using real NaNs
internally rather than the literal string 'NaN'.
"""
import argparse
import os
import sys
import pandas as pd


def make_default_output_path(input_path):
    base, ext = os.path.splitext(input_path)
    return f"{base}_parsed{ext}"


def find_do_column(df, do_col_name=None):
    if do_col_name and do_col_name in df.columns:
        return do_col_name
    if 'DO' in df.columns:
        return 'DO'
    # case-insensitive search for column containing 'do'
    for c in df.columns:
        if 'do' in c.lower():
            return c
    return None


def add_time_lag(df, date_col, hours, source_col, new_col):
    """Add a time-lagged column using merge_asof on datetime."""
    lag_df = df[[date_col, source_col]].copy()
    lag_df[date_col] = lag_df[date_col] + pd.Timedelta(hours=hours)
    lag_df = lag_df.rename(columns={source_col: new_col})
    
    return pd.merge_asof(
        df,
        lag_df,
        on=date_col,
        direction="backward"
    )


def add_aerator_cumulative_lag(df, date_col, hours, aerated_col, new_col):
    """
    Calculate minutes aerator was ON during the past `hours` interval using a
    rolling time window.  Regardless of the data spacing, the result for each
    timestamp is the summed minutes where `aerated_col` equaled 1 within the
    preceding period of length `hours`.
    """
    # sort by date and compute approximate interval length
    df = df.sort_values(date_col)
    median_interval = (df[date_col].diff().median() / pd.Timedelta(minutes=1))

    # create a time-indexed series and apply a rolling window
    series = df.set_index(date_col)[aerated_col].rolling(f"{hours}h").sum()
    # multiply count of "on" rows by minutes per row to get minutes
    df[new_col] = series.values * median_interval
    return df


def process_file(input_path, output_path=None, date_col_name=None, do_col_name=None):
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)

    # Clean column names (strip whitespace)
    df.columns = df.columns.str.strip()

    # Determine which column to parse as date
    if date_col_name and date_col_name in df.columns:
        date_col = date_col_name
    elif 'DATE' in df.columns:
        date_col = 'DATE'
    else:
        date_col = df.columns[0]

    # Parse the date column to datetime (invalid parsing -> NaT)
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')

    # Compute datetime-derived components from the parsed date
    computed = pd.DataFrame(index=df.index)
    computed['hour'] = df[date_col].dt.hour
    computed['minute'] = df[date_col].dt.minute
    computed['day_of_month'] = df[date_col].dt.day
    computed['day_of_year'] = df[date_col].dt.dayofyear
    computed['month'] = df[date_col].dt.month
    computed['year'] = df[date_col].dt.year

    # For each of these fields, fill missing or blank values using the computed values.
    for col in computed.columns:
        if col in df.columns:
            mask_missing = df[col].isna() | (df[col].astype(str).str.strip() == '')
            df.loc[mask_missing, col] = computed.loc[mask_missing, col]
        else:
            df[col] = computed[col]

    # Drop rows where DO is exactly zero (numeric 0 or string "0")
    do_col = find_do_column(df, do_col_name)
    if do_col:
        coerced = pd.to_numeric(df[do_col], errors='coerce')
        zero_mask = coerced == 0
        zero_mask = zero_mask | (df[do_col].astype(str).str.strip() == '0')
        # keep only rows where DO is *not* zero
        df = df[~zero_mask.fillna(False)]

    # Sort by date for lag operations
    df = df.sort_values(date_col)

    # Drop rows with NaT in date_col before creating lags (merge_asof requires no nulls)
    df_for_lags = df.dropna(subset=[date_col])

    # Create lag columns if DO and Air temp columns exist
    if do_col and 'Air temp C' in df_for_lags.columns:
        df_for_lags = add_time_lag(df_for_lags, date_col, 1, do_col, "DO_T_1")
        df_for_lags = add_time_lag(df_for_lags, date_col, 3, do_col, "DO_T_3")
        df_for_lags = add_time_lag(df_for_lags, date_col, 6, do_col, "DO_T_6")
        df_for_lags = add_time_lag(df_for_lags, date_col, 1, 'Air temp C', "AT_T_1")
        df_for_lags = add_time_lag(df_for_lags, date_col, 3, 'Air temp C', "AT_T_3")
        df_for_lags = add_time_lag(df_for_lags, date_col, 6, 'Air temp C', "AT_T_6")
        
        # Add aerator cumulative lags (minutes on) if Aerated column exists
        aerated_col = None
        for c in df_for_lags.columns:
            if 'aerat' in c.lower():
                aerated_col = c
                break
        if aerated_col:
            df_for_lags = add_aerator_cumulative_lag(df_for_lags, date_col, 1, aerated_col, "Aerator_T_1")
            df_for_lags = add_aerator_cumulative_lag(df_for_lags, date_col, 3, aerated_col, "Aerator_T_3")
            df_for_lags = add_aerator_cumulative_lag(df_for_lags, date_col, 6, aerated_col, "Aerator_T_6")
        
        # Update main df with lag columns
        df = df_for_lags

    # after lags exist, remove rows where any of the lag columns contain NaN
    lag_prefixes = ("DO_T_", "AT_T_", "Aerator_T_")
    lag_cols = [c for c in df.columns if c.startswith(lag_prefixes)]
    if lag_cols:
        mask = pd.DataFrame(False, index=df.index, columns=lag_cols)
        for col in lag_cols:
            mask[col] = df[col].isna()
        df = df[~mask.any(axis=1)]

    # explicit requirement: drop rows where the 1‑hour DO lag is missing/NaN
    if 'DO_T_1' in df.columns:
        df = df[~df['DO_T_1'].isna()]

    # Determine output path
    if not output_path:
        output_path = make_default_output_path(input_path)

    # Avoid overwriting input file
    if os.path.abspath(output_path) == os.path.abspath(input_path):
        output_path = make_default_output_path(input_path)

    # Save processed CSV; keep real NaNs in the file
    df.to_csv(output_path, index=False)
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description='Init setup: parse date, fill datetime fields, clean DO zeros, and create lag columns'
    )
    parser.add_argument('--input', '-i', required=False, help='Path to input CSV (will prompt if omitted)')
    parser.add_argument('--output', '-o', required=False, help='Desired output CSV path')
    parser.add_argument('--date-col', '-c', required=False, help='Name of the date column to parse (default: DATE or first column)')
    parser.add_argument('--do-col', '-d', required=False, help='Name of the DO column to clean (default: DO or first col containing "do")')

    args = parser.parse_args()

    input_path = args.input
    if not input_path:
        try:
            input_path = input('Enter path to CSV file: ').strip()
        except (EOFError, KeyboardInterrupt):
            print('No input provided. Exiting.', file=sys.stderr)
            sys.exit(2)

        if not input_path:
            print('No input provided. Exiting.', file=sys.stderr)
            sys.exit(2)

    output_path = args.output
    date_col = args.date_col
    do_col = args.do_col

    try:
        out = process_file(input_path, output_path=output_path, date_col_name=date_col, do_col_name=do_col)
        print(f'Processed file saved to: {out}')
    except Exception as e:
        print(f'Error: {e}', file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
