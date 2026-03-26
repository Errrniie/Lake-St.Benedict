from __future__ import annotations

from Module.Clean.clean_module import drop_zero_do_rows, fill_hour_column
from Module.DORate.dodeltas_module import DEFAULT_HORIZONS, add_do_deltas, apply_nan_string
from Module.Lag.lag_module import apply_lag_features, drop_rows_missing_lags
from Module.Model.predict_module import predict_from_model_bundle
from Module.Model.train_module import train_model_on_csv
from Module.Parse.parse_module import find_do_column, parse_date_column, resolve_date_column


def run_connected_modules(
    df,
    date_col_name: str | None = None,
    do_col_name: str | None = None,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    na_string: str = "NaN",
):
    """Top-level module connector called by the controller/core."""
    date_col = resolve_date_column(df, date_col_name=date_col_name)
    do_col = find_do_column(df, do_col_name=do_col_name)
    if not do_col:
        raise ValueError("Could not find a DO column.")

    out = parse_date_column(df, date_col)
    out = fill_hour_column(out, date_col)
    out = drop_zero_do_rows(out, do_col)
    out = apply_lag_features(out, date_col, do_col)
    out = drop_rows_missing_lags(out)
    out = add_do_deltas(out, date_col, do_col, horizons=horizons)
    out = apply_nan_string(out, horizons=horizons, na_string=na_string)
    return out, date_col, do_col


def run_model_training(
    csv_path: str,
    target_col: str,
    model_output_path: str | None = None,
    include_other_delta_features: bool = True,
) -> dict:
    return train_model_on_csv(
        csv_path=csv_path,
        target_col=target_col,
        model_output_path=model_output_path,
        include_other_delta_features=include_other_delta_features,
    )


def run_model_prediction(
    model_path: str,
    input_csv_path: str,
    output_csv_path: str,
) -> dict:
    return predict_from_model_bundle(
        model_path=model_path,
        input_csv_path=input_csv_path,
        output_csv_path=output_csv_path,
    )

