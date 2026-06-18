import numpy as np
import pandas as pd

from src.retail_forecasting import build_daily_model_table, temporal_split


def make_sales_history(days: int = 240) -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=days, freq="D")
    sales = 100 + np.arange(days) * 0.2 + 10 * (dates.dayofweek >= 5)
    return pd.DataFrame(
        {
            "date": dates,
            "store_nbr": 1,
            "family": "GROCERY I",
            "sales": sales,
            "onpromotion": np.arange(days) % 5,
        }
    )


def test_sales_features_use_only_prior_observations() -> None:
    sales = make_sales_history()
    model_table = build_daily_model_table(sales)

    first_date = model_table.index[0]
    previous_date = first_date - pd.Timedelta(days=1)
    expected_lag = sales.loc[sales["date"] == previous_date, "sales"].iloc[0]

    assert model_table.loc[first_date, "lag_1"] == expected_lag

    prior_week = sales.loc[
        sales["date"].between(first_date - pd.Timedelta(days=7), previous_date),
        "sales",
    ]
    assert np.isclose(model_table.loc[first_date, "rolling_mean_7"], prior_week.mean())


def test_temporal_split_keeps_validation_and_test_windows_locked() -> None:
    model_table = build_daily_model_table(make_sales_history())
    train, validation, test = temporal_split(model_table)

    assert len(validation) == 90
    assert len(test) == 90
    assert train.index.max() < validation.index.min() < test.index.min()
    assert test.index.max() == model_table.index.max()
