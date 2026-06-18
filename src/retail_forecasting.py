from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor


VALIDATION_DAYS = 90
TEST_DAYS = 90
RANDOM_STATE = 42


def load_sales(data_dir: Path) -> pd.DataFrame:
    dtypes = {
        "store_nbr": "category",
        "family": "category",
        "sales": "float32",
        "onpromotion": "uint32",
    }
    return pd.read_csv(
        data_dir / "train.csv",
        dtype=dtypes,
        parse_dates=["date"],
    )


def build_daily_model_table(sales: pd.DataFrame) -> pd.DataFrame:
    daily_sales = sales.groupby("date")["sales"].mean().astype(float)
    daily_promotions = sales.groupby("date")["onpromotion"].sum().astype(float)

    frame = pd.DataFrame(
        {
            "sales": daily_sales,
            "promotion_units": daily_promotions,
        }
    )

    frame["time_step"] = np.arange(len(frame))
    frame["year"] = frame.index.year
    frame["month"] = frame.index.month
    frame["day_of_year"] = frame.index.dayofyear
    frame["day_of_week"] = frame.index.dayofweek
    frame["is_weekend"] = (frame.index.dayofweek >= 5).astype(int)

    for lag in [1, 7, 14, 28]:
        frame[f"lag_{lag}"] = frame["sales"].shift(lag)

    past_sales = frame["sales"].shift(1)
    for window in [7, 14, 28]:
        frame[f"rolling_mean_{window}"] = past_sales.rolling(window).mean()
        frame[f"rolling_std_{window}"] = past_sales.rolling(window).std()

    return frame.dropna()


def feature_columns() -> list[str]:
    return [
        "promotion_units",
        "time_step",
        "year",
        "month",
        "day_of_year",
        "day_of_week",
        "is_weekend",
        "lag_1",
        "lag_7",
        "lag_14",
        "lag_28",
        "rolling_mean_7",
        "rolling_std_7",
        "rolling_mean_14",
        "rolling_std_14",
        "rolling_mean_28",
        "rolling_std_28",
    ]


def temporal_split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_end = -(VALIDATION_DAYS + TEST_DAYS)
    validation_end = -TEST_DAYS
    train = frame.iloc[:train_end]
    validation = frame.iloc[train_end:validation_end]
    test = frame.iloc[validation_end:]
    return train, validation, test


def regression_metrics(name: str, actual: pd.Series, prediction: np.ndarray) -> dict[str, float | str]:
    return {
        "model": name,
        "mae": mean_absolute_error(actual, prediction),
        "rmse": mean_squared_error(actual, prediction) ** 0.5,
        "r2": r2_score(actual, prediction),
    }


def fit_boosting_models(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    features = feature_columns()
    X_train, y_train = train[features], train["sales"]
    X_validation, y_validation = validation[features], validation["sales"]
    X_test, y_test = test[features], test["sales"]

    xgb_params = {
        "objective": "reg:squarederror",
        "learning_rate": 0.03,
        "max_depth": 4,
        "min_child_weight": 5,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.1,
        "reg_lambda": 2.0,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    }
    xgb_tuner = XGBRegressor(
        n_estimators=1_000,
        early_stopping_rounds=50,
        eval_metric="rmse",
        **xgb_params,
    )
    xgb_tuner.fit(
        X_train,
        y_train,
        eval_set=[(X_validation, y_validation)],
        verbose=False,
    )

    lgbm_params = {
        "learning_rate": 0.03,
        "num_leaves": 24,
        "max_depth": 6,
        "min_child_samples": 30,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.1,
        "reg_lambda": 2.0,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "verbosity": -1,
    }
    lgbm_tuner = LGBMRegressor(n_estimators=1_000, **lgbm_params)
    lgbm_tuner.fit(
        X_train,
        y_train,
        eval_set=[(X_validation, y_validation)],
        eval_metric="rmse",
        callbacks=[early_stopping(50, verbose=False), log_evaluation(0)],
    )

    xgb_validation = np.clip(xgb_tuner.predict(X_validation), 0, None)
    lgbm_validation = np.clip(lgbm_tuner.predict(X_validation), 0, None)
    validation_rmse = np.array(
        [
            mean_squared_error(y_validation, xgb_validation) ** 0.5,
            mean_squared_error(y_validation, lgbm_validation) ** 0.5,
        ]
    )
    weights = (1 / validation_rmse) / (1 / validation_rmse).sum()

    development = pd.concat([train, validation])
    X_development = development[features]
    y_development = development["sales"]

    xgb_model = XGBRegressor(
        n_estimators=xgb_tuner.best_iteration + 1,
        eval_metric="rmse",
        **xgb_params,
    )
    xgb_model.fit(X_development, y_development, verbose=False)

    lgbm_model = LGBMRegressor(
        n_estimators=lgbm_tuner.best_iteration_,
        **lgbm_params,
    )
    lgbm_model.fit(X_development, y_development, callbacks=[log_evaluation(0)])

    xgb_test = np.clip(xgb_model.predict(X_test), 0, None)
    lgbm_test = np.clip(lgbm_model.predict(X_test), 0, None)
    ensemble_test = weights[0] * xgb_test + weights[1] * lgbm_test

    linear_model = LinearRegression()
    linear_model.fit(X_development[["lag_1"]], y_development)
    linear_test = np.clip(linear_model.predict(X_test[["lag_1"]]), 0, None)

    metrics = pd.DataFrame(
        [
            regression_metrics("Lag-1 linear regression", y_test, linear_test),
            regression_metrics("Seasonal naive (lag 7)", y_test, X_test["lag_7"].to_numpy()),
            regression_metrics("XGBoost", y_test, xgb_test),
            regression_metrics("LightGBM", y_test, lgbm_test),
            regression_metrics("Validation-weighted ensemble", y_test, ensemble_test),
        ]
    ).sort_values("rmse")

    predictions = pd.DataFrame(
        {
            "actual": y_test,
            "xgboost": xgb_test,
            "lightgbm": lgbm_test,
            "ensemble": ensemble_test,
        },
        index=y_test.index,
    )
    return metrics, predictions


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the retail demand forecasting pipeline.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Directory containing the Kaggle competition train.csv file.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()

    sales = load_sales(args.data_dir)
    model_table = build_daily_model_table(sales)
    train, validation, test = temporal_split(model_table)
    metrics, predictions = fit_boosting_models(train, validation, test)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.output_dir / "model_metrics.csv", index=False)
    predictions.to_csv(args.output_dir / "test_predictions.csv")
    print(metrics.to_string(index=False, float_format=lambda value: f"{value:.3f}"))


if __name__ == "__main__":
    main()
