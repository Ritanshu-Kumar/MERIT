import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler


DATASET = "data/sample/AAPL_m8_analysis_dataset.csv"

FEATURES = [
    "spread",
    "relative_spread",
    "bid_size",
    "ask_size",
    "imbalance",
    "microprice",
    "imbalance_l5",
    "imbalance_l10",
    "weighted_imbalance_l5",
    "weighted_imbalance_l10",
]

TARGET = "post_fill_move_1s"

TRAIN_FRACTION = 0.60
VALIDATION_FRACTION = 0.20
PURGE_SECONDS = 5
EMBARGO_SECONDS = 5


def split_dataset(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df.sort_values("timestamp").reset_index(drop=True)

    n = len(df)

    train_end = int(n * TRAIN_FRACTION)
    validation_end = int(
        n * (TRAIN_FRACTION + VALIDATION_FRACTION)
    )

    train = df.iloc[:train_end].copy()
    validation = df.iloc[train_end:validation_end].copy()
    test = df.iloc[validation_end:].copy()

    purge = pd.Timedelta(seconds=PURGE_SECONDS)
    embargo = pd.Timedelta(seconds=EMBARGO_SECONDS)

    validation = validation[
        validation["timestamp"]
        >= train["timestamp"].iloc[-1] + purge
    ]

    test = test[
        test["timestamp"]
        >= validation["timestamp"].iloc[-1] + embargo
    ]

    return train, validation, test


def evaluate(
    model: LinearRegression,
    x: pd.DataFrame,
    y: pd.Series,
    name: str,
) -> None:
    prediction = model.predict(x)

    mae = mean_absolute_error(y, prediction)
    rmse = mean_squared_error(y, prediction) ** 0.5
    correlation = pd.Series(prediction).corr(
        y.reset_index(drop=True)
    )

    print(f"\n{name}")
    print(f"  MAE:         {mae:.6f}")
    print(f"  RMSE:        {rmse:.6f}")
    print(f"  correlation: {correlation:.6f}")


def main() -> None:
    df = pd.read_csv(
        DATASET,
        parse_dates=["timestamp"],
    )

    train, validation, test = split_dataset(df)

    scaler = StandardScaler()

    x_train = scaler.fit_transform(train[FEATURES])
    x_validation = scaler.transform(validation[FEATURES])
    x_test = scaler.transform(test[FEATURES])

    y_train = train[TARGET]
    y_validation = validation[TARGET]
    y_test = test[TARGET]

    model = LinearRegression()
    model.fit(x_train, y_train)

    print(f"Train observations:      {len(train):,}")
    print(f"Validation observations: {len(validation):,}")
    print(f"Test observations:       {len(test):,}")

    evaluate(
        model,
        x_train,
        y_train,
        "Train",
    )

    evaluate(
        model,
        x_validation,
        y_validation,
        "Validation",
    )

    evaluate(
        model,
        x_test,
        y_test,
        "Test",
    )

    coefficients = pd.Series(
        model.coef_,
        index=FEATURES,
    ).sort_values()

    print("\nStandardized coefficients:")
    print(coefficients.to_string())


if __name__ == "__main__":
    main()