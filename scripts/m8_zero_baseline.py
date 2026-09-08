import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


DATASET = "data/sample/AAPL_m8_analysis_dataset.csv"

TRAIN_FRACTION = 0.60
VALIDATION_FRACTION = 0.20
PURGE_SECONDS = 5
EMBARGO_SECONDS = 5

TARGET = "post_fill_move_1s"


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
    name: str,
    y_true: pd.Series,
    prediction: pd.Series,
) -> None:
    mae = mean_absolute_error(y_true, prediction)
    rmse = mean_squared_error(y_true, prediction) ** 0.5
    correlation = prediction.corr(
        y_true.reset_index(drop=True)
    )

    print(name)
    print(f"  MAE:         {mae:.6f}")
    print(f"  RMSE:        {rmse:.6f}")
    print(f"  correlation: {correlation:.6f}")


def main() -> None:
    df = pd.read_csv(
        DATASET,
        parse_dates=["timestamp"],
    )

    train, validation, test = split_dataset(df)

    train_mean = train[TARGET].mean()

    validation_prediction = pd.Series(
        train_mean,
        index=validation.index,
    )

    test_prediction = pd.Series(
        train_mean,
        index=test.index,
    )

    print(f"Train mean target: {train_mean:.6f}")
    print()

    evaluate(
        "Validation zero-information baseline",
        validation[TARGET].reset_index(drop=True),
        validation_prediction.reset_index(drop=True),
    )

    print()

    evaluate(
        "Test zero-information baseline",
        test[TARGET].reset_index(drop=True),
        test_prediction.reset_index(drop=True),
    )


if __name__ == "__main__":
    main()