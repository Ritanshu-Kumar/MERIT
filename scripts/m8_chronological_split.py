import pandas as pd


DATASET = "data/sample/AAPL_m8_analysis_dataset.csv"

TRAIN_FRACTION = 0.60
VALIDATION_FRACTION = 0.20
PURGE_SECONDS = 5
EMBARGO_SECONDS = 5


def main() -> None:
    df = pd.read_csv(
        DATASET,
        parse_dates=["timestamp"],
    )

    df = df.sort_values("timestamp").reset_index(drop=True)

    n = len(df)

    train_end = int(n * TRAIN_FRACTION)
    validation_end = int(
        n * (TRAIN_FRACTION + VALIDATION_FRACTION)
    )

    train = df.iloc[:train_end].copy()
    validation = df.iloc[train_end:validation_end].copy()
    test = df.iloc[validation_end:].copy()

    train_boundary = train["timestamp"].iloc[-1]
    validation_start = validation["timestamp"].iloc[0]

    validation_boundary = validation["timestamp"].iloc[-1]
    test_start = test["timestamp"].iloc[0]

    purge = pd.Timedelta(seconds=PURGE_SECONDS)
    embargo = pd.Timedelta(seconds=EMBARGO_SECONDS)

    validation = validation[
        validation["timestamp"] >= train_boundary + purge
    ]

    test = test[
        test["timestamp"] >= validation_boundary + embargo
    ]

    print("Dataset")
    print(f"Total observations: {len(df):,}")
    print(f"Start: {df['timestamp'].iloc[0]}")
    print(f"End:   {df['timestamp'].iloc[-1]}")

    print("\nSplits")
    print(
        f"Train:      {len(train):,} "
        f"{train['timestamp'].iloc[0]} -> "
        f"{train['timestamp'].iloc[-1]}"
    )
    print(
        f"Validation: {len(validation):,} "
        f"{validation['timestamp'].iloc[0]} -> "
        f"{validation['timestamp'].iloc[-1]}"
    )
    print(
        f"Test:       {len(test):,} "
        f"{test['timestamp'].iloc[0]} -> "
        f"{test['timestamp'].iloc[-1]}"
    )


if __name__ == "__main__":
    main()