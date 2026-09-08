import pandas as pd


DATASET = "data/sample/AAPL_m8_analysis_dataset.csv"

FEATURE = "spread"
THRESHOLD = 0.08

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


def main() -> None:
    df = pd.read_csv(
        DATASET,
        parse_dates=["timestamp"],
    )

    _, _, test = split_dataset(df)

    selected = test[test[FEATURE] <= THRESHOLD]
    unselected = test[test[FEATURE] > THRESHOLD]

    print(f"Selected: {len(selected):,}")
    print(f"Unselected: {len(unselected):,}")

    for horizon in ("100ms", "1s", "5s"):
        target = f"post_fill_move_{horizon}"

        selected_mean = selected[target].mean()
        unselected_mean = unselected[target].mean()
        difference = selected_mean - unselected_mean

        selected_negative = (
            selected[target] < 0
        ).mean()

        unselected_negative = (
            unselected[target] < 0
        ).mean()

        print(f"\n=== {horizon} ===")
        print(
            f"Selected mean:       {selected_mean:.6f}"
        )
        print(
            f"Unselected mean:     {unselected_mean:.6f}"
        )
        print(
            f"Difference:          {difference:.6f}"
        )
        print(
            f"Selected negative %: "
            f"{selected_negative:.2%}"
        )
        print(
            f"Unselected negative %: "
            f"{unselected_negative:.2%}"
        )


if __name__ == "__main__":
    main()