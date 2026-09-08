import pandas as pd
from scipy.stats import mannwhitneyu


DATASET = "data/sample/AAPL_m8_analysis_dataset.csv"

FEATURE = "spread"
THRESHOLD = 0.08

TRAIN_FRACTION = 0.60
VALIDATION_FRACTION = 0.20
PURGE_SECONDS = 5
EMBARGO_SECONDS = 5

ALPHA = 0.05
N_TESTS = 3


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

    selected = test[
        test[FEATURE] <= THRESHOLD
    ]

    unselected = test[
        test[FEATURE] > THRESHOLD
    ]

    corrected_alpha = ALPHA / N_TESTS

    print(
        f"Selected observations:   {len(selected):,}"
    )
    print(
        f"Unselected observations: {len(unselected):,}"
    )
    print(
        f"Bonferroni alpha:         {corrected_alpha:.6f}"
    )

    for horizon in ("100ms", "1s", "5s"):
        target = f"post_fill_move_{horizon}"

        selected_values = selected[target]
        unselected_values = unselected[target]

        statistic, p_value = mannwhitneyu(
            selected_values,
            unselected_values,
            alternative="two-sided",
        )

        difference = (
            selected_values.mean()
            - unselected_values.mean()
        )

        print(f"\n=== {horizon} ===")
        print(
            f"Selected mean:       "
            f"{selected_values.mean():.6f}"
        )
        print(
            f"Unselected mean:     "
            f"{unselected_values.mean():.6f}"
        )
        print(
            f"Mean difference:     "
            f"{difference:.6f}"
        )
        print(
            f"Mann-Whitney U:      "
            f"{statistic:.3f}"
        )
        print(
            f"Raw p-value:         "
            f"{p_value:.6f}"
        )
        print(
            f"Bonferroni p-value:  "
            f"{min(p_value * N_TESTS, 1.0):.6f}"
        )
        print(
            f"Significant:         "
            f"{p_value < corrected_alpha}"
        )


if __name__ == "__main__":
    main()