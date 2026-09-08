import numpy as np
import pandas as pd


DATASET = "data/sample/AAPL_m8_analysis_dataset.csv"

FEATURE = "spread"
THRESHOLD = 0.08

TRAIN_FRACTION = 0.60
VALIDATION_FRACTION = 0.20
PURGE_SECONDS = 5
EMBARGO_SECONDS = 5

BLOCK_SIZE = 10
BOOTSTRAPS = 10000
SEED = 42


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


def block_bootstrap_difference(
    test: pd.DataFrame,
    target: str,
) -> np.ndarray:
    rng = np.random.default_rng(SEED)

    values = test[target].to_numpy()
    selected_mask = (
        test[FEATURE].to_numpy() <= THRESHOLD
    )

    n = len(test)

    blocks = [
        np.arange(
            start,
            min(start + BLOCK_SIZE, n),
        )
        for start in range(0, n, BLOCK_SIZE)
    ]

    differences = np.empty(BOOTSTRAPS)

    for iteration in range(BOOTSTRAPS):
        sampled_indices: list[int] = []

        while len(sampled_indices) < n:
            block = blocks[
                rng.integers(len(blocks))
            ]
            sampled_indices.extend(block.tolist())

        sampled_indices = sampled_indices[:n]

        sampled_values = values[sampled_indices]
        sampled_selected = selected_mask[sampled_indices]

        selected_values = sampled_values[
            sampled_selected
        ]
        unselected_values = sampled_values[
            ~sampled_selected
        ]

        if (
            len(selected_values) == 0
            or len(unselected_values) == 0
        ):
            differences[iteration] = np.nan
            continue

        differences[iteration] = (
            selected_values.mean()
            - unselected_values.mean()
        )

    return differences[
        ~np.isnan(differences)
    ]


def main() -> None:
    df = pd.read_csv(
        DATASET,
        parse_dates=["timestamp"],
    )

    _, _, test = split_dataset(df)

    print(
        f"Test observations: {len(test):,}"
    )

    for horizon in ("100ms", "1s", "5s"):
        target = f"post_fill_move_{horizon}"

        selected = test[
            test[FEATURE] <= THRESHOLD
        ][target]

        unselected = test[
            test[FEATURE] > THRESHOLD
        ][target]

        observed_difference = (
            selected.mean()
            - unselected.mean()
        )

        bootstrap = block_bootstrap_difference(
            test,
            target,
        )

        lower = np.percentile(
            bootstrap,
            2.5,
        )

        upper = np.percentile(
            bootstrap,
            97.5,
        )

        print(f"\n=== {horizon} ===")
        print(
            f"Selected:            {len(selected):,}"
        )
        print(
            f"Unselected:          {len(unselected):,}"
        )
        print(
            f"Selected mean:       {selected.mean():.6f}"
        )
        print(
            f"Unselected mean:     {unselected.mean():.6f}"
        )
        print(
            f"Observed difference: {observed_difference:.6f}"
        )
        print(
            f"Block bootstrap 95% CI: "
            f"[{lower:.6f}, {upper:.6f}]"
        )


if __name__ == "__main__":
    main()