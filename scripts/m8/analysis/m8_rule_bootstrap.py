import numpy as np
import pandas as pd


DATASET = "data/sample/AAPL_m8_analysis_dataset.csv"

FEATURE = "spread"
THRESHOLD = 0.08
TARGET = "post_fill_move_1s"

TRAIN_FRACTION = 0.60
VALIDATION_FRACTION = 0.20
PURGE_SECONDS = 5
EMBARGO_SECONDS = 5

BLOCK_SIZE = 5
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


def block_bootstrap_mean_difference(
    selected: np.ndarray,
    unselected: np.ndarray,
    block_size: int,
    bootstrap_count: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)

    combined = np.concatenate(
        [
            np.ones(len(selected), dtype=int),
            np.zeros(len(unselected), dtype=int),
        ]
    )

    values = np.concatenate(
        [
            selected,
            unselected,
        ]
    )

    blocks = [
        np.arange(
            start,
            min(start + block_size, len(values)),
        )
        for start in range(
            0,
            len(values),
            block_size,
        )
    ]

    differences = np.empty(bootstrap_count)

    for iteration in range(bootstrap_count):
        sampled_indices = []

        while len(sampled_indices) < len(values):
            block = blocks[
                rng.integers(len(blocks))
            ]
            sampled_indices.extend(block.tolist())

        sampled_indices = np.asarray(
            sampled_indices[:len(values)]
        )

        sampled_values = values[sampled_indices]
        sampled_groups = combined[sampled_indices]

        sampled_selected = sampled_values[
            sampled_groups == 1
        ]

        sampled_unselected = sampled_values[
            sampled_groups == 0
        ]

        differences[iteration] = (
            sampled_selected.mean()
            - sampled_unselected.mean()
        )

    return differences


def main() -> None:
    df = pd.read_csv(
        DATASET,
        parse_dates=["timestamp"],
    )

    _, _, test = split_dataset(df)

    selected = test[
        test[FEATURE] <= THRESHOLD
    ][TARGET].to_numpy()

    unselected = test[
        test[FEATURE] > THRESHOLD
    ][TARGET].to_numpy()

    observed_difference = (
        selected.mean()
        - unselected.mean()
    )

    bootstrap = block_bootstrap_mean_difference(
        selected=selected,
        unselected=unselected,
        block_size=BLOCK_SIZE,
        bootstrap_count=BOOTSTRAPS,
        seed=SEED,
    )

    lower = np.percentile(bootstrap, 2.5)
    upper = np.percentile(bootstrap, 97.5)

    print(f"Selected observations:   {len(selected)}")
    print(f"Unselected observations: {len(unselected)}")
    print(f"Observed difference:      {observed_difference:.6f}")
    print(f"Bootstrap 95% interval:   [{lower:.6f}, {upper:.6f}]")

    print(
        f"Selected mean:            {selected.mean():.6f}"
    )
    print(
        f"Unselected mean:          {unselected.mean():.6f}"
    )


if __name__ == "__main__":
    main()