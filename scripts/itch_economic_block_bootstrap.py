from __future__ import annotations

import argparse
import numpy as np
import pandas as pd


DEFAULT_PATH = (
    "research/itch_economic_minute_equity.csv"
)

BLOCKS = (1, 5, 10)
REPS = 5000
SEED = 20260916


def load_minute_pnl(path: str) -> pd.Series:
    df = pd.read_csv(path)

    wide = (
        df.pivot(
            index="timestamp_ns",
            columns="strategy",
            values="equity",
        )
        .sort_index()
    )

    required = {"BASELINE", "M8"}

    missing = required - set(wide.columns)

    if missing:
        raise ValueError(
            f"Missing strategies: {sorted(missing)}"
        )

    wide = wide[
        ["BASELINE", "M8"]
    ].dropna()

    # Convert equity levels into minute P&L.
    baseline_pnl = (
        wide["BASELINE"].diff()
    )

    m8_pnl = (
        wide["M8"].diff()
    )

    difference = (
        m8_pnl - baseline_pnl
    ).dropna()

    return difference


def circular_block_sample(
    values: np.ndarray,
    block_size: int,
    rng: np.random.Generator,
) -> float:
    n = len(values)

    n_blocks = int(
        np.ceil(
            n / block_size
        )
    )

    total = 0.0

    for _ in range(n_blocks):
        start = int(
            rng.integers(0, n)
        )

        indices = (
            start
            + np.arange(block_size)
        ) % n

        total += values[indices].sum()

    return total


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--path",
        default=DEFAULT_PATH,
    )

    parser.add_argument(
        "--reps",
        type=int,
        default=REPS,
    )

    args = parser.parse_args()

    differences = load_minute_pnl(
        args.path
    )

    observed = float(
        differences.sum()
    )

    print(
        "=== ITCH ECONOMIC BLOCK BOOTSTRAP ==="
    )
    print(
        f"minute observations: "
        f"{len(differences):,}"
    )
    print(
        f"observed M8 - baseline: "
        f"${observed:,.2f}"
    )

    print()

    for block_size in BLOCKS:
        rng = np.random.default_rng(
            SEED + block_size
        )

        values = differences.to_numpy(
            dtype=float
        )

        samples = np.empty(
            args.reps,
            dtype=float,
        )

        for i in range(args.reps):
            samples[i] = (
                circular_block_sample(
                    values,
                    block_size,
                    rng,
                )
            )

        print(
            f"{block_size:>2}-minute blocks:"
        )
        print(
            f"  p025 = "
            f"${np.quantile(samples, 0.025):,.2f}"
        )
        print(
            f"  p50  = "
            f"${np.quantile(samples, 0.50):,.2f}"
        )
        print(
            f"  p975 = "
            f"${np.quantile(samples, 0.975):,.2f}"
        )

        print(
            f"  bootstrap mean = "
            f"${samples.mean():,.2f}"
        )

        print()


if __name__ == "__main__":
    main()