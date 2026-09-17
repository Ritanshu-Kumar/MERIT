from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


DATA_PATH = Path(
    "research/m9_fill_hazard_2019-07-30.csv"
)

TIME_BINS_MS = [
    0,
    1,
    10,
    50,
    100,
    500,
    1000,
]


def main() -> None:
    df = pd.read_csv(DATA_PATH)

    df["time_ms"] = (
        df["time_to_resolution_ns"] / 1_000_000.0
    )

    df["time_bin"] = pd.cut(
        df["time_ms"],
        bins=TIME_BINS_MS,
        right=True,
        include_lowest=True,
    )

    print(f"Rows: {len(df):,}")

    print("\nOVERALL OUTCOMES")
    print(df["outcome"].value_counts())

    print("\nOVERALL PROPORTIONS")
    print(
        df["outcome"]
        .value_counts(normalize=True)
        .mul(100)
        .round(3)
    )

    print("\nOUTCOMES BY SIDE")
    print(
        pd.crosstab(
            df["quote_side"],
            df["outcome"],
        )
    )

    print("\nOUTCOME RATES BY TIME BIN")

    time_table = (
        pd.crosstab(
            df["time_bin"],
            df["outcome"],
            normalize="index",
        )
        .mul(100)
        .round(3)
    )

    print(time_table)

    for variable, label in [
        ("queue_ahead_initial", "QUEUE"),
        ("imbalance", "IMBALANCE"),
        ("relative_spread", "RELATIVE SPREAD"),
        ("delta_microprice", "DELTA MICROPRICE"),
    ]:
        df[f"{variable}_bin"] = pd.qcut(
            df[variable],
            5,
            labels=False,
            duplicates="drop",
        )

        print(
            f"\n{'=' * 72}"
        )
        print(
            f"{label}: QUINTILE OUTCOME RATES"
        )
        print(
            f"{'=' * 72}"
        )

        table = (
            df.groupby(
                [f"{variable}_bin", "quote_side"],
                observed=True,
            )["outcome"]
            .value_counts(
                normalize=True
            )
            .unstack(fill_value=0)
            .mul(100)
            .round(3)
        )

        print(table)

    print(
        "\nFILL RATE BY QUEUE QUINTILE"
    )

    queue_table = (
        df.groupby(
            ["queue_ahead_initial", "quote_side"],
            observed=True,
        )
        .agg(
            fill_rate=(
                "fill_before_adverse",
                "mean",
            ),
            mean_filled=(
                "filled_qty",
                "mean",
            ),
            n=("outcome", "size"),
        )
    )

    print(queue_table)

    print(
        "\nMEAN TIME TO TERMINAL EVENT"
    )

    print(
        df.groupby(
            ["quote_side", "outcome"],
            observed=True,
        )["time_ms"]
        .agg(["count", "mean", "median"])
        .round(3)
    )


if __name__ == "__main__":
    main()