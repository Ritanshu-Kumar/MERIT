from __future__ import annotations

import argparse

import pandas as pd


THRESHOLD = 0.00957966


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        default="data/itch/M8_ITCH_2019-07-30_events.csv",
    )

    parser.add_argument(
        "--output",
        default="data/itch/M8_ITCH_2019-07-30_events_corrected.csv",
    )

    args = parser.parse_args()

    df = pd.read_csv(args.input)

    # ITCH side = resting order side.
    # A passive fill is on the opposite side.
    df["quote_side"] = df["side"].map(
        {
            "S": "BUY",
            "B": "SELL",
        }
    )

    if df["quote_side"].isna().any():
        raise ValueError(
            "Unexpected ITCH side value."
        )

    signed_imbalance = df["imbalance"].where(
        df["quote_side"] == "BUY",
        -df["imbalance"],
    )

    df["score"] = (
        df["relative_spread"]
        * 100.0
        * signed_imbalance
    )

    df["selected"] = (
        df["score"] >= THRESHOLD
    ).astype(int)

    df.to_csv(
        args.output,
        index=False,
    )

    print("=== CORRECTED ITCH M8 LABELS ===")
    print(f"rows: {len(df):,}")
    print(
        f"selected: {int(df['selected'].sum()):,}"
    )
    print(
        f"selection rate: {df['selected'].mean():.6f}"
    )
    print()
    print(
        df.groupby(
            ["symbol", "quote_side"]
        )["selected"]
        .agg(
            n="size",
            selected="sum",
            rate="mean",
        )
        .to_string()
    )
    print()
    print(f"output: {args.output}")


if __name__ == "__main__":
    main()