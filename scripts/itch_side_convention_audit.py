from __future__ import annotations

import argparse

import pandas as pd


DEFAULT_PATH = (
    "data/itch/M8_ITCH_2019-07-30_events.csv"
)

THRESHOLD = 0.00957966


def score(
    relative_spread: float,
    imbalance: float,
    side: str,
) -> float:
    signed_imbalance = (
        imbalance
        if side == "BUY"
        else -imbalance
    )

    return (
        relative_spread
        * 100.0
        * signed_imbalance
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        default=DEFAULT_PATH,
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=20,
    )
    args = parser.parse_args()

    df = pd.read_csv(args.path)

    df = df.sort_values(
        ["timestamp_ns", "symbol"],
        kind="stable",
    ).head(args.rows)

    print("=== ITCH SIDE CONVENTION AUDIT ===")
    print()

    for row in df.itertuples(index=False):
        resting_side = row.side

        # An execution of a resting SELL fills
        # our passive BUY; an execution of a
        # resting BUY fills our passive SELL.
        quote_side = (
            "BUY"
            if resting_side == "S"
            else "SELL"
        )

        buy_score = score(
            row.relative_spread,
            row.imbalance,
            "BUY",
        )

        sell_score = score(
            row.relative_spread,
            row.imbalance,
            "SELL",
        )

        correct_score = (
            buy_score
            if quote_side == "BUY"
            else sell_score
        )

        print(
            f"{row.symbol:>5} "
            f"resting={resting_side} "
            f"quote={quote_side} "
            f"imb={row.imbalance:+.6f} "
            f"rel_spread={row.relative_spread:.8f} "
            f"BUY={buy_score:+.6f} "
            f"SELL={sell_score:+.6f} "
            f"correct={correct_score:+.6f} "
            f"selected={correct_score >= THRESHOLD}"
        )


if __name__ == "__main__":
    main()