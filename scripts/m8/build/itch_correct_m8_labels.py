
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

    # ITCH side is the resting order's side.
    # The passive quote is on the opposite side.
    df["quote_side"] = df["side"].map({
        "S": "BUY",
        "B": "SELL",
    })

    if df["quote_side"].isna().any():
        raise ValueError("Unexpected ITCH side value.")

    # Calculate M8 score from the passive quote's perspective.
    signed_imbalance = df["imbalance"].where(
        df["quote_side"] == "BUY",
        -df["imbalance"],
    )

    df["score"] = (
        df["relative_spread"] * 100.0 * signed_imbalance
    )

    df["selected"] = (
        df["score"] >= THRESHOLD
    ).astype(int)

    # Recalculate markout relative to execution price.
    # BUY: future midpoint - execution price.
    # SELL: execution price - future midpoint.
    execution_price = pd.to_numeric(
        df["execution_price"], errors="coerce"
    )

    for horizon in ("100ms", "1s", "5s"):
        future_mid = pd.to_numeric(
            df[f"future_mid_{horizon}"], errors="coerce"
        )

        signed_move = (
            future_mid - execution_price
        ).where(
            df["quote_side"] == "BUY",
            execution_price - future_mid,
        )

        df[f"post_fill_move_{horizon}"] = signed_move

        df[f"return_bps_{horizon}"] = (
            signed_move / execution_price * 10000.0
        ).where(execution_price > 0)

    df.to_csv(args.output, index=False)

    print("=== CORRECTED ITCH M8 LABELS ===")
    print(f"Rows: {len(df):,}")
    print(f"Selected: {int(df['selected'].sum()):,}")
    print(f"Selection rate: {df['selected'].mean():.6f}")
    print("Returns recalculated relative to execution price.")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
