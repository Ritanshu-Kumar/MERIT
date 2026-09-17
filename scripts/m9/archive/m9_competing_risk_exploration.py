from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


DATA_PATH = Path(
    "research/m9_fill_hazard_2019-07-30.csv"
)

HORIZONS_MS = [
    1,
    5,
    10,
    25,
    50,
    100,
    250,
    500,
    1000,
]


def cumulative_incidence(
    df: pd.DataFrame,
    horizon_ms: int,
) -> tuple[float, float, int]:
    t = df["time_ms"] <= horizon_ms

    fills = (
        t
        & df["outcome"].eq("FILL")
    ).sum()

    adverse = (
        t
        & df["outcome"].eq("ADVERSE")
    ).sum()

    n = len(df)

    return (
        fills / n,
        adverse / n,
        n,
    )


def add_quantiles(
    df: pd.DataFrame,
) -> pd.DataFrame:
    df = df.copy()

    df["queue_q"] = pd.qcut(
        df["queue_ahead_initial"],
        5,
        labels=False,
        duplicates="drop",
    )

    df["imbalance_q"] = pd.qcut(
        df["imbalance"],
        5,
        labels=False,
        duplicates="drop",
    )

    df["delta_microprice_q"] = pd.qcut(
        df["delta_microprice"],
        5,
        labels=False,
        duplicates="drop",
    )

    df["relative_spread_q"] = pd.qcut(
        df["relative_spread"],
        5,
        labels=False,
        duplicates="drop",
    )

    return df


def print_cif(
    df: pd.DataFrame,
    name: str,
) -> None:
    print("\n" + "=" * 80)
    print(name)
    print("=" * 80)

    rows = []

    for horizon in HORIZONS_MS:
        fill, adverse, n = cumulative_incidence(
            df,
            horizon,
        )

        rows.append(
            {
                "horizon_ms": horizon,
                "fill_cif_pct": 100 * fill,
                "adverse_cif_pct": 100 * adverse,
                "remaining_pct": 100 * (
                    1 - fill - adverse
                ),
                "n": n,
            }
        )

    print(
        pd.DataFrame(rows)
        .round(4)
        .to_string(index=False)
    )


def print_grouped_cif(
    df: pd.DataFrame,
    column: str,
) -> None:
    print("\n" + "=" * 80)
    print(f"{column} — 100ms and 1000ms")
    print("=" * 80)

    rows = []

    for group, g in df.groupby(
        column,
        observed=True,
    ):
        fill_100, adverse_100, _ = (
            cumulative_incidence(g, 100)
        )

        fill_1000, adverse_1000, _ = (
            cumulative_incidence(g, 1000)
        )

        rows.append(
            {
                "group": group,
                "n": len(g),
                "fill_100ms_pct": 100 * fill_100,
                "adverse_100ms_pct": (
                    100 * adverse_100
                ),
                "fill_1s_pct": 100 * fill_1000,
                "adverse_1s_pct": (
                    100 * adverse_1000
                ),
            }
        )

    print(
        pd.DataFrame(rows)
        .round(4)
        .to_string(index=False)
    )


def main() -> None:
    df = pd.read_csv(DATA_PATH)

    df["time_ms"] = (
        df["time_to_resolution_ns"]
        / 1_000_000.0
    )

    print(
        f"Rows: {len(df):,}"
    )

    for side in ["BUY", "SELL"]:
        print_cif(
            df[df["quote_side"] == side],
            f"QUOTE SIDE: {side}",
        )

    df = add_quantiles(df)

    for column in [
        "queue_q",
        "imbalance_q",
        "delta_microprice_q",
        "relative_spread_q",
    ]:
        print_grouped_cif(
            df,
            column,
        )

    print("\nQUEUE QUINTILE BOUNDARIES")

    print(
        df[
            "queue_ahead_initial"
        ]
        .quantile(
            [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        )
        .round(2)
        .to_string()
    )


if __name__ == "__main__":
    main()