import pandas as pd


DATASET = "data/sample/M8_replication_research_dataset.csv"

FEATURE = "spread"
THRESHOLD = 0.08

HORIZONS = [
    "100ms",
    "1s",
    "5s",
]


def add_initial_edge(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    result["initial_edge"] = (
        result["mid_price"] - result["fill_price"]
    ).where(
        result["side"] == "BUY",
        result["fill_price"] - result["mid_price"],
    )

    return result


def main() -> None:
    df = pd.read_csv(
        DATASET,
        parse_dates=["timestamp"],
    )

    df = add_initial_edge(df)

    print(f"Total observations: {len(df):,}")
    print()

    rows = []

    for symbol, symbol_df in df.groupby("symbol"):
        selected = symbol_df[
            symbol_df[FEATURE] <= THRESHOLD
        ]

        unselected = symbol_df[
            symbol_df[FEATURE] > THRESHOLD
        ]

        for horizon in HORIZONS:
            markout_column = f"markout_{horizon}"

            selected_values = (
                selected[markout_column]
                - selected["initial_edge"]
            ).dropna()

            unselected_values = (
                unselected[markout_column]
                - unselected["initial_edge"]
            ).dropna()

            if (
                selected_values.empty
                or unselected_values.empty
            ):
                continue

            rows.append(
                {
                    "symbol": symbol,
                    "horizon": horizon,
                    "selected_n": len(selected_values),
                    "unselected_n": len(unselected_values),
                    "selected_mean": selected_values.mean(),
                    "unselected_mean": unselected_values.mean(),
                    "difference": (
                        selected_values.mean()
                        - unselected_values.mean()
                    ),
                    "selected_negative_pct": (
                        (selected_values < 0).mean()
                    ),
                    "unselected_negative_pct": (
                        (unselected_values < 0).mean()
                    ),
                }
            )

    result = pd.DataFrame(rows)

    print("Per-symbol replication results:")
    print(
        result.to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    print("\n1-second summary:")

    one_second = result[
        result["horizon"] == "1s"
    ].copy()

    print(
        one_second[
            [
                "symbol",
                "selected_n",
                "unselected_n",
                "selected_mean",
                "unselected_mean",
                "difference",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    print("\nDirection of effect:")

    positive = (
        one_second["difference"] > 0
    ).sum()

    total = len(one_second)

    print(
        f"Positive difference: "
        f"{positive}/{total}"
    )


if __name__ == "__main__":
    main()