import pandas as pd


DATASET = "data/sample/M8_replication_research_dataset.csv"

TARGET_HORIZONS = [
    "100ms",
    "1s",
    "5s",
]

TIGHT_SPREAD_QUANTILE = 0.20


def add_post_fill_targets(
    df: pd.DataFrame,
) -> pd.DataFrame:
    result = df.copy()

    result["initial_edge"] = (
        result["mid_price"] - result["fill_price"]
    ).where(
        result["side"] == "BUY",
        result["fill_price"] - result["mid_price"],
    )

    for horizon in TARGET_HORIZONS:
        result[f"post_fill_move_{horizon}"] = (
            result[f"markout_{horizon}"]
            - result["initial_edge"]
        )

    return result


def main() -> None:
    df = pd.read_csv(
        DATASET,
        parse_dates=["timestamp"],
    )

    df = add_post_fill_targets(df)
    
    spread_threshold = (
        df.groupby("symbol")["spread"]
        .transform(
            lambda x: x.quantile(
                TIGHT_SPREAD_QUANTILE
            )
        )
    )

    df["tight_spread"] = (
        df["spread"] <= spread_threshold
    )


    print(
        f"Total observations: {len(df):,}"
    )

    print("\nObservations by symbol:")
    print(
        df.groupby("symbol")
        .size()
        .to_string()
    )

    print("\nTight-spread observations by symbol:")
    print(
        df.groupby("symbol")["tight_spread"]
        .agg(["count", "sum"])
        .rename(
            columns={
                "count": "total",
                "sum": "tight",
            }
        )
        .assign(
            tight_pct=lambda x: (
                x["tight"] / x["total"]
            )
        )
        .to_string(
            float_format=lambda value: f"{value:.4f}",
        )
    )

    rows = []

    for symbol, symbol_df in df.groupby("symbol"):
        tight = symbol_df[
            symbol_df["tight_spread"]
        ]

        normal = symbol_df[
            ~symbol_df["tight_spread"]
        ]

        for horizon in TARGET_HORIZONS:
            target = f"post_fill_move_{horizon}"

            tight_values = tight[target].dropna()
            normal_values = normal[target].dropna()

            if (
                tight_values.empty
                or normal_values.empty
            ):
                continue

            rows.append(
                {
                    "symbol": symbol,
                    "horizon": horizon,
                    "tight_n": len(tight_values),
                    "normal_n": len(normal_values),
                    "tight_mean": tight_values.mean(),
                    "normal_mean": normal_values.mean(),
                    "difference": (
                        tight_values.mean()
                        - normal_values.mean()
                    ),
                    "tight_negative_pct": (
                        (tight_values < 0).mean()
                    ),
                    "normal_negative_pct": (
                        (normal_values < 0).mean()
                    ),
                }
            )

    result = pd.DataFrame(rows)

    print("\nNormalized spread replication:")
    print(
        result.to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    print("\n1-second summary:")
    print(
        result[
            result["horizon"] == "1s"
        ][
            [
                "symbol",
                "tight_n",
                "normal_n",
                "tight_mean",
                "normal_mean",
                "difference",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )


if __name__ == "__main__":
    main()