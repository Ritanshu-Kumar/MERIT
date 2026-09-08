import pandas as pd
from scipy.stats import pearsonr, spearmanr


DATASET = "data/sample/M8_replication_research_dataset.csv"

TARGETS = [
    "markout_100ms",
    "markout_1s",
    "markout_5s",
]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    result["initial_edge"] = (
        result["mid_price"] - result["fill_price"]
    ).where(
        result["side"] == "BUY",
        result["fill_price"] - result["mid_price"],
    )

    result["relative_spread_pct"] = (
        result["spread"] / result["mid_price"]
    ) * 100.0

    for target in TARGETS:
        horizon = target.replace("markout_", "")

        result[f"post_fill_move_{horizon}"] = (
            result[target] - result["initial_edge"]
        )

    return result


def main() -> None:
    df = pd.read_csv(
        DATASET,
        parse_dates=["timestamp"],
    )

    df = add_features(df)

    print(f"Total observations: {len(df):,}")

    print("\nRelative spread by symbol:")
    print(
        df.groupby("symbol")["relative_spread_pct"]
        .agg(["count", "mean", "median", "min", "max"])
        .to_string(
            float_format=lambda value: f"{value:.6f}"
        )
    )

    print("\nCorrelation with post-fill movement:")

    for horizon in ("100ms", "1s", "5s"):
        target = f"post_fill_move_{horizon}"

        subset = df[
            ["relative_spread_pct", target]
        ].dropna()

        pearson_r, pearson_p = pearsonr(
            subset["relative_spread_pct"],
            subset[target],
        )

        spearman_r, spearman_p = spearmanr(
            subset["relative_spread_pct"],
            subset[target],
        )

        print(f"\n=== {horizon} ===")
        print(f"Pearson:")
        print(f"  r = {pearson_r:.6f}")
        print(f"  p = {pearson_p:.6f}")
        print("Spearman:")
        print(f"  rho = {spearman_r:.6f}")
        print(f"  p = {spearman_p:.6f}")

        print("\nBy symbol:")
        for symbol, symbol_df in df.groupby("symbol"):
            subset = symbol_df[
                [
                    "relative_spread_pct",
                    target,
                ]
            ].dropna()

            if len(subset) < 3:
                continue

            rho, p_value = spearmanr(
                subset["relative_spread_pct"],
                subset[target],
            )

            print(
                f"  {symbol}: "
                f"rho={rho:.6f}, "
                f"p={p_value:.6f}, "
                f"n={len(subset)}"
            )

    print("\nRelative-spread quintiles:")

    df["relative_spread_quintile"] = (
        df.groupby("symbol")["relative_spread_pct"]
        .transform(
            lambda x: pd.qcut(
                x,
                q=5,
                labels=False,
                duplicates="drop",
            )
        )
    )

    print(
        df.groupby("relative_spread_quintile", observed=True)[
            "post_fill_move_1s"
        ]
        .agg(["count", "mean", "median", "std"])
        .to_string(
            float_format=lambda value: f"{value:.6f}"
        )
    )


if __name__ == "__main__":
    main()