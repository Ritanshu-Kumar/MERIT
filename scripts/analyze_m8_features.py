import pandas as pd


DATASET = "data/sample/AAPL_m8_analysis_dataset.csv"

FEATURES = [
    "spread",
    "relative_spread",
    "bid_size",
    "ask_size",
    "imbalance",
    "microprice",
    "imbalance_l5",
    "imbalance_l10",
    "weighted_imbalance_l5",
    "weighted_imbalance_l10",
]


def main() -> None:
    df = pd.read_csv(DATASET)

    target = df["post_fill_move_1s"]

    print(f"Observations: {len(df):,}")
    print(f"Target mean: {target.mean():.6f}")
    print(f"Target median: {target.median():.6f}")
    print(f"Target std: {target.std():.6f}")
    print(f"Target negative %: {(target < 0).mean():.2%}")

    print("\nFeature correlations with post-fill 1s move:")

    correlations = (
        df[FEATURES]
        .corrwith(target)
        .sort_values()
    )

    print(correlations.to_string())

    print("\nFeature summary:")
    print(
        df[FEATURES]
        .describe()
        .T[
            [
                "mean",
                "std",
                "min",
                "25%",
                "50%",
                "75%",
                "max",
            ]
        ]
    )

    print("\nTarget by side:")
    print(
        df.groupby("side")["post_fill_move_1s"]
        .agg(["count", "mean", "median", "std", "min", "max"])
    )


if __name__ == "__main__":
    main()