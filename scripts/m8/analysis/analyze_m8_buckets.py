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

TARGET = "post_fill_move_1s"


def main() -> None:
    df = pd.read_csv(DATASET)

    print(f"Observations: {len(df):,}")

    for feature in FEATURES:
        print(f"\n=== {feature} ===")

        try:
            buckets = pd.qcut(
                df[feature],
                q=5,
                duplicates="drop",
            )
        except ValueError:
            print("Unable to form quantile buckets.")
            continue

        result = (
            df.assign(bucket=buckets)
            .groupby("bucket", observed=True)[TARGET]
            .agg(
                count="count",
                mean="mean",
                median="median",
                std="std",
            )
        )

        print(result.to_string())


if __name__ == "__main__":
    main()