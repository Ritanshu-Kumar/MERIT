import pandas as pd


DATASET = "data/sample/AAPL_m8_research_dataset.csv"


def main() -> None:
    df = pd.read_csv(DATASET)

    print(f"Rows: {len(df):,}")
    print(f"Unique fills: {df['fill_id'].nunique():,}")
    print(f"Duplicate fills: {df['fill_id'].duplicated().sum():,}")

    print("\nSide counts:")
    print(df["side"].value_counts())

    for column in (
        "markout_100ms",
        "markout_1s",
        "markout_5s",
    ):
        series = df[column]

        print(f"\n{column}:")
        print(f"  mean:        {series.mean():.6f}")
        print(f"  median:      {series.median():.6f}")
        print(f"  std:         {series.std():.6f}")
        print(f"  min:         {series.min():.6f}")
        print(f"  max:         {series.max():.6f}")
        print(f"  negative %:  {(series < 0).mean():.2%}")

    print("\n1s markout by side:")
    print(
        df.groupby("side")["markout_1s"]
        .agg(["count", "mean", "median", "std", "min", "max"])
    )

    print("\n1s markout quantiles:")
    print(
        df["markout_1s"].quantile(
            [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
        )
    )

    print("\nLargest negative 5s markouts:")
    print(
        df.nsmallest(10, "markout_5s")[
            ["timestamp", "side", "fill_price", "mid_price", "markout_5s"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()