import pandas as pd


DATASET = "data/sample/AAPL_m8_research_dataset.csv"


def signed_initial_edge(df: pd.DataFrame) -> pd.Series:
    return (
        (df["mid_price"] - df["fill_price"]).where(
            df["side"] == "BUY",
            df["fill_price"] - df["mid_price"],
        )
    )


def main() -> None:
    df = pd.read_csv(DATASET)

    df["initial_edge"] = signed_initial_edge(df)

    for horizon in ("100ms", "1s", "5s"):
        markout = df[f"markout_{horizon}"]
        post_fill_move = markout - df["initial_edge"]

        print(f"\n=== {horizon} ===")
        print(f"Markout mean:          {markout.mean():.6f}")
        print(f"Initial edge mean:     {df['initial_edge'].mean():.6f}")
        print(f"Post-fill move mean:   {post_fill_move.mean():.6f}")
        print(f"Post-fill move median: {post_fill_move.median():.6f}")
        print(
            "Post-fill negative %:  "
            f"{(post_fill_move < 0).mean():.2%}"
        )

        by_side = pd.DataFrame(
            {
                "markout": markout,
                "initial_edge": df["initial_edge"],
                "post_fill_move": post_fill_move,
                "side": df["side"],
            }
        )

        print("\nBy side:")
        print(
            by_side.groupby("side")[
                ["markout", "initial_edge", "post_fill_move"]
            ].agg(["count", "mean", "median", "std"])
        )

    print("\nLargest adverse 5s post-fill moves:")
    df["post_fill_move_5s"] = (
        df["markout_5s"] - df["initial_edge"]
    )

    print(
        df.nsmallest(10, "post_fill_move_5s")[
            [
                "timestamp",
                "side",
                "fill_price",
                "mid_price",
                "initial_edge",
                "markout_5s",
                "post_fill_move_5s",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()