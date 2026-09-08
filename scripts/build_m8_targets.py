import pandas as pd


DATASET = "data/sample/AAPL_m8_research_dataset.csv"
OUTPUT = "data/sample/AAPL_m8_analysis_dataset.csv"


def main() -> None:
    df = pd.read_csv(DATASET)

    for horizon in ("100ms", "1s", "5s"):
        df[f"initial_edge_{horizon}"] = (
            df["mid_price"] - df["fill_price"]
        ).where(
            df["side"] == "BUY",
            df["fill_price"] - df["mid_price"],
        )

        df[f"post_fill_move_{horizon}"] = (
            df[f"markout_{horizon}"]
            - df[f"initial_edge_{horizon}"]
        )

    df.to_csv(OUTPUT, index=False)

    print(f"Rows: {len(df):,}")
    print(f"Output: {OUTPUT}")


if __name__ == "__main__":
    main()