import numpy as np
import pandas as pd


DATASET = "data/sample/M8_replication_research_dataset.csv"

TIGHT_QUANTILE = 0.20
TARGET = "markout_1s"


def add_post_fill_move(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    result["initial_edge"] = (
        result["mid_price"] - result["fill_price"]
    ).where(
        result["side"] == "BUY",
        result["fill_price"] - result["mid_price"],
    )

    result["post_fill_move"] = (
        result[TARGET] - result["initial_edge"]
    )

    return result


def main() -> None:
    df = pd.read_csv(
        DATASET,
        parse_dates=["timestamp"],
    )

    df = add_post_fill_move(df)

    thresholds = (
        df.groupby("symbol")["spread"]
        .quantile(TIGHT_QUANTILE)
    )

    df["tight_spread"] = df.apply(
        lambda row: (
            row["spread"]
            <= thresholds.loc[row["symbol"]]
        ),
        axis=1,
    )

    rows = []

    for symbol, symbol_df in df.groupby("symbol"):
        tight = symbol_df[
            symbol_df["tight_spread"]
        ]["post_fill_move"].dropna()

        normal = symbol_df[
            ~symbol_df["tight_spread"]
        ]["post_fill_move"].dropna()

        if tight.empty or normal.empty:
            continue

        difference = (
            tight.mean() - normal.mean()
        )

        rows.append(
            {
                "symbol": symbol,
                "tight_n": len(tight),
                "normal_n": len(normal),
                "tight_mean": tight.mean(),
                "normal_mean": normal.mean(),
                "difference": difference,
            }
        )

    result = pd.DataFrame(rows)

    print("Per-symbol effects:")
    print(
        result.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    effects = result["difference"].to_numpy()

    print("\nEqual-weight symbol effect:")
    print(
        f"Mean difference: "
        f"{effects.mean():.6f}"
    )

    print(
        f"Median difference: "
        f"{np.median(effects):.6f}"
    )

    print(
        f"Positive symbols: "
        f"{(effects > 0).sum()}/{len(effects)}"
    )

    print(
        f"Negative symbols: "
        f"{(effects < 0).sum()}/{len(effects)}"
    )


if __name__ == "__main__":
    main()