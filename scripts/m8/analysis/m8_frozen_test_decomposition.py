from __future__ import annotations

import numpy as np
import pandas as pd


DATASET = "data/sample/M8_replication_research_dataset.csv"
OUTPUT = "M8_frozen_test_decomposition.csv"

TRAIN_FRACTION = 0.60
VALIDATION_FRACTION = 0.20
PURGE_SECONDS = 5
EMBARGO_SECONDS = 5

THRESHOLD = 0.00957966
DIRECTION = "high"
N_TIME_BLOCKS = 3


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    required = {
        "timestamp",
        "symbol",
        "side",
        "fill_price",
        "mid_price",
        "relative_spread",
        "imbalance",
        "markout_1s",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    result = df.copy()

    result["timestamp"] = pd.to_datetime(
        result["timestamp"],
        errors="coerce",
    )

    result = result.dropna(
        subset=[
            "timestamp",
            "fill_price",
            "mid_price",
            "relative_spread",
            "imbalance",
            "markout_1s",
        ]
    ).copy()

    result = result[
        (result["fill_price"] > 0)
        & (result["mid_price"] > 0)
    ].copy()

    result["initial_edge"] = np.where(
        result["side"].eq("BUY"),
        result["mid_price"] - result["fill_price"],
        result["fill_price"] - result["mid_price"],
    )

    result["post_fill_move_1s"] = (
        result["markout_1s"] - result["initial_edge"]
    )

    result["post_fill_return_bps"] = (
        result["post_fill_move_1s"]
        / result["fill_price"]
        * 10000.0
    )

    result["signed_imbalance"] = np.where(
        result["side"].eq("BUY"),
        result["imbalance"],
        -result["imbalance"],
    )

    result["relative_spread_pct"] = (
        result["relative_spread"] * 100.0
    )

    result["interaction_score"] = (
        result["relative_spread_pct"]
        * result["signed_imbalance"]
    )

    return result.sort_values(
        "timestamp",
        kind="stable",
    ).reset_index(drop=True)


def split_dataset(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(df)

    train_end = int(n * TRAIN_FRACTION)
    validation_end = int(
        n * (TRAIN_FRACTION + VALIDATION_FRACTION)
    )

    train = df.iloc[:train_end].copy()
    validation = df.iloc[train_end:validation_end].copy()
    test = df.iloc[validation_end:].copy()

    purge = pd.Timedelta(seconds=PURGE_SECONDS)
    embargo = pd.Timedelta(seconds=EMBARGO_SECONDS)

    train_boundary = train["timestamp"].iloc[-1]
    validation_boundary = validation["timestamp"].iloc[-1]

    validation = validation[
        validation["timestamp"]
        >= train_boundary + purge
    ].copy()

    test = test[
        test["timestamp"]
        >= validation_boundary + embargo
    ].copy()

    return train, validation, test


def apply_policy(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    if DIRECTION == "high":
        result["selected"] = (
            result["interaction_score"] >= THRESHOLD
        )
    else:
        result["selected"] = (
            result["interaction_score"] <= THRESHOLD
        )

    return result


def summarize(
    df: pd.DataFrame,
    label: str,
) -> dict:
    selected = df[df["selected"]]

    return {
        "segment": label,
        "n": len(df),
        "selected": len(selected),
        "selection_rate": (
            len(selected) / len(df)
            if len(df)
            else np.nan
        ),
        "selected_mean_bps": (
            selected["post_fill_return_bps"].mean()
            if not selected.empty
            else np.nan
        ),
        "unselected_mean_bps": (
            df.loc[
                ~df["selected"],
                "post_fill_return_bps",
            ].mean()
            if (~df["selected"]).any()
            else np.nan
        ),
        "improvement_bps": (
            selected["post_fill_return_bps"].mean()
            - df.loc[
                ~df["selected"],
                "post_fill_return_bps",
            ].mean()
            if not selected.empty
            and (~df["selected"]).any()
            else np.nan
        ),
        "quote_all_mean_bps": (
            df["post_fill_return_bps"].mean()
        ),
    }


def main() -> None:
    df = prepare_data(
        pd.read_csv(DATASET)
    )

    _, _, test = split_dataset(df)
    test = apply_policy(test)

    rows = []

    rows.append(
        summarize(test, "overall")
    )

    for symbol, symbol_df in test.groupby(
        "symbol",
        sort=True,
    ):
        rows.append(
            summarize(
                symbol_df,
                symbol,
            )
        )

    test = test.sort_values(
        "timestamp",
        kind="stable",
    ).copy()

    test["time_block"] = pd.qcut(
        np.arange(len(test)),
        q=N_TIME_BLOCKS,
        labels=False,
    ) + 1

    for block, block_df in test.groupby(
        "time_block",
        sort=True,
    ):
        rows.append(
            summarize(
                block_df,
                f"time_block_{int(block)}",
            )
        )

    results = pd.DataFrame(rows)

    print("\nM8 FROZEN TEST DECOMPOSITION")
    print("=" * 90)

    print(
        f"Policy: interaction_score >= {THRESHOLD:.8f}"
    )

    print("\nOverall")
    print("-" * 90)
    print(
        results[
            results["segment"] == "overall"
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    print("\nBy symbol")
    print("-" * 90)
    print(
        results[
            results["segment"].isin(
                sorted(test["symbol"].unique())
            )
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    print("\nBy chronological test block")
    print("-" * 90)
    print(
        results[
            results["segment"].str.startswith(
                "time_block_"
            )
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    results.to_csv(
        OUTPUT,
        index=False,
    )

    print(f"\nSaved: {OUTPUT}")


if __name__ == "__main__":
    main()