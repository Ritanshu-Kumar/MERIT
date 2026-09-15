from __future__ import annotations

import numpy as np
import pandas as pd


DATASET = "data/sample/M8_replication_research_dataset.csv"
OUTPUT = "M8_frozen_validation_results.csv"

TRAIN_FRACTION = 0.60
VALIDATION_FRACTION = 0.20
PURGE_SECONDS = 5
EMBARGO_SECONDS = 5

QUANTILES = [0.20, 0.40, 0.50, 0.60, 0.80]

PRIMARY_TARGET = "post_fill_return_bps"


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
    )

    result = result[
        (result["fill_price"] > 0)
        & (result["mid_price"] > 0)
    ].copy()

    # Same target definition used in R4-R6.
    result["initial_edge"] = np.where(
        result["side"].eq("BUY"),
        result["mid_price"] - result["fill_price"],
        result["fill_price"] - result["mid_price"],
    )

    result["post_fill_move_1s"] = (
        result["markout_1s"]
        - result["initial_edge"]
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

    # Frozen candidate interaction score.
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


def evaluate_policy(
    df: pd.DataFrame,
    threshold: float,
    direction: str,
) -> dict[str, float]:
    if direction == "high":
        selected = df[
            df["interaction_score"] >= threshold
        ]
    elif direction == "low":
        selected = df[
            df["interaction_score"] <= threshold
        ]
    else:
        raise ValueError(
            f"Unknown direction: {direction}"
        )

    if selected.empty:
        return {
            "threshold": threshold,
            "direction": direction,
            "selected": 0,
            "selection_rate": 0.0,
            "mean_post_fill_return_bps": np.nan,
            "mean_gross_markout": np.nan,
            "mean_initial_edge": np.nan,
        }

    return {
        "threshold": threshold,
        "direction": direction,
        "selected": len(selected),
        "selection_rate": len(selected) / len(df),
        "mean_post_fill_return_bps": (
            selected["post_fill_return_bps"].mean()
        ),
        "mean_gross_markout": (
            selected["markout_1s"].mean()
        ),
        "mean_initial_edge": (
            selected["initial_edge"].mean()
        ),
    }


def evaluate_quote_all(
    df: pd.DataFrame,
) -> dict[str, float]:
    return {
        "selected": len(df),
        "selection_rate": 1.0,
        "mean_post_fill_return_bps": (
            df["post_fill_return_bps"].mean()
        ),
        "mean_gross_markout": (
            df["markout_1s"].mean()
        ),
        "mean_initial_edge": (
            df["initial_edge"].mean()
        ),
    }


def main() -> None:
    df = prepare_data(
        pd.read_csv(DATASET)
    )

    train, validation, test = split_dataset(df)

    print("\nM8 FROZEN VALIDATION")
    print("=" * 90)

    print("\nDataset")
    print("-" * 90)
    print(f"Total observations: {len(df):,}")
    print(
        f"Start: {df['timestamp'].iloc[0]}"
    )
    print(
        f"End:   {df['timestamp'].iloc[-1]}"
    )

    print("\nSplits")
    print("-" * 90)
    print(
        f"Train:      {len(train):,} | "
        f"{train['timestamp'].iloc[0]} -> "
        f"{train['timestamp'].iloc[-1]}"
    )
    print(
        f"Validation: {len(validation):,} | "
        f"{validation['timestamp'].iloc[0]} -> "
        f"{validation['timestamp'].iloc[-1]}"
    )
    print(
        f"Test:       {len(test):,} | "
        f"{test['timestamp'].iloc[0]} -> "
        f"{test['timestamp'].iloc[-1]}"
    )

    print("\nFrozen mechanism")
    print("-" * 90)
    print(
        "interaction_score = "
        "relative_spread_pct × signed_imbalance"
    )
    print(
        "signed_imbalance = +imbalance for BUY, "
        "-imbalance for SELL"
    )


    thresholds = sorted(
        validation["interaction_score"]
        .quantile(QUANTILES)
        .unique()
    )

    validation_results = []

    for direction in ["high", "low"]:
        for threshold in thresholds:
            result = evaluate_policy(
                validation,
                float(threshold),
                direction,
            )

            validation_results.append(result)

    validation_results_df = pd.DataFrame(
        validation_results
    )

    print("\nValidation search")
    print("-" * 90)
    print(
        validation_results_df.to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    valid = validation_results_df[
        validation_results_df["selection_rate"].between(
            0.05,
            0.95,
        )
    ].dropna(
        subset=["mean_post_fill_return_bps"]
    )

    if valid.empty:
        raise ValueError(
            "No valid validation policy survived "
            "the selection-rate constraints."
        )

    best_row = valid.loc[
        valid["mean_post_fill_return_bps"].idxmax()
    ]

    best_threshold = float(
        best_row["threshold"]
    )

    best_direction = str(
        best_row["direction"]
    )

    print("\nFrozen policy selected from validation")
    print("-" * 90)
    print(
        f"Direction: {best_direction}"
    )
    print(
        f"Threshold: {best_threshold:.8f}"
    )
    print(
        f"Validation selection rate: "
        f"{best_row['selection_rate']:.4f}"
    )
    print(
        f"Validation mean post-fill return: "
        f"{best_row['mean_post_fill_return_bps']:+.6f} bps"
    )


    frozen_test = evaluate_policy(
        test,
        best_threshold,
        best_direction,
    )

    quote_all_test = evaluate_quote_all(test)

    print("\nFinal frozen test evaluation")
    print("-" * 90)

    print("Candidate mechanism:")
    print(
        f"  Selected: {int(frozen_test['selected']):,}"
    )
    print(
        f"  Selection rate: "
        f"{frozen_test['selection_rate']:.4f}"
    )
    print(
        f"  Mean post-fill return: "
        f"{frozen_test['mean_post_fill_return_bps']:+.6f} bps"
    )
    print(
        f"  Mean gross markout: "
        f"{frozen_test['mean_gross_markout']:+.6f}"
    )

    print("\nQuote-all baseline:")
    print(
        f"  Selected: {int(quote_all_test['selected']):,}"
    )
    print(
        f"  Selection rate: "
        f"{quote_all_test['selection_rate']:.4f}"
    )
    print(
        f"  Mean post-fill return: "
        f"{quote_all_test['mean_post_fill_return_bps']:+.6f} bps"
    )
    print(
        f"  Mean gross markout: "
        f"{quote_all_test['mean_gross_markout']:+.6f}"
    )

    improvement = (
        frozen_test["mean_post_fill_return_bps"]
        - quote_all_test["mean_post_fill_return_bps"]
    )

    print("\nTest improvement versus quote-all")
    print("-" * 90)
    print(
        f"{improvement:+.6f} bps"
    )


    validation_results_df["final_policy"] = (
        (validation_results_df["threshold"] == best_threshold)
        & (validation_results_df["direction"] == best_direction)
    )

    validation_results_df["test_mean_post_fill_return_bps"] = (
        frozen_test["mean_post_fill_return_bps"]
    )

    validation_results_df[
        "quote_all_test_mean_post_fill_return_bps"
    ] = quote_all_test[
        "mean_post_fill_return_bps"
    ]

    validation_results_df[
        "test_improvement_vs_quote_all_bps"
    ] = improvement

    validation_results_df.to_csv(
        OUTPUT,
        index=False,
    )

    print(f"\nSaved: {OUTPUT}")


if __name__ == "__main__":
    main()