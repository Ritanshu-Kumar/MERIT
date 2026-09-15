from __future__ import annotations

import numpy as np
import pandas as pd


DATASET = "data/sample/M8_replication_research_dataset.csv"
OUTPUT = "M8_tick_regime_robustness.csv"

TRAIN_FRACTION = 0.60
VALIDATION_FRACTION = 0.20
PURGE_SECONDS = 5
EMBARGO_SECONDS = 5

THRESHOLD = 0.00957966
TICK_SIZE = 0.01


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
        "spread",
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
            "spread",
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

    result["selected"] = (
        result["interaction_score"] >= THRESHOLD
    )

    return result


def summarize(
    df: pd.DataFrame,
    label: str,
) -> dict[str, float]:
    selected = df[df["selected"]]
    unselected = df[~df["selected"]]

    selected_mean = (
        selected["post_fill_return_bps"].mean()
        if not selected.empty
        else np.nan
    )

    unselected_mean = (
        unselected["post_fill_return_bps"].mean()
        if not unselected.empty
        else np.nan
    )

    return {
        "regime": label,
        "n": len(df),
        "selected": len(selected),
        "selection_rate": (
            len(selected) / len(df)
            if len(df)
            else np.nan
        ),
        "selected_mean_bps": selected_mean,
        "unselected_mean_bps": unselected_mean,
        "improvement_bps": (
            selected_mean - unselected_mean
            if np.isfinite(selected_mean)
            and np.isfinite(unselected_mean)
            else np.nan
        ),
        "quote_all_mean_bps": (
            df["post_fill_return_bps"].mean()
        ),
        "median_spread": df["spread"].median(),
    }


def main() -> None:
    df = prepare_data(
        pd.read_csv(DATASET)
    )

    _, _, test = split_dataset(df)
    test = apply_policy(test)

    test["tick_regime"] = np.where(
        test["spread"] <= TICK_SIZE,
        "one_tick",
        "multi_tick",
    )

    rows = [
        summarize(test, "overall"),
    ]

    for regime, regime_df in test.groupby(
        "tick_regime",
        sort=False,
    ):
        rows.append(
            summarize(
                regime_df,
                regime,
            )
        )

    print("\nM8 TICK REGIME ROBUSTNESS")
    print("=" * 90)
    print(
        f"Frozen policy: interaction_score >= {THRESHOLD:.8f}"
    )
    print(
        f"One-tick boundary: spread <= ${TICK_SIZE:.2f}"
    )

    print("\nOverall and tick-regime results")
    print("-" * 90)

    results = pd.DataFrame(rows)

    print(
        results.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    print("\nBy symbol")
    print("-" * 90)

    for symbol, symbol_df in test.groupby(
        "symbol",
        sort=True,
    ):
        one_tick = symbol_df[
            symbol_df["spread"] <= TICK_SIZE
        ]

        multi_tick = symbol_df[
            symbol_df["spread"] > TICK_SIZE
        ]

        print(f"\n{symbol}")

        for label, subset in [
            ("one_tick", one_tick),
            ("multi_tick", multi_tick),
        ]:
            if subset.empty:
                continue

            summary = summarize(
                subset,
                label,
            )

            print(
                f"  {label}: "
                f"n={summary['n']:>5} | "
                f"selected={summary['selected']:>5} | "
                f"selection={summary['selection_rate']:.4f} | "
                f"selected={summary['selected_mean_bps']:+.4f} | "
                f"unselected={summary['unselected_mean_bps']:+.4f} | "
                f"improvement={summary['improvement_bps']:+.4f}"
            )

    results.to_csv(
        OUTPUT,
        index=False,
    )

    print(f"\nSaved: {OUTPUT}")


if __name__ == "__main__":
    main()