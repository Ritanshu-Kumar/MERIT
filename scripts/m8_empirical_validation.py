from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_DATASET = Path(
    "data/sample/M8_replication_research_dataset.csv"
)

DEFAULT_THRESHOLD = 0.00957966

LAGS = (0, 1, 5, 20, 100)


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
        utc=True,
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

    result["interaction_score"] = (
        result["relative_spread_pct"]
        * result["signed_imbalance"]
    )

    return (
        result.sort_values(
            ["timestamp", "symbol"],
            kind="stable",
        )
        .reset_index(drop=True)
    )


def selected_mask(
    df: pd.DataFrame,
    threshold: float,
) -> pd.Series:
    return df["interaction_score"] >= threshold


def selected_minus_unselected_effect(
    df: pd.DataFrame,
    selected: pd.Series,
) -> float:
    if selected.sum() == 0:
        raise ValueError(
            "No selected observations."
        )

    if (~selected).sum() == 0:
        raise ValueError(
            "No unselected observations."
        )

    return float(
        df.loc[
            selected,
            "post_fill_return_bps",
        ].mean()
        - df.loc[
            ~selected,
            "post_fill_return_bps",
        ].mean()
    )


def temporal_placebo(
    df: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []

    for lag in LAGS:
        if lag == 0:
            lagged_imbalance = (
                df["signed_imbalance"]
                .copy()
            )
        else:
            lagged_imbalance = (
                df.groupby(
                    "symbol",
                    sort=False,
                )["signed_imbalance"]
                .shift(lag)
            )

        usable = lagged_imbalance.notna()

        score = (
            df["relative_spread_pct"]
            * lagged_imbalance
        )

        selected = (
            (score >= threshold)
            & usable
        )

        usable_df = (
            df.loc[usable]
            .reset_index(drop=True)
        )

        usable_selected = (
            selected.loc[usable]
            .reset_index(drop=True)
        )

        effect = selected_minus_unselected_effect(
            usable_df,
            usable_selected,
        )

        rows.append(
            {
                "lag_events": lag,
                "n": int(usable.sum()),
                "selected": int(
                    selected.sum()
                ),
                "selection_rate": float(
                    selected.sum()
                    / usable.sum()
                ),
                "effect_bps": effect,
            }
        )

    return pd.DataFrame(rows)


def permutation_placebo(
    df: pd.DataFrame,
    threshold: float,
    repetitions: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    symbols = (
        df["symbol"]
        .to_numpy()
    )

    original = (
        df["signed_imbalance"]
        .to_numpy()
        .copy()
    )

    effects: list[float] = []

    for _ in range(repetitions):
        shuffled = original.copy()

        for symbol in np.unique(symbols):
            indices = np.flatnonzero(
                symbols == symbol
            )

            rng.shuffle(
                shuffled[indices]
            )

        score = (
            df["relative_spread_pct"]
            .to_numpy()
            * shuffled
        )

        selected = score >= threshold

        effects.append(
            selected_minus_unselected_effect(
                df,
                pd.Series(
                    selected,
                    index=df.index,
                ),
            )
        )

    effects_array = np.asarray(
        effects,
        dtype=float,
    )

    return pd.DataFrame(
        {
            "mean_bps": [
                float(
                    effects_array.mean()
                )
            ],
            "p025_bps": [
                float(
                    np.quantile(
                        effects_array,
                        0.025,
                    )
                )
            ],
            "p50_bps": [
                float(
                    np.quantile(
                        effects_array,
                        0.50,
                    )
                )
            ],
            "p975_bps": [
                float(
                    np.quantile(
                        effects_array,
                        0.975,
                    )
                )
            ],
            "repetitions": [
                repetitions
            ],
            "seed": [seed],
        }
    )


def block_bootstrap(
    df: pd.DataFrame,
    threshold: float,
    block_minutes: int,
    repetitions: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    work = df.copy()

    work["block"] = (
        work["timestamp"]
        .dt.floor(
            f"{block_minutes}min"
        )
    )

    blocks = (
        work["block"]
        .drop_duplicates()
        .to_numpy()
    )

    observed = selected_minus_unselected_effect(
        work,
        selected_mask(
            work,
            threshold,
        ),
    )

    bootstrap_effects: list[float] = []

    for _ in range(repetitions):
        sampled_blocks = rng.choice(
            blocks,
            size=len(blocks),
            replace=True,
        )

        pieces = [
            work.loc[
                work["block"] == block
            ]
            for block in sampled_blocks
        ]

        sample = pd.concat(
            pieces,
            ignore_index=True,
        )

        selected = selected_mask(
            sample,
            threshold,
        )

        bootstrap_effects.append(
            selected_minus_unselected_effect(
                sample,
                selected,
            )
        )

    values = np.asarray(
        bootstrap_effects,
        dtype=float,
    )

    return pd.DataFrame(
        {
            "block_minutes": [
                block_minutes
            ],
            "repetitions": [
                repetitions
            ],
            "seed": [seed],
            "observed_effect_bps": [
                observed
            ],
            "p025_bps": [
                float(
                    np.quantile(
                        values,
                        0.025,
                    )
                )
            ],
            "p975_bps": [
                float(
                    np.quantile(
                        values,
                        0.975,
                    )
                )
            ],
        }
    )


def symbol_decomposition(
    df: pd.DataFrame,
    selected: pd.Series,
) -> pd.DataFrame:
    rows = []

    for symbol, group in (
        df.groupby(
            "symbol",
            sort=True,
        )
    ):
        group_selected = (
            selected.loc[
                group.index
            ]
        )

        selected_values = (
            group.loc[
                group_selected,
                "post_fill_return_bps",
            ]
        )

        unselected_values = (
            group.loc[
                ~group_selected,
                "post_fill_return_bps",
            ]
        )

        effect = np.nan

        if (
            not selected_values.empty
            and not unselected_values.empty
        ):
            effect = (
                selected_values.mean()
                - unselected_values.mean()
            )

        rows.append(
            {
                "symbol": symbol,
                "rows": len(group),
                "selected": int(
                    group_selected.sum()
                ),
                "selection_rate": float(
                    group_selected.mean()
                ),
                "selected_mean_bps": (
                    float(
                        selected_values.mean()
                    )
                    if not selected_values.empty
                    else np.nan
                ),
                "unselected_mean_bps": (
                    float(
                        unselected_values.mean()
                    )
                    if not unselected_values.empty
                    else np.nan
                ),
                "effect_bps": (
                    float(effect)
                    if not pd.isna(effect)
                    else np.nan
                ),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
    )

    parser.add_argument(
        "--permutation-reps",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--bootstrap-reps",
        type=int,
        default=2000,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=20260916,
    )

    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(
            "research/m8_validation_outputs"
        ),
    )

    args = parser.parse_args()

    args.out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = prepare_data(
        pd.read_csv(
            args.dataset
        )
    )

    selected = selected_mask(
        df,
        args.threshold,
    )

    primary_effect = (
        selected_minus_unselected_effect(
            df,
            selected,
        )
    )

    print()
    print("M8 EMPIRICAL VALIDATION")
    print("=" * 80)

    print(
        f"Dataset: {args.dataset}"
    )

    print(
        f"Rows: {len(df):,}"
    )

    print(
        f"Frozen threshold: "
        f"{args.threshold:.8f}"
    )

    print(
        f"Selected: "
        f"{int(selected.sum()):,}"
    )

    print(
        f"Selection rate: "
        f"{selected.mean():.4f}"
    )

    print(
        f"Selected - unselected: "
        f"{primary_effect:+.6f} bps"
    )

    temporal = temporal_placebo(
        df,
        args.threshold,
    )

    permutation = (
        permutation_placebo(
            df,
            args.threshold,
            args.permutation_reps,
            args.seed,
        )
    )

    symbols = symbol_decomposition(
        df,
        selected,
    )

    bootstrap_results = []

    for block_minutes in (
        1,
        5,
        10,
    ):
        result = block_bootstrap(
            df,
            args.threshold,
            block_minutes,
            args.bootstrap_reps,
            args.seed
            + block_minutes,
        )

        bootstrap_results.append(
            result.iloc[0]
        )

    bootstrap = pd.DataFrame(
        bootstrap_results
    )

    temporal.to_csv(
        args.out_dir
        / "temporal_placebo.csv",
        index=False,
    )

    permutation.to_csv(
        args.out_dir
        / "permutation_placebo.csv",
        index=False,
    )

    symbols.to_csv(
        args.out_dir
        / "symbol_decomposition.csv",
        index=False,
    )

    bootstrap.to_csv(
        args.out_dir
        / "bootstrap_effect.csv",
        index=False,
    )

    print()
    print("Temporal placebo")
    print("-" * 80)
    print(
        temporal.to_string(
            index=False
        )
    )

    print()
    print("Permutation placebo")
    print("-" * 80)
    print(
        permutation.to_string(
            index=False
        )
    )

    print()
    print("Symbol decomposition")
    print("-" * 80)
    print(
        symbols.to_string(
            index=False
        )
    )

    print()
    print("Block bootstrap")
    print("-" * 80)
    print(
        bootstrap.to_string(
            index=False
        )
    )

    print()
    print(
        "Outputs written to: "
        f"{args.out_dir}"
    )


if __name__ == "__main__":
    main()