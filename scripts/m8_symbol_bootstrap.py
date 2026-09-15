from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_DATASET = Path(
    "data/sample/M8_replication_research_dataset.csv"
)

DEFAULT_THRESHOLD = 0.00957966


def prepare_data(
    df: pd.DataFrame,
) -> pd.DataFrame:
    required = {
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

    result = result.dropna(
        subset=[
            "symbol",
            "side",
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
        result["relative_spread"]
        * 100.0
    )

    result["interaction_score"] = (
        result["relative_spread_pct"]
        * result["signed_imbalance"]
    )

    result = result.dropna(
        subset=[
            "post_fill_return_bps",
            "interaction_score",
        ]
    ).reset_index(drop=True)

    return result


def effect(
    df: pd.DataFrame,
    threshold: float,
) -> float:
    selected = (
        df["interaction_score"]
        >= threshold
    )

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


def bootstrap_symbol(
    df: pd.DataFrame,
    threshold: float,
    repetitions: int,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)

    observed = effect(
        df,
        threshold,
    )

    values = (
        df[
            "post_fill_return_bps"
        ].to_numpy()
    )

    selected = (
        df["interaction_score"]
        >= threshold
    ).to_numpy()

    bootstrap_effects = []

    n = len(df)

    for _ in range(repetitions):
        indices = rng.integers(
            0,
            n,
            size=n,
        )

        sampled_values = values[
            indices
        ]

        sampled_selected = selected[
            indices
        ]

        if (
            sampled_selected.sum() == 0
            or (~sampled_selected).sum() == 0
        ):
            continue

        selected_mean = (
            sampled_values[
                sampled_selected
            ].mean()
        )

        unselected_mean = (
            sampled_values[
                ~sampled_selected
            ].mean()
        )

        bootstrap_effects.append(
            selected_mean
            - unselected_mean
        )

    values = np.asarray(
        bootstrap_effects,
        dtype=float,
    )

    return {
        "rows": len(df),
        "selected": int(
            selected.sum()
        ),
        "selection_rate": float(
            selected.mean()
        ),
        "observed_effect_bps": observed,
        "p025_bps": float(
            np.quantile(
                values,
                0.025,
            )
        ),
        "p50_bps": float(
            np.quantile(
                values,
                0.50,
            )
        ),
        "p975_bps": float(
            np.quantile(
                values,
                0.975,
            )
        ),
        "bootstrap_repetitions": len(
            values
        ),
        "seed": seed,
    }


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
        "--repetitions",
        type=int,
        default=5000,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=20260930,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "research/m8_validation_outputs/"
            "symbol_bootstrap.csv"
        ),
    )

    args = parser.parse_args()

    df = prepare_data(
        pd.read_csv(
            args.dataset
        )
    )

    rows = []

    for index, (
        symbol,
        group,
    ) in enumerate(
        df.groupby(
            "symbol",
            sort=True,
        )
    ):
        result = bootstrap_symbol(
            group.reset_index(drop=True),
            args.threshold,
            args.repetitions,
            args.seed + index,
        )

        result["symbol"] = symbol

        rows.append(result)

    output = pd.DataFrame(
        rows
    )[
        [
            "symbol",
            "rows",
            "selected",
            "selection_rate",
            "observed_effect_bps",
            "p025_bps",
            "p50_bps",
            "p975_bps",
            "bootstrap_repetitions",
            "seed",
        ]
    ]

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.to_csv(
        args.output,
        index=False,
    )

    print()
    print(
        "M8 PER-SYMBOL BOOTSTRAP"
    )
    print("=" * 90)
    print(
        f"Threshold: "
        f"{args.threshold:.8f}"
    )
    print(
        f"Repetitions: "
        f"{args.repetitions:,}"
    )
    print()
    print(
        output.to_string(
            index=False,
            float_format=lambda value:
                f"{value:.6f}",
        )
    )
    print()
    print(
        f"Saved: {args.output}"
    )


if __name__ == "__main__":
    main()