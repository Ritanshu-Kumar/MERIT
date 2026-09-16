from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_DATASET = Path(
    "data/itch/M8_ITCH_2019-07-30_events_corrected.csv"
)

THRESHOLD = 0.00957966
LAGS = (0, 1, 5, 20, 100)
PERMUTATION_REPS = 1000
SEED = 20260916


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    required = {
        "timestamp_ns",
        "symbol",
        "quote_side",
        "relative_spread",
        "imbalance",
        "return_bps_1s",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    result = df.copy()

    result = result.dropna(
        subset=[
            "relative_spread",
            "imbalance",
            "return_bps_1s",
        ]
    ).copy()

    result["timestamp_ns"] = pd.to_numeric(
        result["timestamp_ns"],
        errors="coerce",
    )

    result = result.dropna(
        subset=["timestamp_ns"]
    )

    result["signed_imbalance"] = np.where(
        result["quote_side"].eq("BUY"),
        result["imbalance"],
        -result["imbalance"],
    )

    result["relative_spread_pct"] = (
        result["relative_spread"] * 100.0
    )

    return (
        result.sort_values(
            ["timestamp_ns", "symbol"],
            kind="stable",
        )
        .reset_index(drop=True)
    )


def selected_minus_unselected(
    target: pd.Series,
    selected: pd.Series,
) -> float:
    if selected.sum() == 0:
        raise ValueError("No selected observations.")

    if (~selected).sum() == 0:
        raise ValueError("No unselected observations.")

    return float(
        target.loc[selected].mean()
        - target.loc[~selected].mean()
    )


def temporal_placebo(
    df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for lag in LAGS:
        if lag == 0:
            lagged_signed = (
                df["signed_imbalance"]
                .copy()
            )
        else:
            lagged_signed = (
                df.groupby(
                    "symbol",
                    sort=False,
                )["signed_imbalance"]
                .shift(lag)
            )

        usable = lagged_signed.notna()

        score = (
            df["relative_spread_pct"]
            * lagged_signed
        )

        selected = (
            score >= THRESHOLD
        ) & usable

        effect = selected_minus_unselected(
            df.loc[usable, "return_bps_1s"]
            .reset_index(drop=True),
            selected.loc[usable]
            .reset_index(drop=True),
        )

        rows.append(
            {
                "lag_events": lag,
                "n": int(usable.sum()),
                "selected": int(selected.sum()),
                "selection_rate": float(
                    selected.sum() / usable.sum()
                ),
                "effect_bps": effect,
            }
        )

    return pd.DataFrame(rows)


def permutation_placebo(
    df: pd.DataFrame,
) -> dict[str, float]:
    rng = np.random.default_rng(SEED)

    symbols = df["symbol"].to_numpy()

    original = (
        df["signed_imbalance"]
        .to_numpy()
        .copy()
    )

    effects = []

    observed_score = (
        df["relative_spread_pct"].to_numpy()
        * original
    )

    observed_selected = (
        observed_score >= THRESHOLD
    )

    observed = selected_minus_unselected(
        df["return_bps_1s"],
        pd.Series(
            observed_selected,
            index=df.index,
        ),
    )

    for _ in range(PERMUTATION_REPS):
        shuffled = original.copy()

        for symbol in np.unique(symbols):
            indices = np.flatnonzero(
                symbols == symbol
            )

            values = shuffled[indices].copy()
            rng.shuffle(values)
            shuffled[indices] = values

        score = (
            df["relative_spread_pct"].to_numpy()
            * shuffled
        )

        selected = (
            score >= THRESHOLD
        )

        effects.append(
            selected_minus_unselected(
                df["return_bps_1s"],
                pd.Series(
                    selected,
                    index=df.index,
                ),
            )
        )

    values = np.asarray(
        effects,
        dtype=float,
    )

    return {
        "observed": float(observed),
        "mean": float(values.mean()),
        "p025": float(
            np.quantile(values, 0.025)
        ),
        "p50": float(
            np.quantile(values, 0.50)
        ),
        "p975": float(
            np.quantile(values, 0.975)
        ),
        "repetitions": PERMUTATION_REPS,
        "seed": SEED,
    }


def symbol_decomposition(
    df: pd.DataFrame,
) -> pd.DataFrame:
    score = (
        df["relative_spread_pct"]
        * df["signed_imbalance"]
    )

    selected = score >= THRESHOLD

    rows = []

    for symbol, group in df.groupby(
        "symbol",
        sort=True,
    ):
        mask = selected.loc[
            group.index
        ]

        selected_values = group.loc[
            mask,
            "return_bps_1s",
        ]

        unselected_values = group.loc[
            ~mask,
            "return_bps_1s",
        ]

        rows.append(
            {
                "symbol": symbol,
                "n": len(group),
                "selected": int(mask.sum()),
                "selection_rate": float(
                    mask.mean()
                ),
                "selected_mean_bps": float(
                    selected_values.mean()
                ),
                "unselected_mean_bps": float(
                    unselected_values.mean()
                ),
                "effect_bps": float(
                    selected_values.mean()
                    - unselected_values.mean()
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

    args = parser.parse_args()

    df = prepare_data(
        pd.read_csv(args.dataset)
    )

    score = (
        df["relative_spread_pct"]
        * df["signed_imbalance"]
    )

    selected = score >= THRESHOLD

    primary = selected_minus_unselected(
        df["return_bps_1s"],
        selected,
    )

    print()
    print("=== ITCH CANONICAL M8 VALIDATION ===")
    print("=" * 80)
    print(f"rows: {len(df):,}")
    print(f"threshold: {THRESHOLD:.8f}")
    print(
        f"selected: {int(selected.sum()):,}"
    )
    print(
        f"selection rate: {selected.mean():.6f}"
    )
    print(
        f"primary 1s effect: {primary:+.6f} bps"
    )

    print()
    print("=== TEMPORAL PLACEBO ===")
    temporal = temporal_placebo(df)
    print(
        temporal.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    print()
    print("=== SYMBOL DECOMPOSITION ===")
    symbols = symbol_decomposition(df)
    print(
        symbols.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    print()
    print("=== WITHIN-SYMBOL PERMUTATION ===")
    permutation = permutation_placebo(df)

    for key, value in permutation.items():
        if isinstance(value, float):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()