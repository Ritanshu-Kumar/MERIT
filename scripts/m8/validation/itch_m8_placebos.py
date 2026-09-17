from __future__ import annotations

import argparse

import numpy as np
import pandas as pd


DEFAULT_PATH = "data/itch/M8_ITCH_2019-07-30_events.csv"

PRIMARY_TARGET = "return_bps_1s"
LAGS = [0, 1, 5, 20, 100]

PERMUTATIONS = 1000
SEED = 20260916


def selected_effect(
    selected: pd.Series,
    target: pd.Series,
) -> float:
    selected_mean = target[selected == 1].mean()
    unselected_mean = target[selected == 0].mean()
    return float(selected_mean - unselected_mean)


def temporal_placebo(
    df: pd.DataFrame,
    lags: list[int],
) -> pd.DataFrame:
    rows = []

    for lag in lags:
        effects = []

        for symbol, group in df.groupby("symbol", sort=False):
            group = group.reset_index(drop=True)

            if lag == 0:
                selected = group["selected"]
                target = group[PRIMARY_TARGET]
            else:
                selected = group["selected"].iloc[:-lag].reset_index(drop=True)
                target = group[PRIMARY_TARGET].iloc[lag:].reset_index(drop=True)

            valid = target.notna()

            if valid.sum() == 0:
                continue

            effects.append(
                selected_effect(
                    selected[valid],
                    target[valid],
                )
            )

        rows.append(
            {
                "lag": lag,
                "pooled_equal_row_weight": float(
                    selected_effect(
                        (
                            df.groupby("symbol", sort=False)
                            ["selected"]
                            .apply(
                                lambda x: x.iloc[:-lag]
                                if lag > 0
                                else x
                            )
                            .reset_index(drop=True)
                        ),
                        (
                            df.groupby("symbol", sort=False)
                            [PRIMARY_TARGET]
                            .apply(
                                lambda x: x.iloc[lag:]
                                if lag > 0
                                else x
                            )
                            .reset_index(drop=True)
                        ),
                    )
                ),
                "mean_symbol_effect": float(np.mean(effects)),
                "n_symbols": len(effects),
            }
        )

    return pd.DataFrame(rows)


def permutation_placebo(
    df: pd.DataFrame,
    repetitions: int,
    seed: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)

    observed = selected_effect(
        df["selected"],
        df[PRIMARY_TARGET],
    )

    values = np.empty(repetitions)

    # Shuffle selection labels within each symbol.
    symbol_groups = [
        group.index.to_numpy()
        for _, group in df.groupby("symbol", sort=False)
    ]

    for repetition in range(repetitions):
        shuffled = df["selected"].to_numpy(copy=True)

        for indices in symbol_groups:
            local = shuffled[indices].copy()
            rng.shuffle(local)
            shuffled[indices] = local

        values[repetition] = selected_effect(
            pd.Series(shuffled, index=df.index),
            df[PRIMARY_TARGET],
        )

    return {
        "observed": float(observed),
        "mean": float(values.mean()),
        "p025": float(np.quantile(values, 0.025)),
        "p50": float(np.quantile(values, 0.50)),
        "p975": float(np.quantile(values, 0.975)),
        "repetitions": repetitions,
        "seed": seed,
    }


def symbol_effects(
    df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for symbol, group in df.groupby(
        "symbol",
        sort=True,
    ):
        rows.append(
            {
                "symbol": symbol,
                "n": len(group),
                "selected": int(group["selected"].sum()),
                "selection_rate": float(group["selected"].mean()),
                "effect_100ms": selected_effect(
                    group["selected"],
                    group["return_bps_100ms"],
                ),
                "effect_1s": selected_effect(
                    group["selected"],
                    group["return_bps_1s"],
                ),
                "effect_5s": selected_effect(
                    group["selected"],
                    group["return_bps_5s"],
                ),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        default=DEFAULT_PATH,
    )
    parser.add_argument(
        "--permutations",
        type=int,
        default=PERMUTATIONS,
    )
    args = parser.parse_args()

    df = pd.read_csv(args.path)

    df = df.sort_values(
        ["symbol", "timestamp_ns"],
        kind="stable",
    ).reset_index(drop=True)

    print("=== ITCH M8 PLACEBO ANALYSIS ===")
    print(f"dataset: {args.path}")
    print(f"rows:    {len(df):,}")
    print()

    print("=== TEMPORAL PLACEBO ===")
    temporal = temporal_placebo(
        df,
        LAGS,
    )
    print(
        temporal.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    print()
    print("=== SYMBOL EFFECTS ===")
    symbols = symbol_effects(df)
    print(
        symbols.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    print()
    print("=== WITHIN-SYMBOL PERMUTATION ===")

    permutation = permutation_placebo(
        df,
        args.permutations,
        SEED,
    )

    for key, value in permutation.items():
        if isinstance(value, float):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()