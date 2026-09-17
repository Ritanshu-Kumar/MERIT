from __future__ import annotations

import numpy as np
import pandas as pd


DATASET = "data/sample/M8_replication_research_dataset.csv"
OUTPUT = "M8_spread_imbalance_interaction.csv"

N_QUINTILES = 5
BOOTSTRAP_REPS = 1000
BLOCK_SIZE = 10
SEED = 42


def slope(x: np.ndarray, y: np.ndarray) -> float:
    x_centered = x - x.mean()
    denominator = np.sum(x_centered * x_centered)

    if denominator == 0:
        return np.nan

    return float(
        np.sum(x_centered * (y - y.mean())) / denominator
    )


def block_bootstrap_slope(
    x: np.ndarray,
    y: np.ndarray,
    block_size: int,
    reps: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    n = len(x)

    if n < 20 or np.ptp(x) == 0:
        return np.nan, np.nan, np.nan

    blocks = [
        np.arange(start, min(start + block_size, n))
        for start in range(0, n, block_size)
    ]

    estimates = []

    for _ in range(reps):
        sampled_blocks = []

        while sum(len(block) for block in sampled_blocks) < n:
            sampled_blocks.append(
                blocks[rng.integers(0, len(blocks))]
            )

        indices = np.concatenate(sampled_blocks)[:n]

        estimate = slope(x[indices], y[indices])

        if np.isfinite(estimate):
            estimates.append(estimate)

    if len(estimates) < 50:
        return np.nan, np.nan, np.nan

    estimates = np.asarray(estimates)

    return (
        float(np.std(estimates, ddof=1)),
        float(np.percentile(estimates, 2.5)),
        float(np.percentile(estimates, 97.5)),
    )


def interaction_regression(
    x: np.ndarray,
    imbalance: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    x_mean = x.mean()
    imbalance_mean = imbalance.mean()

    x_centered = x - x_mean
    imbalance_centered = imbalance - imbalance_mean

    interaction = x_centered * imbalance_centered

    design = np.column_stack(
        [
            np.ones(len(x)),
            x_centered,
            imbalance_centered,
            interaction,
        ]
    )

    coefficients, *_ = np.linalg.lstsq(
        design,
        y,
        rcond=None,
    )

    return coefficients


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

    df = df.copy()

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
    )

    df = df.dropna(
        subset=[
            "timestamp",
            "fill_price",
            "mid_price",
            "relative_spread",
            "imbalance",
            "markout_1s",
        ]
    )

    df = df[
        (df["fill_price"] > 0)
        & (df["mid_price"] > 0)
    ].copy()

    df["initial_edge"] = np.where(
        df["side"].eq("BUY"),
        df["mid_price"] - df["fill_price"],
        df["fill_price"] - df["mid_price"],
    )

    df["post_fill_move_1s"] = (
        df["markout_1s"] - df["initial_edge"]
    )

    df["relative_spread_pct"] = (
        df["relative_spread"] * 100.0
    )

    df["post_fill_return_bps"] = (
        df["post_fill_move_1s"]
        / df["fill_price"]
        * 10000.0
    )

    return df.sort_values(
        ["symbol", "timestamp"],
        kind="stable",
    )


def main() -> None:
    rng = np.random.default_rng(SEED)

    df = prepare_data(
        pd.read_csv(DATASET)
    )

    results = []
    interaction_results = []

    print("\nR6 SPREAD × IMBALANCE INTERACTION")
    print("=" * 90)
    print(
        "Target: post-fill return (bps) at 1s"
    )
    print(
        "Effect: relative spread (%) → post-fill return"
    )
    print(
        f"Imbalance buckets: {N_QUINTILES} quintiles | "
        f"bootstrap reps={BOOTSTRAP_REPS} | "
        f"block size={BLOCK_SIZE}"
    )

    for symbol, symbol_df in df.groupby(
        "symbol",
        sort=True,
    ):
        symbol_df = symbol_df.copy()

        # Symbol-specific imbalance quintiles.
        # This avoids cross-symbol scale differences.
        symbol_df["imbalance_quintile"] = pd.qcut(
            symbol_df["imbalance"],
            q=N_QUINTILES,
            labels=False,
            duplicates="drop",
        )

        print(f"\n{symbol}")
        print("-" * 90)

        for quintile, bucket in symbol_df.groupby(
            "imbalance_quintile",
            sort=True,
        ):
            x = bucket[
                "relative_spread_pct"
            ].to_numpy(dtype=float)

            y = bucket[
                "post_fill_return_bps"
            ].to_numpy(dtype=float)

            beta = slope(x, y)

            bootstrap_se, ci_low, ci_high = (
                block_bootstrap_slope(
                    x=x,
                    y=y,
                    block_size=BLOCK_SIZE,
                    reps=BOOTSTRAP_REPS,
                    rng=rng,
                )
            )

            results.append(
                {
                    "symbol": symbol,
                    "imbalance_quintile": int(quintile) + 1,
                    "n": len(bucket),
                    "mean_imbalance": bucket[
                        "imbalance"
                    ].mean(),
                    "mean_relative_spread_pct": bucket[
                        "relative_spread_pct"
                    ].mean(),
                    "beta_bps_per_pct": beta,
                    "bootstrap_se": bootstrap_se,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                }
            )

            print(
                f"  Q{int(quintile) + 1}: "
                f"n={len(bucket):>5} | "
                f"mean imbalance="
                f"{bucket['imbalance'].mean():+.4f} | "
                f"beta={beta:+.4f} | "
                f"95% CI=[{ci_low:+.4f}, "
                f"{ci_high:+.4f}]"
            )

        # Continuous interaction model.
        x = symbol_df[
            "relative_spread_pct"
        ].to_numpy(dtype=float)

        imbalance = symbol_df[
            "imbalance"
        ].to_numpy(dtype=float)

        y = symbol_df[
            "post_fill_return_bps"
        ].to_numpy(dtype=float)

        if len(symbol_df) >= 20:
            coefficients = interaction_regression(
                x=x,
                imbalance=imbalance,
                y=y,
            )

            beta_spread = coefficients[1]
            beta_imbalance = coefficients[2]
            beta_interaction = coefficients[3]

            interaction_results.append(
                {
                    "symbol": symbol,
                    "n": len(symbol_df),
                    "spread_beta": beta_spread,
                    "imbalance_beta": beta_imbalance,
                    "interaction_beta": beta_interaction,
                }
            )

            print("\n  Continuous interaction:")
            print(
                f"    spread beta      = "
                f"{beta_spread:+.6f}"
            )
            print(
                f"    imbalance beta   = "
                f"{beta_imbalance:+.6f}"
            )
            print(
                f"    spread × imbalance= "
                f"{beta_interaction:+.6f}"
            )

    bucket_df = pd.DataFrame(results)
    interaction_df = pd.DataFrame(
        interaction_results
    )

    print("\n")
    print("=" * 90)
    print("R6 INTERACTION SUMMARY")
    print("=" * 90)

    print("\nContinuous interaction coefficients")
    print("-" * 90)

    for _, row in interaction_df.iterrows():
        print(
            f"{row['symbol']:>5} | "
            f"spread={row['spread_beta']:+.6f} | "
            f"imbalance={row['imbalance_beta']:+.6f} | "
            f"interaction="
            f"{row['interaction_beta']:+.6f}"
        )

    print("\nInteraction sign")
    print("-" * 90)

    positive = (
        interaction_df["interaction_beta"] > 0
    ).sum()

    negative = (
        interaction_df["interaction_beta"] < 0
    ).sum()

    zero = (
        interaction_df["interaction_beta"] == 0
    ).sum()

    print(
        f"Positive: {positive} | "
        f"Negative: {negative} | "
        f"Zero: {zero}"
    )

    print("\nQuintile monotonicity")
    print("-" * 90)

    for symbol in sorted(
        bucket_df["symbol"].unique()
    ):
        symbol_buckets = bucket_df[
            bucket_df["symbol"] == symbol
        ].sort_values(
            "imbalance_quintile"
        )

        betas = (
            symbol_buckets[
                "beta_bps_per_pct"
            ]
            .to_numpy()
        )

        print(
            f"{symbol}: "
            + "  ".join(
                f"Q{i + 1}={beta:+.3f}"
                for i, beta in enumerate(betas)
            )
        )

    bucket_output = OUTPUT
    interaction_output = (
        "M8_spread_imbalance_interaction_coefficients.csv"
    )

    bucket_df.to_csv(
        bucket_output,
        index=False,
    )

    interaction_df.to_csv(
        interaction_output,
        index=False,
    )

    print(f"\nSaved: {bucket_output}")
    print(f"Saved: {interaction_output}")


if __name__ == "__main__":
    main()