from __future__ import annotations

import numpy as np
import pandas as pd


DATASET = "data/sample/M8_replication_research_dataset.csv"
OUTPUT = "M8_intraday_stability.csv"

N_BLOCKS = 5
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
        sampled = []

        while sum(len(block) for block in sampled) < n:
            sampled.append(
                blocks[rng.integers(0, len(blocks))]
            )

        indices = np.concatenate(sampled)[:n]

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


def main() -> None:
    rng = np.random.default_rng(SEED)

    df = pd.read_csv(DATASET)

    required = {
        "timestamp",
        "symbol",
        "side",
        "fill_price",
        "mid_price",
        "relative_spread",
        "markout_1s",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

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
            "markout_1s",
        ]
    ).copy()

    df = df[
        (df["fill_price"] > 0)
        & (df["mid_price"] > 0)
    ].copy()

    # Reconstruct the same post-fill movement target used in R4.
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

    results = []

    for symbol, symbol_df in df.groupby("symbol", sort=True):
        symbol_df = symbol_df.sort_values(
            "timestamp",
            kind="stable",
        ).copy()

        n = len(symbol_df)

        # Equal-count chronological blocks.
        symbol_df["intraday_block"] = pd.qcut(
            np.arange(n),
            q=N_BLOCKS,
            labels=False,
        ) + 1

        for block_id, block_df in symbol_df.groupby(
            "intraday_block",
            sort=True,
        ):
            block_df = block_df.copy()

            x = block_df["relative_spread_pct"].to_numpy(
                dtype=float
            )
            y = block_df["post_fill_return_bps"].to_numpy(
                dtype=float
            )

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
                    "intraday_block": int(block_id),
                    "n": len(block_df),
                    "start_timestamp": block_df[
                        "timestamp"
                    ].min(),
                    "end_timestamp": block_df[
                        "timestamp"
                    ].max(),
                    "mean_relative_spread_pct": (
                        block_df["relative_spread_pct"].mean()
                    ),
                    "median_relative_spread_pct": (
                        block_df["relative_spread_pct"].median()
                    ),
                    "beta_bps_per_pct": beta,
                    "bootstrap_se": bootstrap_se,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                }
            )

    results_df = pd.DataFrame(results)

    print("\nR5 INTRADAY STABILITY")
    print("=" * 90)
    print(
        "Effect: post-fill return (bps) per +1 percentage-point "
        "relative spread"
    )
    print(
        f"Chronological blocks: {N_BLOCKS} per symbol | "
        f"bootstrap reps={BOOTSTRAP_REPS} | block size={BLOCK_SIZE}"
    )

    print("\nPer-symbol intraday estimates")
    print("-" * 90)

    for symbol in sorted(results_df["symbol"].unique()):
        print(f"\n{symbol}")

        symbol_results = results_df[
            results_df["symbol"] == symbol
        ].sort_values("intraday_block")

        for _, row in symbol_results.iterrows():
            print(
                f"  Block {int(row['intraday_block'])}: "
                f"n={int(row['n']):>5} | "
                f"beta={row['beta_bps_per_pct']:+.4f} | "
                f"95% CI=[{row['ci_low']:+.4f}, "
                f"{row['ci_high']:+.4f}]"
            )

    print("\nSign consistency")
    print("-" * 90)

    for symbol in sorted(results_df["symbol"].unique()):
        symbol_results = results_df[
            results_df["symbol"] == symbol
        ]

        valid = symbol_results[
            symbol_results["beta_bps_per_pct"].notna()
        ]

        negative = (
            valid["beta_bps_per_pct"] < 0
        ).sum()

        positive = (
            valid["beta_bps_per_pct"] > 0
        ).sum()

        zero = (
            valid["beta_bps_per_pct"] == 0
        ).sum()

        print(
            f"{symbol}: "
            f"negative={negative}, "
            f"positive={positive}, "
            f"zero={zero}"
        )

    print("\nCross-block dispersion")
    print("-" * 90)

    for symbol in sorted(results_df["symbol"].unique()):
        values = results_df.loc[
            results_df["symbol"] == symbol,
            "beta_bps_per_pct",
        ].dropna()

        if len(values) >= 2:
            print(
                f"{symbol}: "
                f"mean={values.mean():+.4f} | "
                f"std={values.std(ddof=1):.4f} | "
                f"min={values.min():+.4f} | "
                f"max={values.max():+.4f}"
            )

    results_df.to_csv(OUTPUT, index=False)

    print(f"\nSaved: {OUTPUT}")


if __name__ == "__main__":
    main()