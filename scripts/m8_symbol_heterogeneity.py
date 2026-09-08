from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


DATASET = "C:/Ritanshu/Project/MERIT/data/sample/M8_replication_research_dataset.csv"
HORIZON = "1s"
BLOCK_SIZE = 10
BOOTSTRAP_REPS = 1000
SEED = 42


@dataclass(frozen=True)
class SymbolEffect:
    symbol: str
    n: int
    beta_bps_per_pct: float
    bootstrap_se: float
    ci_low: float
    ci_high: float
    median_relative_spread_pct: float
    median_spread: float


def slope(x: np.ndarray, y: np.ndarray) -> float:
    x_centered = x - x.mean()
    denominator = np.sum(x_centered * x_centered)

    if denominator == 0:
        return np.nan

    return float(np.sum(x_centered * (y - y.mean())) / denominator)


def block_bootstrap_slope(
    x: np.ndarray,
    y: np.ndarray,
    block_size: int,
    reps: int,
    rng: np.random.Generator,
) -> np.ndarray:
    n = len(x)

    if n < 2 or np.ptp(x) == 0:
        return np.array([])

    blocks = [
        np.arange(start, min(start + block_size, n))
        for start in range(0, n, block_size)
    ]

    estimates = np.empty(reps)

    for i in range(reps):
        sampled_blocks: list[np.ndarray] = []

        while sum(len(block) for block in sampled_blocks) < n:
            sampled_blocks.append(
                blocks[rng.integers(0, len(blocks))]
            )

        indices = np.concatenate(sampled_blocks)[:n]

        estimates[i] = slope(x[indices], y[indices])

    return estimates[np.isfinite(estimates)]


def analyze_symbol(
    symbol_df: pd.DataFrame,
    rng: np.random.Generator,
) -> SymbolEffect | None:
    symbol = str(symbol_df["symbol"].iloc[0])

    df = symbol_df.copy()

    df = df[
        df["relative_spread"].notna()
        & df["markout_1s"].notna()
        & df["fill_price"].notna()
        & df["mid_price"].notna()
        & (df["fill_price"] > 0)
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

    if len(df) < 20:
        return None

    # Independent variables:
    # relative spread in percentage points.
    x = df["relative_spread"].to_numpy(dtype=float) * 100.0

    # Dependent variable:
    # post-fill movement as return, expressed in basis points.
    y = (
        df["post_fill_move_1s"].to_numpy(dtype=float)
        / df["fill_price"].to_numpy(dtype=float)
        * 10000.0
    )

    beta = slope(x, y)

    bootstrap = block_bootstrap_slope(
        x=x,
        y=y,
        block_size=BLOCK_SIZE,
        reps=BOOTSTRAP_REPS,
        rng=rng,
    )

    if len(bootstrap) < 50:
        return None

    bootstrap_se = float(np.std(bootstrap, ddof=1))
    ci_low, ci_high = np.percentile(bootstrap, [2.5, 97.5])

    return SymbolEffect(
        symbol=symbol,
        n=len(df),
        beta_bps_per_pct=beta,
        bootstrap_se=bootstrap_se,
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        median_relative_spread_pct=float(
            df["relative_spread"].median() * 100.0
        ),
        median_spread=float(df["spread"].median()),
    )


def random_effects(effects: pd.DataFrame) -> dict[str, float]:
    beta = effects["beta_bps_per_pct"].to_numpy(dtype=float)
    se = effects["bootstrap_se"].to_numpy(dtype=float)

    valid = np.isfinite(beta) & np.isfinite(se) & (se > 0)

    beta = beta[valid]
    se = se[valid]

    k = len(beta)

    if k < 2:
        raise ValueError("Need at least two symbols for meta-analysis.")

    fixed_weights = 1.0 / (se ** 2)

    fixed_beta = np.sum(fixed_weights * beta) / np.sum(fixed_weights)

    q = np.sum(
        fixed_weights * (beta - fixed_beta) ** 2
    )

    df_q = k - 1

    denominator = (
        np.sum(fixed_weights)
        - np.sum(fixed_weights ** 2) / np.sum(fixed_weights)
    )

    tau2 = max(
        0.0,
        (q - df_q) / denominator,
    )

    random_weights = 1.0 / (se ** 2 + tau2)

    random_beta = (
        np.sum(random_weights * beta)
        / np.sum(random_weights)
    )

    random_se = np.sqrt(1.0 / np.sum(random_weights))

    ci_low = random_beta - 1.96 * random_se
    ci_high = random_beta + 1.96 * random_se

    if q > 0:
        i2 = max(
            0.0,
            (q - df_q) / q * 100.0,
        )
    else:
        i2 = 0.0

    prediction_se = np.sqrt(random_se**2 + tau2)

    prediction_low = random_beta - 1.96 * prediction_se
    prediction_high = random_beta + 1.96 * prediction_se

    return {
        "fixed_effect": fixed_beta,
        "random_effect": random_beta,
        "random_se": random_se,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "tau2": tau2,
        "Q": q,
        "I2_percent": i2,
        "prediction_low": prediction_low,
        "prediction_high": prediction_high,
    }


def main() -> None:
    rng = np.random.default_rng(SEED)

    df = pd.read_csv(DATASET)

    required = {
        "symbol",
        "relative_spread",
        "markout_1s",
        "fill_price",
        "mid_price",
        "spread",
        "side",
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

    df = df.sort_values(
        ["symbol", "timestamp"],
        kind="stable",
    )

    effects: list[SymbolEffect] = []

    for symbol, symbol_df in df.groupby("symbol", sort=True):
        result = analyze_symbol(symbol_df, rng)

        if result is not None:
            effects.append(result)

    if not effects:
        raise ValueError("No symbol produced a valid estimate.")

    effects_df = pd.DataFrame(
        [
            {
                "symbol": effect.symbol,
                "n": effect.n,
                "beta_bps_per_pct": effect.beta_bps_per_pct,
                "bootstrap_se": effect.bootstrap_se,
                "ci_low": effect.ci_low,
                "ci_high": effect.ci_high,
                "median_relative_spread_pct": effect.median_relative_spread_pct,
                "median_spread": effect.median_spread,
            }
            for effect in effects
        ]
    )

    equal_weighted = effects_df["beta_bps_per_pct"].mean()

    fill_weights = effects_df["n"].to_numpy(dtype=float)
    fill_weighted = np.average(
        effects_df["beta_bps_per_pct"].to_numpy(dtype=float),
        weights=fill_weights,
    )

    meta = random_effects(effects_df)

    print("\nR4 SYMBOL HETEROGENEITY")
    print("=" * 72)
    print(
        "Effect: post-fill return (bps) per +1 percentage-point "
        "relative spread"
    )
    print(
        f"Bootstrap: block size={BLOCK_SIZE}, "
        f"reps={BOOTSTRAP_REPS}, seed={SEED}"
    )

    print("\nPer-symbol estimates")
    print("-" * 72)

    for _, row in effects_df.iterrows():
        print(
            f"{row['symbol']:>5} | "
            f"n={int(row['n']):>5} | "
            f"beta={row['beta_bps_per_pct']:+.4f} | "
            f"95% CI=[{row['ci_low']:+.4f}, "
            f"{row['ci_high']:+.4f}] | "
            f"median spread={row['median_spread']:.4f}"
        )

    print("\nAggregate estimates")
    print("-" * 72)
    print(f"Equal-weighted mean : {equal_weighted:+.4f}")
    print(f"Fill-weighted mean  : {fill_weighted:+.4f}")

    print("\nRandom-effects meta-analysis")
    print("-" * 72)
    print(f"Fixed-effect estimate : {meta['fixed_effect']:+.4f}")
    print(f"Random-effect estimate: {meta['random_effect']:+.4f}")
    print(
        f"95% CI                : "
        f"[{meta['ci_low']:+.4f}, {meta['ci_high']:+.4f}]"
    )
    print(f"Tau²                  : {meta['tau2']:.6f}")
    print(f"Q                     : {meta['Q']:.4f}")
    print(f"I²                    : {meta['I2_percent']:.2f}%")
    print(
        f"95% prediction interval: "
        f"[{meta['prediction_low']:+.4f}, "
        f"{meta['prediction_high']:+.4f}]"
    )

    output = "M8_symbol_heterogeneity.csv"
    effects_df.to_csv(output, index=False)

    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()