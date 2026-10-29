from pathlib import Path

import numpy as np
import pandas as pd

SOURCE = Path("research/m9_2019-08-30_regular_fill_markouts.csv")
OUTPUT = Path("research/m9_2019-08-30_regular_markout_freshness_sensitivity.csv")

OPEN_NS = 34_200_000_000_000
CLOSE_NS = 57_600_000_000_000
BLOCK_NS = 5 * 60 * 1_000_000_000
N_BLOCKS = (CLOSE_NS - OPEN_NS) // BLOCK_NS

N_BOOTSTRAPS = 10_000
SEED = 20261028


def summarize(
    frame: pd.DataFrame,
    label: str,
    symbol: str,
    draws: np.ndarray,
    rng: np.random.Generator,
) -> dict:
    frame = frame.dropna(
        subset=[
            "fill_timestamp_ns",
            "snapshot_age_ns",
            "markout_1s_usd_per_share",
        ]
    ).copy()

    n = len(frame)

    if n == 0:
        return {
            "sample": label,
            "symbol": symbol,
            "n_fills": 0,
            "mean_markout": np.nan,
            "median_markout": np.nan,
            "positive_markout_pct": np.nan,
            "block_ci95_low": np.nan,
            "block_ci95_high": np.nan,
        }

    frame["block_id"] = (
        (frame["fill_timestamp_ns"] - OPEN_NS) // BLOCK_NS
    ).astype(int)

    if not frame["block_id"].between(0, N_BLOCKS - 1).all():
        raise ValueError("Fill timestamp outside regular-session blocks.")

    stats = frame.groupby("block_id")[
        "markout_1s_usd_per_share"
    ].agg(["sum", "count"])

    sums = np.zeros(N_BLOCKS, dtype=float)
    counts = np.zeros(N_BLOCKS, dtype=float)

    ids = stats.index.to_numpy(dtype=int)
    sums[ids] = stats["sum"].to_numpy(dtype=float)
    counts[ids] = stats["count"].to_numpy(dtype=float)

    boot_sums = sums[draws].sum(axis=1)
    boot_counts = counts[draws].sum(axis=1)

    valid = boot_counts > 0
    boot_means = boot_sums[valid] / boot_counts[valid]

    low, high = np.quantile(boot_means, [0.025, 0.975])

    return {
        "sample": label,
        "symbol": symbol,
        "n_fills": n,
        "mean_markout": frame["markout_1s_usd_per_share"].mean(),
        "median_markout": frame["markout_1s_usd_per_share"].median(),
        "positive_markout_pct": (
            100 * frame["markout_1s_usd_per_share"].gt(0).mean()
        ),
        "block_ci95_low": low,
        "block_ci95_high": high,
    }


def main() -> None:
    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)

    df = pd.read_csv(SOURCE, low_memory=False)

    for col in [
        "fill_timestamp_ns",
        "snapshot_age_ns",
        "markout_1s_usd_per_share",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[
        df["fill_timestamp_ns"].ge(OPEN_NS)
        & df["fill_timestamp_ns"].lt(CLOSE_NS)
    ].copy()

    fresh = df["snapshot_age_ns"].le(1_000_000_000)

    rng = np.random.default_rng(SEED)
    draws = rng.integers(
        0,
        N_BLOCKS,
        size=(N_BOOTSTRAPS, N_BLOCKS),
    )

    results = []

    for label, mask in [
        ("all_valid_fills", pd.Series(True, index=df.index)),
        ("snapshot_age_le_1s", fresh),
    ]:
        sample = df.loc[mask].copy()

        groups = [("ALL", sample)]
        groups.extend(
            (symbol, group)
            for symbol, group in sample.groupby("symbol")
        )

        for symbol, group in groups:
            results.append(
                summarize(
                    group,
                    label,
                    symbol,
                    draws,
                    rng,
                )
            )

    result = pd.DataFrame(results)
    result.to_csv(OUTPUT, index=False)

    print("Saved:", OUTPUT)
    print("Five-minute block bootstrap, 10,000 replicates")
    print(
        result[
            [
                "sample",
                "symbol",
                "n_fills",
                "mean_markout",
                "median_markout",
                "positive_markout_pct",
                "block_ci95_low",
                "block_ci95_high",
            ]
        ].round(6).to_string(index=False)
    )


if __name__ == "__main__":
    main()