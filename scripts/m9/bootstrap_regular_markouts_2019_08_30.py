from pathlib import Path

import numpy as np
import pandas as pd

SOURCE = Path("research/m9_2019-08-30_regular_fill_markouts.csv")
OUTPUT = Path("research/m9_2019-08-30_regular_markout_block_bootstrap.csv")

OPEN_NS = 34_200_000_000_000
CLOSE_NS = 57_600_000_000_000

HORIZONS = ["10ms", "100ms", "500ms", "1s", "5s"]
BLOCK_LENGTHS_MIN = [1, 5, 15]

N_BOOTSTRAPS = 10_000
RANDOM_SEED = 20261028

# Five horizons x five symbols = 25 predefined symbol-horizon tests.
FAMILY_SIZE = len(HORIZONS) * 5
FWER_ALPHA = 0.05
BONFERRONI_ALPHA = FWER_ALPHA / FAMILY_SIZE

CI95 = (0.025, 0.975)
CI_BONFERRONI = (
    BONFERRONI_ALPHA / 2,
    1 - BONFERRONI_ALPHA / 2,
)


def bootstrap_means(
    block_sums: np.ndarray,
    block_counts: np.ndarray,
    rng: np.random.Generator,
    replicates: int,
) -> tuple[np.ndarray, float]:
    """Resample whole time blocks and calculate weighted means."""
    n_blocks = len(block_sums)

    draws = rng.integers(
        0,
        n_blocks,
        size=(replicates, n_blocks),
        dtype=np.int16,
    )

    sums = block_sums[draws].sum(axis=1)
    counts = block_counts[draws].sum(axis=1)

    valid = counts > 0
    means = sums[valid] / counts[valid]

    valid_fraction = float(valid.mean())
    return means, valid_fraction


def main() -> None:
    if not SOURCE.exists():
        raise FileNotFoundError(f"Missing input: {SOURCE}")

    df = pd.read_csv(SOURCE, low_memory=False)

    if "outcome" in df.columns:
        df = df[df["outcome"].eq("FILL")].copy()

    df["fill_timestamp_ns"] = pd.to_numeric(
        df["fill_timestamp_ns"], errors="coerce"
    )

    df = df[
        df["fill_timestamp_ns"].ge(OPEN_NS)
        & df["fill_timestamp_ns"].lt(CLOSE_NS)
    ].copy()

    df["fill_timestamp_ns"] = df["fill_timestamp_ns"].astype("int64")

    symbols = sorted(df["symbol"].dropna().unique())
    expected_symbols = 5

    if len(symbols) != expected_symbols:
        raise ValueError(
            f"Expected {expected_symbols} symbols, found {symbols}"
        )

    rng = np.random.default_rng(RANDOM_SEED)
    results = []

    print("Filled candidates:", len(df))
    print("Symbols:", symbols)
    print("Bootstrap replicates:", N_BOOTSTRAPS)
    print("Primary block length: 5 minutes")
    print("Bonferroni family size:", FAMILY_SIZE)
    print(
        "Family-wise 95% confidence procedure: "
        f"{100 * (1 - BONFERRONI_ALPHA):.1f}% "
        "per-test percentile intervals"
    )

    for block_minutes in BLOCK_LENGTHS_MIN:
        block_ns = block_minutes * 60 * 1_000_000_000
        n_blocks = (CLOSE_NS - OPEN_NS) // block_ns

        df["block_id"] = (
            (df["fill_timestamp_ns"] - OPEN_NS) // block_ns
        ).astype("int64")

        if not df["block_id"].between(0, n_blocks - 1).all():
            raise ValueError("Found fills outside the session block range.")

        print(f"\n--- {block_minutes}-minute blocks ({n_blocks} blocks) ---")

        # Generate one common resampling plan for this block length.
        # This preserves contemporaneous cross-symbol dependence.
        for horizon in HORIZONS:
            value_col = f"markout_{horizon}_usd_per_share"
            bps_col = f"markout_{horizon}_bps"

            if value_col not in df.columns:
                raise ValueError(f"Missing column: {value_col}")

            for symbol in symbols:
                group = df[df["symbol"].eq(symbol)].copy()

                group[value_col] = pd.to_numeric(
                    group[value_col], errors="coerce"
                )
                group = group.dropna(subset=[value_col])

                if group.empty:
                    print(f"WARNING: no valid data for {symbol}, {horizon}")
                    continue

                block_stats = group.groupby("block_id")[value_col].agg(
                    ["sum", "count"]
                )

                sums = np.zeros(n_blocks, dtype=float)
                counts = np.zeros(n_blocks, dtype=float)

                ids = block_stats.index.to_numpy(dtype=int)
                sums[ids] = block_stats["sum"].to_numpy(dtype=float)
                counts[ids] = block_stats["count"].to_numpy(dtype=float)

                observed_mean = float(group[value_col].mean())
                n_nonempty_blocks = int(np.count_nonzero(counts))

                boot, valid_fraction = bootstrap_means(
                    sums,
                    counts,
                    rng,
                    N_BOOTSTRAPS,
                )

                if len(boot) < 1000:
                    print(
                        f"WARNING: only {len(boot)} valid bootstrap "
                        f"replicates for {symbol}, {horizon}, "
                        f"{block_minutes}-minute blocks"
                    )
                    continue

                ci95_low, ci95_high = np.quantile(boot, CI95)

                row = {
                    "block_minutes": block_minutes,
                    "horizon": horizon,
                    "symbol": symbol,
                    "n_fills": len(group),
                    "nonempty_time_blocks": n_nonempty_blocks,
                    "mean_markout_usd_per_share": observed_mean,
                    "mean_markout_bps": (
                        float(pd.to_numeric(
                            group[bps_col], errors="coerce"
                        ).mean())
                        if bps_col in group.columns
                        else np.nan
                    ),
                    "bootstrap_se": float(np.std(boot, ddof=1)),
                    "ci95_low": float(ci95_low),
                    "ci95_high": float(ci95_high),
                    "valid_bootstrap_fraction": valid_fraction,
                    "bonferroni_ci_low": np.nan,
                    "bonferroni_ci_high": np.nan,
                    "familywise_interval_excludes_zero": np.nan,
                }

                # The five-minute specification is primary.
                if block_minutes == 5:
                    low, high = np.quantile(boot, CI_BONFERRONI)
                    row["bonferroni_ci_low"] = float(low)
                    row["bonferroni_ci_high"] = float(high)
                    row["familywise_interval_excludes_zero"] = bool(
                        high < 0 or low > 0
                    )

                results.append(row)

                if block_minutes == 5 and horizon == "1s":
                    print(
                        f"{symbol}: n={len(group)}, "
                        f"mean={observed_mean:.6f}, "
                        f"95% CI=[{ci95_low:.6f}, {ci95_high:.6f}]"
                    )

    output = pd.DataFrame(results)
    output.to_csv(OUTPUT, index=False)

    print("\nSaved:", OUTPUT)

    primary = output[
        (output["block_minutes"] == 5)
        & (output["horizon"] == "1s")
    ]

    print("\nPRIMARY HORIZON: 1 SECOND, FIVE-MINUTE BLOCKS")
    if not primary.empty:
        columns = [
            "symbol",
            "n_fills",
            "nonempty_time_blocks",
            "mean_markout_usd_per_share",
            "ci95_low",
            "ci95_high",
            "bonferroni_ci_low",
            "bonferroni_ci_high",
            "familywise_interval_excludes_zero",
            "valid_bootstrap_fraction",
        ]
        print(primary[columns].round(6).to_string(index=False))

    print(
        "\nInterpretation note: block bootstrap intervals estimate "
        "uncertainty within this one trading day. They do not establish "
        "out-of-sample profitability."
    )


if __name__ == "__main__":
    main()