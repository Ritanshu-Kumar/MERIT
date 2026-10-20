from pathlib import Path

import pandas as pd

DATA = Path(
    "research/m9_fill_hazard_markouts_2019-07-30_timestamped.csv"
)

HORIZONS_NS = {
    "10ms": 10_000_000,
    "100ms": 100_000_000,
    "500ms": 500_000_000,
    "1s": 1_000_000_000,
}

OVERSHOOT_LIMITS_MS = [25, 50, 100, 250, 500, 1000]


def main():
    if not DATA.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA}")

    df = pd.read_csv(DATA)

    fill_ts = pd.to_numeric(
        df["fill_timestamp_ns"], errors="coerce"
    )

    print(f"Dataset rows: {len(df):,}")

    for horizon, target_ns in HORIZONS_NS.items():
        mark_ts = pd.to_numeric(
            df[f"future_mid_{horizon}_timestamp_ns"],
            errors="coerce",
        )

        valid = mark_ts.notna() & fill_ts.notna()

        overshoot_ms = (
            mark_ts[valid] - fill_ts[valid] - target_ns
        ) / 1_000_000

        print(f"\n--- {horizon} horizon ---")
        print(f"Valid markouts: {len(overshoot_ms):,}")

        if overshoot_ms.empty:
            print("No valid timestamp observations.")
            continue

        print(f"Median overshoot: {overshoot_ms.median():.3f} ms")
        print(
            "95th percentile: "
            f"{overshoot_ms.quantile(0.95):.3f} ms"
        )
        print(f"Maximum overshoot: {overshoot_ms.max():.3f} ms")

        for limit in OVERSHOOT_LIMITS_MS:
            accepted = overshoot_ms <= limit
            count = int(accepted.sum())
            percentage = 100 * accepted.mean()

            print(
                f"Limit {limit:4d} ms: "
                f"{count:6,} candidates "
                f"({percentage:6.2f}%)"
            )


if __name__ == "__main__":
    main()