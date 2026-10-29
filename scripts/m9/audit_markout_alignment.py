from pathlib import Path

import pandas as pd

PATH = Path("research/m9_time_sampled_2019-07-30_regular_asof.csv")
CLOSE_NS = 57_600_000_000_000

HORIZONS = {
    "10ms": 10_000_000,
    "100ms": 100_000_000,
    "500ms": 500_000_000,
    "1s": 1_000_000_000,
    "5s": 5_000_000_000,
}


def main() -> None:
    if not PATH.exists():
        raise FileNotFoundError(f"Cannot find {PATH}")

    df = pd.read_csv(PATH, low_memory=False)
    fills = df[df["outcome"].eq("FILL")].copy()

    fill_ts = pd.to_numeric(
        fills["fill_timestamp_ns"], errors="coerce"
    )

    print("=" * 60)
    print("M9 FILLED-CANDIDATE MARKOUT AUDIT")
    print("=" * 60)
    print("Regular-session rows:", len(df))
    print("FILL rows:", len(fills))
    print("Missing fill timestamps:", int(fill_ts.isna().sum()))

    print("\nMARKOUT COMPLETENESS AND TIMESTAMP ALIGNMENT")

    for horizon, offset_ns in HORIZONS.items():
        mid_col = f"future_mid_{horizon}"
        ts_col = f"future_mid_{horizon}_timestamp_ns"

        if mid_col not in fills.columns or ts_col not in fills.columns:
            print(f"\n{horizon}: required columns are missing")
            continue

        mid = pd.to_numeric(fills[mid_col], errors="coerce")
        ts = pd.to_numeric(fills[ts_col], errors="coerce")

        valid = mid.notna() & ts.notna() & fill_ts.notna()

        # A future midpoint should be observed at or after
        # the requested horizon measured from the fill time.
        target_ts = fill_ts + offset_ns
        delay_ns = ts[valid] - target_ts[valid]

        median_delay_ms = (
            round(float(delay_ns.median()) / 1_000_000, 3)
            if len(delay_ns)
            else None
        )
        p90_delay_ms = (
            round(float(delay_ns.quantile(0.90)) / 1_000_000, 3)
            if len(delay_ns)
            else None
        )

        print(f"\n{horizon}")
        print("  Missing midpoint:", int(mid.isna().sum()))
        print("  Missing midpoint timestamp:", int(ts.isna().sum()))
        print("  Valid midpoint/timestamp pairs:", int(valid.sum()))
        print(
            "  Timestamp before target:",
            int(delay_ns.lt(0).sum()),
        )
        print(
            "  Timestamp exactly at target:",
            int(delay_ns.eq(0).sum()),
        )
        print(
            "  Timestamp at/after regular close:",
            int(ts.ge(CLOSE_NS).sum()),
        )
        print("  Median delay after target (ms):", median_delay_ms)
        print("  90th-percentile delay (ms):", p90_delay_ms)

    print("\nFILL COUNTS BY SYMBOL AND SIDE")
    print(pd.crosstab(fills["symbol"], fills["quote_side"]).to_string())

    print("\nAudit complete.")


if __name__ == "__main__":
    main()