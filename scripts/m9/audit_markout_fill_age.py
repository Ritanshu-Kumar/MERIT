from pathlib import Path

import pandas as pd

PATH = Path("research/m9_time_sampled_2019-07-30_regular_asof.csv")

HORIZONS = {
    "10ms": 10_000_000,
    "100ms": 100_000_000,
    "500ms": 500_000_000,
    "1s": 1_000_000_000,
    "5s": 5_000_000_000,
}


def main():
    df = pd.read_csv(PATH, low_memory=False)
    fills = df[df["outcome"].eq("FILL")].copy()

    fill_ts = pd.to_numeric(
        fills["fill_timestamp_ns"], errors="coerce"
    )

    print("FILL rows:", len(fills))

    for horizon, offset_ns in HORIZONS.items():
        mark_ts = pd.to_numeric(
            fills[f"future_mid_{horizon}_timestamp_ns"],
            errors="coerce",
        )
        mark_mid = pd.to_numeric(
            fills[f"future_mid_{horizon}"],
            errors="coerce",
        )

        valid = fill_ts.notna() & mark_ts.notna() & mark_mid.notna()
        elapsed_ms = (mark_ts[valid] - fill_ts[valid]) / 1e6
        lag_ms = (
            fill_ts[valid] + offset_ns - mark_ts[valid]
        ) / 1e6

        print(f"\n{horizon}")
        print("Valid marks:", int(valid.sum()))
        print("Marks before fill:", int((mark_ts[valid] < fill_ts[valid]).sum()))
        print("Marks equal to fill timestamp:", int((mark_ts[valid] == fill_ts[valid]).sum()))
        print("Marks after target:", int((lag_ms < 0).sum()))
        print("Elapsed time from fill (ms):")
        print(elapsed_ms.describe(percentiles=[0.5, 0.9, 0.99]).round(3))
        print("Gap before target (ms):")
        print(lag_ms.describe(percentiles=[0.5, 0.9, 0.99]).round(3))


if __name__ == "__main__":
    main()