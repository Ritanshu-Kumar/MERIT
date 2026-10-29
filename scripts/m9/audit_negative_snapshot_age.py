from pathlib import Path

import numpy as np
import pandas as pd

PATH = Path("research/m9_time_sampled_2019-08-30_asof.csv")

OPEN_NS = 34_200_000_000_000
CLOSE_NS = 57_600_000_000_000


def main():
    df = pd.read_csv(PATH, low_memory=False)

    for col in ["timestamp_ns", "snapshot_timestamp_ns", "snapshot_age_ns"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["age_ms"] = df["snapshot_age_ns"] / 1e6
    df["snapshot_lead_ms"] = -df["age_ms"]

    ts = df["timestamp_ns"]
    df["session"] = np.select(
        [
            ts < OPEN_NS,
            ts < CLOSE_NS,
        ],
        [
            "pre-open",
            "regular",
        ],
        default="after-close",
    )

    negative = df[df["age_ms"] < 0].copy()

    print("=" * 65)
    print("NEGATIVE SNAPSHOT-AGE DIAGNOSTIC")
    print("=" * 65)

    print("Total rows:", len(df))
    print("Negative-age rows:", len(negative))
    print(
        "Negative-age decision times:",
        negative[["symbol", "timestamp_ns"]]
        .drop_duplicates()
        .shape[0],
    )

    print("\nNEGATIVE-AGE ROWS BY SYMBOL")
    print(negative.groupby("symbol").size().to_string())

    print("\nNEGATIVE-AGE ROWS BY SESSION")
    print(pd.crosstab(negative["session"], negative["symbol"]).to_string())

    print("\nNEGATIVE-AGE PERCENTAGE BY SYMBOL")
    print(
        (100 * df.assign(negative=df["age_ms"] < 0)
         .groupby("symbol")["negative"].mean())
        .round(3)
        .to_string()
    )

    print("\nSNAPSHOT LEAD TIME (ms)")
    print(
        negative["snapshot_lead_ms"]
        .describe(percentiles=[0.5, 0.9, 0.99])
        .round(3)
        .to_string()
    )

    columns = [
        "candidate_id",
        "event_id",
        "symbol",
        "quote_side",
        "timestamp_ns",
        "snapshot_timestamp_ns",
        "snapshot_lead_ms",
    ]
    columns = [c for c in columns if c in negative.columns]

    print("\n20 MOST NEGATIVE AGES")
    print(
        negative.nsmallest(20, "age_ms")[columns]
        .to_string(index=False)
    )

    print("\nFIRST 20 NEGATIVE-AGE ROWS IN FILE ORDER")
    print(negative[columns].head(20).to_string(index=False))


if __name__ == "__main__":
    main()