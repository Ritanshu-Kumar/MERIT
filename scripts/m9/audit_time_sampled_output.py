from pathlib import Path

import pandas as pd

PATH = Path("research/m9_time_sampled_2019-07-30_asof.csv")
EXPECTED_INTERVAL_NS = 1_000_000_000

if not PATH.exists():
    raise FileNotFoundError(f"Output file not found: {PATH}")

df = pd.read_csv(PATH, low_memory=False)

print("=" * 65)
print("M9 FULL-DAY TIME-SAMPLING AUDIT")
print("=" * 65)

print("\n1. OUTPUT SIZE")
print("Rows:", len(df))
print("Columns:", len(df.columns))
print("Column names:", df.columns.tolist())

print("\n2. ROW COUNTS")
if "symbol" in df.columns:
    print(df.groupby("symbol").size().to_string())

if "candidate_id" in df.columns:
    print("Duplicate candidate IDs:", int(df["candidate_id"].duplicated().sum()))

if {"symbol", "timestamp_ns", "quote_side"}.issubset(df.columns):
    duplicates = df.duplicated(
        ["symbol", "timestamp_ns", "quote_side"]
    ).sum()
    print("Duplicate symbol/time/side rows:", int(duplicates))

print("\n3. OUTCOME / RESOLUTION DISTRIBUTION")
status_cols = [
    col for col in df.columns
    if col in {"outcome", "label", "resolution_type", "fill_resolution_type"}
]
if status_cols:
    for col in status_cols:
        print(f"\n{col}:")
        print(df[col].value_counts(dropna=False).to_string())
else:
    print("No recognized outcome column found; inspect column names above.")

print("\n4. SNAPSHOT TIMESTAMPS AND AGES")
required = {"timestamp_ns", "snapshot_timestamp_ns"}
if required.issubset(df.columns):
    ts = pd.to_numeric(df["timestamp_ns"], errors="coerce")
    snapshot = pd.to_numeric(
        df["snapshot_timestamp_ns"], errors="coerce"
    )
    age_ns = ts - snapshot
    age_ms = age_ns / 1_000_000

    print("Missing decision timestamps:", int(ts.isna().sum()))
    print("Missing snapshot timestamps:", int(snapshot.isna().sum()))
    print("Negative snapshot ages:", int(age_ns.lt(0).sum()))

    print("\nCalculated snapshot age in milliseconds:")
    print(age_ms.describe(percentiles=[0.50, 0.75, 0.90, 0.99]).round(3))

    for threshold in [1_000, 5_000, 60_000]:
        print(f"Snapshot age > {threshold} ms:", int(age_ms.gt(threshold).sum()))

    if "snapshot_age_ns" in df.columns:
        reported_age = pd.to_numeric(
            df["snapshot_age_ns"], errors="coerce"
        )
        mismatch = (
            age_ns.notna()
            & reported_age.notna()
            & (age_ns - reported_age).abs().gt(1)
        )
        print("Reported/calculated age mismatches:", int(mismatch.sum()))
else:
    print("Required timestamp columns are missing.")

print("\n5. ONE-SECOND GRID")
if {"symbol", "timestamp_ns"}.issubset(df.columns):
    timestamps = df[["symbol", "timestamp_ns"]].copy()
    timestamps["timestamp_ns"] = pd.to_numeric(
        timestamps["timestamp_ns"], errors="coerce"
    )
    timestamps = timestamps.dropna().drop_duplicates()
    timestamps = timestamps.sort_values(["symbol", "timestamp_ns"])

    any_bad = False

    for symbol, group in timestamps.groupby("symbol"):
        delta_ns = group["timestamp_ns"].diff().dropna()
        bad = delta_ns[delta_ns != EXPECTED_INTERVAL_NS]

        print(
            f"{symbol}: unique times={len(group):,}, "
            f"interval errors={len(bad):,}"
        )

        if not bad.empty:
            any_bad = True
            print("  First 10 unexpected intervals (ms):")
            print((bad.head(10) / 1_000_000).to_list())

    print("Grid result:", "REVIEW NEEDED" if any_bad else "PASS")
else:
    print("Cannot validate grid; required columns are missing.")

print("\n6. REQUIRED FIELD MISSINGNESS")
for col in [
    "candidate_id",
    "symbol",
    "timestamp_ns",
    "snapshot_timestamp_ns",
    "snapshot_age_ns",
    "quote_side",
]:
    if col in df.columns:
        print(f"{col}: {int(df[col].isna().sum()):,} missing")

print("\nAudit complete. Review any errors before downstream research.")