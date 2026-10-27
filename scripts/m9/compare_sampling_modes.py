
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


HORIZONS_NS = {
    "10ms": 10_000_000,
    "100ms": 100_000_000,
    "500ms": 500_000_000,
    "1s": 1_000_000_000,
    "5s": 5_000_000_000,
}

FEATURES = [
    "mid",
    "spread",
    "relative_spread",
    "imbalance",
    "delta_imbalance",
    "delta_microprice",
    "delta_spread",
    "queue_ahead_initial",
]


def load_dataset(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{label} dataset not found: {path}")

    df = pd.read_csv(path, low_memory=False)

    required = {
        "timestamp_ns",
        "symbol",
        "quote_side",
        "quote_price",
        "quote_qty",
        "filled_qty",
        "fill_fraction",
        "outcome",
    }
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"{label} dataset is missing columns: {sorted(missing)}"
        )

    for column in [
        "timestamp_ns",
        "quote_price",
        "quote_qty",
        "filled_qty",
        "fill_fraction",
        "fill_timestamp_ns",
        "mid",
        "spread",
        "relative_spread",
        "imbalance",
        "delta_imbalance",
        "delta_microprice",
        "delta_spread",
        "queue_ahead_initial",
    ]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    df["sampling_mode_label"] = label
    return df


def dataset_summary(df: pd.DataFrame) -> dict:
    outcome_counts = df["outcome"].value_counts()

    full_fill = df["outcome"].eq("FILL")
    any_quantity_filled = pd.to_numeric(
        df["filled_qty"], errors="coerce"
    ).fillna(0).gt(0)

    quote_qty = pd.to_numeric(df["quote_qty"], errors="coerce")
    filled_qty = pd.to_numeric(df["filled_qty"], errors="coerce")

    partial_fill = filled_qty.gt(0) & filled_qty.lt(quote_qty)

    result = {
        "mode": df["sampling_mode_label"].iloc[0],
        "rows": len(df),
        "unique_symbols": df["symbol"].nunique(),
        "first_timestamp_ns": df["timestamp_ns"].min(),
        "last_timestamp_ns": df["timestamp_ns"].max(),
        "unique_decision_times_symbol": (
            df[["symbol", "timestamp_ns"]].drop_duplicates().shape[0]
        ),
        "unique_candidate_ids": (
            df["candidate_id"].nunique()
            if "candidate_id" in df.columns
            else None
        ),
        "duplicate_candidate_ids": (
            int(df["candidate_id"].duplicated().sum())
            if "candidate_id" in df.columns
            else None
        ),
        "full_fill_rows": int(full_fill.sum()),
        "full_fill_rate": float(full_fill.mean()) if len(df) else None,
        "rows_with_any_filled_quantity": int(any_quantity_filled.sum()),
        "partial_fill_rows": int(partial_fill.sum()),
        "BUY_rows": int(df["quote_side"].eq("BUY").sum()),
        "SELL_rows": int(df["quote_side"].eq("SELL").sum()),
    }

    for outcome in ["FILL", "ADVERSE", "TIMEOUT"]:
        result[f"outcome_{outcome}"] = int(outcome_counts.get(outcome, 0))

    return result


def outcome_by_symbol(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby(["sampling_mode_label", "symbol", "outcome"])
        .size()
        .rename("rows")
        .reset_index()
    )

    totals = (
        df.groupby(["sampling_mode_label", "symbol"])
        .size()
        .rename("symbol_rows")
        .reset_index()
    )

    grouped = grouped.merge(
        totals,
        on=["sampling_mode_label", "symbol"],
        how="left",
    )
    grouped["rate_within_symbol"] = grouped["rows"] / grouped["symbol_rows"]

    return grouped


def feature_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for feature in FEATURES:
        if feature not in df.columns:
            continue

        values = pd.to_numeric(df[feature], errors="coerce").dropna()

        rows.append({
            "mode": df["sampling_mode_label"].iloc[0],
            "feature": feature,
            "valid_count": int(values.size),
            "missing_count": int(df[feature].isna().sum()),
            "mean": values.mean() if len(values) else None,
            "std": values.std() if len(values) else None,
            "p01": values.quantile(0.01) if len(values) else None,
            "p05": values.quantile(0.05) if len(values) else None,
            "median": values.median() if len(values) else None,
            "p95": values.quantile(0.95) if len(values) else None,
            "p99": values.quantile(0.99) if len(values) else None,
        })

    return pd.DataFrame(rows)


def cluster_summary(df: pd.DataFrame) -> pd.DataFrame:
    keys = ["symbol", "timestamp_ns"]

    clusters = (
        df.groupby(keys, dropna=False)
        .agg(
            rows=("quote_side", "size"),
            unique_sides=("quote_side", "nunique"),
            distinct_prices=("quote_price", "nunique"),
        )
        .reset_index()
    )

    clusters.insert(0, "mode", df["sampling_mode_label"].iloc[0])

    summary = clusters.groupby("mode").agg(
        cluster_count=("rows", "size"),
        mean_rows_per_cluster=("rows", "mean"),
        median_rows_per_cluster=("rows", "median"),
        p95_rows_per_cluster=("rows", lambda x: x.quantile(0.95)),
        max_rows_per_cluster=("rows", "max"),
        clusters_with_both_sides=(
            "unique_sides",
            lambda x: int((x == 2).sum()),
        ),
        clusters_not_containing_two_rows=(
            "rows",
            lambda x: int((x != 2).sum()),
        ),
    ).reset_index()

    return clusters, summary


def sampling_intervals(df: pd.DataFrame) -> pd.DataFrame:
    # One timestamp per decision point; BUY and SELL rows at the same
    # timestamp must not be counted as separate sampling intervals.
    decisions = (
        df[["symbol", "timestamp_ns"]]
        .drop_duplicates()
        .sort_values(["symbol", "timestamp_ns"])
        .copy()
    )

    decisions["interval_ns"] = (
        decisions.groupby("symbol")["timestamp_ns"].diff()
    )

    rows = []

    for symbol, group in decisions.groupby("symbol"):
        intervals_ms = group["interval_ns"].dropna() / 1_000_000

        rows.append({
            "mode": df["sampling_mode_label"].iloc[0],
            "symbol": symbol,
            "decision_times": len(group),
            "interval_count": len(intervals_ms),
            "interval_ms_median": (
                intervals_ms.median() if len(intervals_ms) else None
            ),
            "interval_ms_p05": (
                intervals_ms.quantile(0.05) if len(intervals_ms) else None
            ),
            "interval_ms_p95": (
                intervals_ms.quantile(0.95) if len(intervals_ms) else None
            ),
            "interval_ms_min": (
                intervals_ms.min() if len(intervals_ms) else None
            ),
            "interval_ms_max": (
                intervals_ms.max() if len(intervals_ms) else None
            ),
        })

    return pd.DataFrame(rows)


def markout_quality(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    if "fill_timestamp_ns" not in df.columns:
        return pd.DataFrame()

    full_fills = df[df["outcome"].eq("FILL")].copy()
    fill_ts = pd.to_numeric(
        full_fills["fill_timestamp_ns"], errors="coerce"
    )

    for horizon, horizon_ns in HORIZONS_NS.items():
        value_col = f"future_mid_{horizon}"
        time_col = f"future_mid_{horizon}_timestamp_ns"

        if value_col not in full_fills.columns:
            continue

        values = pd.to_numeric(full_fills[value_col], errors="coerce")
        has_value = values.notna()

        result = {
            "mode": df["sampling_mode_label"].iloc[0],
            "horizon": horizon,
            "full_fill_rows": len(full_fills),
            "nonmissing_midpoint_rows": int(has_value.sum()),
            "midpoint_coverage": (
                float(has_value.mean()) if len(full_fills) else None
            ),
        }

        if time_col in full_fills.columns:
            mark_ts = pd.to_numeric(
                full_fills[time_col], errors="coerce"
            )
            eligible = has_value & fill_ts.notna() & mark_ts.notna()

            target_ts = fill_ts + horizon_ns
            staleness_ms = (
                target_ts[eligible] - mark_ts[eligible]
            ) / 1_000_000

            future_violation = (
                mark_ts[eligible] > target_ts[eligible]
            )

            before_fill_violation = (
                mark_ts[eligible] < fill_ts[eligible]
            )

            result.update({
                "timestamped_rows": int(eligible.sum()),
                "staleness_ms_median": (
                    staleness_ms.median() if len(staleness_ms) else None
                ),
                "staleness_ms_p95": (
                    staleness_ms.quantile(0.95)
                    if len(staleness_ms) else None
                ),
                "staleness_ms_max": (
                    staleness_ms.max() if len(staleness_ms) else None
                ),
                "midpoint_after_target_rows": int(future_violation.sum()),
                "midpoint_before_fill_rows": int(
                    before_fill_violation.sum()
                ),
            })

        rows.append(result)

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare M9 execution-triggered and time-sampled datasets."
    )

    parser.add_argument(
        "--execution",
        type=Path,
        default=Path("research/m9_execution_smoke.csv"),
    )
    parser.add_argument(
        "--time",
        type=Path,
        default=Path("research/m9_time_sampled_smoke.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("research/m9_sampling_diagnostics"),
    )

    args = parser.parse_args()

    execution = load_dataset(args.execution, "execution")
    time_sampled = load_dataset(args.time, "time")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    datasets = [execution, time_sampled]

    pd.DataFrame(
        [dataset_summary(df) for df in datasets]
    ).to_csv(args.output_dir / "dataset_summary.csv", index=False)

    pd.concat(
        [outcome_by_symbol(df) for df in datasets],
        ignore_index=True,
    ).to_csv(args.output_dir / "outcomes_by_symbol.csv", index=False)

    pd.concat(
        [feature_summary(df) for df in datasets],
        ignore_index=True,
    ).to_csv(args.output_dir / "feature_summary.csv", index=False)

    all_clusters = []
    cluster_summaries = []

    for df in datasets:
        clusters, summary = cluster_summary(df)
        all_clusters.append(clusters)
        cluster_summaries.append(summary)

    pd.concat(all_clusters, ignore_index=True).to_csv(
        args.output_dir / "clusters.csv", index=False
    )
    pd.concat(cluster_summaries, ignore_index=True).to_csv(
        args.output_dir / "cluster_summary.csv", index=False
    )

    pd.concat(
        [sampling_intervals(df) for df in datasets],
        ignore_index=True,
    ).to_csv(args.output_dir / "sampling_intervals.csv", index=False)

    markouts = pd.concat(
        [markout_quality(df) for df in datasets],
        ignore_index=True,
    )
    markouts.to_csv(
        args.output_dir / "markout_quality.csv", index=False
    )

    print("\n=== DATASET SUMMARY ===")
    print(pd.read_csv(args.output_dir / "dataset_summary.csv").to_string(index=False))

    print("\n=== OUTCOMES BY SYMBOL ===")
    print(pd.read_csv(args.output_dir / "outcomes_by_symbol.csv").to_string(index=False))

    print("\n=== SAMPLING INTERVALS (ms) ===")
    print(pd.read_csv(args.output_dir / "sampling_intervals.csv").to_string(index=False))

    print("\n=== CLUSTER SUMMARY ===")
    print(pd.read_csv(args.output_dir / "cluster_summary.csv").to_string(index=False))

    print("\n=== MARKOUT QUALITY ===")
    if not markouts.empty:
        print(markouts.to_string(index=False))
    else:
        print("No markout quality results available.")

    print(f"\nReports written to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
