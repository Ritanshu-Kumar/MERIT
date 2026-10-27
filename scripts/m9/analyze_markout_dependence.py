
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DAY_NS = 86_400_000_000_000

OUTCOME_KEY = [
    "symbol",
    "quote_side",
    "quote_price",
    "fill_timestamp_ns",
    "future_mid_5s_timestamp_ns",
    "future_mid_5s",
]


def session_name(value: object) -> str:
    if pd.isna(value):
        return "unknown"

    seconds = (int(value) % DAY_NS) / 1_000_000_000

    if seconds < 4 * 3600 or seconds >= 20 * 3600:
        return "outside_04_20"
    if seconds < 9.5 * 3600:
        return "premarket"
    if seconds < 16 * 3600:
        return "regular"
    return "postmarket"


def load_fills(path: Path, mode: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path, low_memory=False)

    required = {
        *OUTCOME_KEY,
        "timestamp_ns",
        "outcome",
        "mid",
        "spread",
        "queue_ahead_initial",
    }
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"{path} is missing columns: {sorted(missing)}"
        )

    numeric = [
        "timestamp_ns",
        "quote_price",
        "fill_timestamp_ns",
        "future_mid_5s",
        "future_mid_5s_timestamp_ns",
        "queue_ahead_initial",
    ]

    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.loc[
        df["outcome"].eq("FILL")
        & df["fill_timestamp_ns"].notna()
        & df["quote_price"].gt(0)
        & df["future_mid_5s"].notna()
        & df["future_mid_5s_timestamp_ns"].notna()
    ].copy()

    df["mode"] = mode
    df["fill_session"] = df["fill_timestamp_ns"].map(session_name)

    is_buy = df["quote_side"].eq("BUY")
    is_sell = df["quote_side"].eq("SELL")

    df["markout_5s_bps"] = np.select(
        [is_buy, is_sell],
        [
            (
                df["future_mid_5s"] - df["quote_price"]
            ) / df["quote_price"] * 10_000,
            (
                df["quote_price"] - df["future_mid_5s"]
            ) / df["quote_price"] * 10_000,
        ],
        default=np.nan,
    )

    df = df.loc[df["markout_5s_bps"].notna()].copy()

    return df


def candidate_summary(fills: pd.DataFrame) -> pd.DataFrame:
    keys = ["mode", "fill_session", "symbol"]

    return (
        fills.groupby(keys)
        .agg(
            candidate_rows=("markout_5s_bps", "size"),
            row_weighted_mean_bps=("markout_5s_bps", "mean"),
            row_weighted_median_bps=("markout_5s_bps", "median"),
            positive_over_25bps=(
                "markout_5s_bps", lambda s: int(s.gt(25).sum())
            ),
            negative_below_minus_25bps=(
                "markout_5s_bps", lambda s: int(s.lt(-25).sum())
            ),
        )
        .reset_index()
    )


def outcome_clusters(fills: pd.DataFrame) -> pd.DataFrame:
    """
    Sensitivity proxy: group candidate rows with the same symbol,
    side, quote price, fill timestamp, and 5-second midpoint outcome.

    This is NOT a verified exchange-event ID. Distinct ITCH executions
    can share exchange timestamps, so these groups are diagnostic only.
    """
    group_cols = ["mode", "fill_session", *OUTCOME_KEY]

    clusters = (
        fills.groupby(group_cols, dropna=False)
        .agg(
            candidate_count=("markout_5s_bps", "size"),
            cluster_mean_bps=("markout_5s_bps", "mean"),
            cluster_min_bps=("markout_5s_bps", "min"),
            cluster_max_bps=("markout_5s_bps", "max"),
            queue_min=("queue_ahead_initial", "min"),
            queue_max=("queue_ahead_initial", "max"),
            distinct_decision_times=("timestamp_ns", "nunique"),
            candidate_ids=("candidate_id", "nunique")
            if "candidate_id" in fills.columns
            else ("timestamp_ns", "size"),
        )
        .reset_index()
    )

    return clusters


def cluster_summary(
    fills: pd.DataFrame,
    clusters: pd.DataFrame,
) -> pd.DataFrame:
    keys = ["mode", "fill_session", "symbol"]

    row_stats = candidate_summary(fills)

    cluster_stats = (
        clusters.groupby(keys)
        .agg(
            outcome_clusters=("cluster_mean_bps", "size"),
            cluster_weighted_mean_bps=("cluster_mean_bps", "mean"),
            median_cluster_mean_bps=("cluster_mean_bps", "median"),
            largest_outcome_cluster=("candidate_count", "max"),
            median_outcome_cluster_size=("candidate_count", "median"),
            p95_outcome_cluster_size=(
                "candidate_count", lambda s: s.quantile(0.95)
            ),
            candidate_rows_in_clusters_gt1=(
                "candidate_count",
                lambda s: int(s[s.gt(1)].sum()),
            ),
            clusters_with_more_than_one_candidate=(
                "candidate_count", lambda s: int(s.gt(1).sum())
            ),
            clusters_mean_above_25bps=(
                "cluster_mean_bps", lambda s: int(s.gt(25).sum())
            ),
        )
        .reset_index()
    )

    result = row_stats.merge(
        cluster_stats,
        on=keys,
        how="left",
        validate="one_to_one",
    )

    result["cluster_minus_row_mean_bps"] = (
        result["cluster_weighted_mean_bps"]
        - result["row_weighted_mean_bps"]
    )

    result["fraction_rows_in_repeated_clusters"] = (
        result["candidate_rows_in_clusters_gt1"]
        / result["candidate_rows"]
    )

    return result


def tail_counts(
    fills: pd.DataFrame,
    clusters: pd.DataFrame,
) -> pd.DataFrame:
    keys = ["mode", "fill_session", "symbol"]
    rows = []

    for group_key, group in fills.groupby(keys):
        mode, session, symbol = group_key
        group_clusters = clusters.loc[
            (clusters["mode"] == mode)
            & (clusters["fill_session"] == session)
            & (clusters["symbol"] == symbol)
        ]

        marks = group["markout_5s_bps"]

        rows.append({
            "mode": mode,
            "fill_session": session,
            "symbol": symbol,
            "candidate_rows": len(group),
            "rows_gt_10bps": int(marks.gt(10).sum()),
            "rows_gt_25bps": int(marks.gt(25).sum()),
            "rows_gt_50bps": int(marks.gt(50).sum()),
            "rows_lt_minus_10bps": int(marks.lt(-10).sum()),
            "rows_lt_minus_25bps": int(marks.lt(-25).sum()),
            "rows_lt_minus_50bps": int(marks.lt(-50).sum()),
            "clusters_gt_10bps": int(
                group_clusters["cluster_mean_bps"].gt(10).sum()
            ),
            "clusters_gt_25bps": int(
                group_clusters["cluster_mean_bps"].gt(25).sum()
            ),
            "clusters_gt_50bps": int(
                group_clusters["cluster_mean_bps"].gt(50).sum()
            ),
        })

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Audit repeated candidate outcomes and 5-second "
            "markout tail concentration."
        )
    )

    parser.add_argument(
        "--execution",
        type=Path,
        default=Path("research/m9_execution_2019-07-30_asof.csv"),
    )
    parser.add_argument(
        "--time",
        type=Path,
        default=Path("research/m9_time_sampled_2019-07-30_asof.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("research/m9_markout_dependence"),
    )

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    execution = load_fills(args.execution, "execution")
    time_sampled = load_fills(args.time, "time")
    fills = pd.concat(
        [execution, time_sampled],
        ignore_index=True,
    )

    if fills.empty:
        raise ValueError("No eligible full-fill markouts found.")

    clusters = outcome_clusters(fills)
    summary = cluster_summary(fills, clusters)
    tails = tail_counts(fills, clusters)

    summary.to_csv(
        args.output_dir / "dependence_summary.csv",
        index=False,
    )
    clusters.to_csv(
        args.output_dir / "outcome_clusters.csv",
        index=False,
    )
    tails.to_csv(
        args.output_dir / "tail_counts.csv",
        index=False,
    )

    top_positive = clusters.sort_values(
        ["cluster_mean_bps", "candidate_count"],
        ascending=[False, False],
    ).head(50)

    top_negative = clusters.sort_values(
        ["cluster_mean_bps", "candidate_count"],
        ascending=[True, False],
    ).head(50)

    top_positive.to_csv(
        args.output_dir / "top_positive_outcome_clusters.csv",
        index=False,
    )
    top_negative.to_csv(
        args.output_dir / "top_negative_outcome_clusters.csv",
        index=False,
    )

    print("\n=== ROW-WEIGHTED VS CLUSTER-WEIGHTED SUMMARY ===")
    print(summary.to_string(index=False))

    print("\n=== 5-SECOND TAIL COUNTS ===")
    print(tails.to_string(index=False))

    print("\n=== LARGEST POSITIVE OUTCOME CLUSTERS ===")
    print(top_positive.head(20).to_string(index=False))

    print(f"\nReports written to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
