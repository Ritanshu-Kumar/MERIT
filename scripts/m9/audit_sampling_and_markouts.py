
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DAY_NS = 86_400_000_000_000

HORIZONS_NS = {
    "10ms": 10_000_000,
    "100ms": 100_000_000,
    "500ms": 500_000_000,
    "1s": 1_000_000_000,
    "5s": 5_000_000_000,
}


def session_from_timestamp(timestamp_ns: object) -> str:
    """Classify a NASDAQ ITCH timestamp (nanoseconds since midnight ET)."""
    if pd.isna(timestamp_ns):
        return "unknown"

    seconds = (int(timestamp_ns) % DAY_NS) / 1_000_000_000

    if seconds < 4 * 3600 or seconds >= 20 * 3600:
        return "outside_04_20"
    if seconds < 9.5 * 3600:
        return "premarket"
    if seconds < 16 * 3600:
        return "regular"
    return "postmarket"


def load_dataset(path: Path, mode: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path, low_memory=False)
    df["mode"] = mode

    required = {
        "timestamp_ns",
        "symbol",
        "quote_side",
        "quote_price",
        "mid",
        "spread",
        "outcome",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{path} is missing required columns: {sorted(missing)}"
        )

    numeric_columns = [
        "timestamp_ns",
        "quote_price",
        "quote_qty",
        "filled_qty",
        "fill_fraction",
        "fill_timestamp_ns",
        "mid",
        "spread",
        "relative_spread",
        "resolution_mid",
        "resolution_mid_timestamp_ns",
        "resolution_mid_age_ns",
        "queue_ahead_initial",
        "cluster_size",
    ]

    for horizon in HORIZONS_NS:
        numeric_columns.extend([
            f"future_mid_{horizon}",
            f"future_mid_{horizon}_timestamp_ns",
        ])

    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    df["decision_session"] = df["timestamp_ns"].map(
        session_from_timestamp
    )

    if "fill_timestamp_ns" in df.columns:
        df["fill_session"] = df["fill_timestamp_ns"].map(
            session_from_timestamp
        )
    else:
        df["fill_session"] = "unknown"

    df["is_full_fill"] = df["outcome"].eq("FILL")

    if {"filled_qty", "quote_qty"}.issubset(df.columns):
        qty = df["filled_qty"].fillna(0)
        quote_qty = df["quote_qty"]
        df["has_any_fill"] = qty.gt(0)
        df["is_partial_fill"] = qty.gt(0) & qty.lt(quote_qty)
    else:
        df["has_any_fill"] = False
        df["is_partial_fill"] = False

    # Book validity diagnostics. Do not filter these rows yet.
    df["locked_book"] = df["spread"].eq(0)
    df["crossed_book"] = df["spread"].lt(0)
    df["invalid_mid"] = df["mid"].isna() | df["mid"].le(0)

    return df


def dataset_summary(df: pd.DataFrame) -> dict:
    n = len(df)

    return {
        "mode": df["mode"].iloc[0],
        "rows": n,
        "symbols": df["symbol"].nunique(),
        "decision_clusters": (
            df[["symbol", "timestamp_ns"]]
            .drop_duplicates()
            .shape[0]
        ),
        "full_fills": int(df["is_full_fill"].sum()),
        "full_fill_rate": (
            df["is_full_fill"].mean() if n else np.nan
        ),
        "any_fill_quantity": int(df["has_any_fill"].sum()),
        "partial_fill_candidates": int(df["is_partial_fill"].sum()),
        "locked_rows": int(df["locked_book"].sum()),
        "crossed_rows": int(df["crossed_book"].sum()),
        "invalid_mid_rows": int(df["invalid_mid"].sum()),
        "missing_resolution_mid": (
            int(df["resolution_mid"].isna().sum())
            if "resolution_mid" in df.columns else None
        ),
        "duplicate_candidate_ids": (
            int(df["candidate_id"].duplicated().sum())
            if "candidate_id" in df.columns else None
        ),
    }


def session_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    result = (
        df.groupby(
            ["mode", "decision_session", "symbol"],
            dropna=False,
        )
        .agg(
            candidates=("outcome", "size"),
            full_fills=("is_full_fill", "sum"),
            candidates_with_any_fill=("has_any_fill", "sum"),
            partial_fill_candidates=("is_partial_fill", "sum"),
            mean_spread=("spread", "mean"),
            median_spread=("spread", "median"),
            p90_spread=("spread", lambda s: s.quantile(0.90)),
            mean_relative_spread=(
                "relative_spread", "mean"
            ) if "relative_spread" in df.columns else ("spread", "mean"),
            locked_rows=("locked_book", "sum"),
            crossed_rows=("crossed_book", "sum"),
        )
        .reset_index()
    )

    result["full_fill_rate"] = (
        result["full_fills"] / result["candidates"]
    )
    result["any_fill_quantity_rate"] = (
        result["candidates_with_any_fill"] / result["candidates"]
    )

    return result


def make_markout_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Build signed markouts and their spread-capture/drift decomposition."""
    fills = df.loc[df["is_full_fill"]].copy()

    if fills.empty:
        return fills

    side = fills["quote_side"].astype(str).str.upper()
    buy = side.eq("BUY")
    sell = side.eq("SELL")

    price = pd.to_numeric(fills["quote_price"], errors="coerce")
    fill_mid = pd.to_numeric(
        fills.get("resolution_mid", pd.Series(np.nan, index=fills.index)),
        errors="coerce",
    )

    # Use quote price as the passive fill price.
    fills["fill_price_for_markout"] = price

    # Signed favorable spread capture at the recorded fill-time midpoint.
    fills["spread_capture_bps"] = np.select(
        [buy, sell],
        [
            (fill_mid - price) / price * 10_000,
            (price - fill_mid) / price * 10_000,
        ],
        default=np.nan,
    )

    # Decision midpoint -> fill midpoint drift is not used here:
    # decomposition is based on the recorded fill-time midpoint.
    for horizon, horizon_ns in HORIZONS_NS.items():
        mid_col = f"future_mid_{horizon}"
        time_col = f"future_mid_{horizon}_timestamp_ns"

        if mid_col not in fills.columns:
            continue

        future_mid = pd.to_numeric(
            fills[mid_col], errors="coerce"
        )

        fills[f"markout_{horizon}_bps"] = np.select(
            [buy, sell],
            [
                (future_mid - price) / price * 10_000,
                (price - future_mid) / price * 10_000,
            ],
            default=np.nan,
        )

        if time_col in fills.columns and "fill_timestamp_ns" in fills.columns:
            mark_ts = pd.to_numeric(fills[time_col], errors="coerce")
            fill_ts = pd.to_numeric(
                fills["fill_timestamp_ns"], errors="coerce"
            )
            target_ts = fill_ts + horizon_ns

            # Positive staleness means the as-of midpoint predates target T.
            fills[f"staleness_{horizon}_ms"] = (
                target_ts - mark_ts
            ) / 1_000_000

            fills[f"after_target_{horizon}"] = (
                mark_ts.notna()
                & target_ts.notna()
                & mark_ts.gt(target_ts)
            )

        if fill_mid.notna().any():
            fills[f"drift_{horizon}_bps"] = np.select(
                [buy, sell],
                [
                    (future_mid - fill_mid) / price * 10_000,
                    (fill_mid - future_mid) / price * 10_000,
                ],
                default=np.nan,
            )

    return fills


def markout_summaries(fills: pd.DataFrame) -> pd.DataFrame:
    rows = []

    if fills.empty:
        return pd.DataFrame()

    for group_keys, group in fills.groupby(
        ["mode", "fill_session", "symbol"],
        dropna=False,
    ):
        mode, session, symbol = group_keys

        for horizon in HORIZONS_NS:
            col = f"markout_{horizon}_bps"
            if col not in group.columns:
                continue

            values = pd.to_numeric(group[col], errors="coerce").dropna()
            if values.empty:
                continue

            stale_col = f"staleness_{horizon}_ms"
            stale = (
                pd.to_numeric(group[stale_col], errors="coerce").dropna()
                if stale_col in group.columns else pd.Series(dtype=float)
            )

            capture = pd.to_numeric(
                group["spread_capture_bps"], errors="coerce"
            ).dropna()

            drift_col = f"drift_{horizon}_bps"
            drift = (
                pd.to_numeric(group[drift_col], errors="coerce").dropna()
                if drift_col in group.columns else pd.Series(dtype=float)
            )

            late_col = f"after_target_{horizon}"
            late_count = (
                int(group[late_col].sum())
                if late_col in group.columns else None
            )

            rows.append({
                "mode": mode,
                "fill_session": session,
                "symbol": symbol,
                "horizon": horizon,
                "n": int(values.size),
                "mean_markout_bps": values.mean(),
                "median_markout_bps": values.median(),
                "std_markout_bps": values.std(),
                "p01_markout_bps": values.quantile(0.01),
                "p05_markout_bps": values.quantile(0.05),
                "p95_markout_bps": values.quantile(0.95),
                "p99_markout_bps": values.quantile(0.99),
                "min_markout_bps": values.min(),
                "max_markout_bps": values.max(),
                "mean_spread_capture_bps": (
                    capture.mean() if len(capture) else np.nan
                ),
                "mean_drift_bps": (
                    drift.mean() if len(drift) else np.nan
                ),
                "staleness_p95_ms": (
                    stale.quantile(0.95) if len(stale) else np.nan
                ),
                "staleness_max_ms": (
                    stale.max() if len(stale) else np.nan
                ),
                "mark_after_target_count": late_count,
            })

    return pd.DataFrame(rows)


def spread_bucket_audit(fills: pd.DataFrame) -> pd.DataFrame:
    """
    Descriptive spread-bin analysis.

    These bins are diagnostic only. Do not use them to choose model
    thresholds or exclude observations from the final test set.
    """
    if fills.empty:
        return pd.DataFrame()

    work = fills.copy()
    work["spread_bucket"] = "unavailable"

    grouping = ["mode", "fill_session", "symbol"]

    for _, idx in work.groupby(grouping, dropna=False).groups.items():
        spreads = pd.to_numeric(
            work.loc[idx, "spread"], errors="coerce"
        )
        valid = spreads.notna()

        if valid.sum() < 4:
            work.loc[idx, "spread_bucket"] = "too_few_rows"
            continue

        try:
            bucket = pd.qcut(
                spreads[valid],
                q=4,
                labels=["Q1_low", "Q2", "Q3", "Q4_high"],
                duplicates="drop",
            )
            work.loc[bucket.index, "spread_bucket"] = (
                bucket.astype(str)
            )
        except ValueError:
            work.loc[idx, "spread_bucket"] = "constant_spread"

    summaries = []

    for keys, group in work.groupby(
        grouping + ["spread_bucket"],
        dropna=False,
    ):
        mode, session, symbol, bucket = keys

        row = {
            "mode": mode,
            "fill_session": session,
            "symbol": symbol,
            "spread_bucket": bucket,
            "n": len(group),
            "mean_spread": group["spread"].mean(),
        }

        for horizon in ["100ms", "1s", "5s"]:
            col = f"markout_{horizon}_bps"
            if col in group.columns:
                row[f"{horizon}_mean_bps"] = group[col].mean()
                row[f"{horizon}_median_bps"] = group[col].median()

        row["mean_spread_capture_bps"] = (
            group["spread_capture_bps"].mean()
        )
        summaries.append(row)

    return pd.DataFrame(summaries)


def sampling_gap_audit(df: pd.DataFrame) -> pd.DataFrame:
    # Do not count BUY and SELL rows at the same decision timestamp twice.
    points = (
        df[["mode", "symbol", "timestamp_ns"]]
        .drop_duplicates()
        .sort_values(["mode", "symbol", "timestamp_ns"])
        .copy()
    )

    points["gap_ms"] = (
        points.groupby(["mode", "symbol"])["timestamp_ns"].diff()
        / 1_000_000
    )

    rows = []

    for (mode, symbol), group in points.groupby(["mode", "symbol"]):
        gaps = group["gap_ms"].dropna()

        rows.append({
            "mode": mode,
            "symbol": symbol,
            "decision_points": len(group),
            "interval_count": len(gaps),
            "median_gap_ms": gaps.median() if len(gaps) else np.nan,
            "p95_gap_ms": gaps.quantile(0.95) if len(gaps) else np.nan,
            "p99_gap_ms": gaps.quantile(0.99) if len(gaps) else np.nan,
            "max_gap_ms": gaps.max() if len(gaps) else np.nan,
            "gaps_over_1500ms": int(gaps.gt(1500).sum()),
            "gaps_over_5s": int(gaps.gt(5000).sum()),
            "gaps_over_60s": int(gaps.gt(60000).sum()),
        })

    return pd.DataFrame(rows)


def export_extremes(fills: pd.DataFrame, output_dir: Path) -> None:
    if fills.empty:
        return

    common_columns = [
        "mode",
        "symbol",
        "quote_side",
        "timestamp_ns",
        "fill_timestamp_ns",
        "quote_price",
        "mid",
        "resolution_mid",
        "spread",
        "relative_spread",
        "queue_ahead_initial",
        "filled_qty",
        "outcome",
        "decision_session",
        "fill_session",
    ]

    for horizon in HORIZONS_NS:
        markout_col = f"markout_{horizon}_bps"
        if markout_col not in fills.columns:
            continue

        stale_col = f"staleness_{horizon}_ms"
        selected_columns = [
            c for c in common_columns if c in fills.columns
        ] + [markout_col]

        if stale_col in fills.columns:
            selected_columns.append(stale_col)

        valid = fills.loc[fills[markout_col].notna()].copy()
        if valid.empty:
            continue

        valid.nlargest(20, markout_col)[selected_columns].to_csv(
            output_dir / f"top_positive_{horizon}.csv",
            index=False,
        )
        valid.nsmallest(20, markout_col)[selected_columns].to_csv(
            output_dir / f"top_negative_{horizon}.csv",
            index=False,
        )

        # Count the tail rather than automatically excluding it.
        tail = valid.groupby("mode").agg(
            n=("mode", "size"),
            above_10bps=(markout_col, lambda s: int(s.gt(10).sum())),
            below_minus_10bps=(markout_col, lambda s: int(s.lt(-10).sum())),
            above_25bps=(markout_col, lambda s: int(s.gt(25).sum())),
            below_minus_25bps=(markout_col, lambda s: int(s.lt(-25).sum())),
            above_50bps=(markout_col, lambda s: int(s.gt(50).sum())),
            below_minus_50bps=(markout_col, lambda s: int(s.lt(-50).sum())),
        )
        tail.to_csv(
            output_dir / f"markout_tail_counts_{horizon}.csv"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Audit M9 sampling, sessions, midpoint markouts, "
            "spread decomposition, and extreme observations."
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
        default=Path("research/m9_markout_audit"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    execution = load_dataset(args.execution, "execution")
    time_sampled = load_dataset(args.time, "time")
    datasets = [execution, time_sampled]

    pd.DataFrame(
        [dataset_summary(df) for df in datasets]
    ).to_csv(args.output_dir / "dataset_quality.csv", index=False)

    sessions = pd.concat(
        [session_outcomes(df) for df in datasets],
        ignore_index=True,
    )
    sessions.to_csv(
        args.output_dir / "outcomes_by_session_symbol.csv",
        index=False,
    )

    gaps = pd.concat(
        [sampling_gap_audit(df) for df in datasets],
        ignore_index=True,
    )
    gaps.to_csv(
        args.output_dir / "sampling_gap_audit.csv",
        index=False,
    )

    fill_frames = [
        make_markout_frame(df) for df in datasets
    ]
    fills = pd.concat(fill_frames, ignore_index=True)

    marks = markout_summaries(fills)
    marks.to_csv(
        args.output_dir / "markout_summary_by_session_symbol.csv",
        index=False,
    )

    spread_audit = spread_bucket_audit(fills)
    spread_audit.to_csv(
        args.output_dir / "markouts_by_spread_bucket.csv",
        index=False,
    )

    export_extremes(fills, args.output_dir)

    print("\n=== DATA QUALITY ===")
    print(pd.read_csv(
        args.output_dir / "dataset_quality.csv"
    ).to_string(index=False))

    print("\n=== OUTCOMES BY SESSION AND SYMBOL ===")
    print(sessions.to_string(index=False))

    print("\n=== SAMPLING GAP AUDIT ===")
    print(gaps.to_string(index=False))

    print("\n=== MARKOUTS BY FILL SESSION AND SYMBOL ===")
    if marks.empty:
        print("No usable markout observations.")
    else:
        print(marks.to_string(index=False))

    print("\n=== MARKOUTS BY SPREAD BUCKET ===")
    if spread_audit.empty:
        print("No usable spread-bucket diagnostics.")
    else:
        print(spread_audit.to_string(index=False))

    print(f"\nAudit outputs saved to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
