from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("research/m9_fill_hazard_markouts_2019-07-30_timestamped.csv")
OUT = Path("research/m9_economic_units_audit.txt")

HORIZONS = {
    "10ms": "future_mid_10ms",
    "100ms": "future_mid_100ms",
    "500ms": "future_mid_500ms",
    "1s": "future_mid_1s",
}

REQUIRED = [
    "timestamp_ns",
    "symbol",
    "quote_side",
    "quote_price",
    "quote_qty",
    "mid",
    "filled_qty",
    "fill_fraction",
    "resolution_mid",
    "outcome",
    "fill_timestamp_ns",
    *HORIZONS.values(),
    "future_mid_10ms_timestamp_ns",
    "future_mid_100ms_timestamp_ns",
    "future_mid_500ms_timestamp_ns",
    "future_mid_1s_timestamp_ns",
]


def describe(series: pd.Series) -> str:
    values = (
        pd.to_numeric(series, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    if values.empty:
        return "n=0"

    q = values.quantile([0.01, 0.10, 0.50, 0.90, 0.99])

    return (
        f"n={len(values):,}; "
        f"min={values.min():.8g}; "
        f"p01={q.loc[0.01]:.8g}; "
        f"p10={q.loc[0.10]:.8g}; "
        f"median={q.loc[0.50]:.8g}; "
        f"p90={q.loc[0.90]:.8g}; "
        f"p99={q.loc[0.99]:.8g}; "
        f"max={values.max():.8g}"
    )


def main() -> None:
    if not DATA.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA.resolve()}"
        )

    df = pd.read_csv(DATA)

    missing = sorted(set(REQUIRED) - set(df.columns))
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    numeric_columns = [
        "quote_price",
        "quote_qty",
        "mid",
        "filled_qty",
        "fill_fraction",
        "resolution_mid",
        "fill_timestamp_ns",
        *HORIZONS.values(),
        "future_mid_10ms_timestamp_ns",
        "future_mid_100ms_timestamp_ns",
        "future_mid_500ms_timestamp_ns",
        "future_mid_1s_timestamp_ns",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column], errors="coerce"
        )

    df["quote_side"] = df["quote_side"].astype(str).str.upper()
    df["outcome"] = df["outcome"].astype(str).str.upper()

    lines = [
        "MERIT M9 economic units and P&L audit",
        "=" * 48,
        f"Input: {DATA}",
        f"Rows: {len(df):,}",
        "",
        "1) Raw price and quantity scales",
    ]

    for column in [
        "quote_price",
        "mid",
        "resolution_mid",
        *HORIZONS.values(),
        "quote_qty",
        "filled_qty",
        "fill_fraction",
    ]:
        lines.append(f"{column}: {describe(df[column])}")

    lines.extend(["", "1a) Markout timestamp integrity and event-time overshoot"])
    horizon_ns = {
        "10ms": 10_000_000,
        "100ms": 100_000_000,
        "500ms": 500_000_000,
        "1s": 1_000_000_000,
    }
    fill_ts = df["fill_timestamp_ns"]
    for horizon, offset_ns in horizon_ns.items():
        price_col = HORIZONS[horizon]
        ts_col = f"future_mid_{horizon}_timestamp_ns"
        has_price = df[price_col].notna()
        has_ts = df[ts_col].notna()
        mismatches = int(has_price.ne(has_ts).sum())
        orphan_ts = int((has_ts & fill_ts.isna()).sum())
        before = has_ts & fill_ts.notna() & ((df[ts_col] - fill_ts) < offset_ns)
        before_count = int(before.sum())
        valid = has_ts & fill_ts.notna()
        overshoot_ms = (
            df.loc[valid, ts_col] - fill_ts.loc[valid] - offset_ns
        ) / 1_000_000
        if mismatches or orphan_ts or before_count:
            raise ValueError(
                f"{horizon} timestamp audit failed: mismatches={mismatches}, "
                f"timestamps_without_fill={orphan_ts}, before_horizon={before_count}"
            )
        if len(overshoot_ms):
            lines.append(
                f"{horizon}: mismatches=0; timestamps_without_fill=0; before_horizon=0; "
                f"overshoot_ms p50={overshoot_ms.quantile(0.50):.6f}, "
                f"p95={overshoot_ms.quantile(0.95):.6f}, max={overshoot_ms.max():.6f}; "
                f"valid timestamps={len(overshoot_ms):,}"
            )
        else:
            lines.append(
                f"{horizon}: mismatches=0; timestamps_without_fill=0; "
                f"before_horizon=0; no observed markouts"
            )

    valid = (
        df["quote_price"].gt(0)
        & df["mid"].gt(0)
        & np.isfinite(df["quote_price"])
        & np.isfinite(df["mid"])
    )

    ratio = (
        df.loc[valid, "quote_price"]
        / df.loc[valid, "mid"]
    )

    relative_gap = (
        df.loc[valid, "quote_price"]
        - df.loc[valid, "mid"]
    ).abs() / df.loc[valid, "mid"]

    lines.extend([
        "",
        "2) Quote price versus initial midpoint",
        f"quote_price / mid: {describe(ratio)}",
        f"Relative absolute gap: {describe(relative_gap)}",
        "Rows with quote/mid ratio outside [0.8, 1.2]: "
        f"{int((~ratio.between(0.8, 1.2)).sum()):,}",
    ])

    lines.extend([
        "",
        "3) Signed movement per share and markout bps",
        "Positive means favorable to the passive quote.",
    ])

    for horizon, column in HORIZONS.items():
        mask = (
            df["quote_price"].gt(0)
            & df[column].notna()
            & df["quote_side"].isin(["BUY", "SELL"])
        )

        sample = df.loc[mask].copy()

        sample["signed_move_usd_per_share"] = np.where(
            sample["quote_side"].eq("BUY"),
            sample[column] - sample["quote_price"],
            sample["quote_price"] - sample[column],
        )

        sample["signed_move_bps"] = (
            sample["signed_move_usd_per_share"]
            / sample["quote_price"]
            * 10_000
        )

        lines.append(
            f"\n--- {horizon}: {len(sample):,} valid rows ---"
        )

        for (outcome, side), group in sample.groupby(
            ["outcome", "quote_side"],
            sort=True,
        ):
            lines.append(
                f"{outcome:7s} {side:4s}: "
                f"USD/share "
                f"[{describe(group['signed_move_usd_per_share'])}]; "
                f"bps [{describe(group['signed_move_bps'])}]"
            )

    lines.extend([
        "",
        "4) Independent central-scenario P&L — full-fill primary and partial sensitivity separated",
        "Full fills use the 100ms midpoint. Partial fills use resolution_mid only as a separate sensitivity.",
    ])

    df["evaluated_qty"] = pd.to_numeric(
        df["filled_qty"], errors="coerce"
    ).fillna(0).clip(lower=0)

    full_fill = df["outcome"].eq("FILL")
    has_fill = df["evaluated_qty"].gt(0)
    partial_fill = (
        has_fill & ~full_fill
        & df["evaluated_qty"].lt(df["quote_qty"])
    )

    df["audit_mark_mid"] = np.nan
    df.loc[full_fill, "audit_mark_mid"] = df.loc[
        full_fill, HORIZONS["100ms"]
    ]

    valued = (
        full_fill
        & has_fill
        & df["audit_mark_mid"].notna()
        & df["quote_price"].gt(0)
        & df["quote_side"].isin(["BUY", "SELL"])
    )

    df["audit_signed_move"] = np.where(
        df["quote_side"].eq("BUY"),
        df["audit_mark_mid"] - df["quote_price"],
        df["quote_price"] - df["audit_mark_mid"],
    )

    df["audit_gross"] = np.where(
        valued,
        df["audit_signed_move"] * df["evaluated_qty"],
        0.0,
    )

    df["audit_fee_05bps"] = np.where(
        valued,
        df["quote_price"] * df["evaluated_qty"] * 0.5 / 10_000,
        0.0,
    )

    df["audit_net"] = df["audit_gross"] - df["audit_fee_05bps"]

    df["partial_audit_mark_mid"] = np.where(
        partial_fill, df["resolution_mid"], np.nan
    )

    partial_valued = (
        partial_fill
        & df["partial_audit_mark_mid"].notna()
        & df["quote_price"].gt(0)
        & df["quote_side"].isin(["BUY", "SELL"])
    )

    df["partial_audit_signed_move"] = np.where(
        df["quote_side"].eq("BUY"),
        df["partial_audit_mark_mid"] - df["quote_price"],
        df["quote_price"] - df["partial_audit_mark_mid"],
    )

    df["partial_audit_gross"] = np.where(
        partial_valued,
        df["partial_audit_signed_move"] * df["evaluated_qty"],
        0.0,
    )

    df["partial_audit_fee_05bps"] = np.where(
        partial_valued,
        df["quote_price"] * df["evaluated_qty"] * 0.5 / 10_000,
        0.0,
    )

    df["partial_audit_net"] = (
        df["partial_audit_gross"] - df["partial_audit_fee_05bps"]
    )

    lines.extend([
        f"Rows with any positive filled quantity: {int(has_fill.sum()):,}",
        f"Full-fill candidates: {int((full_fill & has_fill).sum()):,}",
        f"Partial-fill candidates: {int(partial_fill.sum()):,}",
        f"Full fills missing valid primary valuation: {int((full_fill & has_fill & ~valued).sum()):,}",
        f"Partial fills missing sensitivity valuation: {int((partial_fill & ~partial_valued).sum()):,}",
        f"Primary full-fill shares: {df.loc[valued, 'evaluated_qty'].sum():,.2f}",
        f"Primary full-fill gross P&L: ${df.loc[valued, 'audit_gross'].sum():,.6f}",
        f"Primary full-fill fees at 0.5 bps: ${df.loc[valued, 'audit_fee_05bps'].sum():,.6f}",
        f"Primary full-fill net P&L: ${df.loc[valued, 'audit_net'].sum():,.6f}",
        f"Partial sensitivity shares: {df.loc[partial_valued, 'evaluated_qty'].sum():,.2f}",
        f"Partial sensitivity gross P&L: ${df.loc[partial_valued, 'partial_audit_gross'].sum():,.6f}",
        f"Partial sensitivity fees at 0.5 bps: ${df.loc[partial_valued, 'partial_audit_fee_05bps'].sum():,.6f}",
        f"Partial sensitivity net P&L: ${df.loc[partial_valued, 'partial_audit_net'].sum():,.6f}",
        "Mean signed movement USD/share (full fills): "
        f"{df.loc[valued, 'audit_signed_move'].mean():.8f}",
        "Median signed movement USD/share (full fills): "
        f"{df.loc[valued, 'audit_signed_move'].median():.8f}",
        "99th percentile absolute movement USD/share (full fills): "
        f"{df.loc[valued, 'audit_signed_move'].abs().quantile(0.99):.8f}",
    ])

    top = df.loc[valued].copy()
    top["abs_gross"] = top["audit_gross"].abs()
    top = top.nlargest(12, "abs_gross")

    lines.extend([
        "",
        "5) Largest absolute candidate contributions",
    ])

    columns = [
        "timestamp_ns",
        "symbol",
        "quote_side",
        "outcome",
        "quote_price",
        "mid",
        "audit_mark_mid",
        "quote_qty",
        "filled_qty",
        "fill_fraction",
        "audit_signed_move",
        "audit_gross",
    ]

    lines.append(
        top[columns].to_string(
            index=False,
            float_format=lambda x: f"{x:.8f}",
        )
    )

    candidate_file = Path(
        "research/m9_economic_decision_test_candidates.csv"
    )

    lines.extend([
        "",
        "6) Previous economic-test candidate export",
    ])

    if candidate_file.exists():
        candidates = pd.read_csv(candidate_file)

        lines.append(
            f"Candidate export rows: {len(candidates):,}"
        )
        lines.append(
            f"Columns: {', '.join(candidates.columns)}"
        )

        for column in [
            "quote_price",
            "mid",
            "resolution_mid",
            "future_mid_100ms",
            "filled_qty",
        ]:
            if column in candidates.columns:
                lines.append(
                    f"{column}: {describe(candidates[column])}"
                )
    else:
        lines.append("Candidate export not found; skipped.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n".join(lines))
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()