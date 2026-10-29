from pathlib import Path

import numpy as np
import pandas as pd

SOURCE = Path("research/m9_time_sampled_2019-08-30_regular_asof.csv")
DETAIL_OUTPUT = Path("research/m9_2019-08-30_regular_fill_markouts.csv")
SUMMARY_OUTPUT = Path("research/m9_2019-08-30_regular_fill_markout_summary.csv")

CLOSE_NS = 57_600_000_000_000

HORIZONS = {
    "10ms": 10_000_000,
    "100ms": 100_000_000,
    "500ms": 500_000_000,
    "1s": 1_000_000_000,
    "5s": 5_000_000_000,
}


def summarize(group, horizon, scope, group_name):
    markout_col = f"markout_{horizon}_usd_per_share"
    bps_col = f"markout_{horizon}_bps"
    movement_col = f"post_fill_mid_move_{horizon}"

    g = group[markout_col].dropna()
    if g.empty:
        return None

    bps = group.loc[g.index, bps_col].dropna()
    movement = group.loc[g.index, movement_col].dropna()

    event_ids = pd.to_numeric(
        group.loc[g.index, "fill_resolution_message_index"],
        errors="coerce",
    )

    return {
        "horizon": horizon,
        "scope": scope,
        "group": group_name,
        "n_fills": len(g),
        "unique_resolution_events": int(event_ids.nunique()),
        "mean_markout_per_share": g.mean(),
        "median_markout_per_share": g.median(),
        "p05_markout_per_share": g.quantile(0.05),
        "p95_markout_per_share": g.quantile(0.95),
        "positive_markout_pct": 100 * g.gt(0).mean(),
        "negative_markout_pct": 100 * g.lt(0).mean(),
        "mean_markout_bps": bps.mean(),
        "mean_signed_post_fill_mid_move": movement.mean(),
    }


def main():
    if not SOURCE.exists():
        raise FileNotFoundError(f"Source file not found: {SOURCE}")

    df = pd.read_csv(SOURCE, low_memory=False)
    fills = df.loc[df["outcome"].eq("FILL")].copy()

    numeric_cols = [
        "timestamp_ns",
        "fill_timestamp_ns",
        "quote_price",
        "resolution_mid",
        "fill_resolution_message_index",
    ]

    for col in numeric_cols:
        fills[col] = pd.to_numeric(fills[col], errors="coerce")

    fills["side_sign"] = fills["quote_side"].map(
        {"BUY": 1.0, "SELL": -1.0}
    )

    if fills["side_sign"].isna().any():
        raise ValueError("Unexpected quote_side in FILL rows.")

    summary_rows = []

    print("=" * 65)
    print("REGULAR-SESSION SIDE-ADJUSTED MARKOUT ANALYSIS")
    print("=" * 65)
    print("Input rows:", len(df))
    print("FILL rows:", len(fills))
    print("Missing fill timestamps:", int(fills["fill_timestamp_ns"].isna().sum()))
    print("Missing fill-time midpoints:", int(fills["resolution_mid"].isna().sum()))

    for horizon, offset_ns in HORIZONS.items():
        mid_col = f"future_mid_{horizon}"
        ts_col = f"future_mid_{horizon}_timestamp_ns"

        future_mid = pd.to_numeric(fills[mid_col], errors="coerce")
        observation_ts = pd.to_numeric(fills[ts_col], errors="coerce")
        fill_ts = fills["fill_timestamp_ns"]
        target_ts = fill_ts + offset_ns

        before_fill = observation_ts.lt(fill_ts)
        after_target = observation_ts.gt(target_ts)
        at_or_after_close = target_ts.ge(CLOSE_NS)

        valid = (
            future_mid.notna()
            & observation_ts.notna()
            & fill_ts.notna()
            & fills["quote_price"].notna()
            & fills["resolution_mid"].notna()
            & fills["side_sign"].notna()
            & ~before_fill
            & ~after_target
            & ~at_or_after_close
        )

        # M_h from the research protocol:
        # BUY: future midpoint - fill price
        # SELL: fill price - future midpoint
        markout = fills["side_sign"] * (
            future_mid - fills["quote_price"]
        )

        # Separate diagnostic: signed midpoint movement after the fill.
        # This excludes the spread component of the entry price.
        post_fill_move = fills["side_sign"] * (
            future_mid - fills["resolution_mid"]
        )

        markout_col = f"markout_{horizon}_usd_per_share"
        bps_col = f"markout_{horizon}_bps"
        movement_col = f"post_fill_mid_move_{horizon}"
        lag_col = f"markout_{horizon}_gap_before_target_ms"

        fills[markout_col] = markout.where(valid)
        fills[bps_col] = (
            10_000 * markout / fills["quote_price"]
        ).where(valid)
        fills[movement_col] = post_fill_move.where(valid)
        fills[lag_col] = (
            (target_ts - observation_ts) / 1_000_000
        ).where(valid)

        print(f"\n{horizon}")
        print("  Valid marks before session close:", int(valid.sum()))
        print("  Targets at/after regular close:", int(at_or_after_close.sum()))
        print("  Observations before fill:", int(before_fill.sum()))
        print("  Observations after target:", int(after_target.sum()))
        print(
            "  Median gap before target (ms):",
            round(float(fills.loc[valid, lag_col].median()), 3)
            if valid.any() else None,
        )

        eligible = fills.loc[valid]

        groups = [
            ("overall", "ALL", eligible.groupby(lambda _: "ALL")),
            ("by_side", None, eligible.groupby("quote_side")),
            ("by_symbol", None, eligible.groupby("symbol")),
            (
                "by_symbol_side",
                None,
                eligible.groupby(["symbol", "quote_side"]),
            ),
        ]

        for scope, _, grouped in groups:
            for key, group in grouped:
                if isinstance(key, tuple):
                    group_name = "|".join(map(str, key))
                else:
                    group_name = str(key)

                row = summarize(group, horizon, scope, group_name)
                if row is not None:
                    summary_rows.append(row)

    # Preserve the complete fill-level diagnostics.
    fills.to_csv(DETAIL_OUTPUT, index=False)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(SUMMARY_OUTPUT, index=False)

    # Multiple candidates can resolve on the same exchange event.
    event_counts = fills[
        "fill_resolution_message_index"
    ].value_counts()

    print("\n" + "=" * 65)
    print("PRIMARY HORIZON: 1 SECOND")
    print("=" * 65)

    primary = summary[
        (summary["horizon"] == "1s")
        & summary["scope"].isin(["overall", "by_side", "by_symbol"])
    ]

    if not primary.empty:
        print(
            primary[
                [
                    "scope",
                    "group",
                    "n_fills",
                    "unique_resolution_events",
                    "mean_markout_per_share",
                    "median_markout_per_share",
                    "p05_markout_per_share",
                    "p95_markout_per_share",
                    "positive_markout_pct",
                    "mean_markout_bps",
                    "mean_signed_post_fill_mid_move",
                ]
            ].round(6).to_string(index=False)
        )

    print("\nFILL EVENT DEPENDENCE")
    print("Unique resolution events:", int(event_counts.size))
    print("Events resolving multiple candidates:", int(event_counts.gt(1).sum()))
    print("Maximum candidates per resolution event:", int(event_counts.max())
          if not event_counts.empty else 0)

    print("\nSaved detailed results:", DETAIL_OUTPUT)
    print("Saved summary results:", SUMMARY_OUTPUT)
    print(
        "\nThese are descriptive statistics, not an IID significance test "
        "or proof of strategy profitability."
    )


if __name__ == "__main__":
    main()