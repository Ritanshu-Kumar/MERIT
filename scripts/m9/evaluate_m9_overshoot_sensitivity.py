from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("research/m9_economic_decision_test_candidates.csv")
OUT = Path("research/m9_overshoot_sensitivity.csv")

SCENARIOS = {
    "optimistic": "10ms",
    "central": "100ms",
    "conservative": "1s",
}

HORIZONS_NS = {
    "10ms": 10_000_000,
    "100ms": 100_000_000,
    "500ms": 500_000_000,
    "1s": 1_000_000_000,
}

LIMITS_MS = [25, 50, 100, 250, 500, 1000]

POLICIES = [
    "m8_selected",
    "ml_selected",
    "ml_stratified_selected",
]


def parse_bool(series):
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)

    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map({
            "true": True,
            "false": False,
            "1": True,
            "0": False,
        })
        .fillna(False)
    )


def main():
    if not DATA.exists():
        raise FileNotFoundError(
            f"Missing {DATA}. Run evaluate_m8_ml_economics.py first."
        )

    df = pd.read_csv(DATA)

    required = {
        "scenario",
        "outcome",
        "quote_side",
        "quote_price",
        "filled_qty",
        "evaluated_qty",
        "fees",
        "fill_timestamp_ns",
        *POLICIES,
    }

    for horizon in HORIZONS_NS:
        required.update({
            f"future_mid_{horizon}",
            f"future_mid_{horizon}_timestamp_ns",
        })

    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    for policy in POLICIES:
        df[policy] = parse_bool(df[policy])

    for column in [
        "quote_price",
        "filled_qty",
        "evaluated_qty",
        "fees",
        "fill_timestamp_ns",
    ]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df["outcome"] = df["outcome"].astype(str).str.upper()
    results = []

    for scenario, horizon in SCENARIOS.items():
        block = df[df["scenario"].eq(scenario)].copy()

        price_col = f"future_mid_{horizon}"
        time_col = f"future_mid_{horizon}_timestamp_ns"
        target_ns = HORIZONS_NS[horizon]

        block[price_col] = pd.to_numeric(
            block[price_col], errors="coerce"
        )
        block[time_col] = pd.to_numeric(
            block[time_col], errors="coerce"
        )

        block["overshoot_ms"] = (
            block[time_col]
            - block["fill_timestamp_ns"]
            - target_ns
        ) / 1_000_000

        is_full_fill = (
            block["outcome"].eq("FILL")
            & block["filled_qty"].gt(0)
            & block["evaluated_qty"].gt(0)
        )

        full_fills = block[is_full_fill].copy()

        valid = (
            full_fills["fill_timestamp_ns"].notna()
            & full_fills[time_col].notna()
            & full_fills[price_col].notna()
            & full_fills["quote_price"].notna()
            & full_fills["fees"].notna()
            & full_fills["quote_side"].isin(["BUY", "SELL"])
        )

        if not valid.all():
            raise ValueError(
                f"{scenario}: {int((~valid).sum())} full fills lack "
                "required P&L or timestamp data. Investigate rather "
                "than silently dropping them."
            )

        if (full_fills["overshoot_ms"] < 0).any():
            raise ValueError(
                f"{scenario}: observations before target horizon found."
            )

        buy = full_fills["quote_side"].eq("BUY")

        full_fills["signed_move"] = np.where(
            buy,
            full_fills[price_col] - full_fills["quote_price"],
            full_fills["quote_price"] - full_fills[price_col],
        )

        full_fills["gross_pnl"] = (
            full_fills["signed_move"]
            * full_fills["evaluated_qty"]
        )

        full_fills["net_pnl"] = (
            full_fills["gross_pnl"] - full_fills["fees"]
        )

        print(f"\nScenario: {scenario} | Horizon: {horizon}")

        for policy in POLICIES:
            selected_count = int(block[policy].sum())

            selected_fills = full_fills[
                full_fills[policy]
            ].copy()

            baseline_net = float(
                selected_fills["net_pnl"].sum()
            )
            fill_count = len(selected_fills)

            for limit in LIMITS_MS:
                retained = selected_fills[
                    selected_fills["overshoot_ms"] <= limit
                ]

                late = selected_fills[
                    selected_fills["overshoot_ms"] > limit
                ]

                retained_count = len(retained)

                results.append({
                    "scenario": scenario,
                    "horizon": horizon,
                    "policy": policy,
                    "overshoot_limit_ms": limit,
                    "selected_candidates": selected_count,
                    "full_fill_candidates": fill_count,
                    "retained_markouts": retained_count,
                    "retained_coverage_pct": (
                        100 * retained_count / fill_count
                        if fill_count else np.nan
                    ),
                    "baseline_unfiltered_full_fill_net_pnl_usd":
                        baseline_net,
                    "retained_observation_net_pnl_usd": float(
                        retained["net_pnl"].sum()
                    ),
                    "late_observation_net_pnl_usd": float(
                        late["net_pnl"].sum()
                    ),
                })

                print(
                    f"  {policy:24s} limit={limit:4d} ms | "
                    f"coverage={100 * retained_count / fill_count if fill_count else 0:6.2f}% | "
                    f"retained net=${retained['net_pnl'].sum():10.2f} | "
                    f"late net=${late['net_pnl'].sum():10.2f}"
                )

    result_df = pd.DataFrame(results)
    result_df.to_csv(OUT, index=False)

    print(f"\nSaved: {OUT}")
    print(
        "Important: retained and late P&L are sensitivity decompositions, "
        "not executable portfolio P&L. Do not compare retained totals "
        "without considering the different coverage and selection counts."
    )


if __name__ == "__main__":
    main()