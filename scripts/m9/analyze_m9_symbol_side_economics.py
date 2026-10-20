from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("research/m9_economic_decision_test_candidates.csv")
OUT = Path("research/m9_symbol_side_economics.csv")
DIFF_OUT = Path("research/m9_symbol_side_policy_differences.csv")

SCENARIOS = ["conservative", "central", "optimistic"]

POLICIES = [
    "m8_selected",
    "ml_selected",
    "ml_stratified_selected",
    "ml_activity_matched_selected",
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
        "symbol",
        "quote_side",
        "timestamp_ns",
        "ml_score",
        "outcome",
        "filled_qty",
        "net_mtm_pnl",
        "m8_selected",
        "ml_selected",
        "ml_stratified_selected",
    }

    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    for column in [
        "m8_selected",
        "ml_selected",
        "ml_stratified_selected",
    ]:
        df[column] = parse_bool(df[column])

    for column in [
        "timestamp_ns",
        "ml_score",
        "filled_qty",
        "net_mtm_pnl",
    ]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    if df[["timestamp_ns", "ml_score", "filled_qty", "net_mtm_pnl"]].isna().any().any():
        raise ValueError(
            "Missing timestamp, ML score, fill quantity, or P&L. "
            "Investigate before continuing."
        )

    df["outcome"] = df["outcome"].astype(str).str.upper()
    summaries = []

    for scenario in SCENARIOS:
        part = df[df["scenario"].eq(scenario)].copy()

        if part.empty:
            raise ValueError(f"No rows for scenario {scenario}")

        # Same total activity budget as M8, ranked by ML score.
        # This is a retrospective activity-matched diagnostic.
        k = int(part["m8_selected"].sum())
        part["ml_activity_matched_selected"] = False

        chosen = (
            part.sort_values(
                ["ml_score", "timestamp_ns"],
                ascending=[False, True],
                kind="mergesort",
            )
            .head(k)
            .index
        )

        part.loc[chosen, "ml_activity_matched_selected"] = True

        if int(part["ml_activity_matched_selected"].sum()) != k:
            raise AssertionError("Activity-matched selection count differs from M8.")

        part["full_fill"] = (
            part["outcome"].eq("FILL") & part["filled_qty"].gt(0)
        )
        part["partial_fill"] = (
            part["filled_qty"].gt(0) & ~part["outcome"].eq("FILL")
        )

        for policy in POLICIES:
            selected = part[part[policy]]

            for (symbol, side), group in selected.groupby(
                ["symbol", "quote_side"], sort=True
            ):
                full = group[group["full_fill"]]

                summaries.append({
                    "scenario": scenario,
                    "symbol": symbol,
                    "quote_side": side,
                    "policy": policy,
                    "selected_candidates": len(group),
                    "full_fill_candidates": int(group["full_fill"].sum()),
                    "partial_fill_candidates": int(group["partial_fill"].sum()),
                    "full_fill_rate_pct": (
                        100 * group["full_fill"].mean()
                        if len(group) else np.nan
                    ),
                    "net_full_fill_pnl_usd": float(
                        group["net_mtm_pnl"].sum()
                    ),
                    "net_pnl_per_selected_candidate_usd": (
                        float(group["net_mtm_pnl"].mean())
                        if len(group) else np.nan
                    ),
                    "net_pnl_per_full_fill_candidate_usd": (
                        float(full["net_mtm_pnl"].mean())
                        if len(full) else np.nan
                    ),
                    "gross_pnl_per_full_fill_candidate_usd": (
                        float(full["gross_pnl"].mean())
                        if len(full) and "gross_pnl" in full.columns
                        else np.nan
                    ),
                    "partial_sensitivity_net_pnl_usd": (
                        float(group["partial_sensitivity_net_pnl"].sum())
                        if "partial_sensitivity_net_pnl" in group.columns
                        else np.nan
                    ),
                })

    result = pd.DataFrame(summaries)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT, index=False)

    # Compare each policy with M8 within the same scenario, symbol, and side.
    metric = "net_pnl_per_selected_candidate_usd"

    pivot = result.pivot(
        index=["scenario", "symbol", "quote_side"],
        columns="policy",
        values=metric,
    ).reset_index()

    for policy in POLICIES:
        if policy != "m8_selected" and policy in pivot.columns:
            pivot[f"{policy}_minus_m8_usd_per_selected"] = (
                pivot[policy] - pivot["m8_selected"]
            )

    pivot.to_csv(DIFF_OUT, index=False)

    print("\nSymbol/side economics:")
    print(
        result.sort_values(
            ["scenario", "symbol", "quote_side", "policy"]
        ).to_string(index=False, float_format=lambda x: f"{x:.5f}")
    )

    print("\nPolicy differences against M8:")
    print(
        pivot.to_string(index=False, float_format=lambda x: f"{x:.5f}")
    )

    print(f"\nSaved: {OUT}")
    print(f"Saved: {DIFF_OUT}")
    print(
        "\nReminder: symbol/side subsets are diagnostic and can be small. "
        "Do not claim a winning symbol or side from these results alone."
    )


if __name__ == "__main__":
    main()