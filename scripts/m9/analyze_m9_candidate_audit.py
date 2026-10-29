
from pathlib import Path

import numpy as np
import pandas as pd

INPUT = Path(
    "research/m9_cross_date_candidate_audit_2019-08-30.csv"
)
OUTPUT_DIR = Path("research")

df = pd.read_csv(INPUT)

required = {
    "symbol",
    "quote_side",
    "m8_selected",
    "m9_ml_selected",
    "m9_ml_activity_matched_selected",
    "outcome",
    "filled_qty",
    "fill_fraction",
    "fill_before_adverse",
    "quote_price",
    "future_mid_1s",
    "mid",
}

missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {sorted(missing)}")

for col in [
    "m8_selected",
    "m9_ml_selected",
    "m9_ml_activity_matched_selected",
    "fill_before_adverse",
]:
    df[col] = df[col].fillna(False).astype(bool)

# Compare the original frozen M8 rule with the frozen M9 threshold.
df["selection_group"] = np.select(
    [
        df["m8_selected"] & df["m9_ml_selected"],
        df["m8_selected"] & ~df["m9_ml_selected"],
        ~df["m8_selected"] & df["m9_ml_selected"],
    ],
    ["both_selected", "m8_only", "m9_only"],
    default="neither",
)

# Side-adjusted one-second markout, in dollars per share.
# Buy-side quotes benefit from a rising midpoint; sell-side quotes
# benefit from a falling midpoint.
side = (
    df["quote_side"]
    .astype(str)
    .str.lower()
    .str.strip()
    .map({
        "buy": 1,
        "bid": 1,
        "b": 1,
        "sell": -1,
        "ask": -1,
        "s": -1,
    })
)

if side.isna().any():
    bad = sorted(
        df.loc[side.isna(), "quote_side"].astype(str).unique()
    )
    raise ValueError(f"Unrecognized quote_side values: {bad}")

df["markout_1s_per_share"] = (
    side * (df["future_mid_1s"] - df["quote_price"])
)

df["is_full_fill"] = (
    df["outcome"].astype(str).str.upper().eq("FILL")
    & df["filled_qty"].gt(0)
)

df["is_adverse"] = (
    df["outcome"].astype(str).str.upper().eq("ADVERSE")
)

# All candidates, grouped by the selection decision.
group_summary = (
    df.groupby("selection_group", dropna=False)
    .agg(
        candidates=("candidate_id", "size"),
        full_fills=("is_full_fill", "sum"),
        adverse_outcomes=("is_adverse", "sum"),
        fill_before_adverse_rate=("fill_before_adverse", "mean"),
        mean_markout_all=("markout_1s_per_share", "mean"),
    )
    .reset_index()
)

group_summary["full_fill_rate_pct"] = (
    100 * group_summary["full_fills"] / group_summary["candidates"]
)
group_summary["adverse_rate_pct"] = (
    100 * group_summary["adverse_outcomes"] / group_summary["candidates"]
)

# Markouts conditional on a full fill.
filled = df.loc[df["is_full_fill"]].copy()

fill_summary = (
    filled.groupby("selection_group", dropna=False)
    .agg(
        full_fills=("candidate_id", "size"),
        mean_markout_1s=("markout_1s_per_share", "mean"),
        median_markout_1s=("markout_1s_per_share", "median"),
        mean_filled_qty=("filled_qty", "mean"),
        mean_fill_fraction=("fill_fraction", "mean"),
    )
    .reset_index()
)

# Break down selection groups by symbol and quote side.
by_symbol_side = (
    df.groupby(
        ["symbol", "quote_side", "selection_group"],
        dropna=False,
    )
    .agg(
        candidates=("candidate_id", "size"),
        full_fills=("is_full_fill", "sum"),
        adverse_outcomes=("is_adverse", "sum"),
        mean_markout_all=("markout_1s_per_share", "mean"),
    )
    .reset_index()
)

by_symbol_side["full_fill_rate_pct"] = (
    100 * by_symbol_side["full_fills"] / by_symbol_side["candidates"]
)
by_symbol_side["adverse_rate_pct"] = (
    100 * by_symbol_side["adverse_outcomes"] / by_symbol_side["candidates"]
)

# The activity-matched selector chooses the same count as M8 within
# symbol × quote_side. Compare its full-fill markouts with M8's.
df["matched_group"] = np.select(
    [
        df["m8_selected"] & df["m9_ml_activity_matched_selected"],
        df["m8_selected"] & ~df["m9_ml_activity_matched_selected"],
        ~df["m8_selected"] & df["m9_ml_activity_matched_selected"],
    ],
    ["both_selected", "m8_only", "matched_m9_only"],
    default="neither",
)

matched_by_symbol_side = (
    df.groupby(
        ["symbol", "quote_side", "matched_group"],
        dropna=False,
    )
    .agg(
        candidates=("candidate_id", "size"),
        full_fills=("is_full_fill", "sum"),
        adverse_outcomes=("is_adverse", "sum"),
    )
    .reset_index()
)

matched_by_symbol_side["full_fill_rate_pct"] = (
    100 * matched_by_symbol_side["full_fills"]
    / matched_by_symbol_side["candidates"]
)

group_summary.to_csv(
    OUTPUT_DIR / "m9_candidate_audit_group_summary.csv", index=False
)
fill_summary.to_csv(
    OUTPUT_DIR / "m9_candidate_audit_fill_markouts.csv", index=False
)
by_symbol_side.to_csv(
    OUTPUT_DIR / "m9_candidate_audit_by_symbol_side.csv", index=False
)
matched_by_symbol_side.to_csv(
    OUTPUT_DIR / "m9_candidate_audit_activity_matched.csv", index=False
)

print("\nSELECTION GROUP SUMMARY")
print(group_summary.to_string(index=False))

print("\nFULL-FILL MARKOUTS BY SELECTION GROUP")
print(fill_summary.to_string(index=False))

print("\nSYMBOL × SIDE BREAKDOWN")
print(by_symbol_side.to_string(index=False))

print("\nACTIVITY-MATCHED BREAKDOWN")
print(matched_by_symbol_side.to_string(index=False))

# Policy-level economic breakdown by symbol and quote side.
# Central scenario: 100ms markout, 0.5 bps fee, full filled quantity.
# Conservative scenario: 1s markout, 1 bps fee, 50% quantity.
# Optimistic scenario: 10ms markout, no fee, full quantity.

df["gross_central"] = (
    side * (df["future_mid_100ms"] - df["quote_price"])
    * df["filled_qty"]
)
df["fee_central"] = (
    df["quote_price"] * df["filled_qty"] * 0.5 / 10000
)

df["gross_conservative"] = (
    side * (df["future_mid_1s"] - df["quote_price"])
    * df["filled_qty"] * 0.5
)
df["fee_conservative"] = (
    df["quote_price"] * df["filled_qty"] * 0.5
    * 1.0 / 10000
)

df["gross_optimistic"] = (
    side * (df["future_mid_10ms"] - df["quote_price"])
    * df["filled_qty"]
)

policies = {
    "M8": df["m8_selected"],
    "M9_frozen": df["m9_ml_selected"],
    "M9_activity_matched": df["m9_ml_activity_matched_selected"],
}

economic_rows = []

for policy_name, selected_mask in policies.items():
    selected = df.loc[selected_mask].copy()
    fills = selected.loc[selected["is_full_fill"]].copy()

    counts = (
        selected.groupby(["symbol", "quote_side"])
        .size()
        .rename("selected_candidates")
        .reset_index()
    )

    economics = (
        fills.groupby(["symbol", "quote_side"])
        .agg(
            full_fills=("candidate_id", "size"),
            central_gross=("gross_central", "sum"),
            central_fees=("fee_central", "sum"),
            conservative_gross=("gross_conservative", "sum"),
            conservative_fees=("fee_conservative", "sum"),
            optimistic_gross=("gross_optimistic", "sum"),
        )
        .reset_index()
    )

    result = counts.merge(
        economics,
        on=["symbol", "quote_side"],
        how="left",
    ).fillna(0)

    result["central_net"] = (
        result["central_gross"] - result["central_fees"]
    )
    result["conservative_net"] = (
        result["conservative_gross"]
        - result["conservative_fees"]
    )
    result["policy"] = policy_name

    economic_rows.append(result)

economic_breakdown = pd.concat(
    economic_rows,
    ignore_index=True,
)

economic_path = (
    OUTPUT_DIR / "m9_candidate_audit_economics_by_symbol_side.csv"
)
economic_breakdown.to_csv(economic_path, index=False)

print("\nECONOMIC BREAKDOWN BY POLICY, SYMBOL AND SIDE")
print(
    economic_breakdown.sort_values(
        ["policy", "symbol", "quote_side"]
    ).to_string(index=False)
)
print("\nSaved:", economic_path)


# Test whether higher M9 scores correspond to better
# candidate-level economics, separately by symbol and quote side.

df["m9_score_decile"] = (
    df.groupby(["symbol", "quote_side"])["m9_ml_score"]
    .transform(
        lambda s: pd.qcut(
            s.rank(method="first"),
            q=10,
            labels=False,
        ) + 1
    )
)

# Assign economic value only to full-fill candidates.
# Non-fills receive zero in this candidate-level proxy.
df["central_net_proxy"] = 0.0

filled_mask = df["is_full_fill"]

df.loc[filled_mask, "central_net_proxy"] = (
    side.loc[filled_mask]
    * (
        df.loc[filled_mask, "future_mid_100ms"]
        - df.loc[filled_mask, "quote_price"]
    )
    * df.loc[filled_mask, "filled_qty"]
    - (
        df.loc[filled_mask, "quote_price"]
        * df.loc[filled_mask, "filled_qty"]
        * 0.5 / 10000
    )
)

score_bins = (
    df.groupby(
        ["symbol", "quote_side", "m9_score_decile"],
        observed=True,
    )
    .agg(
        candidates=("candidate_id", "size"),
        mean_score=("m9_ml_score", "mean"),
        full_fills=("is_full_fill", "sum"),
        adverse_outcomes=("is_adverse", "sum"),
        net_proxy_usd=("central_net_proxy", "sum"),
    )
    .reset_index()
)

score_bins["full_fill_rate_pct"] = (
    100 * score_bins["full_fills"] / score_bins["candidates"]
)

score_bins["adverse_rate_pct"] = (
    100 * score_bins["adverse_outcomes"] / score_bins["candidates"]
)

score_bins["net_proxy_per_candidate_usd"] = (
    score_bins["net_proxy_usd"] / score_bins["candidates"]
)

score_path = (
    OUTPUT_DIR / "m9_candidate_score_deciles_by_symbol_side.csv"
)
score_bins.to_csv(score_path, index=False)

print("\nM9 SCORE DECILES BY SYMBOL AND SIDE")
print(score_bins.to_string(index=False))
print("\nSaved:", score_path)


print("\nSaved four diagnostic CSV files in:", OUTPUT_DIR)
