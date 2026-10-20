from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("research/m9_economic_decision_test_candidates.csv")

SUMMARY_OUT = Path("research/m9_activity_matched_summary.csv")
DIFF_OUT = Path("research/m9_activity_matched_differences.csv")

SCENARIOS = ["conservative", "central", "optimistic"]

POLICIES = [
    "m8_selected",
    "ml_selected",
    "ml_stratified_selected",
    "ml_activity_matched_selected",
]

BLOCK_NS = 60_000_000_000
BOOTSTRAP_REPS = 2000
SEED = 20261019


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


def add_activity_matched_policy(df):
    """Select the same total candidate count as M8 using ML scores only."""
    result = df.copy()
    result["ml_activity_matched_selected"] = False

    k = int(result["m8_selected"].sum())

    if k == 0:
        return result

    if result["ml_score"].isna().any():
        raise ValueError(
            "Missing ML scores; cannot construct an activity-matched policy."
        )

    chosen = (
        result.sort_values(
            ["ml_score", "timestamp_ns"],
            ascending=[False, True],
            kind="mergesort",
        )
        .head(k)
        .index
    )

    result.loc[chosen, "ml_activity_matched_selected"] = True

    if int(result["ml_activity_matched_selected"].sum()) != k:
        raise AssertionError("ML selection count does not match M8.")

    return result


def summarize_policy(df, policy, scenario):
    selected = df[df[policy]]
    n = len(selected)

    full_fill = (
        selected["outcome"].eq("FILL")
        & selected["filled_qty"].gt(0)
    )
    partial_fill = (
        selected["filled_qty"].gt(0)
        & ~selected["outcome"].eq("FILL")
    )

    net = selected["net_mtm_pnl"]

    return {
        "scenario": scenario,
        "policy": policy,
        "selected_candidates": n,
        "selection_rate_pct": (
            100 * n / len(df) if len(df) else np.nan
        ),
        "candidates_with_any_fill": int(
            selected["filled_qty"].gt(0).sum()
        ),
        "full_fill_candidates": int(full_fill.sum()),
        "partial_fill_candidates": int(partial_fill.sum()),
        "full_fill_rate_pct": (
            100 * full_fill.sum() / n if n else np.nan
        ),
        "net_full_fill_pnl_usd": float(net.sum()),
        "net_pnl_per_selected_candidate_usd": (
            float(net.mean()) if n else np.nan
        ),
        "partial_sensitivity_net_pnl_usd": float(
            selected["partial_sensitivity_net_pnl"].sum()
        ) if "partial_sensitivity_net_pnl" in selected.columns else np.nan,
    }


def block_bootstrap_difference(df, left, right, rng):
    """Difference in net P&L per selected candidate, resampling 60s blocks."""
    blocks = np.sort(df["time_block"].unique())

    if len(blocks) == 0:
        return np.nan, np.nan, np.nan, 0

    stats = {}

    for policy in [left, right]:
        p = df[df[policy]].groupby("time_block")["net_mtm_pnl"].agg(
            total="sum",
            count="size",
        )

        sums = p["total"].reindex(blocks, fill_value=0).to_numpy(float)
        counts = p["count"].reindex(blocks, fill_value=0).to_numpy(float)

        stats[policy] = (sums, counts)

    left_sum, left_count = stats[left]
    right_sum, right_count = stats[right]

    if left_count.sum() == 0 or right_count.sum() == 0:
        return np.nan, np.nan, np.nan, 0

    observed = (
        left_sum.sum() / left_count.sum()
        - right_sum.sum() / right_count.sum()
    )

    draws = []
    n_blocks = len(blocks)

    for _ in range(BOOTSTRAP_REPS):
        picks = rng.integers(0, n_blocks, size=n_blocks)

        l_count = left_count[picks].sum()
        r_count = right_count[picks].sum()

        if l_count == 0 or r_count == 0:
            continue

        difference = (
            left_sum[picks].sum() / l_count
            - right_sum[picks].sum() / r_count
        )
        draws.append(difference)

    if not draws:
        return float(observed), np.nan, np.nan, 0

    low, high = np.quantile(draws, [0.025, 0.975])

    return float(observed), float(low), float(high), len(draws)


def main():
    if not DATA.exists():
        raise FileNotFoundError(
            f"Missing {DATA}. Run evaluate_m8_ml_economics.py first."
        )

    df = pd.read_csv(DATA)

    required = {
        "scenario",
        "timestamp_ns",
        "symbol",
        "quote_side",
        "outcome",
        "filled_qty",
        "ml_score",
        "m8_selected",
        "ml_selected",
        "ml_stratified_selected",
        "net_mtm_pnl",
    }

    missing = sorted(required - set(df.columns))

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    for col in [
        "m8_selected",
        "ml_selected",
        "ml_stratified_selected",
    ]:
        df[col] = parse_bool(df[col])

    for col in [
        "timestamp_ns",
        "filled_qty",
        "ml_score",
        "net_mtm_pnl",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if df[["timestamp_ns", "ml_score", "net_mtm_pnl"]].isna().any().any():
        raise ValueError(
            "Missing timestamp, ML score, or net P&L. Investigate before comparing."
        )

    df["outcome"] = df["outcome"].astype(str).str.upper()
    df["time_block"] = (
        df["timestamp_ns"].astype("int64") // BLOCK_NS
    )

    summary_rows = []
    difference_rows = []
    rng = np.random.default_rng(SEED)

    for scenario in SCENARIOS:
        part = df[df["scenario"].eq(scenario)].copy()

        if part.empty:
            raise ValueError(f"No rows for scenario: {scenario}")

        part = add_activity_matched_policy(part)

        n_m8 = int(part["m8_selected"].sum())
        n_ml = int(part["ml_activity_matched_selected"].sum())

        print(f"\n{'=' * 64}")
        print(f"Scenario: {scenario}")
        print(f"M8 selection count: {n_m8:,}")
        print(f"Activity-matched ML selection count: {n_ml:,}")

        for policy in POLICIES:
            row = summarize_policy(part, policy, scenario)
            summary_rows.append(row)

            print(
                f"{policy:28s} "
                f"selected={row['selected_candidates']:6,d} | "
                f"full fills={row['full_fill_candidates']:5,d} | "
                f"net/selected=${row['net_pnl_per_selected_candidate_usd']:.6f}"
            )

        comparisons = [
            (
                "ML_train_threshold_minus_M8",
                "ml_selected",
                "m8_selected",
            ),
            (
                "ML_activity_matched_minus_M8",
                "ml_activity_matched_selected",
                "m8_selected",
            ),
            (
                "ML_symbol_side_matched_minus_M8",
                "ml_stratified_selected",
                "m8_selected",
            ),
        ]

        for label, left, right in comparisons:
            observed, low, high, n_boot = block_bootstrap_difference(
                part, left, right, rng
            )

            difference_rows.append({
                "scenario": scenario,
                "comparison": label,
                "difference_net_pnl_per_selected_candidate_usd": observed,
                "ci95_low_usd": low,
                "ci95_high_usd": high,
                "bootstrap_replicates_used": n_boot,
            })

    summary = pd.DataFrame(summary_rows)
    differences = pd.DataFrame(difference_rows)

    SUMMARY_OUT.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY_OUT, index=False)
    differences.to_csv(DIFF_OUT, index=False)

    print("\nBootstrap differences (left policy minus M8):")
    print(differences.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    print(f"\nSaved summary: {SUMMARY_OUT}")
    print(f"Saved differences: {DIFF_OUT}")

    print(
        "\nInterpretation: activity-matched ML uses test-period scores to "
        "enforce the M8 selection budget, without consulting test outcomes "
        "or P&L. Treat it as an activity-matched ranking diagnostic, not "
        "a frozen-threshold deployment result. The single-day sample remains exploratory."
    )


if __name__ == "__main__":
    main()