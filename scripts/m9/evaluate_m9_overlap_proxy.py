from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("research/m9_economic_decision_test_candidates.csv")
SUMMARY_OUT = Path("research/m9_overlap_proxy_summary.csv")
DIFF_OUT = Path("research/m9_overlap_proxy_differences.csv")

SCENARIOS = ["conservative", "central", "optimistic"]

POLICIES = [
    "m8_selected",
    "ml_selected",
    "ml_stratified_selected",
    "ml_activity_matched_selected",
]

BLOCK_NS = 60_000_000_000
BOOTSTRAP_REPS = 2000
SEED = 20261020


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


def apply_activity_matched_policy(df):
    result = df.copy()
    result["ml_activity_matched_selected"] = False

    target_count = int(result["m8_selected"].sum())

    if target_count == 0:
        return result

    if result["ml_score"].isna().any():
        raise ValueError("Missing ML scores.")

    selected = (
        result.sort_values(
            ["ml_score", "timestamp_ns"],
            ascending=[False, True],
            kind="mergesort",
        )
        .head(target_count)
        .index
    )

    result.loc[selected, "ml_activity_matched_selected"] = True
    return result


def apply_one_active_proxy(df, policy):
    """Greedily allow one active candidate per symbol/side."""
    selected = df[df[policy]].copy()

    selected["timestamp_ns"] = pd.to_numeric(
        selected["timestamp_ns"], errors="raise"
    ).astype("int64")

    selected["time_to_resolution_ns"] = pd.to_numeric(
        selected["time_to_resolution_ns"], errors="coerce"
    )

    if selected["time_to_resolution_ns"].isna().any():
        raise ValueError(
            f"{policy}: missing resolution durations among selected candidates."
        )

    if (selected["time_to_resolution_ns"] < 0).any():
        raise ValueError(f"{policy}: negative resolution duration detected.")

    selected = selected.sort_values(
        ["symbol", "quote_side", "timestamp_ns"],
        kind="mergesort",
    )

    accepted_indices = []

    for _, group in selected.groupby(
        ["symbol", "quote_side"], sort=False
    ):
        active_until = -1

        for idx, row in group.iterrows():
            timestamp = int(row["timestamp_ns"])
            duration = int(row["time_to_resolution_ns"])

            # If the previous order has not resolved, skip this candidate.
            if timestamp < active_until:
                continue

            accepted_indices.append(idx)

            # Treat zero-duration candidates as occupying at least one ns
            # so duplicate-timestamp candidates cannot all be accepted.
            active_until = timestamp + max(duration, 1)

    return df.loc[accepted_indices].copy()


def summarize(scenario, policy, original, accepted):
    n_before = len(original)
    n_after = len(accepted)

    full = (
        accepted["outcome"].eq("FILL")
        & accepted["filled_qty"].gt(0)
    )
    partial = (
        accepted["filled_qty"].gt(0)
        & ~accepted["outcome"].eq("FILL")
    )

    return {
        "scenario": scenario,
        "policy": policy,
        "selected_before_proxy": n_before,
        "selected_after_proxy": n_after,
        "retained_selection_pct": (
            100 * n_after / n_before if n_before else np.nan
        ),
        "full_fill_candidates_after_proxy": int(full.sum()),
        "partial_fill_candidates_after_proxy": int(partial.sum()),
        "full_fill_rate_after_proxy_pct": (
            100 * full.mean() if n_after else np.nan
        ),
        "primary_net_pnl_usd": float(
            accepted["net_mtm_pnl"].sum()
        ),
        "primary_net_pnl_per_accepted_candidate_usd": (
            float(accepted["net_mtm_pnl"].mean())
            if n_after else np.nan
        ),
        "partial_sensitivity_net_pnl_usd": float(
            accepted["partial_sensitivity_net_pnl"].sum()
        ) if "partial_sensitivity_net_pnl" in accepted.columns else np.nan,
    }


def bootstrap_difference(left, right, rng):
    """Difference in P&L per accepted candidate, paired by time block."""
    blocks = np.sort(
        pd.concat([left["time_block"], right["time_block"]]).unique()
    )

    if len(blocks) == 0:
        return np.nan, np.nan, np.nan, 0

    def block_stats(frame):
        grouped = frame.groupby("time_block")["net_mtm_pnl"].agg(
            total="sum", count="size"
        )
        totals = grouped["total"].reindex(
            blocks, fill_value=0
        ).to_numpy(float)
        counts = grouped["count"].reindex(
            blocks, fill_value=0
        ).to_numpy(float)
        return totals, counts

    left_sum, left_count = block_stats(left)
    right_sum, right_count = block_stats(right)

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

        lc = left_count[picks].sum()
        rc = right_count[picks].sum()

        if lc == 0 or rc == 0:
            continue

        draws.append(
            left_sum[picks].sum() / lc
            - right_sum[picks].sum() / rc
        )

    if not draws:
        return float(observed), np.nan, np.nan, 0

    low, high = np.quantile(draws, [0.025, 0.975])
    return float(observed), float(low), float(high), len(draws)


def main():
    if not DATA.exists():
        raise FileNotFoundError(f"Missing dataset: {DATA}")

    df = pd.read_csv(DATA)

    required = {
        "scenario", "timestamp_ns", "time_to_resolution_ns",
        "symbol", "quote_side", "outcome", "filled_qty",
        "ml_score", "m8_selected", "ml_selected",
        "ml_stratified_selected", "net_mtm_pnl",
    }

    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    for col in [
        "m8_selected", "ml_selected", "ml_stratified_selected"
    ]:
        df[col] = parse_bool(df[col])

    for col in [
        "timestamp_ns", "time_to_resolution_ns",
        "filled_qty", "ml_score", "net_mtm_pnl"
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if df[
        ["timestamp_ns", "filled_qty", "ml_score", "net_mtm_pnl"]
    ].isna().any().any():
        raise ValueError("Missing numeric inputs; inspect the candidate export.")

    df["outcome"] = df["outcome"].astype(str).str.upper()
    df["time_block"] = (
        df["timestamp_ns"].astype("int64") // BLOCK_NS
    )

    summary_rows = []
    diff_rows = []
    rng = np.random.default_rng(SEED)

    for scenario in SCENARIOS:
        part = df[df["scenario"].eq(scenario)].copy()
        if part.empty:
            raise ValueError(f"No rows for scenario: {scenario}")

        part = apply_activity_matched_policy(part)

        accepted_by_policy = {}

        print(f"\n--- {scenario.upper()} ---")

        for policy in POLICIES:
            accepted = apply_one_active_proxy(part, policy)
            accepted_by_policy[policy] = accepted

            row = summarize(
                scenario,
                policy,
                part[part[policy]],
                accepted,
            )
            summary_rows.append(row)

            print(
                f"{policy:28s} "
                f"before={row['selected_before_proxy']:6,d} "
                f"after={row['selected_after_proxy']:6,d} "
                f"retained={row['retained_selection_pct']:6.2f}% "
                f"full_fills={row['full_fill_candidates_after_proxy']:5,d} "
                f"net/accepted=${row['primary_net_pnl_per_accepted_candidate_usd']:.6f}"
            )

        for policy in [
            "ml_selected",
            "ml_activity_matched_selected",
            "ml_stratified_selected",
        ]:
            observed, low, high, n_boot = bootstrap_difference(
                accepted_by_policy[policy],
                accepted_by_policy["m8_selected"],
                rng,
            )

            diff_rows.append({
                "scenario": scenario,
                "comparison": f"{policy}_minus_m8_after_overlap_proxy",
                "difference_net_pnl_per_accepted_candidate_usd": observed,
                "ci95_low_usd": low,
                "ci95_high_usd": high,
                "bootstrap_replicates_used": n_boot,
            })

    summary = pd.DataFrame(summary_rows)
    differences = pd.DataFrame(diff_rows)

    SUMMARY_OUT.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY_OUT, index=False)
    differences.to_csv(DIFF_OUT, index=False)

    print("\nOverlap-proxy bootstrap differences:")
    print(
        differences.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    print(f"\nSaved: {SUMMARY_OUT}")
    print(f"Saved: {DIFF_OUT}")

    print(
        "\nLIMITATION: This is a greedy one-active-quote-per-symbol/side "
        "proxy using candidate resolution times. It does not model inventory, "
        "cancellations, queue priority, capital constraints, or liquidation."
    )


if __name__ == "__main__":
    main()