from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA = Path("research/m9_fill_hazard_markouts_2019-07-30_timestamped.csv")
OUT = Path("research/m9_economic_decision_test.csv")
REPORT = Path("research/m9_economic_decision_test.txt")

M8_THRESHOLD = 0.00957966
EMBARGO_NS = 2_000_000_000
BLOCK_NS = 60_000_000_000
BOOTSTRAP_REPS = 2000
SEED = 20261018

NUMERIC = [
    "imbalance",
    "relative_spread",
    "delta_imbalance",
    "delta_microprice",
    "delta_spread",
    "queue_ahead_initial",
]
CATEGORICAL = ["symbol", "quote_side"]
FEATURES = NUMERIC + CATEGORICAL

SCENARIOS = {
    "conservative": {"horizon": "1s", "fee_bps": 1.0, "qty_haircut": 0.50},
    "central": {"horizon": "100ms", "fee_bps": 0.5, "qty_haircut": 1.00},
    "optimistic": {"horizon": "10ms", "fee_bps": 0.0, "qty_haircut": 1.00},
}
HORIZON_COLUMNS = {
    "10ms": "future_mid_10ms",
    "100ms": "future_mid_100ms",
    "500ms": "future_mid_500ms",
    "1s": "future_mid_1s",
}


def chronological_split(df: pd.DataFrame):
    timestamps = np.sort(df["timestamp_ns"].unique())
    boundary = timestamps[int(len(timestamps) * 0.70)]
    train = df[df["timestamp_ns"] < boundary - EMBARGO_NS].copy()
    test = df[df["timestamp_ns"] > boundary + EMBARGO_NS].copy()
    return train, test, boundary


def make_model() -> Pipeline:
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocess = ColumnTransformer([
        ("numeric", numeric_pipe, NUMERIC),
        ("categorical", categorical_pipe, CATEGORICAL),
    ])
    return Pipeline([
        ("preprocess", preprocess),
        ("classifier", LogisticRegression(
            C=1.0, max_iter=2000, random_state=42
        )),
    ])


def parse_boolish(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().map({
        "true": True, "false": False, "1": True, "0": False,
    })


def validate_markout_timestamps(df: pd.DataFrame) -> None:
    """Fail closed if markout prices and event timestamps are inconsistent."""
    horizons_ns = {
        "10ms": 10_000_000,
        "100ms": 100_000_000,
        "500ms": 500_000_000,
        "1s": 1_000_000_000,
    }
    for column in ["fill_timestamp_ns"] + [
        f"future_mid_{h}_timestamp_ns" for h in horizons_ns
    ]:
        if column not in df.columns:
            raise ValueError(f"Missing timestamp column: {column}")

    fill_ts = pd.to_numeric(df["fill_timestamp_ns"], errors="coerce")
    for horizon, offset_ns in horizons_ns.items():
        price_col = f"future_mid_{horizon}"
        ts_col = f"future_mid_{horizon}_timestamp_ns"
        prices = pd.to_numeric(df[price_col], errors="coerce")
        mark_ts = pd.to_numeric(df[ts_col], errors="coerce")

        mismatch = prices.notna().ne(mark_ts.notna())
        if mismatch.any():
            raise ValueError(
                f"{horizon}: {int(mismatch.sum())} midpoint/timestamp mismatches"
            )

        orphan = mark_ts.notna() & fill_ts.isna()
        if orphan.any():
            raise ValueError(
                f"{horizon}: {int(orphan.sum())} markout timestamps lack fill time"
            )

        before_horizon = (
            mark_ts.notna()
            & fill_ts.notna()
            & ((mark_ts - fill_ts) < offset_ns)
        )
        if before_horizon.any():
            raise ValueError(
                f"{horizon}: {int(before_horizon.sum())} observations precede target horizon"
            )

def add_candidate_pnl(
    df: pd.DataFrame,
    horizon: str,
    fee_bps: float,
    qty_haircut: float,
) -> pd.DataFrame:
    """Candidate-level mark-to-market proxy; primary P&L uses full fills only.

    Partial-fill contributions are separately reported as a sensitivity
    using resolution_mid, because exact partial-fill timestamps are unavailable.
    Neither estimate is realized portfolio P&L.
    """
    x = df.copy()
    x["quote_price"] = pd.to_numeric(x["quote_price"], errors="coerce")
    x["quote_qty"] = pd.to_numeric(x["quote_qty"], errors="coerce")
    x["filled_qty"] = pd.to_numeric(
        x["filled_qty"], errors="coerce"
    ).fillna(0)
    x["resolution_mid"] = pd.to_numeric(
        x["resolution_mid"], errors="coerce"
    )

    future_col = HORIZON_COLUMNS[horizon]
    x[future_col] = pd.to_numeric(x[future_col], errors="coerce")

    is_full_fill = x["outcome"].eq("FILL")
    x["has_any_fill"] = x["filled_qty"].gt(0)

    x["primary_full_fill_candidate"] = (
        is_full_fill & x["has_any_fill"]
    )

    x["partial_filled_candidate"] = (
        x["has_any_fill"]
        & ~is_full_fill
        & x["filled_qty"].lt(x["quote_qty"])
    )

    # Primary economic comparison: complete fills only.
    x["filled_qty_haircut"] = x["filled_qty"] * qty_haircut
    x["evaluated_qty"] = np.where(
        x["primary_full_fill_candidate"],
        x["filled_qty_haircut"],
        0.0,
    )

    x["mark_mid"] = np.nan
    x.loc[
        x["primary_full_fill_candidate"], "mark_mid"
    ] = x.loc[
        x["primary_full_fill_candidate"], future_col
    ]

    unvalued_full = (
        x["primary_full_fill_candidate"] & x["mark_mid"].isna()
    )
    if unvalued_full.any():
        raise ValueError(
            f"{int(unvalued_full.sum())} full-fill candidates lack "
            f"a mark price for horizon={horizon}. "
            "Do not silently value them as zero."
        )

    buy = x["quote_side"].eq("BUY")
    signed_move = np.where(
        buy,
        x["mark_mid"] - x["quote_price"],
        x["quote_price"] - x["mark_mid"],
    )

    x["gross_pnl"] = np.where(
        x["primary_full_fill_candidate"],
        signed_move * x["evaluated_qty"],
        0.0,
    )

    x["fees"] = np.where(
        x["primary_full_fill_candidate"],
        x["quote_price"]
        * x["evaluated_qty"]
        * fee_bps
        / 10_000.0,
        0.0,
    )

    x["net_mtm_pnl"] = x["gross_pnl"] - x["fees"]

    # Secondary partial-fill sensitivity using resolution_mid.
    partial = x["partial_filled_candidate"]

    x["partial_sensitivity_qty"] = np.where(
        partial,
        x["filled_qty_haircut"],
        0.0,
    )

    x["partial_sensitivity_mark_mid"] = np.where(
        partial,
        x["resolution_mid"],
        np.nan,
    )

    x["partial_sensitivity_valued"] = (
        partial
        & x["partial_sensitivity_mark_mid"].notna()
        & x["quote_price"].notna()
        & x["quote_side"].isin(["BUY", "SELL"])
    )

    partial_move = np.where(
        buy,
        x["partial_sensitivity_mark_mid"] - x["quote_price"],
        x["quote_price"] - x["partial_sensitivity_mark_mid"],
    )

    x["partial_sensitivity_gross_pnl"] = np.where(
        x["partial_sensitivity_valued"],
        partial_move * x["partial_sensitivity_qty"],
        0.0,
    )

    x["partial_sensitivity_fees"] = np.where(
        x["partial_sensitivity_valued"],
        x["quote_price"]
        * x["partial_sensitivity_qty"]
        * fee_bps
        / 10_000.0,
        0.0,
    )

    x["partial_sensitivity_net_pnl"] = (
        x["partial_sensitivity_gross_pnl"]
        - x["partial_sensitivity_fees"]
    )

    x["time_block"] = (
        pd.to_numeric(x["timestamp_ns"], errors="raise").astype("int64")
        // BLOCK_NS
    )

    return x


def summarize_policy(
    df: pd.DataFrame, policy: str, scenario: str
) -> dict:
    selected = df[df[policy]].copy()
    full_filled = selected[selected["primary_full_fill_candidate"]]
    partial = selected[selected["partial_filled_candidate"]]

    n = len(selected)
    any_fill_count = int(selected["has_any_fill"].sum())
    partial_valued_count = int(
        partial["partial_sensitivity_valued"].sum()
    )

    return {
        "scenario": scenario,
        "policy": policy,
        "selected_candidates": n,
        "candidates_with_any_fill": any_fill_count,
        "full_fill_candidates": int(
            selected["primary_full_fill_candidate"].sum()
        ),
        "partial_fill_candidates": int(
            selected["partial_filled_candidate"].sum()
        ),
        "partial_sensitivity_unvalued_candidates": int(
            (
                selected["partial_filled_candidate"]
                & ~selected["partial_sensitivity_valued"]
            ).sum()
        ),
        "fill_candidate_rate_pct": (
            100 * any_fill_count / n if n else np.nan
        ),
        "full_fill_candidate_rate_pct": (
            100 * float(
                selected["primary_full_fill_candidate"].mean()
            ) if n else np.nan
        ),
        "evaluated_shares_full_fills_only": float(
            selected["evaluated_qty"].sum()
        ),
        "gross_mark_to_market_pnl_usd": float(
            selected["gross_pnl"].sum()
        ),
        "estimated_fees_usd": float(selected["fees"].sum()),
        "net_mark_to_market_pnl_usd": float(
            selected["net_mtm_pnl"].sum()
        ),
        "net_pnl_per_selected_candidate_usd": (
            float(selected["net_mtm_pnl"].mean()) if n else np.nan
        ),
        "net_pnl_per_100_selected_candidates_usd": (
            float(100 * selected["net_mtm_pnl"].mean())
            if n else np.nan
        ),
        "net_pnl_per_full_fill_candidate_usd": (
            float(full_filled["net_mtm_pnl"].mean())
            if len(full_filled) else np.nan
        ),
        "partial_sensitivity_valued_candidates": partial_valued_count,
        "partial_sensitivity_net_pnl_usd": float(
            selected["partial_sensitivity_net_pnl"].sum()
        ),
        "partial_sensitivity_net_pnl_per_valued_partial_usd": (
            float(
                selected.loc[
                    selected["partial_filled_candidate"]
                    & selected["partial_sensitivity_valued"],
                    "partial_sensitivity_net_pnl",
                ].mean()
            ) if partial_valued_count else np.nan
        ),
    }


def paired_block_bootstrap_difference(
    df: pd.DataFrame, left: str, right: str, reps: int, rng: np.random.Generator
) -> tuple[float, float, float, int]:
    """Bootstrap difference in net P&L per selected candidate using shared time blocks."""
    selected = df[df[left] | df[right]].copy()
    blocks = sorted(selected["time_block"].unique())
    if not blocks:
        return np.nan, np.nan, np.nan, 0

    # Keep zero-selection blocks in the shared resampling universe. For each
    # policy, bootstrap total P&L divided by selected candidate count.
    stats = {}
    all_blocks = sorted(df["time_block"].unique())
    for policy in (left, right):
        p = df[df[policy]].groupby("time_block")["net_mtm_pnl"].agg(["sum", "count"])
        stats[policy] = (
            p["sum"].reindex(all_blocks, fill_value=0).to_numpy(float),
            p["count"].reindex(all_blocks, fill_value=0).to_numpy(float),
        )

    left_sum, left_count = stats[left]
    right_sum, right_count = stats[right]
    obs_left = left_sum.sum() / left_count.sum() if left_count.sum() else np.nan
    obs_right = right_sum.sum() / right_count.sum() if right_count.sum() else np.nan
    observed = obs_left - obs_right

    draws = []
    n_blocks = len(all_blocks)
    for _ in range(reps):
        picks = rng.integers(0, n_blocks, size=n_blocks)
        lc, rc = left_count[picks].sum(), right_count[picks].sum()
        if lc == 0 or rc == 0:
            continue
        draws.append(
            left_sum[picks].sum() / lc - right_sum[picks].sum() / rc
        )
    if not draws:
        return float(observed), np.nan, np.nan, 0
    low, high = np.quantile(draws, [0.025, 0.975])
    return float(observed), float(low), float(high), len(draws)


def main() -> None:
    if not DATA.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA}")

    df = pd.read_csv(DATA)
    required = {
        "timestamp_ns", "symbol", "quote_side", "quote_price", "quote_qty",
        "filled_qty", "resolution_mid", "outcome", "fill_timestamp_ns", *FEATURES,
        *HORIZON_COLUMNS.values(),
        "future_mid_10ms_timestamp_ns", "future_mid_100ms_timestamp_ns",
        "future_mid_500ms_timestamp_ns", "future_mid_1s_timestamp_ns",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    validate_markout_timestamps(df)

    df["quote_side"] = df["quote_side"].astype(str).str.upper()
    df["outcome"] = df["outcome"].astype(str).str.upper()
    df = df.dropna(subset=FEATURES + ["timestamp_ns", "outcome"]).copy()
    if not df["outcome"].isin(["FILL", "ADVERSE", "TIMEOUT"]).all():
        raise ValueError("Unexpected outcome label(s) in input.")

    signed_imbalance = np.where(
        df["quote_side"].eq("BUY"), df["imbalance"], -df["imbalance"]
    )
    df["m8_score"] = df["relative_spread"] * 100.0 * signed_imbalance
    df["m8_selected"] = df["m8_score"] >= M8_THRESHOLD

    train, test, boundary = chronological_split(df)
    if train.empty or test.empty:
        raise ValueError("Chronological split produced an empty partition.")

    model = make_model()
    model.fit(train[FEATURES], train["outcome"])
    classes = list(model.named_steps["classifier"].classes_)
    if "FILL" not in classes or "ADVERSE" not in classes:
        raise ValueError(f"Training classes missing FILL/ADVERSE: {classes}")

    train_probs = model.predict_proba(train[FEATURES])
    train_score = (
        train_probs[:, classes.index("FILL")]
        - train_probs[:, classes.index("ADVERSE")]
    )
    target_rate = float(train["m8_selected"].mean())
    ml_threshold = float(np.quantile(train_score, 1.0 - target_rate))

    test_probs = model.predict_proba(test[FEATURES])
    test["ml_score"] = (
        test_probs[:, classes.index("FILL")]
        - test_probs[:, classes.index("ADVERSE")]
    )
    test["ml_selected"] = test["ml_score"] >= ml_threshold

    # Secondary diagnostic only: retrospective score ranking matched to M8's
    # test count within symbol × side. This is not a prospective policy.
    test["ml_stratified_selected"] = False
    for _, group in test.groupby(["symbol", "quote_side"], sort=True):
        k = int(group["m8_selected"].sum())
        if k:
            chosen = group.sort_values(
                ["ml_score", "timestamp_ns"],
                ascending=[False, True],
                kind="mergesort",
            ).head(k).index
            test.loc[chosen, "ml_stratified_selected"] = True

    rng = np.random.default_rng(SEED)
    summaries = []
    differences = []
    enriched_scenarios = []

    for scenario, settings in SCENARIOS.items():
        econ = add_candidate_pnl(
            test,
            settings["horizon"],
            settings["fee_bps"],
            settings["qty_haircut"],
        )
        enriched_scenarios.append(econ.assign(scenario=scenario))
        for policy in ("m8_selected", "ml_selected", "ml_stratified_selected"):
            summaries.append(summarize_policy(econ, policy, scenario))

        for left, right, label in [
            ("ml_selected", "m8_selected", "ML_train_threshold_minus_M8"),
            ("ml_stratified_selected", "m8_selected", "ML_retrospective_stratified_minus_M8"),
        ]:
            observed, low, high, n_boot = paired_block_bootstrap_difference(
                econ, left, right, BOOTSTRAP_REPS, rng
            )
            differences.append({
                "scenario": scenario,
                "comparison": label,
                "difference_net_pnl_per_selected_candidate_usd": observed,
                "ci95_low_usd": low,
                "ci95_high_usd": high,
                "bootstrap_replicates_used": n_boot,
            })

    summary_df = pd.DataFrame(summaries)
    difference_df = pd.DataFrame(differences)
    details = pd.concat(enriched_scenarios, ignore_index=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(OUT, index=False)
    difference_path = OUT.with_name("m9_economic_decision_test_differences.csv")
    difference_df.to_csv(difference_path, index=False)
    detail_path = OUT.with_name("m9_economic_decision_test_candidates.csv")
    details.to_csv(detail_path, index=False)

    report = [
        "MERIT M9 economic decision test — candidate-level estimate",
        "=" * 68,
        f"Input rows after feature validation: {len(df):,}",
        f"Train rows: {len(train):,}",
        f"Test rows: {len(test):,}",
        f"Chronological boundary timestamp_ns: {boundary}",
        f"Embargo: {EMBARGO_NS:,} ns",
        f"Frozen M8 threshold: {M8_THRESHOLD}",
        f"ML threshold learned on training data: {ml_threshold:.8f}",
        f"Training M8 selection rate: {target_rate:.6%}",
        "",
        "Scenario assumptions:",
        "- Conservative: 1s markout for full fills, 50% quantity haircut, 1.0 bps fee.",
        "- Central: 100ms markout for full fills, recorded quantity, 0.5 bps fee.",
        "- Optimistic: 10ms markout for full fills, recorded quantity, zero fees.",
        "- Primary economics use fully filled candidates only; partial-fill contributions are excluded from primary P&L.",
        "- Partial-fill resolution_mid estimates are reported separately as sensitivity only; exact partial-fill timestamps are unavailable.",
        "",
        "Candidate-level economic proxy (USD):",
        summary_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"),
        "",
        "Paired 60-second block-bootstrap differences (per selected candidate):",
        difference_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"),
        "",
        "Interpretation limits:",
        "- These are overlapping hypothetical quote candidates, not an executable portfolio.",
        "- Primary P&L is mark-to-market from passive quote price to a later midpoint for full fills only, less estimated entry fees.",
        "- Per-selected-candidate metrics retain every selected candidate in the denominator; partial fills do not contribute to primary P&L.",
        "- No inventory limits, overlapping live-order constraints, exit costs, or realized liquidation are modeled.",
        "- ML train-threshold policy is prospective relative to this split; the symbol×side-matched ML policy uses test scores retrospectively and is diagnostic only.",
        "- Single trading day: exploratory only. A CI spanning zero is inconclusive, not proof of equivalence.",
        "",
        f"Summary CSV: {OUT}",
        f"Difference CSV: {difference_path}",
        f"Candidate details CSV: {detail_path}",
    ]
    text = "\n".join(report)
    REPORT.write_text(text, encoding="utf-8")
    print(text)
    print(f"\nSaved: {OUT}")
    print(f"Saved: {difference_path}")
    print(f"Saved: {detail_path}")
    print(f"Saved: {REPORT}")


if __name__ == "__main__":
    main()
