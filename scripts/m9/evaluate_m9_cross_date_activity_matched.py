from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# July trains the model. August is the cross-date evaluation.
TRAIN_PATH = Path(
    "research/m9_time_sampled_2019-07-30_regular_asof.csv"
)
TEST_PATH = Path(
    "research/m9_time_sampled_2019-08-30_regular_asof.csv"
)

OUTPUT = Path(
    "research/m9_cross_date_economic_summary_activity_matched.csv"
)
DIFFERENCES_OUTPUT = Path(
    "research/m9_cross_date_economic_differences_activity_matched.csv"
)

M8_THRESHOLD = 0.00957966
EMBARGO_NS = 2_000_000_000

BLOCK_NS = 60_000_000_000
BOOTSTRAP_REPS = 2_000
SEED = 20261029

NUMERIC_FEATURES = [
    "imbalance",
    "relative_spread",
    "delta_imbalance",
    "delta_microprice",
    "delta_spread",
    "queue_ahead_initial",
]

CATEGORICAL_FEATURES = [
    "symbol",
    "quote_side",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

SCENARIOS = {
    "conservative": {
        "horizon": "1s",
        "fee_bps": 1.0,
        "qty_haircut": 0.50,
    },
    "central": {
        "horizon": "100ms",
        "fee_bps": 0.5,
        "qty_haircut": 1.00,
    },
    "optimistic": {
        "horizon": "10ms",
        "fee_bps": 0.0,
        "qty_haircut": 1.00,
    },
}

HORIZONS_NS = {
    "10ms": 10_000_000,
    "100ms": 100_000_000,
    "500ms": 500_000_000,
    "1s": 1_000_000_000,
}


def load_candidates(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset: {path}")

    df = pd.read_csv(path, low_memory=False)

    required = set(FEATURES) | {
        "candidate_id",
        "timestamp_ns",
        "snapshot_timestamp_ns",
        "snapshot_age_ns",
        "quote_price",
        "quote_qty",
        "filled_qty",
        "outcome",
        "fill_timestamp_ns",
    }

    for horizon in HORIZONS_NS:
        required.add(f"future_mid_{horizon}")
        required.add(f"future_mid_{horizon}_timestamp_ns")

    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(
            f"{path} is missing required columns: {missing}"
        )

    if df["candidate_id"].duplicated().any():
        raise ValueError(f"Duplicate candidate IDs in {path}")

    numeric_columns = [
        "timestamp_ns",
        "snapshot_timestamp_ns",
        "snapshot_age_ns",
        "quote_price",
        "quote_qty",
        "filled_qty",
        "fill_timestamp_ns",
        *NUMERIC_FEATURES,
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df["outcome"] = df["outcome"].astype(str).str.upper()
    df["quote_side"] = df["quote_side"].astype(str).str.upper()
    df["symbol"] = df["symbol"].astype(str).str.upper()

    if not df["outcome"].isin(
        ["FILL", "ADVERSE", "TIMEOUT"]
    ).all():
        raise ValueError(f"Unexpected outcome labels in {path}")

    if df["timestamp_ns"].isna().any():
        raise ValueError(f"Missing decision timestamps in {path}")

    if df["snapshot_timestamp_ns"].isna().any():
        raise ValueError(f"Missing snapshot timestamps in {path}")

    # As-of snapshots must never be from after the decision.
    age = df["timestamp_ns"] - df["snapshot_timestamp_ns"]

    if age.lt(0).any():
        raise ValueError(
            f"{int(age.lt(0).sum())} negative snapshot ages in {path}"
        )

    if (
        df["snapshot_age_ns"].notna()
        & (df["snapshot_age_ns"] - age).abs().gt(1)
    ).any():
        raise ValueError(
            f"Snapshot-age column disagrees with timestamps in {path}"
        )

    # Verify as-of markout timestamp semantics:
    # fill time <= observed midpoint time <= fill time + horizon.
    fills = df["outcome"].eq("FILL")
    fill_ts = pd.to_numeric(
        df["fill_timestamp_ns"], errors="coerce"
    )

    if fill_ts.loc[fills].isna().any():
        raise ValueError(f"FILL rows without fill timestamps in {path}")

    for horizon, horizon_ns in HORIZONS_NS.items():
        price_col = f"future_mid_{horizon}"
        ts_col = f"future_mid_{horizon}_timestamp_ns"

        prices = pd.to_numeric(df[price_col], errors="coerce")
        mark_ts = pd.to_numeric(df[ts_col], errors="coerce")

        mismatch = prices.notna().ne(mark_ts.notna())
        if mismatch.any():
            raise ValueError(
                f"{horizon}: midpoint/timestamp mismatch in {path}"
            )

        missing_fills = fills & (prices.isna() | mark_ts.isna())
        if missing_fills.any():
            raise ValueError(
                f"{horizon}: {int(missing_fills.sum())} FILL rows "
                f"lack markout data in {path}"
            )

        before_fill = (
            fills
            & mark_ts.notna()
            & fill_ts.notna()
            & mark_ts.lt(fill_ts)
        )

        after_target = (
            fills
            & mark_ts.notna()
            & fill_ts.notna()
            & mark_ts.gt(fill_ts + horizon_ns)
        )

        if before_fill.any() or after_target.any():
            raise ValueError(
                f"{horizon}: invalid as-of markout timestamps in {path}; "
                f"before fill={int(before_fill.sum())}, "
                f"after target={int(after_target.sum())}"
            )

    return df


def make_model() -> Pipeline:
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="most_frequent"),
        ),
        (
            "onehot",
            OneHotEncoder(handle_unknown="ignore"),
        ),
    ])

    preprocessing = ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
    ])

    return Pipeline([
        ("preprocess", preprocessing),
        (
            "classifier",
            LogisticRegression(
                C=1.0,
                max_iter=2000,
                random_state=42,
            ),
        ),
    ])


def add_selection_scores(
    training_data: pd.DataFrame,
    evaluation_data: pd.DataFrame,
) -> tuple[pd.DataFrame, float, float]:
    train = training_data.copy()
    test = evaluation_data.copy()

    def m8_score(df):
        signed_imbalance = np.where(
            df["quote_side"].eq("BUY"),
            df["imbalance"],
            -df["imbalance"],
        )
        return df["relative_spread"] * 100.0 * signed_imbalance

    train["m8_score"] = m8_score(train)
    test["m8_score"] = m8_score(test)

    train["m8_selected"] = train["m8_score"].ge(M8_THRESHOLD)
    test["m8_selected"] = test["m8_score"].ge(M8_THRESHOLD)

    model = make_model()
    model.fit(train[FEATURES], train["outcome"])

    classes = list(model.named_steps["classifier"].classes_)

    if "FILL" not in classes or "ADVERSE" not in classes:
        raise ValueError(
            f"Training set lacks required classes: {classes}"
        )

    train_probs = model.predict_proba(train[FEATURES])
    test_probs = model.predict_proba(test[FEATURES])

    fill_index = classes.index("FILL")
    adverse_index = classes.index("ADVERSE")

    train_score = (
        train_probs[:, fill_index]
        - train_probs[:, adverse_index]
    )

    test["m9_ml_score"] = (
        test_probs[:, fill_index]
        - test_probs[:, adverse_index]
    )

    # Freeze the M9 activity threshold using July training data only.
    target_rate = float(train["m8_selected"].mean())

    if not 0.0 < target_rate < 1.0:
        raise ValueError(
            f"Invalid training M8 selection rate: {target_rate}"
        )

    ml_threshold = float(
        np.quantile(train_score, 1.0 - target_rate)
    )

    test["m9_ml_selected"] = (
        test["m9_ml_score"] >= ml_threshold
    )

    test["all_candidates_selected"] = True

    return test, ml_threshold, target_rate


def add_scenario_economics(
    df: pd.DataFrame,
    horizon: str,
    fee_bps: float,
    qty_haircut: float,
) -> pd.DataFrame:
    x = df.copy()

    future_col = f"future_mid_{horizon}"
    x[future_col] = pd.to_numeric(
        x[future_col], errors="coerce"
    )

    x["filled_qty"] = pd.to_numeric(
        x["filled_qty"], errors="coerce"
    ).fillna(0)

    x["quote_price"] = pd.to_numeric(
        x["quote_price"], errors="coerce"
    )

    full_fill = (
        x["outcome"].eq("FILL")
        & x["filled_qty"].gt(0)
    )

    missing_mark = full_fill & x[future_col].isna()

    if missing_mark.any():
        raise ValueError(
            f"{int(missing_mark.sum())} full fills lack {horizon} markouts"
        )

    x["full_fill"] = full_fill
    x["evaluated_qty"] = np.where(
        full_fill,
        x["filled_qty"] * qty_haircut,
        0.0,
    )

    signed_move = np.where(
        x["quote_side"].eq("BUY"),
        x[future_col] - x["quote_price"],
        x["quote_price"] - x[future_col],
    )

    x["gross_pnl_proxy"] = np.where(
        full_fill,
        signed_move * x["evaluated_qty"],
        0.0,
    )

    x["estimated_fees"] = np.where(
        full_fill,
        x["quote_price"] * x["evaluated_qty"] * fee_bps / 10_000.0,
        0.0,
    )

    x["net_pnl_proxy"] = (
        x["gross_pnl_proxy"] - x["estimated_fees"]
    )

    x["time_block"] = (
        x["timestamp_ns"].astype("int64") // BLOCK_NS
    )

    return x


def summarize_policy(
    data: pd.DataFrame,
    mask: pd.Series,
    policy: str,
    scenario: str,
    settings: dict,
) -> dict:
    selected = data.loc[mask]
    n = len(selected)

    return {
        "evaluation_date": "2019-08-30",
        "scenario": scenario,
        "horizon": settings["horizon"],
        "fee_bps": settings["fee_bps"],
        "quantity_haircut": settings["qty_haircut"],
        "policy": policy,
        "selected_candidates": n,
        "selection_rate_pct": (
            100.0 * n / len(data) if len(data) else np.nan
        ),
        "full_fill_candidates": int(selected["full_fill"].sum()),
        "full_fill_rate_pct": (
            100.0 * selected["full_fill"].mean()
            if n else np.nan
        ),
        "gross_pnl_proxy_usd": float(
            selected["gross_pnl_proxy"].sum()
        ),
        "estimated_fees_usd": float(
            selected["estimated_fees"].sum()
        ),
        "net_pnl_proxy_usd": float(
            selected["net_pnl_proxy"].sum()
        ),
        "net_pnl_proxy_per_selected_candidate_usd": (
            float(selected["net_pnl_proxy"].mean())
            if n else np.nan
        ),
    }


def bootstrap_difference(
    data: pd.DataFrame,
    left_mask: pd.Series,
    right_mask: pd.Series,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    blocks = np.sort(data["time_block"].unique())

    if len(blocks) == 0:
        return np.nan, np.nan, np.nan

    stats = {}

    for label, mask in [
        ("left", left_mask),
        ("right", right_mask),
    ]:
        selected = data.loc[mask]

        grouped = selected.groupby("time_block")[
            "net_pnl_proxy"
        ].agg(["sum", "count"])

        stats[label] = (
            grouped["sum"].reindex(blocks, fill_value=0).to_numpy(float),
            grouped["count"].reindex(blocks, fill_value=0).to_numpy(float),
        )

    left_sum, left_count = stats["left"]
    right_sum, right_count = stats["right"]

    if left_count.sum() == 0 or right_count.sum() == 0:
        return np.nan, np.nan, np.nan

    observed = (
        left_sum.sum() / left_count.sum()
        - right_sum.sum() / right_count.sum()
    )

    draws = []

    for _ in range(BOOTSTRAP_REPS):
        picks = rng.integers(0, len(blocks), size=len(blocks))

        lc = left_count[picks].sum()
        rc = right_count[picks].sum()

        if lc == 0 or rc == 0:
            continue

        draws.append(
            left_sum[picks].sum() / lc
            - right_sum[picks].sum() / rc
        )

    if not draws:
        return float(observed), np.nan, np.nan

    lo, hi = np.quantile(draws, [0.025, 0.975])

    return float(observed), float(lo), float(hi)


def main() -> None:
    july = load_candidates(TRAIN_PATH)
    august = load_candidates(TEST_PATH)

    # Train using July's first 70% of unique decision timestamps.
    # Two seconds are embargoed around the chronological boundary.
    timestamps = np.sort(july["timestamp_ns"].unique())

    if len(timestamps) < 10:
        raise ValueError("Too few July decision timestamps")

    boundary = timestamps[int(0.70 * len(timestamps))]

    train = july[
        july["timestamp_ns"] < boundary - EMBARGO_NS
    ].copy()

    if train.empty:
        raise ValueError("July training partition is empty")

    evaluation, ml_threshold, train_m8_rate = add_selection_scores(
        train,
        august,
    )

    # Retrospective activity-matched diagnostic:
    # match August M8 selection counts within symbol × quote_side.
    # Ranking uses M9 scores only, not evaluation outcomes or P&L.
    evaluation["m9_ml_activity_matched_selected"] = False

    for _, group in evaluation.groupby(
        ["symbol", "quote_side"],
        sort=True,
    ):
        target_count = int(group["m8_selected"].sum())

        if target_count == 0:
            continue

        chosen = (
            group.sort_values(
                ["m9_ml_score", "timestamp_ns"],
                ascending=[False, True],
                kind="mergesort",
            )
            .head(target_count)
            .index
        )

        evaluation.loc[
            chosen,
            "m9_ml_activity_matched_selected",
        ] = True

    expected_count = int(evaluation["m8_selected"].sum())
    actual_count = int(
        evaluation["m9_ml_activity_matched_selected"].sum()
    )

    if expected_count != actual_count:
        raise AssertionError(
            f"Activity match failed: M8={expected_count}, "
            f"M9={actual_count}"
        )

    print("Activity-matched candidate count:", actual_count)
    candidate_audit_path = Path(
        "research/m9_cross_date_candidate_audit_2019-08-30.csv"
    )
    evaluation.to_csv(candidate_audit_path, index=False)

    print("Saved candidate-level audit:", candidate_audit_path)
    print("Audit columns:")
    print(evaluation.columns.tolist())

    print("=" * 70)
    print("M9 CROSS-DATE ECONOMIC PROXY")
    print("=" * 70)
    print("Training date: 2019-07-30")
    print("Evaluation date: 2019-08-30")
    print("July training rows:", len(train))
    print("August evaluation rows:", len(evaluation))
    print("July chronological boundary:", int(boundary))
    print("Embargo (seconds):", EMBARGO_NS / 1e9)
    print("Frozen M8 threshold:", M8_THRESHOLD)
    print("July training M8 selection rate:", f"{train_m8_rate:.4%}")
    print("Frozen M9 ML score threshold:", f"{ml_threshold:.8f}")
    print("Bootstrap replicates:", BOOTSTRAP_REPS)
    print("Bootstrap block size: 60 seconds")

    policy_masks = {
        "all_sampled_candidates": evaluation[
            "all_candidates_selected"
        ].astype(bool),
        "m8_frozen_rule": evaluation["m8_selected"].astype(bool),
        "m9_ml_frozen_threshold": evaluation[
            "m9_ml_selected"
        ].astype(bool),
        "m9_ml_activity_matched": evaluation[
        "m9_ml_activity_matched_selected"
        ].astype(bool),
    }

    rng = np.random.default_rng(SEED)
    summary_rows = []
    difference_rows = []

    for scenario, settings in SCENARIOS.items():
        economics = add_scenario_economics(
            evaluation,
            horizon=settings["horizon"],
            fee_bps=settings["fee_bps"],
            qty_haircut=settings["qty_haircut"],
        )

        for policy, mask in policy_masks.items():
            summary_rows.append(
                summarize_policy(
                    economics,
                    mask,
                    policy,
                    scenario,
                    settings,
                )
            )

        comparisons = [
            (
                "M9_ML_minus_M8",
                policy_masks["m9_ml_frozen_threshold"],
                policy_masks["m8_frozen_rule"],
            ),
            (
                "M9_ML_minus_all_sampled_candidates",
                policy_masks["m9_ml_frozen_threshold"],
                policy_masks["all_sampled_candidates"],
            ),
            (
                "M9_ML_activity_matched_minus_M8",
                policy_masks["m9_ml_activity_matched"],
                policy_masks["m8_frozen_rule"],
            ),
        ]

        for label, left_mask, right_mask in comparisons:
            observed, low, high = bootstrap_difference(
                economics,
                left_mask,
                right_mask,
                rng,
            )

            difference_rows.append({
                "evaluation_date": "2019-08-30",
                "scenario": scenario,
                "comparison": label,
                "difference_net_pnl_proxy_per_selected_candidate_usd": observed,
                "ci95_low_usd": low,
                "ci95_high_usd": high,
            })

    summary = pd.DataFrame(summary_rows)
    differences = pd.DataFrame(difference_rows)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT, index=False)
    differences.to_csv(DIFFERENCES_OUTPUT, index=False)

    print("\nECONOMIC SUMMARY")
    print(
        summary[
            [
                "scenario",
                "policy",
                "selected_candidates",
                "selection_rate_pct",
                "full_fill_candidates",
                "gross_pnl_proxy_usd",
                "estimated_fees_usd",
                "net_pnl_proxy_usd",
                "net_pnl_proxy_per_selected_candidate_usd",
            ]
        ].round(6).to_string(index=False)
    )

    print("\nBLOCK-BOOTSTRAP DIFFERENCES")
    print(differences.round(6).to_string(index=False))

    print("\nSaved:", OUTPUT)
    print("Saved:", DIFFERENCES_OUTPUT)

    print(
        "\nLIMITATIONS: These are overlapping candidate-level mark-to-market "
        "proxies, not portfolio P&L. They exclude inventory constraints, "
        "order replacement interactions, exit costs, and realized liquidation. "
        "The all-candidates policy is a sampled opportunity benchmark, not "
        "the event-driven M5 portfolio replay. Confidence intervals are "
        "exploratory and are not multiple-testing adjusted."
    )


if __name__ == "__main__":
    main()