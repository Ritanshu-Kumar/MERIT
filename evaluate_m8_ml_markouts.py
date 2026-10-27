
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA = Path("research/m9_fill_hazard_markouts_2019-07-30.csv")
SUMMARY_OUT = Path("research/m9_m8_ml_markout_summary.csv")
STRATA_OUT = Path("research/m9_m8_ml_markout_by_symbol_side.csv")
REPORT_OUT = Path("research/m9_m8_ml_markout_report.txt")

M8_THRESHOLD = 0.00957966
EMBARGO_NS = 2_000_000_000
BLOCK_NS = 60_000_000_000
BOOTSTRAP_REPS = 1000
SEED = 20261015

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

HORIZONS = {
    "10ms": "future_mid_10ms",
    "100ms": "future_mid_100ms",
    "500ms": "future_mid_500ms",
    "1s": "future_mid_1s",
    "5s": "future_mid_5s",
}


def available_horizons(columns):
    available = {
        label: column
        for label, column in HORIZONS.items()
        if column in columns
    }
    if not available:
        raise ValueError(
            "No supported future-mid horizon columns found in dataset. "
            f"Expected one of: {sorted(HORIZONS.values())}"
        )
    return available


def chronological_split(df):
    timestamps = np.sort(df["timestamp_ns"].unique())
    boundary = timestamps[int(len(timestamps) * 0.70)]

    train = df[df["timestamp_ns"] < boundary - EMBARGO_NS].copy()
    test = df[df["timestamp_ns"] > boundary + EMBARGO_NS].copy()

    return train, test, boundary


def make_preprocessor():
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer([
        ("numeric", numeric_pipe, NUMERIC),
        ("categorical", categorical_pipe, CATEGORICAL),
    ])


def make_markouts(subset, future_mid_column):
    """Return fully filled candidates with observed signed markouts.

    Positive bps = favorable to the passive quote.
    Negative bps = unfavorable to the passive quote.
    """
    filled = subset[subset["outcome"].eq("FILL")].copy()

    if filled.empty:
        return pd.DataFrame(
            columns=[
                "timestamp_ns",
                "fill_timestamp_ns",
                "symbol",
                "quote_side",
                "markout_bps",
            ]
        )

    quote = pd.to_numeric(filled["quote_price"], errors="coerce")
    future = pd.to_numeric(filled[future_mid_column], errors="coerce")
    fill_time = pd.to_numeric(filled["fill_timestamp_ns"], errors="coerce")

    valid = (
        quote.gt(0)
        & future.notna()
        & np.isfinite(future)
        & fill_time.notna()
    )

    filled = filled.loc[valid].copy()
    quote = quote.loc[valid].to_numpy(dtype=float)
    future = future.loc[valid].to_numpy(dtype=float)

    buy = filled["quote_side"].eq("BUY").to_numpy()

    favorable_move = np.where(
        buy,
        future - quote,
        quote - future,
    )

    result = filled[
        [
            "timestamp_ns",
            "fill_timestamp_ns",
            "symbol",
            "quote_side",
        ]
    ].copy()

    result["markout_bps"] = favorable_move / quote * 10_000.0
    result["fill_timestamp_ns"] = result["fill_timestamp_ns"].astype("int64")

    return result


def block_bootstrap_mean_ci(markouts, rng):
    """Approximate 95% CI for a mean using 60-second fill-time blocks."""
    if markouts.empty:
        return np.nan, np.nan

    work = markouts.copy()
    work["block"] = work["fill_timestamp_ns"] // BLOCK_NS

    stats = work.groupby("block")["markout_bps"].agg(["sum", "count"])
    sums = stats["sum"].to_numpy(dtype=float)
    counts = stats["count"].to_numpy(dtype=float)

    draws = np.empty(BOOTSTRAP_REPS, dtype=float)
    n_blocks = len(stats)

    for i in range(BOOTSTRAP_REPS):
        picks = rng.integers(0, n_blocks, size=n_blocks)
        denominator = counts[picks].sum()
        draws[i] = (
            sums[picks].sum() / denominator
            if denominator > 0
            else np.nan
        )

    low, high = np.nanquantile(draws, [0.025, 0.975])
    return float(low), float(high)


def block_bootstrap_difference_ci(left, right, rng):
    """CI for mean(left) - mean(right), resampling common time blocks."""
    if left.empty or right.empty:
        return np.nan, np.nan

    def block_stats(frame):
        work = frame.copy()
        work["block"] = work["fill_timestamp_ns"] // BLOCK_NS
        return work.groupby("block")["markout_bps"].agg(["sum", "count"])

    left_stats = block_stats(left)
    right_stats = block_stats(right)

    blocks = sorted(set(left_stats.index) | set(right_stats.index))
    left_sum = left_stats["sum"].reindex(blocks, fill_value=0).to_numpy(float)
    left_count = left_stats["count"].reindex(blocks, fill_value=0).to_numpy(float)
    right_sum = right_stats["sum"].reindex(blocks, fill_value=0).to_numpy(float)
    right_count = right_stats["count"].reindex(blocks, fill_value=0).to_numpy(float)

    n_blocks = len(blocks)
    draws = np.empty(BOOTSTRAP_REPS, dtype=float)

    for i in range(BOOTSTRAP_REPS):
        picks = rng.integers(0, n_blocks, size=n_blocks)

        n_left = left_count[picks].sum()
        n_right = right_count[picks].sum()

        if n_left == 0 or n_right == 0:
            draws[i] = np.nan
            continue

        left_mean = left_sum[picks].sum() / n_left
        right_mean = right_sum[picks].sum() / n_right
        draws[i] = left_mean - right_mean

    low, high = np.nanquantile(draws, [0.025, 0.975])
    return float(low), float(high)


def main():
    if not DATA.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA}")

    df = pd.read_csv(DATA)

    horizons = available_horizons(df.columns)

    required = {
        "timestamp_ns",
        "symbol",
        "quote_side",
        "quote_price",
        "outcome",
        "fill_timestamp_ns",
        *FEATURES,
        *horizons.values(),
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df.dropna(
        subset=FEATURES + ["timestamp_ns", "outcome"]
    ).copy()

    signed_imbalance = np.where(
        df["quote_side"].eq("BUY"),
        df["imbalance"],
        -df["imbalance"],
    )

    df["m8_score"] = (
        df["relative_spread"] * 100.0 * signed_imbalance
    )
    df["m8_selected"] = df["m8_score"] >= M8_THRESHOLD

    train, test, boundary = chronological_split(df)

    if train.empty or test.empty:
        raise ValueError("Chronological split produced an empty partition.")

    model = Pipeline([
        ("preprocess", make_preprocessor()),
        ("classifier", LogisticRegression(
            C=1.0,
            max_iter=2000,
            random_state=42,
        )),
    ])

    model.fit(train[FEATURES], train["outcome"])

    classes = list(model.named_steps["classifier"].classes_)
    probabilities = pd.DataFrame(
        model.predict_proba(test[FEATURES]),
        columns=classes,
        index=test.index,
    )

    test["ml_score"] = (
        probabilities["FILL"] - probabilities["ADVERSE"]
    )

    # Match M8's count within each symbol × quote-side stratum.
    ml_parts = []

    for _, group in test.groupby(
        ["symbol", "quote_side"],
        sort=True,
    ):
        target_count = int(group["m8_selected"].sum())

        ranked = group.sort_values(
            ["ml_score", "timestamp_ns"],
            ascending=[False, True],
            kind="mergesort",
        )

        ml_parts.append(ranked.head(target_count))

    ml = pd.concat(ml_parts).sort_index()

    m8 = test[test["m8_selected"]].copy()

    if len(ml) != len(m8):
        raise AssertionError("ML and M8 selection counts do not match.")

    policies = {
        "All test candidates": test,
        "Frozen M8": m8,
        "ML matched by symbol + side": ml,
    }

    rng = np.random.default_rng(SEED)
    summary_rows = []

    # First calculate markout diagnostics at each horizon.
    marks_by_policy_horizon = {}

    for horizon, future_column in horizons.items():
        for policy, selected in policies.items():
            marks = make_markouts(selected, future_column)
            marks_by_policy_horizon[(policy, horizon)] = marks

            full_fills = int(selected["outcome"].eq("FILL").sum())
            n_selected = len(selected)

            ci_low, ci_high = block_bootstrap_mean_ci(marks, rng)

            summary_rows.append({
                "policy": policy,
                "horizon": horizon,
                "selected_candidates": n_selected,
                "full_fill_count": full_fills,
                "full_fill_rate_pct": 100 * full_fills / max(n_selected, 1),
                "markout_observations": len(marks),
                "markout_coverage_pct_of_full_fills": (
                    100 * len(marks) / max(full_fills, 1)
                ),
                "mean_favorable_markout_bps": (
                    marks["markout_bps"].mean()
                    if not marks.empty else np.nan
                ),
                "mean_markout_ci95_low": ci_low,
                "mean_markout_ci95_high": ci_high,
                "median_favorable_markout_bps": (
                    marks["markout_bps"].median()
                    if not marks.empty else np.nan
                ),
                "positive_markout_pct": (
                    100 * marks["markout_bps"].gt(0).mean()
                    if not marks.empty else np.nan
                ),
            })

    summary = pd.DataFrame(summary_rows)

    # Compare the ML and M8 conditional markout means using common
    # time-block resampling. Positive difference favors ML.
    diff_rows = []

    for horizon in horizons:
        m8_marks = marks_by_policy_horizon[("Frozen M8", horizon)]
        ml_marks = marks_by_policy_horizon[
            ("ML matched by symbol + side", horizon)
        ]

        if m8_marks.empty or ml_marks.empty:
            observed_diff = np.nan
            ci_low, ci_high = np.nan, np.nan
        else:
            observed_diff = (
                ml_marks["markout_bps"].mean()
                - m8_marks["markout_bps"].mean()
            )
            ci_low, ci_high = block_bootstrap_difference_ci(
                ml_marks, m8_marks, rng
            )

        diff_rows.append({
            "horizon": horizon,
            "ml_minus_m8_mean_markout_bps": observed_diff,
            "difference_ci95_low": ci_low,
            "difference_ci95_high": ci_high,
        })

    differences = pd.DataFrame(diff_rows)

    # Per-symbol/side outcomes and 1-second markouts.
    stratum_rows = []

    for (symbol, side), _ in test.groupby(
        ["symbol", "quote_side"],
        sort=True,
    ):
        for policy, selected in [
            ("Frozen M8", m8),
            ("ML matched", ml),
        ]:
            group = selected[
                selected["symbol"].eq(symbol)
                & selected["quote_side"].eq(side)
            ]

            fills = int(group["outcome"].eq("FILL").sum())
            marks_1s = make_markouts(group, horizons["1s"])

            stratum_rows.append({
                "symbol": symbol,
                "quote_side": side,
                "policy": policy,
                "selected": len(group),
                "full_fills": fills,
                "full_fill_rate_pct": 100 * fills / max(len(group), 1),
                "markout_observations_1s": len(marks_1s),
                "mean_favorable_markout_1s_bps": (
                    marks_1s["markout_bps"].mean()
                    if not marks_1s.empty else np.nan
                ),
                "positive_markout_1s_pct": (
                    100 * marks_1s["markout_bps"].gt(0).mean()
                    if not marks_1s.empty else np.nan
                ),
            })

    strata = pd.DataFrame(stratum_rows)

    SUMMARY_OUT.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY_OUT, index=False)
    strata.to_csv(STRATA_OUT, index=False)

    report = (
        "MERIT: M8 vs ML markout diagnostic\n"
        + "=" * 65 + "\n"
        + f"Rows: {len(df):,}\n"
        + f"Train rows: {len(train):,}\n"
        + f"Test rows: {len(test):,}\n"
        + f"Split boundary timestamp_ns: {boundary}\n"
        + f"M8-selected test candidates: {len(m8):,}\n"
        + f"ML-selected test candidates: {len(ml):,}\n\n"
        + "Conditional markout summary\n"
        + summary.to_string(index=False, float_format=lambda x: f"{x:.4f}")
        + "\n\nML minus M8 conditional markout\n"
        + differences.to_string(index=False, float_format=lambda x: f"{x:.4f}")
        + "\n\nPer-symbol/side summary at 1 second\n"
        + strata.to_string(index=False, float_format=lambda x: f"{x:.4f}")
        + "\n\nInterpretation limits:\n"
        + "- Positive markout means favorable to the passive quote.\n"
        + "- Markouts are conditional on completed full fills.\n"
        + "- Partial-fill P&L and unfilled candidates are not valued here.\n"
        + "- Fees, inventory costs, hedging, and exit costs are excluded.\n"
        + "- Confidence intervals resample 60-second fill-time blocks; they do not\n"
        + "  capture model refitting uncertainty or independent-day variation.\n"
        + "- One trading date only; results remain exploratory.\n"
        + f"\nSummary saved to: {SUMMARY_OUT}\n"
        + f"Stratum summary saved to: {STRATA_OUT}\n"
    )

    REPORT_OUT.write_text(report, encoding="utf-8")

    print("MERIT: M8 vs ML markout diagnostic")
    print("=" * 65)
    print("\nConditional markout summary")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nML minus M8 conditional markout")
    print(differences.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nSaved: {SUMMARY_OUT}")
    print(f"Saved: {STRATA_OUT}")
    print(f"Saved: {REPORT_OUT}")


if __name__ == "__main__":
    main()
