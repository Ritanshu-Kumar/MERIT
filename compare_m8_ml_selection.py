
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA = Path("research/m9_fill_hazard_markouts_2019-07-30.csv")
OUTPUT = Path("research/m9_m8_vs_ml_selection.csv")
REPORT = Path("research/m9_m8_vs_ml_selection.txt")

M8_THRESHOLD = 0.00957966
EMBARGO_NS = 2_000_000_000

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


def split_chronologically(df):
    timestamps = np.sort(df["timestamp_ns"].unique())
    boundary = timestamps[int(len(timestamps) * 0.70)]

    train = df[df["timestamp_ns"] < boundary - EMBARGO_NS].copy()
    test = df[df["timestamp_ns"] > boundary + EMBARGO_NS].copy()

    return train, test, boundary


def main():
    if not DATA.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA}")

    df = pd.read_csv(DATA)

    required = [
        "timestamp_ns",
        "symbol",
        "quote_side",
        "relative_spread",
        "imbalance",
        "outcome",
        *NUMERIC,
    ]
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df.dropna(subset=FEATURES + ["timestamp_ns", "outcome"]).copy()

    # Frozen M8 score, calculated from pre-placement state.
    signed_imbalance = np.where(
        df["quote_side"].eq("BUY"),
        df["imbalance"],
        -df["imbalance"],
    )
    df["m8_score"] = (
        df["relative_spread"] * 100.0 * signed_imbalance
    )
    df["m8_selected"] = df["m8_score"] >= M8_THRESHOLD

    train, test, boundary = split_chronologically(df)

    if train.empty or test.empty:
        raise ValueError("Empty train or test split.")

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

    model = Pipeline([
        ("preprocess", preprocess),
        ("classifier", LogisticRegression(
            C=1.0,
            max_iter=2000,
            random_state=42,
        )),
    ])

    model.fit(train[FEATURES], train["outcome"])

    classes = list(model.named_steps["classifier"].classes_)
    probabilities = model.predict_proba(test[FEATURES])
    probability_df = pd.DataFrame(
        probabilities,
        columns=classes,
        index=test.index,
    )

    # Higher score means more predicted FILL and less predicted ADVERSE.
    test["ml_score"] = (
        probability_df["FILL"] - probability_df["ADVERSE"]
    )

    # Match the ML selection rate to M8's training selection rate.
    # This threshold is learned only from the training period.
    target_rate = train["m8_selected"].mean()
    if target_rate <= 0:
        raise ValueError("M8 selected no training candidates.")
    if target_rate >= 1:
        raise ValueError("M8 selected every training candidate.")

    ml_threshold = train.assign(
        _score=(
            model.predict_proba(train[FEATURES])[
                :, classes.index("FILL")
            ]
            - model.predict_proba(train[FEATURES])[
                :, classes.index("ADVERSE")
            ]
        )
    )["_score"].quantile(1.0 - target_rate)

    test["ml_selected"] = test["ml_score"] >= ml_threshold

    # Outcomes are reported for every selected candidate.
    # FILL means a full simulated fill; ADVERSE/TIMEOUT may
    # include partial fills. This is not yet a P&L estimate.
    rows = []
    for policy, mask in [
        ("All test candidates", pd.Series(True, index=test.index)),
        ("Frozen M8", test["m8_selected"]),
        ("ML, matched training selection rate", test["ml_selected"]),
    ]:
        subset = test.loc[mask]
        counts = subset["outcome"].value_counts()

        rows.append({
            "policy": policy,
            "selected_candidates": len(subset),
            "selection_rate_pct": 100 * len(subset) / len(test),
            "full_fill_rate_pct": 100 * counts.get("FILL", 0) / max(len(subset), 1),
            "adverse_rate_pct": 100 * counts.get("ADVERSE", 0) / max(len(subset), 1),
            "timeout_rate_pct": 100 * counts.get("TIMEOUT", 0) / max(len(subset), 1),
        })

    summary = pd.DataFrame(rows)

    # Also measure ranking ability for the full-fill outcome.
    y_fill = test["outcome"].eq("FILL").astype(int)
    fill_auc = roc_auc_score(y_fill, probability_df["FILL"])

    test_output = test[
        [
            "timestamp_ns", "symbol", "quote_side", "outcome",
            "m8_score", "m8_selected", "ml_score", "ml_selected",
        ]
    ].copy()

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    test_output.to_csv(OUTPUT, index=False)

    report = [
        "MERIT M8 vs ML candidate-selection comparison",
        "=" * 55,
        f"Rows after feature validation: {len(df):,}",
        f"Train rows: {len(train):,}",
        f"Test rows: {len(test):,}",
        f"Boundary timestamp_ns: {boundary}",
        f"Embargo: {EMBARGO_NS:,} ns",
        f"Frozen M8 threshold: {M8_THRESHOLD}",
        f"Training M8 selection rate: {target_rate:.6%}",
        f"ML threshold (training only): {ml_threshold:.8f}",
        f"Test FILL-probability ROC AUC: {fill_auc:.6f}",
        "",
        summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"),
        "",
        "Interpretation limits:",
        "- One trading date; exploratory temporal holdout only.",
        "- Outcome rates are not net P&L.",
        "- ADVERSE and TIMEOUT candidates may have partial fills.",
        "- The ML score ranks P(FILL) - P(ADVERSE); it is not a calibrated utility.",
        "- No policy threshold was selected using test outcomes.",
        "",
        f"Candidate-level output: {OUTPUT}",
    ]

    text = "\n".join(report)
    REPORT.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
