from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

INPUT = Path("research/m9_fill_hazard_markouts_2019-07-30.csv")
OUTPUT = Path("research/m9_fill_outcome_baseline_metrics.txt")
EMBARGO_NS = 2_000_000_000
LABELS = ["ADVERSE", "FILL", "TIMEOUT"]
NUMERIC = [
    "imbalance",
    "relative_spread",
    "delta_imbalance",
    "delta_microprice",
    "delta_spread",
    "queue_log",
]
CATEGORICAL = ["symbol", "quote_side"]


def main() -> None:
    if not INPUT.exists():
        raise FileNotFoundError(f"Dataset not found: {INPUT}")

    df = pd.read_csv(INPUT)
    required = ["timestamp_ns", "outcome", "symbol", "quote_side", "queue_ahead_initial", *NUMERIC[:-1]]
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Pre-placement predictors only. Do not use outcome-derived columns,
    # fill timestamps, resolution fields, or future midpoint/markout fields.
    df["queue_log"] = np.log1p(
        pd.to_numeric(df["queue_ahead_initial"], errors="coerce").clip(lower=0)
    )
    df["outcome"] = df["outcome"].astype(str)
    df = df[df["outcome"].isin(LABELS)].copy()
    df = df.dropna(subset=[*NUMERIC, *CATEGORICAL, "timestamp_ns", "outcome"])

    timestamps = np.sort(df["timestamp_ns"].unique())
    if len(timestamps) < 10:
        raise ValueError("Too few unique timestamps for a chronological split")
    boundary = int(timestamps[int(len(timestamps) * 0.70)])
    train = df[df["timestamp_ns"] < boundary - EMBARGO_NS].copy()
    test = df[df["timestamp_ns"] > boundary + EMBARGO_NS].copy()
    excluded = len(df) - len(train) - len(test)
    if train.empty or test.empty:
        raise ValueError("Train or test split is empty")

    majority = train["outcome"].value_counts().idxmax()
    baseline_pred = np.full(len(test), majority, dtype=object)
    baseline_proba = np.zeros((len(test), len(LABELS)), dtype=float)
    train_priors = train["outcome"].value_counts(normalize=True)
    for i, label in enumerate(LABELS):
        baseline_proba[:, i] = float(train_priors.get(label, 0.0))

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ]
    )
    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("classifier", LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced", random_state=42,)),
        ]
    )
    features = NUMERIC + CATEGORICAL
    model.fit(train[features], train["outcome"])
    pred = model.predict(test[features])
    raw_proba = model.predict_proba(test[features])
    model_classes = list(model.named_steps["classifier"].classes_)
    # Reorder probability columns to a stable LABELS order for log loss.
    proba = np.zeros((len(test), len(LABELS)), dtype=float)
    for j, label in enumerate(model_classes):
        proba[:, LABELS.index(label)] = raw_proba[:, j]

    lines: list[str] = []
    lines.append("MERIT M9 three-class fill-outcome baseline")
    lines.append("Target: outcome (ADVERSE / FILL / TIMEOUT)")
    lines.append("Predictors: " + ", ".join(features))
    lines.append("Excluded by design: fill_before_adverse, filled_qty, fill_fraction, time_to_resolution_ns, resolution_mid, level_resets, fill_timestamp_ns, all future_mid_* columns, adverse_threshold")
    lines.append(f"Rows loaded: {len(df):,}")
    lines.append(f"Unique timestamps: {len(timestamps):,}")
    lines.append(f"Boundary timestamp_ns: {boundary}")
    lines.append(f"Embargo: {EMBARGO_NS:,} ns (2 seconds)")
    lines.append(f"Train rows: {len(train):,}; test rows: {len(test):,}; excluded rows: {excluded:,}")
    lines.append("\nTrain outcome counts:\n" + train["outcome"].value_counts().reindex(LABELS, fill_value=0).to_string())
    lines.append("\nTest outcome counts:\n" + test["outcome"].value_counts().reindex(LABELS, fill_value=0).to_string())
    lines.append("\nTest metrics (majority-class baseline):")
    lines.append(f"Majority label: {majority}")
    lines.append(f"Balanced accuracy: {balanced_accuracy_score(test['outcome'], baseline_pred):.6f}")
    lines.append(f"Macro F1: {f1_score(test['outcome'], baseline_pred, labels=LABELS, average='macro', zero_division=0):.6f}")
    lines.append(f"Log loss (training class priors): {log_loss(test['outcome'], baseline_proba, labels=LABELS):.6f}")
    lines.append("\nTest metrics (regularized logistic regression):")
    lines.append(f"Balanced accuracy: {balanced_accuracy_score(test['outcome'], pred):.6f}")
    lines.append(f"Macro F1: {f1_score(test['outcome'], pred, labels=LABELS, average='macro', zero_division=0):.6f}")
    lines.append(f"Log loss: {log_loss(test['outcome'], proba, labels=LABELS):.6f}")
    lines.append("\nClassification report:\n" + classification_report(test["outcome"], pred, labels=LABELS, zero_division=0, digits=4))
    lines.append("\nConfusion matrix (rows=true, columns=predicted; order ADVERSE, FILL, TIMEOUT):\n" + str(confusion_matrix(test["outcome"], pred, labels=LABELS)))
    lines.append("\nTest fill rate by symbol (%):\n" + (100 * test.assign(is_fill=test["outcome"].eq("FILL")).groupby("symbol")["is_fill"].mean()).round(4).to_string())
    lines.append("\nCaution: one trading date only; this is an exploratory within-day temporal holdout, not independent-day validation or proof of policy P&L improvement.")

    report = "\n".join(lines) + "\n"
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(report, encoding="utf-8")
    print(report)
    print(f"\nSaved report to: {OUTPUT}")


if __name__ == "__main__":
    main()
