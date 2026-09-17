from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


HAZARD_PATH = Path(
    "research/m9_fill_hazard_2019-07-30.csv"
)

STATE_PATH = Path(
    "research/m9_state_transitions_2019-07-30.csv"
)

INTERVALS = [
    ("0_1ms", 0.0, 1.0),
    ("1_10ms", 1.0, 10.0),
    ("10_50ms", 10.0, 50.0),
    ("50_100ms", 50.0, 100.0),
    ("100_500ms", 100.0, 500.0),
    ("500_1000ms", 500.0, 1000.0),
]

MODELS = {
    "A_baseline": [
        "imbalance",
        "relative_spread",
        "queue_log",
    ],
    "B_dynamic": [
        "imbalance",
        "relative_spread",
        "queue_log",
        "delta_imbalance",
        "delta_microprice",
    ],
    "C_dynamic_flow": [
        "imbalance",
        "relative_spread",
        "queue_log",
        "delta_imbalance",
        "delta_microprice",
        "events_10ms",
        "executions_10ms",
        "buy_flow_10ms",
        "sell_flow_10ms",
    ],
}


def load_data() -> pd.DataFrame:
    hazard = pd.read_csv(HAZARD_PATH)

    state_columns = [
        "symbol",
        "timestamp_ns",
        "events_10ms",
        "executions_10ms",
        "buy_flow_10ms",
        "sell_flow_10ms",
    ]

    state = pd.read_csv(
        STATE_PATH,
        usecols=state_columns,
    )

    # ITCH timestamps can contain multiple executions at the same
    # timestamp. Preserve their occurrence order within each timestamp.
    state["_exec_seq"] = (
        state.groupby(
            ["symbol", "timestamp_ns"],
            sort=False,
        ).cumcount()
    )

    # The hazard builder creates exactly two candidates per execution:
    # one hypothetical BUY and one hypothetical SELL.
    hazard["_candidate_seq"] = (
        hazard.groupby(
            ["symbol", "timestamp_ns"],
            sort=False,
        ).cumcount()
    )

    hazard["_exec_seq"] = (
        hazard["_candidate_seq"] // 2
    )

    # Verify that every timestamp group contains pairs of candidates.
    candidate_counts = (
        hazard.groupby(
            ["symbol", "timestamp_ns"],
            sort=False,
        )
        .size()
    )

    if not candidate_counts.mod(2).eq(0).all():
        bad = candidate_counts[
            ~candidate_counts.mod(2).eq(0)
        ]

        raise RuntimeError(
            "Hazard dataset does not contain "
            "an even number of candidates for "
            f"{len(bad):,} timestamp groups."
        )

    merged = hazard.merge(
        state,
        on=[
            "symbol",
            "timestamp_ns",
            "_exec_seq",
        ],
        how="left",
        validate="many_to_one",
    )

    flow_columns = [
        "events_10ms",
        "executions_10ms",
        "buy_flow_10ms",
        "sell_flow_10ms",
    ]

    missing_flow = merged[
        flow_columns
    ].isna().any(axis=1).sum()

    if missing_flow:
        raise RuntimeError(
            f"{missing_flow:,} hazard rows "
            "failed to match M9 flow features."
        )

    merged = merged.drop(
        columns=[
            "_candidate_seq",
            "_exec_seq",
        ]
    )

    return merged


def chronological_split(
    df: pd.DataFrame,
    fraction: float = 0.70,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_parts = []
    test_parts = []

    for _, group in df.groupby(
        "symbol",
        sort=False,
    ):
        group = group.sort_values(
            "timestamp_ns"
        )

        split = int(
            len(group) * fraction
        )

        train_parts.append(
            group.iloc[:split]
        )
        test_parts.append(
            group.iloc[split:]
        )

    return (
        pd.concat(
            train_parts,
            ignore_index=True,
        ),
        pd.concat(
            test_parts,
            ignore_index=True,
        ),
    )


def build_risk_set(
    df: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []

    for interval_name, lower, upper in INTERVALS:
        risk = df[
            df["time_ms"] >= lower
        ].copy()

        if risk.empty:
            continue

        if lower == 0.0:
            terminal = (
                (risk["time_ms"] >= lower)
                & (risk["time_ms"] <= upper)
            )
        else:
            terminal = (
                (risk["time_ms"] > lower)
                & (risk["time_ms"] <= upper)
            )

        event = np.zeros(
            len(risk),
            dtype=np.int8,
        )

        fill = (
            terminal
            & risk["outcome"].eq("FILL")
        )

        adverse = (
            terminal
            & risk["outcome"].eq("ADVERSE")
        )

        event[fill.to_numpy()] = 1
        event[adverse.to_numpy()] = 2

        risk["interval"] = interval_name
        risk["event"] = event

        rows.append(risk)

    return pd.concat(
        rows,
        ignore_index=True,
    )


def prepare(
    df: pd.DataFrame,
) -> pd.DataFrame:
    df = df.copy()

    df["time_ms"] = (
        df["time_to_resolution_ns"]
        / 1_000_000.0
    )

    df["queue_log"] = np.log1p(
        df["queue_ahead_initial"]
    )

    return df


def build_model(
    numeric_features: list[str],
) -> Pipeline:
    categorical_features = [
        "quote_side",
        "symbol",
        "interval",
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                StandardScaler(),
                numeric_features,
            ),
            (
                "categorical",
                OneHotEncoder(
                    drop="first",
                    handle_unknown="ignore",
                ),
                categorical_features,
            ),
        ]
    )

    return Pipeline(
        [
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "logistic",
                LogisticRegression(
                    max_iter=2000,
                    C=1.0,
                ),
            ),
        ]
    )


def evaluate(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    event_code: int,
) -> dict[str, float]:
    model = build_model(features)

    y_train = (
        train["event"]
        .eq(event_code)
        .astype(int)
    )

    y_test = (
        test["event"]
        .eq(event_code)
        .astype(int)
    )

    columns = (
        features
        + [
            "quote_side",
            "symbol",
            "interval",
        ]
    )

    model.fit(
        train[columns],
        y_train,
    )

    probability = model.predict_proba(
        test[columns]
    )[:, 1]

    return {
        "event_rate": float(
            y_test.mean()
        ),
        "auc": float(
            roc_auc_score(
                y_test,
                probability,
            )
        ),
        "log_loss": float(
            log_loss(
                y_test,
                probability,
            )
        ),
    }


def print_results(
    results: dict[str, dict[str, float]],
    event_name: str,
) -> None:
    print(
        f"\n{event_name} HAZARD"
    )
    print("-" * 72)

    baseline_auc = results[
        "A_baseline"
    ]["auc"]

    baseline_loss = results[
        "A_baseline"
    ]["log_loss"]

    for name, metrics in results.items():
        print(f"\n{name}")

        print(
            f"  event rate: {metrics['event_rate']:.6f}"
        )

        print(
            f"  AUC:        {metrics['auc']:.6f}"
        )

        print(
            f"  ΔAUC:       "
            f"{metrics['auc'] - baseline_auc:+.6f}"
        )

        print(
            f"  log-loss:   {metrics['log_loss']:.6f}"
        )

        print(
            f"  Δlog-loss:  "
            f"{metrics['log_loss'] - baseline_loss:+.6f}"
        )


def main() -> None:
    df = load_data()

    print(
        f"Hazard candidates: {len(df):,}"
    )

    print(
        "Flow-feature matches: "
        "100.000%"
    )

    df = prepare(df)

    train_base, test_base = chronological_split(
        df
    )

    train = build_risk_set(
        train_base
    )

    test = build_risk_set(
        test_base
    )

    print(
        f"Train candidates: "
        f"{len(train_base):,}"
    )

    print(
        f"Test candidates: "
        f"{len(test_base):,}"
    )

    print(
        f"Train risk rows: "
        f"{len(train):,}"
    )

    print(
        f"Test risk rows: "
        f"{len(test):,}"
    )

    observed = (
        test.groupby(
            "interval",
            sort=False,
        )["event"]
        .agg(
            fill_hazard=lambda x: (
                x.eq(1).mean()
            ),
            adverse_hazard=lambda x: (
                x.eq(2).mean()
            ),
            risk_rows="size",
        )
        .reset_index()
    )

    print("\nTEST INTERVAL HAZARDS")

    print(
        observed.round(6).to_string(
            index=False
        )
    )

    fill_results = {}
    adverse_results = {}

    for name, features in MODELS.items():
        fill_results[name] = evaluate(
            train,
            test,
            features,
            event_code=1,
        )

        adverse_results[name] = evaluate(
            train,
            test,
            features,
            event_code=2,
        )

    print_results(
        fill_results,
        "FILL",
    )

    print_results(
        adverse_results,
        "ADVERSE",
    )


if __name__ == "__main__":
    main()