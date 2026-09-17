from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler


INPUT = Path(
    "research/m9_fill_hazard_markouts_2019-07-30.csv"
)

OUTPUT = Path(
    "research/m9_joint_quote_value_2019-07-30.csv"
)

HORIZONS = {
    "10ms": "future_mid_10ms",
    "100ms": "future_mid_100ms",
    "500ms": "future_mid_500ms",
    "1s": "future_mid_1s",
}

STATE_FEATURES = [
    "imbalance",
    "relative_spread",
    "queue_log",
    "delta_imbalance",
    "delta_microprice",
]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["queue_log"] = np.log1p(
        np.maximum(
            pd.to_numeric(
                df["queue_ahead_initial"],
                errors="coerce",
            ),
            0.0,
        )
    )

    for col in [
        "imbalance",
        "relative_spread",
        "delta_imbalance",
        "delta_microprice",
        "queue_log",
    ]:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    return df


def chronological_split(
    df: pd.DataFrame,
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
            len(group) * 0.70
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


def make_preprocessor(
    numeric_features: list[str],
) -> ColumnTransformer:

    return ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                numeric_features,
            ),
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                [
                    "quote_side",
                    "symbol",
                ],
            ),
        ]
    )


def signed_quote_payoff_bps(
    df: pd.DataFrame,
    future_mid_col: str,
) -> np.ndarray:

    quote = df[
        "quote_price"
    ].to_numpy(dtype=float)

    future_mid = df[
        future_mid_col
    ].to_numpy(dtype=float)

    buy = df[
        "quote_side"
    ].eq("BUY").to_numpy()

    payoff = np.where(
        buy,
        future_mid - quote,
        quote - future_mid,
    )

    return payoff / quote * 10_000.0


def baseline_side_means(
    train_fills: pd.DataFrame,
    future_mid_col: str,
) -> dict[str, float]:

    means = {}

    for side in ("BUY", "SELL"):

        subset = train_fills[
            train_fills["quote_side"].eq(side)
        ]

        values = signed_quote_payoff_bps(
            subset,
            future_mid_col,
        )

        values = values[
            np.isfinite(values)
        ]

        means[side] = float(
            values.mean()
        )

    return means


def main() -> None:

    print(
        "Loading M9.6 dataset..."
    )

    df = pd.read_csv(INPUT)

    df = add_features(df)

    train, test = chronological_split(
        df
    )

    train = train.copy()
    test = test.copy()

    train["is_fill"] = (
        train["outcome"]
        .eq("FILL")
        .astype(int)
    )

    test["is_fill"] = (
        test["outcome"]
        .eq("FILL")
        .astype(int)
    )

    print(
        f"Train candidates: {len(train):,}"
    )

    print(
        f"Test candidates:  {len(test):,}"
    )

    # ---------------------------------------------------------------
    # Fill-probability model
    # ---------------------------------------------------------------

    fill_features = [
        "imbalance",
        "relative_spread",
        "queue_log",
        "delta_imbalance",
        "delta_microprice",
    ]

    fill_required = fill_features + [
        "quote_side",
        "symbol",
    ]

    train_fill = train.dropna(
        subset=fill_required
    ).copy()

    test_fill = test.dropna(
        subset=fill_required
    ).copy()

    fill_preprocessor = make_preprocessor(
        fill_features
    )

    x_train_fill = (
        fill_preprocessor.fit_transform(
            train_fill
        )
    )

    x_test_fill = (
        fill_preprocessor.transform(
            test_fill
        )
    )

    fill_model = LogisticRegression(
        C=1.0,
        max_iter=2000,
    )

    fill_model.fit(
        x_train_fill,
        train_fill["is_fill"],
    )

    test_fill_prob = (
        fill_model.predict_proba(
            x_test_fill
        )[:, 1]
    )

    fill_auc = roc_auc_score(
        test_fill["is_fill"],
        test_fill_prob,
    )

    print(
        f"\nFill probability AUC: "
        f"{fill_auc:.6f}"
    )

    # ---------------------------------------------------------------
    # Main horizon loop
    # ---------------------------------------------------------------

    rows = []

    for horizon, future_mid_col in HORIZONS.items():

        print(
            f"\n===== {horizon} ====="
        )

        train_fills = train[
            train["outcome"].eq("FILL")
        ].dropna(
            subset=[
                future_mid_col,
                *fill_features,
                "quote_side",
                "symbol",
            ]
        ).copy()

        test_with_fill_prob = test_fill[
            test_fill.index.isin(
                test_fill.index
            )
        ].copy()

        # Align prediction dataframe with the
        # test candidates that have fill probabilities.
        test_candidates = test_fill.copy()

        # -----------------------------------------------------------
        # State-aware markout model
        # -----------------------------------------------------------

        markout_features = [
            "imbalance",
            "relative_spread",
            "queue_log",
            "delta_imbalance",
            "delta_microprice",
        ]

        markout_preprocessor = make_preprocessor(
            markout_features
        )

        x_train_markout = (
            markout_preprocessor.fit_transform(
                train_fills
            )
        )

        train_payoff = signed_quote_payoff_bps(
            train_fills,
            future_mid_col,
        )

        markout_model = Ridge(
            alpha=10.0
        )

        markout_model.fit(
            x_train_markout,
            train_payoff,
        )

        x_test_markout = (
            markout_preprocessor.transform(
                test_candidates
            )
        )

        predicted_state_payoff = (
            markout_model.predict(
                x_test_markout
            )
        )

        # -----------------------------------------------------------
        # Baseline markout
        # -----------------------------------------------------------

        side_means = baseline_side_means(
            train_fills,
            future_mid_col,
        )

        baseline_payoff = (
            test_candidates[
                "quote_side"
            ]
            .map(side_means)
            .to_numpy(dtype=float)
        )

        # -----------------------------------------------------------
        # Joint expected value
        # -----------------------------------------------------------

        # test_fill_prob and test_candidates
        # have identical row ordering.
        state_ev = (
            test_fill_prob
            * predicted_state_payoff
        )

        baseline_ev = (
            test_fill_prob
            * baseline_payoff
        )

        actual_payoff = np.zeros(
            len(test_candidates),
            dtype=float,
        )

        actual_fill = (
            test_candidates[
                "outcome"
            ].eq("FILL").to_numpy()
        )

        if actual_fill.any():

            filled_subset = (
                test_candidates.loc[
                    actual_fill
                ]
            )

            filled_payoff = (
                signed_quote_payoff_bps(
                    filled_subset,
                    future_mid_col,
                )
            )

            actual_payoff[
                actual_fill
            ] = filled_payoff

        actual_value = (
            actual_payoff
            * actual_fill.astype(float)
        )

        # -----------------------------------------------------------
        # Summary
        # -----------------------------------------------------------

        mean_state_ev = float(
            state_ev.mean()
        )

        mean_baseline_ev = float(
            baseline_ev.mean()
        )

        mean_actual = float(
            actual_value.mean()
        )

        state_positive = (
            state_ev > 0
        ).mean()

        baseline_positive = (
            baseline_ev > 0
        ).mean()

        correlation_state = np.corrcoef(
            state_ev,
            actual_value,
        )[0, 1]

        correlation_baseline = np.corrcoef(
            baseline_ev,
            actual_value,
        )[0, 1]

        print(
            f"Baseline EV:       "
            f"{mean_baseline_ev:+.6f} bps"
        )

        print(
            f"State-aware EV:    "
            f"{mean_state_ev:+.6f} bps"
        )

        print(
            f"Actual realized:   "
            f"{mean_actual:+.6f} bps"
        )

        print(
            f"Δ EV:              "
            f"{mean_state_ev - mean_baseline_ev:+.6f} bps"
        )

        print(
            f"Positive baseline: "
            f"{baseline_positive:.4f}"
        )

        print(
            f"Positive state:    "
            f"{state_positive:.4f}"
        )

        print(
            f"State correlation: "
            f"{correlation_state:+.6f}"
        )

        print(
            f"Base correlation:  "
            f"{correlation_baseline:+.6f}"
        )

        rows.append(
            {
                "horizon": horizon,
                "n_test_candidates": len(
                    test_candidates
                ),
                "fill_auc": fill_auc,
                "baseline_ev_bps": mean_baseline_ev,
                "state_aware_ev_bps": mean_state_ev,
                "delta_ev_bps": (
                    mean_state_ev
                    - mean_baseline_ev
                ),
                "actual_realized_bps": mean_actual,
                "baseline_positive_fraction": (
                    baseline_positive
                ),
                "state_positive_fraction": (
                    state_positive
                ),
                "baseline_actual_corr": (
                    correlation_baseline
                ),
                "state_actual_corr": (
                    correlation_state
                ),
            }
        )

    result = pd.DataFrame(rows)

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        OUTPUT,
        index=False,
    )

    print(
        "\n===== M9.6 Summary ====="
    )

    print(
        result.to_string(
            index=False
        )
    )

    print(
        f"\nSaved: {OUTPUT}"
    )


if __name__ == "__main__":
    main()