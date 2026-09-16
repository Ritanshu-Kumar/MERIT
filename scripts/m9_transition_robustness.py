from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_PATH = Path(
    "research/m9_state_transitions_2019-07-30.csv"
)

MODELS = {
    "A_static": [
        "imbalance",
        "relative_spread",
    ],
    "B1_static_dimbalance": [
        "imbalance",
        "relative_spread",
        "delta_imbalance",
    ],
    "B2_static_dmicroprice": [
        "imbalance",
        "relative_spread",
        "delta_microprice",
    ],
    "B3_both_transitions": [
        "imbalance",
        "relative_spread",
        "delta_imbalance",
        "delta_microprice",
    ],
}


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
        group = group.sort_values("timestamp_ns")

        split = int(len(group) * fraction)

        train_parts.append(group.iloc[:split])
        test_parts.append(group.iloc[split:])

    return (
        pd.concat(train_parts),
        pd.concat(test_parts),
    )


def evaluate(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    target: str,
) -> float:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                StandardScaler(),
                features,
            ),
            (
                "symbol",
                OneHotEncoder(
                    drop="first",
                    handle_unknown="ignore",
                ),
                ["symbol"],
            ),
        ]
    )

    x_train = preprocessor.fit_transform(
        train[features + ["symbol"]]
    )

    x_test = preprocessor.transform(
        test[features + ["symbol"]]
    )

    model = LinearRegression()
    model.fit(
        x_train,
        train[target],
    )

    prediction = model.predict(x_test)

    return r2_score(
        test[target],
        prediction,
    )


def main() -> None:
    df = pd.read_csv(DATA_PATH)

    train, test = chronological_split(df)

    # Define the robustness filter strictly from the training sample.
    lower, upper = train[
        "delta_spread"
    ].quantile(
        [0.005, 0.995]
    )

    train = train[
        train["delta_spread"].between(
            lower,
            upper,
        )
    ]

    test = test[
        test["delta_spread"].between(
            lower,
            upper,
        )
    ]

    print(
        f"Delta-spread bounds: "
        f"[{lower:.6f}, {upper:.6f}]"
    )
    print(
        f"Train retained: {len(train):,}"
    )
    print(
        f"Test retained:  {len(test):,}"
    )

    for target in (
        "mid_return_bps_100ms",
        "mid_return_bps_1s",
    ):
        print("\n" + "=" * 80)
        print(f"TARGET: {target}")
        print("=" * 80)

        baseline_r2 = None

        for name, features in MODELS.items():
            r2 = evaluate(
                train,
                test,
                features,
                target,
            )

            if baseline_r2 is None:
                baseline_r2 = r2

            print(f"\n{name}")
            print(f"  R2:      {r2:.6f}")
            print(
                f"  ΔR2:     "
                f"{r2 - baseline_r2:+.6f}"
            )


if __name__ == "__main__":
    main()