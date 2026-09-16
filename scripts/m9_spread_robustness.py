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

FEATURES_A = [
    "imbalance",
    "relative_spread",
]

FEATURES_B3 = [
    "imbalance",
    "relative_spread",
    "delta_spread",
]


def chronological_split(
    df: pd.DataFrame,
    train_fraction: float = 0.70,
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
            len(group) * train_fraction
        )

        train_parts.append(
            group.iloc[:split]
        )
        test_parts.append(
            group.iloc[split:]
        )

    return (
        pd.concat(train_parts),
        pd.concat(test_parts),
    )


def evaluate(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
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
        train["mid_return_bps_100ms"],
    )

    prediction = model.predict(x_test)

    return r2_score(
        test["mid_return_bps_100ms"],
        prediction,
    )


def main() -> None:
    df = pd.read_csv(DATA_PATH)

    train, test = chronological_split(df)

    print(f"Total rows: {len(df):,}")
    print(f"Train rows: {len(train):,}")
    print(f"Test rows:  {len(test):,}")

    print("\nDelta-spread robustness")
    print("=" * 72)

    for trim in (0.0, 0.005, 0.01):
        lower, upper = train[
            "delta_spread"
        ].quantile(
            [trim, 1.0 - trim]
        )

        train_trimmed = train[
            train["delta_spread"].between(
                lower,
                upper,
            )
        ]

        test_trimmed = test[
            test["delta_spread"].between(
                lower,
                upper,
            )
        ]

        r2_a = evaluate(
            train_trimmed,
            test_trimmed,
            FEATURES_A,
        )

        r2_b3 = evaluate(
            train_trimmed,
            test_trimmed,
            FEATURES_B3,
        )

        print(
            f"\nTrim: {trim:.3%}"
        )
        print(
            f"Delta-spread bounds: "
            f"[{lower:.6f}, {upper:.6f}]"
        )
        print(
            f"Train retained: "
            f"{len(train_trimmed):,} "
            f"({len(train_trimmed) / len(train):.2%})"
        )
        print(
            f"Test retained:  "
            f"{len(test_trimmed):,} "
            f"({len(test_trimmed) / len(test):.2%})"
        )
        print(
            f"A static R2:      {r2_a:.6f}"
        )
        print(
            f"B3 + delta spread R2: "
            f"{r2_b3:.6f}"
        )
        print(
            f"Incremental R2:   "
            f"{r2_b3 - r2_a:+.6f}"
        )


if __name__ == "__main__":
    main()