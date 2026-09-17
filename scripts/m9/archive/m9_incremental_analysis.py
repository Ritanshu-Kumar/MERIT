from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_PATH = Path(
    "research/m9_state_transitions_2019-07-30.csv"
)

TARGETS = [
    "mid_return_bps_100ms",
    "mid_return_bps_1s",
]

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
    "B3_static_dspread": [
        "imbalance",
        "relative_spread",
        "delta_spread",
    ],
    "B4_all_transitions": [
        "imbalance",
        "relative_spread",
        "delta_imbalance",
        "delta_microprice",
        "delta_spread",
    ],
}


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
        group = group.sort_values("timestamp_ns")

        split = int(
            len(group) * train_fraction
        )

        train_parts.append(group.iloc[:split])
        test_parts.append(group.iloc[split:])

    return (
        pd.concat(train_parts),
        pd.concat(test_parts),
    )


def spearman(
    y_true: pd.Series,
    prediction: np.ndarray,
) -> float:
    return float(
        y_true.corr(
            pd.Series(
                prediction,
                index=y_true.index,
            ),
            method="spearman",
        )
    )


def evaluate(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    target: str,
) -> dict[str, float]:
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
    y = test[target]

    return {
        "r2": r2_score(y, prediction),
        "mae": mean_absolute_error(
            y,
            prediction,
        ),
        "spearman": spearman(
            y,
            prediction,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=Path,
        default=DATA_PATH,
    )

    args = parser.parse_args()

    df = pd.read_csv(args.input)

    train, test = chronological_split(df)

    print(
        f"Loaded {len(df):,} rows"
    )
    print(
        f"Train: {len(train):,}"
    )
    print(
        f"Test:  {len(test):,}"
    )

    for target in TARGETS:
        print("\n" + "=" * 80)
        print(f"TARGET: {target}")
        print("=" * 80)

        baseline_r2 = None

        for name, features in MODELS.items():
            metrics = evaluate(
                train,
                test,
                features,
                target,
            )

            if baseline_r2 is None:
                baseline_r2 = metrics["r2"]

            print(
                f"\n{name}"
            )
            print(
                f"  R2:        {metrics['r2']:.6f}"
            )
            print(
                f"  ΔR2:       "
                f"{metrics['r2'] - baseline_r2:+.6f}"
            )
            print(
                f"  MAE:       {metrics['mae']:.6f}"
            )
            print(
                f"  Spearman:  "
                f"{metrics['spearman']:.6f}"
            )


if __name__ == "__main__":
    main()