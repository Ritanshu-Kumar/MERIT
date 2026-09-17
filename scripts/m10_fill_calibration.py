from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss,
    roc_auc_score,
)
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)


INPUT = Path(
    "research/m9_fill_hazard_2019-07-30.csv"
)

OUTPUT = Path(
    "research/m10_fill_calibration_2019-07-30.csv"
)


NUMERIC_SETS = {
    "queue_only": [
        "queue_log",
    ],
    "state_queue": [
        "imbalance",
        "relative_spread",
        "queue_log",
    ],
    "state_transition_queue": [
        "imbalance",
        "relative_spread",
        "queue_log",
        "delta_imbalance",
        "delta_microprice",
    ],
}


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

    numeric = [
        "imbalance",
        "relative_spread",
        "delta_imbalance",
        "delta_microprice",
        "queue_log",
    ]

    for col in numeric:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    df["is_fill"] = (
        df["outcome"].eq("FILL").astype(int)
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
        ).reset_index(drop=True)

        cut = int(
            len(group) * 0.70
        )

        train_parts.append(
            group.iloc[:cut]
        )

        test_parts.append(
            group.iloc[cut:]
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


def calibration_table(
    y: np.ndarray,
    p: np.ndarray,
    bins: int = 10,
) -> pd.DataFrame:

    temp = pd.DataFrame(
        {
            "y": y,
            "p": p,
        }
    )

    temp["bin"] = pd.qcut(
        temp["p"],
        q=bins,
        labels=False,
        duplicates="drop",
    )

    rows = []

    for b, group in temp.groupby(
        "bin",
        sort=True,
    ):
        rows.append(
            {
                "bin": int(b),
                "n": len(group),
                "mean_predicted": float(
                    group["p"].mean()
                ),
                "observed_fill_rate": float(
                    group["y"].mean()
                ),
                "calibration_gap": float(
                    group["y"].mean()
                    - group["p"].mean()
                ),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:

    print(
        "Loading M10 fill dataset..."
    )

    df = pd.read_csv(INPUT)
    df = add_features(df)

    train, test = chronological_split(df)

    print(
        f"Train candidates: {len(train):,}"
    )
    print(
        f"Test candidates:  {len(test):,}"
    )

    results = []
    calibration_rows = []

    for model_name, numeric_features in NUMERIC_SETS.items():

        required = numeric_features + [
            "quote_side",
            "symbol",
        ]

        train_m = train.dropna(
            subset=required
        ).copy()

        test_m = test.dropna(
            subset=required
        ).copy()

        preprocessor = make_preprocessor(
            numeric_features
        )

        x_train = (
            preprocessor.fit_transform(
                train_m
            )
        )

        x_test = (
            preprocessor.transform(
                test_m
            )
        )

        model = LogisticRegression(
            max_iter=2000
        )

        model.fit(
            x_train,
            train_m["is_fill"],
        )

        p_test = model.predict_proba(
            x_test
        )[:, 1]

        y_test = (
            test_m["is_fill"]
            .to_numpy()
        )

        auc = roc_auc_score(
            y_test,
            p_test,
        )

        brier = brier_score_loss(
            y_test,
            p_test,
        )

        cal = calibration_table(
            y_test,
            p_test,
        )

        cal["model"] = model_name

        calibration_rows.append(
            cal
        )

        results.append(
            {
                "model": model_name,
                "n_test": len(test_m),
                "auc": auc,
                "brier": brier,
                "mean_predicted_fill": float(
                    p_test.mean()
                ),
                "observed_fill_rate": float(
                    y_test.mean()
                ),
            }
        )

        print(
            f"\n{model_name}"
        )
        print(
            f"  AUC:    {auc:.6f}"
        )
        print(
            f"  Brier:  {brier:.6f}"
        )
        print(
            f"  Mean predicted: "
            f"{p_test.mean():.6f}"
        )
        print(
            f"  Observed:       "
            f"{y_test.mean():.6f}"
        )

    summary = pd.DataFrame(
        results
    )

    calibration = pd.concat(
        calibration_rows,
        ignore_index=True,
    )

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_path = OUTPUT

    calibration_path = OUTPUT.with_name(
        "m10_fill_calibration_bins_2019-07-30.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    calibration.to_csv(
        calibration_path,
        index=False,
    )

    print(
        "\n===== M10 Summary ====="
    )

    print(
        summary.to_string(
            index=False
        )
    )

    print(
        f"\nSaved: {summary_path}"
    )

    print(
        f"Saved: {calibration_path}"
    )


if __name__ == "__main__":
    main()