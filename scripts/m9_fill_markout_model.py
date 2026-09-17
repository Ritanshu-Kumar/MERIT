from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler


INPUT = Path(
    "research/m9_fill_hazard_markouts_2019-07-30.csv"
)

OUTPUT = Path(
    "research/m9_fill_markout_model_2019-07-30.csv"
)

HORIZONS = {
    "10ms": "future_mid_10ms",
    "100ms": "future_mid_100ms",
    "500ms": "future_mid_500ms",
    "1s": "future_mid_1s",
}

FEATURE_SETS = {
    "A_state": [
        "imbalance",
        "relative_spread",
        "queue_log",
    ],
    "B_state_transition": [
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

    return df


def signed_markout_bps(
    df: pd.DataFrame,
    future_mid_col: str,
) -> np.ndarray:

    quote_price = df["quote_price"].to_numpy(
        dtype=float
    )

    future_mid = df[future_mid_col].to_numpy(
        dtype=float
    )

    buy = df["quote_side"].eq("BUY").to_numpy()

    signed = np.where(
        buy,
        future_mid - quote_price,
        quote_price - future_mid,
    )

    return signed / quote_price * 10_000.0


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
            "fill_timestamp_ns"
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


def metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> tuple[float, float, float]:

    r2 = float(
        r2_score(
            y_true,
            y_pred,
        )
    )

    mae = float(
        mean_absolute_error(
            y_true,
            y_pred,
        )
    )

    rho = float(
        spearmanr(
            y_true,
            y_pred,
        ).statistic
    )

    return r2, mae, rho


def main() -> None:

    print("Loading M9.5 fill dataset...")

    df = pd.read_csv(INPUT)

    df = df[
        df["outcome"].eq("FILL")
    ].copy()

    df = add_features(df)

    print(
        f"Completed fills: {len(df):,}"
    )

    train, test = chronological_split(df)

    print(
        f"Train: {len(train):,}"
    )

    print(
        f"Test:  {len(test):,}"
    )

    rows = []

    for horizon, future_mid_col in HORIZONS.items():

        print(
            f"\n===== {horizon} ====="
        )

        train_h = train.dropna(
            subset=[future_mid_col]
        ).copy()

        test_h = test.dropna(
            subset=[future_mid_col]
        ).copy()

        for model_name, numeric_features in FEATURE_SETS.items():

            required = (
                numeric_features
                + [
                    "quote_side",
                    "symbol",
                ]
            )

            train_h2 = train_h.dropna(
                subset=required
            ).copy()

            test_h2 = test_h.dropna(
                subset=required
            ).copy()

            y_train = signed_markout_bps(
                train_h2,
                future_mid_col,
            )

            y_test = signed_markout_bps(
                test_h2,
                future_mid_col,
            )

            preprocessor = make_preprocessor(
                numeric_features
            )

            x_train = preprocessor.fit_transform(
                train_h2
            )

            x_test = preprocessor.transform(
                test_h2
            )

            model = Ridge(
                alpha=10.0
            )

            model.fit(
                x_train,
                y_train,
            )

            y_pred = model.predict(
                x_test
            )

            r2, mae, rho = metrics(
                y_test,
                y_pred,
            )

            rows.append(
                {
                    "horizon": horizon,
                    "model": model_name,
                    "side": "ALL",
                    "n_train": len(train_h2),
                    "n_test": len(test_h2),
                    "r2": r2,
                    "mae_bps": mae,
                    "spearman": rho,
                }
            )

            print(
                f"{model_name:22s}"
                f" R2={r2:+.6f}"
                f" MAE={mae:.6f}"
                f" Spearman={rho:+.6f}"
            )

            for side in ("BUY", "SELL"):

                mask = (
                    test_h2[
                        "quote_side"
                    ].eq(side).to_numpy()
                )

                if mask.sum() < 10:
                    continue

                side_r2, side_mae, side_rho = metrics(
                    y_test[mask],
                    y_pred[mask],
                )

                rows.append(
                    {
                        "horizon": horizon,
                        "model": model_name,
                        "side": side,
                        "n_train": len(train_h2),
                        "n_test": int(mask.sum()),
                        "r2": side_r2,
                        "mae_bps": side_mae,
                        "spearman": side_rho,
                    }
                )

                print(
                    f"  {side:4s}"
                    f" R2={side_r2:+.6f}"
                    f" MAE={side_mae:.6f}"
                    f" Spearman={side_rho:+.6f}"
                )

    result = pd.DataFrame(rows)

    result["delta_r2_vs_A"] = np.nan

    for horizon in HORIZONS:

        base = result[
            (result["horizon"] == horizon)
            & (result["model"] == "A_state")
            & (result["side"] == "ALL")
        ]

        if base.empty:
            continue

        base_r2 = float(
            base.iloc[0]["r2"]
        )

        mask = (
            (result["horizon"] == horizon)
            & (result["side"] == "ALL")
        )

        result.loc[
            mask,
            "delta_r2_vs_A",
        ] = (
            result.loc[mask, "r2"]
            - base_r2
        )

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        OUTPUT,
        index=False,
    )

    print(
        "\n===== Summary ====="
    )

    print(
        result[
            result["side"] == "ALL"
        ].to_string(index=False)
    )

    print(
        f"\nSaved: {OUTPUT}"
    )


if __name__ == "__main__":
    main()