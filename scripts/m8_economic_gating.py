import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


DATASET = "data/sample/AAPL_m8_analysis_dataset.csv"

FEATURES = [
    "spread",
    "bid_size",
    "ask_size",
    "signed_imbalance",
    "signed_microprice_edge",
    "signed_imbalance_l5",
    "signed_imbalance_l10",
    "signed_weighted_imbalance_l5",
    "signed_weighted_imbalance_l10",
]

TARGET = "post_fill_move_1s"

TRAIN_FRACTION = 0.60
VALIDATION_FRACTION = 0.20
PURGE_SECONDS = 5
EMBARGO_SECONDS = 5


def add_side_aware_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    side_sign = result["side"].map(
        {
            "BUY": 1,
            "SELL": -1,
        }
    )

    result["signed_imbalance"] = (
        result["imbalance"] * side_sign
    )

    result["signed_microprice_edge"] = (
        (result["microprice"] - result["mid_price"])
        * side_sign
    )

    result["signed_imbalance_l5"] = (
        result["imbalance_l5"] * side_sign
    )

    result["signed_imbalance_l10"] = (
        result["imbalance_l10"] * side_sign
    )

    result["signed_weighted_imbalance_l5"] = (
        result["weighted_imbalance_l5"] * side_sign
    )

    result["signed_weighted_imbalance_l10"] = (
        result["weighted_imbalance_l10"] * side_sign
    )

    return result


def split_dataset(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df.sort_values("timestamp").reset_index(drop=True)

    n = len(df)

    train_end = int(n * TRAIN_FRACTION)
    validation_end = int(
        n * (TRAIN_FRACTION + VALIDATION_FRACTION)
    )

    train = df.iloc[:train_end].copy()
    validation = df.iloc[train_end:validation_end].copy()
    test = df.iloc[validation_end:].copy()

    purge = pd.Timedelta(seconds=PURGE_SECONDS)
    embargo = pd.Timedelta(seconds=EMBARGO_SECONDS)

    validation = validation[
        validation["timestamp"]
        >= train["timestamp"].iloc[-1] + purge
    ]

    test = test[
        test["timestamp"]
        >= validation["timestamp"].iloc[-1] + embargo
    ]

    return train, validation, test


def add_predictions(
    model: LinearRegression,
    scaler: StandardScaler,
    df: pd.DataFrame,
) -> pd.DataFrame:
    result = df.copy()

    result["predicted_post_fill_move_1s"] = model.predict(
        scaler.transform(result[FEATURES])
    )

    return result


def evaluate_threshold(
    df: pd.DataFrame,
    threshold: float,
) -> dict[str, float]:
    selected = df[
        df["predicted_post_fill_move_1s"] >= threshold
    ]

    if selected.empty:
        return {
            "threshold": threshold,
            "selected": 0,
            "selection_rate": 0.0,
            "mean_post_fill_move": float("nan"),
            "mean_gross_markout": float("nan"),
            "mean_net_economic_value": float("nan"),
        }

    initial_edge = selected["initial_edge_1s"]
    post_fill_move = selected["post_fill_move_1s"]
    gross_markout = selected["markout_1s"]

    economic_value = initial_edge + post_fill_move

    return {
        "threshold": threshold,
        "selected": len(selected),
        "selection_rate": len(selected) / len(df),
        "mean_post_fill_move": post_fill_move.mean(),
        "mean_gross_markout": gross_markout.mean(),
        "mean_net_economic_value": economic_value.mean(),
    }


def main() -> None:
    df = pd.read_csv(
        DATASET,
        parse_dates=["timestamp"],
    )

    df = add_side_aware_features(df)

    train, validation, test = split_dataset(df)

    scaler = StandardScaler()

    x_train = scaler.fit_transform(train[FEATURES])

    model = LinearRegression()
    model.fit(x_train, train[TARGET])

    validation = add_predictions(
        model,
        scaler,
        validation,
    )

    test = add_predictions(
        model,
        scaler,
        test,
    )

    thresholds = sorted(
        validation["predicted_post_fill_move_1s"]
        .quantile(
            [
                0.0,
                0.1,
                0.2,
                0.3,
                0.4,
                0.5,
                0.6,
                0.7,
                0.8,
                0.9,
            ]
        )
        .unique()
    )

    validation_results = pd.DataFrame(
        evaluate_threshold(validation, threshold)
        for threshold in thresholds
    )

    print("Validation threshold search:")
    print(
        validation_results.to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    valid_results = validation_results.dropna(
        subset=["mean_net_economic_value"]
    )

    best_row = valid_results.loc[
        valid_results["mean_net_economic_value"].idxmax()
    ]

    best_threshold = float(best_row["threshold"])

    print(f"\nSelected threshold: {best_threshold:.6f}")

    test_result = evaluate_threshold(
        test,
        best_threshold,
    )

    print("\nFrozen test evaluation:")
    for key, value in test_result.items():
        if key == "selected":
            print(f"{key}: {int(value)}")
        else:
            print(f"{key}: {value:.6f}")


if __name__ == "__main__":
    main()