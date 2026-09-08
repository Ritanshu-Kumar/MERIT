import pandas as pd


DATASET = "data/sample/AAPL_m8_analysis_dataset.csv"

TARGET = "post_fill_move_1s"

FEATURES = [
    "spread",
    "ask_size",
    "bid_size",
    "signed_imbalance",
    "signed_microprice_edge",
    "signed_imbalance_l5",
    "signed_imbalance_l10",
    "signed_weighted_imbalance_l5",
    "signed_weighted_imbalance_l10",
]

TRAIN_FRACTION = 0.60
VALIDATION_FRACTION = 0.20
PURGE_SECONDS = 5
EMBARGO_SECONDS = 5


def add_side_aware_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
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


def find_feature_threshold(
    validation: pd.DataFrame,
    feature: str,
) -> tuple[float, float, int]:
    quantiles = validation[feature].quantile(
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    )

    best_threshold = float("nan")
    best_value = float("inf")
    best_count = 0

    for threshold in quantiles:
        selected = validation[
            validation[feature] <= threshold
        ]

        if len(selected) < 10:
            continue

        value = selected[TARGET].mean()

        if value < best_value:
            best_value = value
            best_threshold = float(threshold)
            best_count = len(selected)

    return best_threshold, best_value, best_count


def main() -> None:
    df = pd.read_csv(
        DATASET,
        parse_dates=["timestamp"],
    )

    df = add_side_aware_features(df)

    train, validation, test = split_dataset(df)

    print(
        f"Train: {len(train):,} | "
        f"Validation: {len(validation):,} | "
        f"Test: {len(test):,}"
    )

    results = []

    for feature in FEATURES:
        threshold, validation_value, count = (
            find_feature_threshold(
                validation,
                feature,
            )
        )

        results.append(
            {
                "feature": feature,
                "threshold": threshold,
                "validation_mean": validation_value,
                "validation_count": count,
            }
        )

    result = (
        pd.DataFrame(results)
        .sort_values("validation_mean")
        .reset_index(drop=True)
    )

    print("\nValidation feature selection:")
    print(
        result.to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    best = result.iloc[0]

    feature = str(best["feature"])
    threshold = float(best["threshold"])

    print(
        f"\nSelected rule: {feature} <= {threshold:.6f}"
    )

    selected_test = test[
        test[feature] <= threshold
    ]

    all_test_mean = test[TARGET].mean()

    print("\nFrozen test evaluation:")
    print(f"All test observations: {len(test):,}")
    print(f"Selected observations: {len(selected_test):,}")
    print(
        f"All test mean target: "
        f"{all_test_mean:.6f}"
    )
    print(
        f"Selected mean target: "
        f"{selected_test[TARGET].mean():.6f}"
    )
    print(
        f"Selected negative rate: "
        f"{(selected_test[TARGET] < 0).mean():.2%}"
    )


if __name__ == "__main__":
    main()