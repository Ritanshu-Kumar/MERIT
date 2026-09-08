import pandas as pd
from scipy.stats import pearsonr, spearmanr


DATASET = "data/sample/AAPL_m8_analysis_dataset.csv"

FEATURES = [
    "spread",
    "relative_spread",
    "bid_size",
    "ask_size",
    "imbalance",
    "microprice",
    "imbalance_l5",
    "imbalance_l10",
    "weighted_imbalance_l5",
    "weighted_imbalance_l10",
]

TARGET = "post_fill_move_1s"


def main() -> None:
    df = pd.read_csv(DATASET)

    rows = []

    for feature in FEATURES:
        x = df[feature]
        y = df[TARGET]

        pearson_r, pearson_p = pearsonr(x, y)
        spearman_r, spearman_p = spearmanr(x, y)

        rows.append(
            {
                "feature": feature,
                "pearson_r": pearson_r,
                "pearson_p": pearson_p,
                "spearman_r": spearman_r,
                "spearman_p": spearman_p,
            }
        )

    result = (
        pd.DataFrame(rows)
        .sort_values("spearman_p")
        .reset_index(drop=True)
    )

    print(result.to_string(index=False))


if __name__ == "__main__":
    main()