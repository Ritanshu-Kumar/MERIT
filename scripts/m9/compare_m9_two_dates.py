from pathlib import Path

import pandas as pd

DATES = {
    "2019-07-30": {
        "summary": Path(
            "research/m9_regular_fill_markout_summary.csv"
        ),
        "bootstrap": Path(
            "research/m9_regular_markout_block_bootstrap.csv"
        ),
        "data": Path(
            "research/m9_time_sampled_2019-07-30_regular_asof.csv"
        ),
    },
    "2019-08-30": {
        "summary": Path(
            "research/m9_2019-08-30_regular_fill_markout_summary.csv"
        ),
        "bootstrap": Path(
            "research/m9_2019-08-30_regular_markout_block_bootstrap.csv"
        ),
        "data": Path(
            "research/m9_time_sampled_2019-08-30_regular_asof.csv"
        ),
    },
}

OUTPUT = Path("research/m9_two_date_one_second_comparison.csv")


def load_date(date, paths):
    summary = pd.read_csv(paths["summary"], low_memory=False)
    summary = summary[
        (summary["horizon"] == "1s")
        & (summary["scope"] == "by_symbol")
    ].copy()
    summary = summary.rename(
        columns={
            "group": "symbol",
            "n_fills": "summary_fills",
            "mean_markout_per_share": "mean_markout",
            "mean_markout_bps": "mean_markout_bps",
        }
    )

    bootstrap = pd.read_csv(paths["bootstrap"], low_memory=False)
    bootstrap = bootstrap[
        (bootstrap["horizon"] == "1s")
        & (bootstrap["block_minutes"] == 5)
    ].copy()

    result = summary.merge(
        bootstrap[
            [
                "symbol",
                "n_fills",
                "ci95_low",
                "ci95_high",
                "bonferroni_ci_low",
                "bonferroni_ci_high",
                "familywise_interval_excludes_zero",
            ]
        ],
        on="symbol",
        how="left",
    )

    data = pd.read_csv(paths["data"], low_memory=False)

    symbol_stats = (
        data.groupby("symbol")
        .agg(
            regular_rows=("outcome", "size"),
            regular_fill_count=(
                "outcome",
                lambda s: s.eq("FILL").sum(),
            ),
        )
        .reset_index()
    )

    symbol_stats["regular_fill_rate_pct"] = (
        100
        * symbol_stats["regular_fill_count"]
        / symbol_stats["regular_rows"]
    )

    result = result.merge(
        symbol_stats,
        on="symbol",
        how="left",
        validate="one_to_one",
    )

    result["date"] = date

    return result


def main():
    parts = []

    for date, paths in DATES.items():
        for path in paths.values():
            if not path.exists():
                raise FileNotFoundError(path)

        parts.append(load_date(date, paths))

    combined = pd.concat(parts, ignore_index=True)

    july = combined[combined["date"] == "2019-07-30"].copy()
    august = combined[combined["date"] == "2019-08-30"].copy()

    comparison = july.merge(
        august,
        on="symbol",
        suffixes=("_jul30", "_aug30"),
    )

    comparison["mean_change_aug_minus_jul"] = (
        comparison["mean_markout_aug30"]
        - comparison["mean_markout_jul30"]
    )

    columns = [
        "symbol",
        "n_fills_jul30",
        "mean_markout_jul30",
        "bonferroni_ci_low_jul30",
        "bonferroni_ci_high_jul30",
        "familywise_interval_excludes_zero_jul30",
        "n_fills_aug30",
        "mean_markout_aug30",
        "bonferroni_ci_low_aug30",
        "bonferroni_ci_high_aug30",
        "familywise_interval_excludes_zero_aug30",
        "mean_change_aug_minus_jul",
    ]

    comparison[columns].to_csv(OUTPUT, index=False)

    print("Saved:", OUTPUT)
    print("\nPER-DATE FILL RATES")
    print(
        combined[
            [
                "date",
                "symbol",
                "regular_rows",
                "regular_fill_count",
                "regular_fill_rate_pct",
            ]
        ].drop_duplicates(
            ["date", "symbol"]
        ).to_string(index=False)
    )

    print("\nONE-SECOND CROSS-DATE COMPARISON")
    print(comparison[columns].round(6).to_string(index=False))


if __name__ == "__main__":
    main()