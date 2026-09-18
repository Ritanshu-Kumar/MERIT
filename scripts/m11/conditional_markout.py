from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler


HAZARD_DEFAULT = Path(
    "research/m9_fill_hazard_markouts_2019-07-30.csv"
)

STATE_DEFAULT = Path(
    "research/m9_state_transitions_2019-07-30.csv"
)

SUMMARY_DEFAULT = Path(
    "research/m11/results/conditional_markout_2019-07-30.csv"
)

COEF_DEFAULT = Path(
    "research/m11/results/conditional_markout_coefficients_2019-07-30.csv"
)

HORIZONS = {
    "10ms": "future_mid_10ms",
    "100ms": "future_mid_100ms",
    "500ms": "future_mid_500ms",
    "1s": "future_mid_1s",
}

PRIMARY_HORIZON = "1s"

MODEL_FEATURES = [
    "imbalance",
    "relative_spread_bps",
    "queue_log",
    "adverse_flow_10ms",
    "adverse_flow_100ms",
    "time_to_fill_ms",
    "fill_fraction",
    "side_code",
    "side_x_imbalance",
]


def signed_markout_bps(
    quote_side: pd.Series,
    quote_price: pd.Series,
    future_mid: pd.Series,
) -> np.ndarray:
    quote = quote_price.to_numpy(dtype=float)
    future = future_mid.to_numpy(dtype=float)
    buy = quote_side.eq("BUY").to_numpy()

    signed = np.where(
        buy,
        quote - future,
        future - quote,
    )

    return signed / quote * 10_000.0


def assign_common_split(
    df: pd.DataFrame,
) -> pd.DataFrame:
    df = df.sort_values(
        ["symbol", "timestamp_ns"],
        kind="mergesort",
    ).copy()

    df["rank"] = (
        df.groupby("symbol", sort=False)
        .cumcount()
    )

    df["n_symbol"] = (
        df.groupby("symbol", sort=False)["timestamp_ns"]
        .transform("size")
    )

    df["split"] = np.where(
        df["rank"]
        < np.floor(df["n_symbol"] * 0.70),
        "TRAIN",
        "TEST",
    )

    return df.drop(
        columns=["rank", "n_symbol"]
    )


def link_pre_fill_flow(
    fills: pd.DataFrame,
    state: pd.DataFrame,
) -> pd.DataFrame:
    fills = fills.copy()
    state = state.copy()

    fills["fill_timestamp_ns"] = pd.to_numeric(
        fills["fill_timestamp_ns"],
        errors="coerce",
    ).astype("int64")

    fills["match_side"] = np.where(
        fills["quote_side"].eq("BUY"),
        "SELL",
        "BUY",
    )

    fills["match_price"] = (
        pd.to_numeric(
            fills["quote_price"],
            errors="coerce",
        ).round(6)
    )

    state["timestamp_ns"] = pd.to_numeric(
        state["timestamp_ns"],
        errors="coerce",
    ).astype("int64")

    state["match_price"] = (
        pd.to_numeric(
            state["execution_price"],
            errors="coerce",
        ).round(6)
    )

    key = [
        "timestamp_ns",
        "symbol",
        "execution_side",
        "match_price",
    ]

    state_key_counts = (
        state.groupby(key, dropna=False)
        .size()
        .rename("match_count")
        .reset_index()
    )

    state_flow = state[
        [
            *key,
            "buy_flow_10ms",
            "sell_flow_10ms",
            "buy_flow_100ms",
            "sell_flow_100ms",
        ]
    ].merge(
        state_key_counts,
        on=key,
        how="left",
    )

    state_flow = (
        state_flow
        .sort_values(
            key,
            kind="mergesort",
        )
        .drop_duplicates(
            key,
            keep="first",
        )
    )

    exact = fills.merge(
        state_flow,
        left_on=[
            "fill_timestamp_ns",
            "symbol",
            "match_side",
            "match_price",
        ],
        right_on=[
            "timestamp_ns",
            "symbol",
            "execution_side",
            "match_price",
        ],
        how="left",
        suffixes=("", "_state"),
    )

    exact["flow_lag_ns"] = np.where(
        exact["buy_flow_10ms"].notna(),
        0,
        np.nan,
    )

    exact["flow_match_type"] = np.where(
        exact["buy_flow_10ms"].notna(),
        "EXACT",
        "FALLBACK",
    )

    unmatched = exact[
        exact["buy_flow_10ms"].isna()
    ][fills.columns].copy()

    if not unmatched.empty:
        fallback_state = state[
            [
                "timestamp_ns",
                "symbol",
                "buy_flow_10ms",
                "sell_flow_10ms",
                "buy_flow_100ms",
                "sell_flow_100ms",
            ]
        ].rename(
            columns={
                "timestamp_ns": "state_timestamp_ns",
            }
        )

        fallback_state = fallback_state.sort_values(
            ["state_timestamp_ns", "symbol"],
            kind="mergesort",
        )

        unmatched = unmatched.sort_values(
            ["fill_timestamp_ns", "symbol"],
            kind="mergesort",
        )

        fallback = pd.merge_asof(
            unmatched,
            fallback_state,
            left_on="fill_timestamp_ns",
            right_on="state_timestamp_ns",
            by="symbol",
            direction="backward",
            allow_exact_matches=False,
        )

        fallback["flow_lag_ns"] = (
            fallback["fill_timestamp_ns"]
            - fallback["state_timestamp_ns"]
        )

        fallback["flow_match_type"] = "FALLBACK"

        exact = exact[
            exact["buy_flow_10ms"].notna()
        ]

        exact = pd.concat(
            [exact, fallback],
            ignore_index=True,
        )

    if len(exact) != len(fills):
        raise RuntimeError(
            f"Fill linkage changed row count: "
            f"{len(fills):,} -> {len(exact):,}"
        )

    return exact

def build_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    df = df.copy()

    numeric_columns = [
        "imbalance",
        "relative_spread",
        "queue_ahead_initial",
        "fill_fraction",
        "timestamp_ns",
        "fill_timestamp_ns",
        "buy_flow_10ms",
        "sell_flow_10ms",
        "buy_flow_100ms",
        "sell_flow_100ms",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df["queue_log"] = np.log1p(
        np.maximum(
            df["queue_ahead_initial"],
            0.0,
        )
    )

    df["relative_spread_bps"] = (
        df["relative_spread"] * 10_000.0
    )

    df["time_to_fill_ms"] = (
        df["fill_timestamp_ns"]
        - df["timestamp_ns"]
    ) / 1_000_000.0

    # BUY quote = executions against resting BUYs
    # SELL quote = executions against resting SELLs.
    # Therefore positive adverse_flow means flow against
    # the passive quote side.
    df["adverse_flow_10ms"] = np.where(
        df["quote_side"].eq("BUY"),
        df["buy_flow_10ms"],
        df["sell_flow_10ms"],
    )

    df["adverse_flow_100ms"] = np.where(
        df["quote_side"].eq("BUY"),
        df["buy_flow_100ms"],
        df["sell_flow_100ms"],
    )

    df["side_code"] = np.where(
        df["quote_side"].eq("BUY"),
        1.0,
        -1.0,
    )

    df["side_x_imbalance"] = (
        df["side_code"] * df["imbalance"]
    )

    return df


def block_ids(
    df: pd.DataFrame,
    timestamp_column: str,
    block_ms: int,
) -> np.ndarray:
    block_ns = block_ms * 1_000_000

    return (
        df["symbol"].astype(str)
        + ":"
        + (
            df[timestamp_column] // block_ns
        ).astype("int64").astype(str)
    ).to_numpy()





def make_block_groups(
    block_ids: np.ndarray,
) -> list[np.ndarray]:
    unique = np.unique(block_ids)
    return [
        np.flatnonzero(block_ids == block)
        for block in unique
    ]


def bootstrap_test_r2(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: list[np.ndarray],
    reps: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    n_blocks = len(groups)

    block_n = np.array(
        [len(index) for index in groups],
        dtype=float,
    )

    block_sum_y = np.array(
        [y_true[index].sum() for index in groups],
        dtype=float,
    )

    block_sum_y2 = np.array(
        [
            np.square(y_true[index]).sum()
            for index in groups
        ],
        dtype=float,
    )

    block_sse = np.array(
        [
            np.square(
                y_true[index] - y_pred[index]
            ).sum()
            for index in groups
        ],
        dtype=float,
    )

    observed = float(
        r2_score(y_true, y_pred)
    )

    draws = np.empty(
        reps,
        dtype=float,
    )

    for i in range(reps):
        picks = rng.integers(
            0,
            n_blocks,
            size=n_blocks,
        )

        n = block_n[picks].sum()
        sum_y = block_sum_y[picks].sum()
        sum_y2 = block_sum_y2[picks].sum()
        sse = block_sse[picks].sum()

        sst = (
            sum_y2
            - (sum_y * sum_y) / n
        )

        draws[i] = (
            1.0 - sse / sst
            if sst > 0
            else np.nan
        )

    low, high = np.nanquantile(
        draws,
        [0.025, 0.975],
    )

    return (
        observed,
        float(low),
        float(high),
    )


def bootstrap_statistic(
    values: np.ndarray,
    groups: list[np.ndarray],
    statistic: str,
    reps: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    n_blocks = len(groups)

    if statistic == "mean":
        block_sum = np.array(
            [values[index].sum() for index in groups],
            dtype=float,
        )
    elif statistic == "positive_fraction":
        block_sum = np.array(
            [
                np.sum(values[index] > 0.0)
                for index in groups
            ],
            dtype=float,
        )
    else:
        raise ValueError(
            f"Unknown statistic: {statistic}"
        )

    block_n = np.array(
        [len(index) for index in groups],
        dtype=float,
    )

    if statistic == "mean":
        observed = float(
            values.mean()
        )
    else:
        observed = float(
            np.mean(values > 0.0)
        )

    draws = np.empty(
        reps,
        dtype=float,
    )

    for i in range(reps):
        picks = rng.integers(
            0,
            n_blocks,
            size=n_blocks,
        )

        denominator = block_n[picks].sum()

        draws[i] = (
            block_sum[picks].sum()
            / denominator
        )

    low, high = np.quantile(
        draws,
        [0.025, 0.975],
    )

    return (
        observed,
        float(low),
        float(high),
    )


def bootstrap_coefficients(
    x: np.ndarray,
    y: np.ndarray,
    groups: list[np.ndarray],
    feature_names: list[str],
    reps: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    n_blocks = len(groups)

    draws = np.empty(
        (reps, len(feature_names)),
        dtype=float,
    )

    for i in range(reps):
        picks = rng.integers(
            0,
            n_blocks,
            size=n_blocks,
        )

        indices = np.concatenate(
            [groups[index] for index in picks]
        )

        model = LinearRegression()
        model.fit(
            x[indices],
            y[indices],
        )

        draws[i] = model.coef_

    rows = []

    for index, feature in enumerate(
        feature_names
    ):
        low, high = np.quantile(
            draws[:, index],
            [0.025, 0.975],
        )

        rows.append(
            {
                "feature": feature,
                "ci95_low": float(low),
                "ci95_high": float(high),
                "ci_excludes_zero": bool(
                    low > 0.0
                    or high < 0.0
                ),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "M11 conditional adverse-selection "
            "markout model."
        )
    )

    parser.add_argument(
        "--hazard",
        type=Path,
        default=HAZARD_DEFAULT,
    )

    parser.add_argument(
        "--state",
        type=Path,
        default=STATE_DEFAULT,
    )

    parser.add_argument(
        "--bootstrap-reps",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--block-ms",
        type=int,
        default=60_000,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=20260918,
    )

    parser.add_argument(
        "--summary-output",
        type=Path,
        default=SUMMARY_DEFAULT,
    )

    parser.add_argument(
        "--coefficient-output",
        type=Path,
        default=COEF_DEFAULT,
    )

    args = parser.parse_args()

    print("Loading M9 fill-hazard dataset...")
    hazard = pd.read_csv(args.hazard)

    print(
        f"Candidates: {len(hazard):,}"
    )

    print(
        "Loading corrected M9 "
        "state-transition dataset..."
    )

    state = pd.read_csv(args.state)

    print(
        f"State observations: {len(state):,}"
    )

    required_hazard = {
        "timestamp_ns",
        "symbol",
        "quote_side",
        "quote_price",
        "queue_ahead_initial",
        "fill_fraction",
        "fill_timestamp_ns",
        "outcome",
        *HORIZONS.values(),
    }

    missing_hazard = sorted(
        required_hazard
        - set(hazard.columns)
    )

    if missing_hazard:
        raise RuntimeError(
            "Hazard file missing: "
            + ", ".join(missing_hazard)
        )

    required_state = {
        "timestamp_ns",
        "symbol",
        "buy_flow_10ms",
        "sell_flow_10ms",
        "buy_flow_100ms",
        "sell_flow_100ms",
    }

    missing_state = sorted(
        required_state
        - set(state.columns)
    )

    if missing_state:
        raise RuntimeError(
            "State file missing: "
            + ", ".join(missing_state)
        )

    # ------------------------------------------------------------
    # Freeze the same candidate-level 70/30 split used by M10.
    # ------------------------------------------------------------

    candidates = assign_common_split(
        hazard
    )

    fills = candidates[
        candidates["outcome"].eq("FILL")
        & candidates[
            "fill_timestamp_ns"
        ].notna()
    ].copy()

    print(
        f"Completed fills: {len(fills):,}"
    )

    # ------------------------------------------------------------
    # Link each fill to the latest state observation strictly
    # before the fill timestamp.
    # ------------------------------------------------------------

    linked = link_pre_fill_flow(
        fills,
        state,
    )

    print(
        "Linked fills:",
        f"{len(linked):,}",
    )

    print(
        "Exact matches:",
        int(
            (linked["flow_match_type"] == "EXACT").sum()
        ),
    )

    print(
        "Fallback matches:",
        int(
            (linked["flow_match_type"] == "FALLBACK").sum()
        ),
    )

    
    flow_columns = [
        "buy_flow_10ms",
        "sell_flow_10ms",
        "buy_flow_100ms",
        "sell_flow_100ms",
    ]

    missing_flow = int(
        linked[flow_columns]
        .isna()
        .any(axis=1)
        .sum()
    )

    if missing_flow:
        raise RuntimeError(
            "Missing pre-fill flow linkage: "
            f"{missing_flow:,} fills"
        )

    lag_ms = (
        linked["flow_lag_ns"]
        / 1_000_000.0
    )

    print(
        "Pre-fill state linkage lag:"
    )
    print(
        f"  median = {lag_ms.median():.3f} ms"
    )
    print(
        f"  p95    = {lag_ms.quantile(.95):.3f} ms"
    )
    print(
        f"  max    = {lag_ms.max():.3f} ms"
    )

    linked = build_features(
        linked
    )

    if len(linked) != 13_805:
        raise RuntimeError(
            f"Expected 13,805 linked fills, "
            f"got {len(linked):,}"
        )

    if linked.index.duplicated().any():
        raise RuntimeError(
            "Duplicate linked fill rows detected."
        )

    train = linked[
        linked["split"].eq("TRAIN")
    ].copy()

    test = linked[
        linked["split"].eq("TEST")
    ].copy()

    print(
        f"Train fills: {len(train):,}"
    )

    print(
        f"Test fills:  {len(test):,}"
    )

    # ------------------------------------------------------------
    # Keep predictors exactly as predefined. A feature with no
    # variation in training data is dropped as a data property,
    # not as model selection.
    # ------------------------------------------------------------

    usable_features = []

    for feature in MODEL_FEATURES:
        unique = (
            train[feature]
            .dropna()
            .nunique()
        )

        if unique > 1:
            usable_features.append(
                feature
            )
        else:
            print(
                f"Dropping constant predictor: "
                f"{feature}"
            )

    print(
        "\nModel features:"
    )

    for feature in usable_features:
        print(
            f"  {feature}"
        )

    rng = np.random.default_rng(
        args.seed
    )

    summary_rows = []
    coefficient_frames = []

    primary_coefficient_frame = None

    # ------------------------------------------------------------
    # Same linear model specification at every markout horizon.
    # ------------------------------------------------------------

    for horizon, future_column in HORIZONS.items():
        print(
            f"\n===== {horizon} ====="
        )

        train_h = train.dropna(
            subset=[
                future_column,
                *usable_features,
            ]
        ).copy()

        test_h = test.dropna(
            subset=[
                future_column,
                *usable_features,
            ]
        ).copy()

        if len(train_h) < 50:
            raise RuntimeError(
                f"Too few training fills for "
                f"{horizon}: {len(train_h)}"
            )

        if len(test_h) < 20:
            raise RuntimeError(
                f"Too few test fills for "
                f"{horizon}: {len(test_h)}"
            )

        y_train = signed_markout_bps(
            train_h["quote_side"],
            train_h["quote_price"],
            train_h[future_column],
        )

        y_test = signed_markout_bps(
            test_h["quote_side"],
            test_h["quote_price"],
            test_h[future_column],
        )

        scaler = StandardScaler()

        x_train = scaler.fit_transform(
            train_h[usable_features]
        )

        x_test = scaler.transform(
            test_h[usable_features]
        )

        model = LinearRegression()
        model.fit(
            x_train,
            y_train,
        )

        prediction = model.predict(
            x_test
        )

        test_blocks = block_ids(
            test_h,
            "fill_timestamp_ns",
            args.block_ms,
        )

        test_groups = make_block_groups(
            test_blocks
        )

        r2, r2_low, r2_high = (
            bootstrap_test_r2(
                y_test,
                prediction,
                test_groups,
                args.bootstrap_reps,
                rng,
            )
        )

        mae = float(
            mean_absolute_error(
                y_test,
                prediction,
            )
        )

        mean_markout, mean_low, mean_high = (
            bootstrap_statistic(
                y_test,
                test_groups,
                "mean",
                args.bootstrap_reps,
                rng,
            )
        )

        adverse_probability, adverse_low, adverse_high = (
            bootstrap_statistic(
                (y_test > 0.0).astype(float),
                test_groups,
                "positive_fraction",
                args.bootstrap_reps,
                rng,
            )
        )

        buy_mask = (
            test_h["quote_side"]
            .eq("BUY")
            .to_numpy()
        )

        sell_mask = (
            test_h["quote_side"]
            .eq("SELL")
            .to_numpy()
        )

        buy_mean = float(
            np.mean(y_test[buy_mask])
        )

        sell_mean = float(
            np.mean(y_test[sell_mask])
        )

        print(
            f"R2: {r2:+.6f} "
            f"[{r2_low:+.6f}, "
            f"{r2_high:+.6f}]"
        )

        print(
            f"MAE: {mae:.6f} bps"
        )

        print(
            f"Mean markout: "
            f"{mean_markout:+.6f} bps "
            f"[{mean_low:+.6f}, "
            f"{mean_high:+.6f}]"
        )

        print(
            f"P(adverse): "
            f"{adverse_probability:.4f} "
            f"[{adverse_low:.4f}, "
            f"{adverse_high:.4f}]"
        )

        print(
            f"BUY mean:  {buy_mean:+.6f} bps"
        )

        print(
            f"SELL mean: {sell_mean:+.6f} bps"
        )

        summary_rows.append(
            {
                "horizon": horizon,
                "n_train": len(train_h),
                "n_test": len(test_h),
                "r2_oos": r2,
                "r2_ci95_low": r2_low,
                "r2_ci95_high": r2_high,
                "mae_bps": mae,
                "mean_markout_bps": mean_markout,
                "mean_markout_ci95_low": mean_low,
                "mean_markout_ci95_high": mean_high,
                "p_adverse_markout": adverse_probability,
                "p_adverse_ci95_low": adverse_low,
                "p_adverse_ci95_high": adverse_high,
                "buy_mean_markout_bps": buy_mean,
                "sell_mean_markout_bps": sell_mean,
                "flow_lag_median_ms": float(
                    test_h["flow_lag_ns"].median()
                    / 1_000_000.0
                ),
                "flow_lag_p95_ms": float(
                    test_h["flow_lag_ns"].quantile(.95)
                    / 1_000_000.0
                ),
            }
        )

        coefficient_frame = pd.DataFrame(
            {
                "horizon": horizon,
                "feature": usable_features,
                "coefficient_standardized": (
                    model.coef_
                ),
            }
        )

        # Coefficient bootstrap is done on TRAINING blocks,
        # preserving the frozen test set for OOS evaluation.
        if horizon == PRIMARY_HORIZON:
            train_blocks = block_ids(
                train_h,
                "fill_timestamp_ns",
                args.block_ms,
            )

            train_groups = make_block_groups(
                train_blocks
            )

            coefficient_ci = (
                bootstrap_coefficients(
                    x_train,
                    y_train,
                    train_groups,
                    usable_features,
                    args.bootstrap_reps,
                    rng,
                )
            )

            coefficient_frame = (
                coefficient_frame.merge(
                    coefficient_ci,
                    on="feature",
                    how="left",
                )
            )

            primary_coefficient_frame = (
                coefficient_frame.copy()
            )

        coefficient_frames.append(
            coefficient_frame
        )

    # ------------------------------------------------------------
    # Primary reviewer stopping diagnostics.
    # ------------------------------------------------------------

    summary = pd.DataFrame(
        summary_rows
    )

    coefficients = pd.concat(
        coefficient_frames,
        ignore_index=True,
    )

    primary_row = summary[
        summary["horizon"].eq(
            PRIMARY_HORIZON
        )
    ].iloc[0]

    side_row = primary_coefficient_frame[
        primary_coefficient_frame["feature"].eq(
            "side_x_imbalance"
        )
    ]

    side_significant = (
        not side_row.empty
        and bool(
            side_row.iloc[0][
                "ci_excludes_zero"
            ]
        )
    )

    other_significant = coefficients[
        (coefficients["horizon"] == PRIMARY_HORIZON)
        & (
            coefficients["feature"]
            != "side_x_imbalance"
        )
        & coefficients["ci_excludes_zero"].fillna(False)
    ]

    baseline_r2_1s = 0.046317
    r2_1s = float(
        primary_row["r2_oos"]
    )

    side_significant = (
        not side_row.empty
        and bool(side_row.iloc[0]["ci_excludes_zero"])
    )

    other_significant = coefficients[
        (coefficients["horizon"] == PRIMARY_HORIZON)
        & (
            coefficients["feature"]
            != "side_x_imbalance"
        )
        & coefficients["ci_excludes_zero"].fillna(False)
    ]

    r2_improved = r2_1s > baseline_r2_1s

    if (
        0.03 <= r2_1s <= 0.08
        and not side_significant
        and len(other_significant) == 0
    ):
        stopping_diagnostic = "NULL"

    elif (
        side_significant
        and len(other_significant) == 0
        and not r2_improved
    ):
        stopping_diagnostic = "SIDE_ONLY"

    elif (
        r2_improved
        and len(other_significant) >= 2
    ):
        stopping_diagnostic = "MODEST_PREDICTIVE"

    else:
        stopping_diagnostic = "NO_IMPROVEMENT"

    print(
        "\n===== M11 primary diagnostics ====="
    )

    print(
        f"1s OOS R2: {r2_1s:+.6f}"
    )

    print(
        "M9.5 reference range: 0.03–0.08"
    )

    print(
        "Side × imbalance CI excludes zero: "
        f"{side_significant}"
    )

    print(
        "Other significant coefficients: "
        f"{len(other_significant)}"
    )

    print(
        "Stopping diagnostic: "
        f"{stopping_diagnostic}"
    )

    summary[
        "stopping_diagnostic"
    ] = stopping_diagnostic

    summary[
        "side_interaction_significant"
    ] = side_significant

    summary[
        "other_significant_coefficients"
    ] = len(other_significant)

    # ------------------------------------------------------------
    # Save compact outputs.
    # ------------------------------------------------------------

    args.summary_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary.to_csv(
        args.summary_output,
        index=False,
    )

    coefficients.to_csv(
        args.coefficient_output,
        index=False,
    )

    print(
        f"\nSaved: {args.summary_output}"
    )

    print(
        f"Saved: {args.coefficient_output}"
    )


if __name__ == "__main__":
    main()