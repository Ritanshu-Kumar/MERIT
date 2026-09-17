from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler


INPUT = Path(
    "research/m9_fill_hazard_markouts_2019-07-30.csv"
)

OUTPUT = Path(
    "research/m9_final_common_split_2019-07-30.csv"
)

HORIZONS = {
    "10ms": "future_mid_10ms",
    "100ms": "future_mid_100ms",
    "500ms": "future_mid_500ms",
    "1s": "future_mid_1s",
}

FEATURES = [
    "imbalance",
    "relative_spread",
    "queue_log",
    "delta_imbalance",
    "delta_microprice",
]


def add_features(df):
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

    for col in [
        "imbalance",
        "relative_spread",
        "delta_imbalance",
        "delta_microprice",
        "queue_log",
    ]:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    return df


def split_by_symbol(df, time_col):
    train_parts = []
    test_parts = []
    boundaries = []

    for symbol, group in df.groupby(
        "symbol",
        sort=False,
    ):
        group = group.sort_values(
            time_col
        ).reset_index(drop=True)

        cut = int(len(group) * 0.70)

        train = group.iloc[:cut].copy()
        test = group.iloc[cut:].copy()

        train_parts.append(train)
        test_parts.append(test)

        boundaries.append(
            {
                "symbol": symbol,
                "n_total": len(group),
                "train_n": len(train),
                "test_n": len(test),
                "train_end": (
                    train.iloc[-1][time_col]
                    if len(train)
                    else np.nan
                ),
                "test_start": (
                    test.iloc[0][time_col]
                    if len(test)
                    else np.nan
                ),
            }
        )

    return (
        pd.concat(train_parts, ignore_index=True),
        pd.concat(test_parts, ignore_index=True),
        pd.DataFrame(boundaries),
    )


def signed_markout(df, future_col):
    quote = df["quote_price"].to_numpy(float)
    future = df[future_col].to_numpy(float)

    buy = (
        df["quote_side"]
        .eq("BUY")
        .to_numpy()
    )

    value = np.where(
        buy,
        future - quote,
        quote - future,
    )

    return value / quote * 10_000.0


def make_preprocessor():
    return ColumnTransformer(
        [
            (
                "num",
                StandardScaler(),
                FEATURES,
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


def bootstrap_gap(
    values,
    blocks,
    rng,
    reps=2000,
):
    blocks_unique = np.unique(blocks)

    grouped = [
        values[blocks == b]
        for b in blocks_unique
    ]

    block_means = np.array(
        [x.mean() for x in grouped]
    )

    weights = np.array(
        [len(x) for x in grouped],
        dtype=float,
    )

    weights /= weights.sum()

    observed = np.average(
        block_means,
        weights=weights,
    )

    boot = np.empty(reps)

    for i in range(reps):
        idx = rng.integers(
            0,
            len(grouped),
            size=len(grouped),
        )

        boot[i] = np.average(
            block_means[idx],
            weights=weights[idx],
        )

    lo, hi = np.quantile(
        boot,
        [0.025, 0.975],
    )

    return (
        float(observed),
        float(lo),
        float(hi),
    )


def main():
    print(
        "Loading M9 dataset..."
    )

    df = pd.read_csv(INPUT)
    df = add_features(df)

    # --------------------------------------------------------------
    # Freeze one common candidate-level chronological split.
    # --------------------------------------------------------------

    train_candidates, test_candidates, boundaries = (
        split_by_symbol(
            df,
            "timestamp_ns",
        )
    )

    print("\nFrozen common split:")
    print(
        boundaries.to_string(
            index=False
        )
    )

    # --------------------------------------------------------------
    # Fill model: trained ONLY on candidate TRAIN period.
    # --------------------------------------------------------------

    train_candidates["is_fill"] = (
        train_candidates["outcome"]
        .eq("FILL")
        .astype(int)
    )

    test_candidates["is_fill"] = (
        test_candidates["outcome"]
        .eq("FILL")
        .astype(int)
    )

    required = FEATURES + [
        "quote_side",
        "symbol",
    ]

    train_fill_model = (
        train_candidates
        .dropna(subset=required)
        .copy()
    )

    test_fill_model = (
        test_candidates
        .dropna(subset=required)
        .copy()
    )

    processor = make_preprocessor()

    x_train = processor.fit_transform(
        train_fill_model
    )

    x_test = processor.transform(
        test_fill_model
    )

    fill_model = LogisticRegression(
        max_iter=2000
    )

    fill_model.fit(
        x_train,
        train_fill_model["is_fill"],
    )

    test_fill_prob = (
        fill_model.predict_proba(
            x_test
        )[:, 1]
    )

    fill_auc = roc_auc_score(
        test_fill_model["is_fill"],
        test_fill_prob,
    )

    print(
        f"\nCommon-split fill AUC: "
        f"{fill_auc:.6f}"
    )

    # --------------------------------------------------------------
    # Attach fill probabilities to the test candidate universe.
    # --------------------------------------------------------------

    test_eval = test_fill_model.copy()
    test_eval["fill_prob"] = test_fill_prob

    # --------------------------------------------------------------
    # Markout models and joint EV.
    # --------------------------------------------------------------

    rng = np.random.default_rng(
        20260917
    )

    results = []

    for horizon, future_col in HORIZONS.items():

        print(
            f"\n===== {horizon} ====="
        )

        train_fills = (
            train_candidates[
                train_candidates["outcome"]
                .eq("FILL")
            ]
            .dropna(
                subset=[
                    future_col,
                    *FEATURES,
                    "quote_side",
                    "symbol",
                ]
            )
            .copy()
        )

        test_eval_h = test_eval.dropna(
            subset=[future_col]
        ).copy()

        # ----------------------------------------------------------
        # Markout model trained ONLY on training-period fills.
        # ----------------------------------------------------------

        mark_processor = make_preprocessor()

        x_train_mark = (
            mark_processor.fit_transform(
                train_fills
            )
        )

        y_train_mark = signed_markout(
            train_fills,
            future_col,
        )

        mark_model = Ridge(
            alpha=10.0
        )

        mark_model.fit(
            x_train_mark,
            y_train_mark,
        )

        x_test_mark = (
            mark_processor.transform(
                test_eval_h
            )
        )

        predicted_markout = (
            mark_model.predict(
                x_test_mark
            )
        )

        # ----------------------------------------------------------
        # Baseline = training-period side-specific markout.
        # ----------------------------------------------------------

        baseline = {}

        for side in ["BUY", "SELL"]:

            side_train = train_fills[
                train_fills["quote_side"]
                .eq(side)
            ]

            baseline[side] = float(
                signed_markout(
                    side_train,
                    future_col,
                ).mean()
            )

        baseline_markout = (
            test_eval_h["quote_side"]
            .map(baseline)
            .to_numpy(float)
        )

        # ----------------------------------------------------------
        # Joint expected value.
        # ----------------------------------------------------------

        baseline_ev = (
            test_eval_h["fill_prob"].to_numpy()
            * baseline_markout
        )

        state_ev = (
            test_eval_h["fill_prob"].to_numpy()
            * predicted_markout
        )

        gap = (
            state_ev
            - baseline_ev
        )

        # ----------------------------------------------------------
        # Actual candidate-level realized value.
        # ----------------------------------------------------------

        actual = np.zeros(
            len(test_eval_h)
        )

        filled = (
            test_eval_h["outcome"]
            .eq("FILL")
            .to_numpy()
        )

        if filled.any():
            actual[filled] = signed_markout(
                test_eval_h.loc[filled],
                future_col,
            )

        # ----------------------------------------------------------
        # Bootstrap the state-vs-baseline EV gap.
        # ----------------------------------------------------------

        blocks = (
            test_eval_h["symbol"].astype(str)
            + ":"
            + (
                test_eval_h[
                    "timestamp_ns"
                ]
                // 60_000_000_000
            ).astype(str)
        ).to_numpy()

        gap_mean, ci_lo, ci_hi = (
            bootstrap_gap(
                gap,
                blocks,
                rng,
            )
        )

        # ----------------------------------------------------------
        # Summary.
        # ----------------------------------------------------------

        baseline_mean = float(
            baseline_ev.mean()
        )

        state_mean = float(
            state_ev.mean()
        )

        actual_mean = float(
            actual.mean()
        )

        print(
            f"Baseline EV:      "
            f"{baseline_mean:+.6f} bps"
        )

        print(
            f"State-aware EV:   "
            f"{state_mean:+.6f} bps"
        )

        print(
            f"Gap:              "
            f"{gap_mean:+.6f} bps"
        )

        print(
            f"Gap 95% CI:       "
            f"[{ci_lo:+.6f}, "
            f"{ci_hi:+.6f}] bps"
        )

        print(
            f"Actual realized:  "
            f"{actual_mean:+.6f} bps"
        )

        results.append(
            {
                "horizon": horizon,
                "n_test": len(test_eval_h),
                "fill_auc": fill_auc,
                "baseline_ev_bps": baseline_mean,
                "state_aware_ev_bps": state_mean,
                "delta_ev_bps": gap_mean,
                "ci95_low_bps": ci_lo,
                "ci95_high_bps": ci_hi,
                "actual_realized_bps": actual_mean,
            }
        )

    results = pd.DataFrame(results)

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        OUTPUT,
        index=False,
    )

    print(
        "\n===== Final common-split result ====="
    )

    print(
        results.to_string(
            index=False
        )
    )

    print(
        f"\nSaved: {OUTPUT}"
    )


if __name__ == "__main__":
    main()