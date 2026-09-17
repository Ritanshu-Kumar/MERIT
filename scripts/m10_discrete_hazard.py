from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scipy.special import expit
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler


HAZARD_FILE = Path(
    "research/m9_fill_hazard_2019-07-30.csv"
)

STATE_FILE = Path(
    "research/m9_state_transitions_2019-07-30.csv"
)

OUTPUT = Path(
    "research/m10_discrete_hazard_2019-07-30.csv"
)

BOOTSTRAP_OUTPUT = Path(
    "research/m10_discrete_hazard_model_comparison_2019-07-30.csv"
)

CAL_OUTPUT = Path(
    "research/m10_discrete_hazard_calibration_2019-07-30.csv"
)

INTERVALS = [
    ("0_10ms", 0, 10_000_000),
    ("10_100ms", 10_000_000, 100_000_000),
    ("100_500ms", 100_000_000, 500_000_000),
    ("500_1000ms", 500_000_000, 1_000_000_000),
]

BASE_FEATURES = [
    "imbalance",
    "relative_spread",
    "queue_log",
]

FLOW_FEATURES = [
    "events_10ms",
    "executions_10ms",
    "buy_flow_10ms",
    "sell_flow_10ms",
]


def load_and_link() -> pd.DataFrame:
    print("Loading M9 hazard dataset...")
    hazard = pd.read_csv(HAZARD_FILE)

    print(
        f"Hazard candidates: {len(hazard):,}"
    )

    hazard = hazard.reset_index(drop=True)
    hazard["candidate_id"] = np.arange(
        len(hazard),
        dtype=np.int64,
    )

    print("Loading M9 state-transition dataset...")
    state = pd.read_csv(STATE_FILE)

    print(
        f"State observations: {len(state):,}"
    )

    # A single execution observation generates two hypothetical
    # quote candidates. Therefore execution_side is NOT part of
    # the state-observation join key.

    join_cols = [
        "symbol",
        "timestamp_ns",
        "mid",
        "spread",
        "relative_spread",
        "imbalance",
        "delta_imbalance",
        "delta_microprice",
        "delta_spread",
    ]

    flow_cols = [
        "events_10ms",
        "executions_10ms",
        "buy_flow_10ms",
        "sell_flow_10ms",
        "bid_size",
        "ask_size",
    ]

    # Normalize numeric precision before matching.
    float_cols = [
        "mid",
        "spread",
        "relative_spread",
        "imbalance",
        "delta_imbalance",
        "delta_microprice",
        "delta_spread",
    ]

    for col in float_cols:
        hazard[col] = pd.to_numeric(
            hazard[col],
            errors="coerce",
        ).round(12)

        state[col] = pd.to_numeric(
            state[col],
            errors="coerce",
        ).round(12)

    state_subset = state[
        join_cols + flow_cols
    ].copy()

    key_counts = (
        state_subset
        .groupby(
            join_cols,
            dropna=False,
        )
        .size()
    )

    duplicate_keys = int(
        (key_counts > 1).sum()
    )

    print(
        f"Duplicated corrected state keys: "
        f"{duplicate_keys:,}"
    )

    if duplicate_keys:
        raise RuntimeError(
            "Corrected state key is not unique."
        )

    merged = hazard.merge(
        state_subset,
        on=join_cols,
        how="left",
        indicator=True,
        validate="many_to_one",
        suffixes=("", "_state"),
    )

    unmatched = int(
        (merged["_merge"] != "both").sum()
    )

    print(
        f"Unmatched hazard candidates: "
        f"{unmatched:,}"
    )

    if unmatched:
        print(
            merged.loc[
                merged["_merge"] != "both",
                [
                    "candidate_id",
                    "symbol",
                    "timestamp_ns",
                    "quote_side",
                ],
            ].head(20).to_string(index=False)
        )

        raise RuntimeError(
            "Some hazard candidates could not be "
            "linked to state-transition observations."
        )

    merged = merged.drop(
        columns=["_merge"]
    )

    return merged


def add_features(
    df: pd.DataFrame,
) -> pd.DataFrame:

    df = df.copy()

    df["queue_ahead_initial"] = pd.to_numeric(
        df["queue_ahead_initial"],
        errors="coerce",
    )

    df["queue_log"] = np.log1p(
        np.maximum(
            df["queue_ahead_initial"],
            0.0,
        )
    )

    df["is_fill"] = (
        df["outcome"]
        .eq("FILL")
        .astype(int)
    )

    df["time_to_resolution_ns"] = pd.to_numeric(
        df["time_to_resolution_ns"],
        errors="coerce",
    )

    return df


def common_split(
    df: pd.DataFrame,
):
    train_parts = []
    test_parts = []
    boundaries = []

    for symbol, group in df.groupby(
        "symbol",
        sort=False,
    ):
        group = group.sort_values(
            "timestamp_ns"
        ).reset_index(drop=True)

        cut = int(
            len(group) * 0.70
        )

        train = group.iloc[:cut].copy()
        test = group.iloc[cut:].copy()

        train_parts.append(train)
        test_parts.append(test)

        boundaries.append(
            {
                "symbol": symbol,
                "train_n": len(train),
                "test_n": len(test),
                "train_end": int(
                    train.iloc[-1]["timestamp_ns"]
                ),
                "test_start": int(
                    test.iloc[0]["timestamp_ns"]
                ),
            }
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
        pd.DataFrame(boundaries),
    )


def split_training_for_calibration(
    train: pd.DataFrame,
):
    fit_parts = []
    cal_parts = []

    for _, group in train.groupby(
        "symbol",
        sort=False,
    ):
        group = group.sort_values(
            "timestamp_ns"
        ).reset_index(drop=True)

        cut = int(
            len(group) * 0.85
        )

        fit_parts.append(
            group.iloc[:cut]
        )

        cal_parts.append(
            group.iloc[cut:]
        )

    return (
        pd.concat(
            fit_parts,
            ignore_index=True,
        ),
        pd.concat(
            cal_parts,
            ignore_index=True,
        ),
    )


def person_period(
    df: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    resolution_time = (
        df["time_to_resolution_ns"]
        .to_numpy(dtype=np.int64)
    )

    for interval_name, lower, upper in INTERVALS:

        if lower == 0:
            risk_mask = resolution_time >= 0
        else:
            risk_mask = resolution_time > lower

        subset = df.loc[
            risk_mask
        ].copy()

        if subset.empty:
            continue

        t = (
            subset[
                "time_to_resolution_ns"
            ]
            .to_numpy(dtype=np.int64)
        )

        fills = (
            subset["outcome"]
            .eq("FILL")
            .to_numpy()
        )

        event = (
            fills
            & (t <= upper)
        )

        subset["interval"] = interval_name
        subset["fill_hazard_event"] = (
            event.astype(int)
        )

        cols = [
            "candidate_id",
            "symbol",
            "quote_side",
            "timestamp_ns",
            "relative_spread",
            "imbalance",
            "queue_log",
            "events_10ms",
            "executions_10ms",
            "buy_flow_10ms",
            "sell_flow_10ms",
            "interval",
            "fill_hazard_event",
        ]

        rows.append(
            subset[cols]
        )

    if not rows:
        raise RuntimeError(
            "No person-period rows were generated."
        )

    return pd.concat(
        rows,
        ignore_index=True,
    )


def make_preprocessor(
    numeric_features: list[str],
):
    return ColumnTransformer(
        [
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
                    "interval",
                ],
            ),
        ]
    )


def fit_hazard(
    train_pp: pd.DataFrame,
    features: list[str],
):
    preprocessor = make_preprocessor(
        features
    )

    x_train = preprocessor.fit_transform(
        train_pp
    )

    model = LogisticRegression(
        max_iter=2000
    )

    model.fit(
        x_train,
        train_pp["fill_hazard_event"],
    )

    return (
        preprocessor,
        model,
    )


def predict_hazard(
    preprocessor,
    model,
    pp: pd.DataFrame,
):
    x = preprocessor.transform(
        pp
    )

    return model.predict_proba(
        x
    )[:, 1]


def fit_calibrators(
    calibration_pp: pd.DataFrame,
    raw_probability: np.ndarray,
):
    temp = calibration_pp.copy()
    temp["raw_p"] = np.clip(
        raw_probability,
        1e-8,
        1.0 - 1e-8,
    )

    calibrators = {}

    for interval_name, _, _ in INTERVALS:

        sub = temp[
            temp["interval"]
            .eq(interval_name)
        ]

        y = sub[
            "fill_hazard_event"
        ].to_numpy()

        p = sub[
            "raw_p"
        ].to_numpy()

        if (
            len(sub) < 100
            or np.unique(y).size < 2
        ):
            calibrators[
                interval_name
            ] = (0.0, 1.0)
            continue

        z = np.log(
            p / (1.0 - p)
        ).reshape(-1, 1)

        model = LogisticRegression(
            C=1e6,
            max_iter=1000,
        )

        model.fit(
            z,
            y,
        )

        calibrators[
            interval_name
        ] = (
            float(model.intercept_[0]),
            float(model.coef_[0, 0]),
        )

    return calibrators


def apply_calibrators(
    pp: pd.DataFrame,
    raw_probability: np.ndarray,
    calibrators,
):
    out = np.empty(
        len(pp),
        dtype=float,
    )

    for interval_name, _, _ in INTERVALS:

        mask = (
            pp["interval"]
            .eq(interval_name)
            .to_numpy()
        )

        p = np.clip(
            raw_probability[mask],
            1e-8,
            1.0 - 1e-8,
        )

        intercept, slope = calibrators[
            interval_name
        ]

        logits = np.log(
            p / (1.0 - p)
        )

        out[mask] = expit(
            intercept
            + slope * logits
        )

    return out


def cumulative_fill_probability(
    pp: pd.DataFrame,
    hazard: np.ndarray,
):
    temp = pp[
        [
            "candidate_id",
            "interval",
        ]
    ].copy()

    temp["hazard"] = hazard

    interval_order = {
        name: i
        for i, (name, _, _) in enumerate(
            INTERVALS
        )
    }

    temp["interval_order"] = (
        temp["interval"].map(
            interval_order
        )
    )

    temp = temp.sort_values(
        [
            "candidate_id",
            "interval_order",
        ]
    )

    rows = []

    for horizon_index, (
        name,
        _,
        _,
    ) in enumerate(INTERVALS):

        sub = temp[
            temp["interval_order"]
            <= horizon_index
        ]

        survival = (
            1.0 - sub["hazard"]
        )

        survival_product = (
            survival
            .groupby(
                sub["candidate_id"]
            )
            .prod()
        )

        cumulative_fill = (
            1.0 - survival_product
        )

        cumulative_fill.name = (
            f"cum_fill_{name}"
        )

        rows.append(
            cumulative_fill
        )

    result = pd.concat(
        rows,
        axis=1,
    )

    result["candidate_id"] = (
        result.index
    )

    return result.reset_index(
        drop=True
    )


def observed_cumulative(
    candidates: pd.DataFrame,
):
    out = candidates[
        ["candidate_id"]
    ].copy()

    t = (
        candidates[
            "time_to_resolution_ns"
        ]
        .to_numpy(dtype=np.int64)
    )

    fill = (
        candidates[
            "outcome"
        ]
        .eq("FILL")
        .to_numpy()
    )

    for name, _, upper in INTERVALS:
        out[
            f"obs_fill_{name}"
        ] = (
            fill
            & (t <= upper)
        ).astype(int)

    return out


def evaluate_candidate_level(
    candidates: pd.DataFrame,
    cumulative: pd.DataFrame,
):
    observed = observed_cumulative(
        candidates
    )

    merged = observed.merge(
        cumulative,
        on="candidate_id",
        how="left",
    )

    rows = []

    for name, _, _ in INTERVALS:

        p = merged[
            f"cum_fill_{name}"
        ].to_numpy()

        y = merged[
            f"obs_fill_{name}"
        ].to_numpy()

        rows.append(
            {
                "horizon": name,
                "auc": roc_auc_score(
                    y,
                    p,
                ),
                "brier": brier_score_loss(
                    y,
                    p,
                ),
                "mean_predicted": float(
                    p.mean()
                ),
                "observed_rate": float(
                    y.mean()
                ),
            }
        )

    return (
        pd.DataFrame(rows),
        merged,
    )


def block_bootstrap_difference(
    merged_model1: pd.DataFrame,
    merged_model2: pd.DataFrame,
    candidates: pd.DataFrame,
    metric: str,
    reps: int = 500,
    seed: int = 20260917,
):
    rng = np.random.default_rng(seed)

    # Align all candidate-level arrays once.
    meta = candidates[
        ["candidate_id", "symbol", "timestamp_ns"]
    ].copy()

    meta["block"] = (
        meta["symbol"].astype(str)
        + ":"
        + (
            meta["timestamp_ns"] // 60_000_000_000
        ).astype(str)
    )

    # Candidate ordering used by the merged prediction tables.
    m1 = merged_model1.set_index("candidate_id")
    m2 = merged_model2.set_index("candidate_id")

    meta = meta[
        meta["candidate_id"].isin(m1.index)
        & meta["candidate_id"].isin(m2.index)
    ].copy()

    meta = meta.set_index("candidate_id")

    blocks = meta["block"].to_numpy()
    unique_blocks, block_inverse = np.unique(
        blocks,
        return_inverse=True,
    )

    block_indices = [
        np.flatnonzero(
            block_inverse == i
        )
        for i in range(len(unique_blocks))
    ]

    results = []

    for horizon, _, _ in INTERVALS:

        y = m1.loc[
            meta.index,
            f"obs_fill_{horizon}",
        ].to_numpy()

        p1 = m1.loc[
            meta.index,
            f"cum_fill_{horizon}",
        ].to_numpy()

        p2 = m2.loc[
            meta.index,
            f"cum_fill_{horizon}",
        ].to_numpy()

        block_stats = []

        # Precompute per-block Brier contributions.
        if metric == "brier":
            for idx in block_indices:
                yy = y[idx]
                d1 = np.mean(
                    (yy - p1[idx]) ** 2
                )
                d2 = np.mean(
                    (yy - p2[idx]) ** 2
                )
                block_stats.append(
                    d2 - d1
                )

            block_stats = np.asarray(
                block_stats,
                dtype=float,
            )

            observed = float(
                block_stats.mean()
            )

            # Fast bootstrap over block statistics.
            draws = rng.integers(
                0,
                len(block_stats),
                size=(
                    reps,
                    len(block_stats),
                ),
            )

            boot = block_stats[
                draws
            ].mean(axis=1)

        elif metric == "auc":
            # AUC is not additive across blocks, so we must
            # reconstruct each bootstrap sample. To keep this
            # tractable, use 200 replications and vectorized
            # block-index construction.
            reps = min(reps, 200)

            valid_blocks = []

            for idx in block_indices:
                yy = y[idx]

                if np.unique(yy).size >= 2:
                    valid_blocks.append(idx)

            if not valid_blocks:
                results.append(
                    {
                        "horizon": horizon,
                        "metric": metric,
                        "model2_minus_model1": np.nan,
                        "ci95_low": np.nan,
                        "ci95_high": np.nan,
                        "n_blocks": 0,
                    }
                )
                continue

            # Observed AUC on the full test set.
            if np.unique(y).size >= 2:
                observed = float(
                    roc_auc_score(y, p2)
                    - roc_auc_score(y, p1)
                )
            else:
                observed = np.nan

            boot = []

            for _ in range(reps):

                selected = rng.integers(
                    0,
                    len(valid_blocks),
                    size=len(valid_blocks),
                )

                sampled_parts = [
                    valid_blocks[i]
                    for i in selected
                ]

                idx = np.concatenate(
                    sampled_parts
                )

                yy = y[idx]

                if np.unique(yy).size < 2:
                    continue

                auc1 = roc_auc_score(
                    yy,
                    p1[idx],
                )

                auc2 = roc_auc_score(
                    yy,
                    p2[idx],
                )

                boot.append(
                    auc2 - auc1
                )

            boot = np.asarray(
                boot,
                dtype=float,
            )

        else:
            raise ValueError(
                f"Unknown metric: {metric}"
            )

        if len(boot) == 0:
            lo = np.nan
            hi = np.nan
        else:
            lo, hi = np.quantile(
                boot,
                [0.025, 0.975],
            )

        results.append(
            {
                "horizon": horizon,
                "metric": metric,
                "model2_minus_model1": observed,
                "ci95_low": float(lo),
                "ci95_high": float(hi),
                "n_blocks": len(unique_blocks),
            }
        )

    return pd.DataFrame(results)


def calibration_table(
    merged: pd.DataFrame,
    model_name: str,
):
    rows = []

    for horizon, _, _ in INTERVALS:

        p = merged[
            f"cum_fill_{horizon}"
        ].to_numpy()

        y = merged[
            f"obs_fill_{horizon}"
        ].to_numpy()

        temp = pd.DataFrame(
            {
                "p": p,
                "y": y,
            }
        )

        temp["bin"] = pd.qcut(
            temp["p"],
            10,
            labels=False,
            duplicates="drop",
        )

        for b, group in temp.groupby(
            "bin",
            sort=True,
        ):
            rows.append(
                {
                    "model": model_name,
                    "horizon": horizon,
                    "bin": int(b),
                    "n": len(group),
                    "predicted": float(
                        group["p"].mean()
                    ),
                    "observed": float(
                        group["y"].mean()
                    ),
                    "gap": float(
                        group["y"].mean()
                        - group["p"].mean()
                    ),
                }
            )

    return pd.DataFrame(rows)


def main():

    df = add_features(
        load_and_link()
    )

    print(
        "\nCreating common 70/30 split..."
    )

    train, test, boundaries = (
        common_split(df)
    )

    print(
        boundaries.to_string(
            index=False
        )
    )

    fit_candidates, calibration_candidates = (
        split_training_for_calibration(
            train
        )
    )

    print(
        f"\nModel-fit candidates: "
        f"{len(fit_candidates):,}"
    )
    print(
        f"Calibration candidates: "
        f"{len(calibration_candidates):,}"
    )
    print(
        f"Test candidates: "
        f"{len(test):,}"
    )

    print(
        "\nExpanding person-period data..."
    )

    fit_pp = person_period(
        fit_candidates
    )

    calibration_pp = person_period(
        calibration_candidates
    )

    test_pp = person_period(
        test
    )

    print(
        f"Model-fit rows: "
        f"{len(fit_pp):,}"
    )
    print(
        f"Calibration rows: "
        f"{len(calibration_pp):,}"
    )
    print(
        f"Test rows: "
        f"{len(test_pp):,}"
    )

    model_specs = {
        "state_queue": BASE_FEATURES,
        "state_queue_flow": (
            BASE_FEATURES
            + FLOW_FEATURES
        ),
    }

    all_summary = []
    all_calibration = {}
    merged_results = {}

    for model_name, features in model_specs.items():

        print(
            f"\n===== {model_name} ====="
        )

        preprocessor, model = fit_hazard(
            fit_pp,
            features,
        )

        p_fit = predict_hazard(
            preprocessor,
            model,
            fit_pp,
        )

        p_cal = predict_hazard(
            preprocessor,
            model,
            calibration_pp,
        )

        p_test = predict_hazard(
            preprocessor,
            model,
            test_pp,
        )

        calibrators = fit_calibrators(
            calibration_pp,
            p_cal,
        )

        p_test_cal = apply_calibrators(
            test_pp,
            p_test,
            calibrators,
        )

        raw_cumulative = cumulative_fill_probability(
            test_pp,
            p_test,
        )

        calibrated_cumulative = cumulative_fill_probability(
            test_pp,
            p_test_cal,
        )

        raw_metrics, raw_merged = (
            evaluate_candidate_level(
                test,
                raw_cumulative,
            )
        )

        calibrated_metrics, calibrated_merged = (
            evaluate_candidate_level(
                test,
                calibrated_cumulative,
            )
        )

        raw_metrics["model"] = (
            model_name + "_raw"
        )

        calibrated_metrics["model"] = (
            model_name + "_calibrated"
        )

        all_summary.extend(
            raw_metrics.to_dict(
                orient="records"
            )
        )

        all_summary.extend(
            calibrated_metrics.to_dict(
                orient="records"
            )
        )

        merged_results[
            model_name
        ] = calibrated_merged

        cal_table = calibration_table(
            calibrated_merged,
            model_name,
        )

        all_calibration[
            model_name
        ] = cal_table

        print(
            "\nRaw cumulative fill:"
        )
        print(
            raw_metrics.to_string(
                index=False
            )
        )

        print(
            "\nCalibrated cumulative fill:"
        )
        print(
            calibrated_metrics.to_string(
                index=False
            )
        )

    print(
        "\n===== Model 2 vs Model 1 bootstrap ====="
    )

    brier_diff = block_bootstrap_difference(
        merged_results["state_queue"],
        merged_results[
            "state_queue_flow"
        ],
        test,
        "brier",
    )

    auc_diff = block_bootstrap_difference(
        merged_results["state_queue"],
        merged_results[
            "state_queue_flow"
        ],
        test,
        "auc",
    )

    bootstrap = pd.concat(
        [
            brier_diff,
            auc_diff,
        ],
        ignore_index=True,
    )

    print(
        bootstrap.to_string(
            index=False
        )
    )

    summary = pd.DataFrame(
        all_summary
    )

    calibration = pd.concat(
        all_calibration.values(),
        ignore_index=True,
    )

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary.to_csv(
        OUTPUT,
        index=False,
    )

    bootstrap.to_csv(
        BOOTSTRAP_OUTPUT,
        index=False,
    )

    calibration.to_csv(
        CAL_OUTPUT,
        index=False,
    )

    print(
        f"\nSaved: {OUTPUT}"
    )
    print(
        f"Saved: {BOOTSTRAP_OUTPUT}"
    )
    print(
        f"Saved: {CAL_OUTPUT}"
    )


if __name__ == "__main__":
    main()