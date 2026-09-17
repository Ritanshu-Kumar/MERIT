from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


HORIZONS = {
    "10ms": "future_mid_10ms",
    "100ms": "future_mid_100ms",
    "500ms": "future_mid_500ms",
    "1s": "future_mid_1s",
}

FEATURES = [
    "imbalance",
    "delta_imbalance",
    "delta_microprice",
    "relative_spread",
    "queue_ahead_initial",
]


def signed_markout_bps(
    quote_side: pd.Series,
    quote_price: pd.Series,
    future_mid: pd.Series,
) -> np.ndarray:
    signed = np.where(
        quote_side.eq("BUY"),
        future_mid - quote_price,
        quote_price - future_mid,
    )
    return signed / quote_price.to_numpy() * 10_000.0


def bootstrap_block_mean(
    values: np.ndarray,
    block_ids: np.ndarray,
    rng: np.random.Generator,
    reps: int,
) -> tuple[float, float, float]:
    unique_blocks = np.unique(block_ids)
    grouped = [values[block_ids == block] for block in unique_blocks]
    means = np.array([group.mean() for group in grouped], dtype=float)
    weights = np.array([len(group) for group in grouped], dtype=float)
    weights /= weights.sum()

    observed = float(np.average(means, weights=weights))
    boot = np.empty(reps, dtype=float)

    for i in range(reps):
        picks = rng.choice(len(grouped), size=len(grouped), replace=True, p=None)
        boot[i] = float(np.average(means[picks], weights=weights[picks]))

    lo, hi = np.quantile(boot, [0.025, 0.975])
    return observed, float(lo), float(hi)


def qcut_summary(
    fills: pd.DataFrame,
    value: pd.Series,
    feature: str,
    horizon: str,
) -> pd.DataFrame:
    if fills[feature].nunique(dropna=True) < 5:
        return pd.DataFrame()

    bins = pd.qcut(fills[feature], 5, labels=False, duplicates="drop")
    tmp = fills[[feature]].copy()
    tmp["bin"] = bins.to_numpy()
    tmp["markout_bps"] = value.to_numpy()
    tmp = tmp.dropna(subset=["bin", "markout_bps"])

    rows = []
    for b, group in tmp.groupby("bin", sort=True):
        rows.append(
            {
                "horizon": horizon,
                "feature": feature,
                "bin": int(b),
                "n": len(group),
                "feature_mean": float(group[feature].mean()),
                "markout_mean_bps": float(group["markout_bps"].mean()),
                "markout_median_bps": float(group["markout_bps"].median()),
                "positive_fraction": float((group["markout_bps"] > 0).mean()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inferential analysis for M9.5 post-fill markouts."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("research/m9_fill_hazard_markouts_2019-07-30.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("research/m9_fill_markout_inference_2019-07-30.csv"),
    )
    parser.add_argument("--bootstrap-reps", type=int, default=2000)
    parser.add_argument("--block-ms", type=int, default=60_000)
    parser.add_argument("--seed", type=int, default=20260917)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    fills = df[df["outcome"].eq("FILL")].copy()
    if fills.empty:
        raise RuntimeError("No completed fills found.")

    if "fill_timestamp_ns" not in fills.columns:
        raise RuntimeError("Input does not contain fill_timestamp_ns.")

    fills["block_id"] = (
        fills.groupby("symbol")["fill_timestamp_ns"].transform("min") * 0
        + (fills["fill_timestamp_ns"] // (args.block_ms * 1_000_000)).astype("int64")
    )
    fills["symbol_block"] = fills["symbol"].astype(str) + ":" + fills["block_id"].astype(str)

    rng = np.random.default_rng(args.seed)
    rows: list[dict[str, object]] = []
    conditional: list[pd.DataFrame] = []

    for horizon, column in HORIZONS.items():
        valid = fills.dropna(subset=[column]).copy()
        markout = signed_markout_bps(
            valid["quote_side"], valid["quote_price"], valid[column]
        )

        observed, lo, hi = bootstrap_block_mean(
            markout,
            valid["symbol_block"].to_numpy(),
            rng,
            args.bootstrap_reps,
        )

        rows.append(
            {
                "horizon": horizon,
                "n": len(valid),
                "mean_bps": observed,
                "ci95_low_bps": lo,
                "ci95_high_bps": hi,
            }
        )

        for side in ("BUY", "SELL"):
            side_mask = valid["quote_side"].eq(side).to_numpy()
            side_valid = valid.iloc[np.flatnonzero(side_mask)].copy()
            side_markout = markout[side_mask]
            observed, lo, hi = bootstrap_block_mean(
                side_markout,
                side_valid["symbol_block"].to_numpy(),
                rng,
                args.bootstrap_reps,
            )
            rows.append(
                {
                    "horizon": f"{horizon}_{side}",
                    "n": len(side_valid),
                    "mean_bps": observed,
                    "ci95_low_bps": lo,
                    "ci95_high_bps": hi,
                }
            )

        for feature in FEATURES:
            conditional.append(qcut_summary(valid, pd.Series(markout), feature, horizon))

    summary = pd.DataFrame(rows)
    conditional_df = pd.concat([x for x in conditional if not x.empty], ignore_index=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output, index=False)
    conditional_path = args.output.with_name(args.output.stem + "_conditional.csv")
    conditional_df.to_csv(conditional_path, index=False)

    print("M9.5 block-bootstrap inference")
    print(summary.to_string(index=False))
    print(f"\nSaved: {args.output}")
    print(f"Saved: {conditional_path}")


if __name__ == "__main__":
    main()
