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


def signed_markout_bps(
    quote_side: pd.Series,
    quote_price: pd.Series,
    future_mid: pd.Series,
) -> pd.Series:
    """Calculate favorable-positive signed markout in basis points."""
    side = quote_side.astype(str).str.upper()
    quote = pd.to_numeric(quote_price, errors="coerce")
    future = pd.to_numeric(future_mid, errors="coerce")

    valid = (
        side.isin(["BUY", "SELL"])
        & quote.gt(0)
        & np.isfinite(quote)
        & np.isfinite(future)
    )

    result = pd.Series(np.nan, index=quote.index, dtype=float)

    buy = side.eq("BUY")
    sell = side.eq("SELL")

    result.loc[valid & buy] = (
        (future.loc[valid & buy] - quote.loc[valid & buy])
        / quote.loc[valid & buy]
        * 10_000.0
    )

    result.loc[valid & sell] = (
        (quote.loc[valid & sell] - future.loc[valid & sell])
        / quote.loc[valid & sell]
        * 10_000.0
    )

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze post-fill signed midpoint markouts."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "research/m9_fill_hazard_markouts_2019-07-30.csv"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "research/m9_fill_markout_summary_2019-07-30.csv"
        ),
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    df = pd.read_csv(args.input)

    required = {"outcome", "quote_side", "quote_price"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    fills = df.loc[df["outcome"].eq("FILL")].copy()

    if fills.empty:
        raise RuntimeError("No completed fills found.")

    invalid_sides = ~fills["quote_side"].astype(str).str.upper().isin(
        ["BUY", "SELL"]
    )
    if invalid_sides.any():
        raise ValueError(
            f"{int(invalid_sides.sum())} fill rows have invalid quote_side values."
        )

    rows: list[dict[str, object]] = []

    for horizon, column in HORIZONS.items():
        if column not in fills.columns:
            print(f"Skipping {horizon}: missing column {column}")
            continue

        valid = fills.loc[
            fills[column].notna()
            & fills["quote_price"].notna()
            & fills["quote_price"].gt(0)
        ].copy()

        if valid.empty:
            continue

        markout = signed_markout_bps(
            valid["quote_side"],
            valid["quote_price"],
            valid[column],
        )

        valid = valid.assign(markout_bps=markout).dropna(
            subset=["markout_bps"]
        )

        def add_summary(label: str, subset: pd.DataFrame) -> None:
            values = subset["markout_bps"]
            rows.append(
                {
                    "horizon": label,
                    "n": len(values),
                    "mean_bps": float(values.mean()) if len(values) else np.nan,
                    "median_bps": float(values.median()) if len(values) else np.nan,
                    "std_bps": (
                        float(values.std(ddof=1))
                        if len(values) > 1
                        else np.nan
                    ),
                    "positive_fraction": (
                        float(values.gt(0).mean())
                        if len(values)
                        else np.nan
                    ),
                }
            )

        add_summary(horizon, valid)

        for side in ("BUY", "SELL"):
            add_summary(
                f"{horizon}_{side}",
                valid.loc[valid["quote_side"].eq(side)],
            )

    if not rows:
        raise RuntimeError("No valid markouts could be calculated.")

    summary = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output, index=False)

    print("Fill markout summary — favorable-positive convention")
    print(summary.to_string(index=False))
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()