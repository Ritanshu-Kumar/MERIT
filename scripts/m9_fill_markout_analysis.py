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
    signed = np.where(
        quote_side.eq("BUY"),
        future_mid - quote_price,
        quote_price - future_mid,
    )
    return signed / quote_price * 10_000.0


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

    df = pd.read_csv(args.input)
    fills = df[df["outcome"].eq("FILL")].copy()

    if fills.empty:
        raise RuntimeError("No completed fills found.")

    rows: list[dict[str, object]] = []

    for horizon, column in HORIZONS.items():
        valid = fills[fills[column].notna()].copy()
        if valid.empty:
            continue

        markout = signed_markout_bps(
            valid["quote_side"],
            valid["quote_price"],
            valid[column],
        )

        rows.append(
            {
                "horizon": horizon,
                "n": len(valid),
                "mean_bps": float(markout.mean()),
                "median_bps": float(markout.median()),
                "std_bps": float(markout.std(ddof=1)),
                "positive_fraction": float((markout > 0).mean()),
            }
        )

        for side in ("BUY", "SELL"):
            side_mask = valid["quote_side"].eq(side)
            side_markout = markout[side_mask]
            rows.append(
                {
                    "horizon": f"{horizon}_{side}",
                    "n": len(side_markout),
                    "mean_bps": float(side_markout.mean()),
                    "median_bps": float(side_markout.median()),
                    "std_bps": float(side_markout.std(ddof=1)),
                    "positive_fraction": float((side_markout > 0).mean()),
                }
            )

    summary = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output, index=False)

    print("Fill markout summary")
    print(summary.to_string(index=False))
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
