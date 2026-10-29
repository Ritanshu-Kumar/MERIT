from pathlib import Path

import pandas as pd

EQUITY_PATH = Path("research/m9_economic_2019-08-30_equity.csv")
FILLS_PATH = Path("research/m9_economic_2019-08-30_fills.csv")

STARTING_EQUITY = 1_000_000.0
OPEN_NS = 34_200_000_000_000
CLOSE_NS = 57_600_000_000_000


def main():
    equity = pd.read_csv(EQUITY_PATH)
    fills = pd.read_csv(FILLS_PATH)

    for col in ["timestamp_ns", "equity"]:
        equity[col] = pd.to_numeric(equity[col], errors="coerce")

    fills["timestamp_ns"] = pd.to_numeric(
        fills["timestamp_ns"], errors="coerce"
    )
    fills["quantity"] = pd.to_numeric(
        fills["quantity"], errors="coerce"
    )

    equity = equity.dropna(
        subset=["timestamp_ns", "equity", "strategy"]
    ).sort_values(["strategy", "timestamp_ns"])

    fills = fills.dropna(
        subset=["timestamp_ns", "quantity", "strategy"]
    ).copy()

    def session(ts):
        if ts < OPEN_NS:
            return "pre-open"
        if ts < CLOSE_NS:
            return "regular"
        return "after-close"

    equity["session"] = equity["timestamp_ns"].map(session)
    fills["session"] = fills["timestamp_ns"].map(session)

    print("=" * 68)
    print("AUGUST 30 ECONOMIC REPLAY AUDIT")
    print("P&L is pre-fee mark-to-market, not net trading profit.")
    print("=" * 68)

    for strategy, group in equity.groupby("strategy"):
        group = group.sort_values("timestamp_ns").copy()
        values = group["equity"]
        timestamps = group["timestamp_ns"]

        # Include initial capital when calculating the all-day drawdown.
        running_peak = values.cummax().clip(lower=STARTING_EQUITY)
        drawdown = values - running_peak

        pre = group[group["timestamp_ns"] < OPEN_NS]
        regular = group[
            group["timestamp_ns"].ge(OPEN_NS)
            & group["timestamp_ns"].lt(CLOSE_NS)
        ]
        after = group[group["timestamp_ns"].ge(CLOSE_NS)]

        pre_close = pre["equity"].iloc[-1] if not pre.empty else None
        regular_close = (
            regular["equity"].iloc[-1] if not regular.empty else None
        )
        after_close = (
            after["equity"].iloc[-1] if not after.empty else None
        )

        print(f"\n--- {strategy} ---")
        print("Final equity:", round(float(values.iloc[-1]), 2))
        print(
            "All-day P&L:",
            round(float(values.iloc[-1] - STARTING_EQUITY), 2),
        )
        print("Maximum drawdown ($):", round(float(drawdown.min()), 2))
        print("Equity observations:", len(group))

        if pre_close is not None:
            print(
                "Pre-open P&L:",
                round(float(pre_close - STARTING_EQUITY), 2),
            )

        if pre_close is not None and regular_close is not None:
            print(
                "Regular-session equity change:",
                round(float(regular_close - pre_close), 2),
            )

        if regular_close is not None and after_close is not None:
            print(
                "After-hours equity change:",
                round(float(after_close - regular_close), 2),
            )

    print("\nFILL COUNTS BY SESSION AND STRATEGY")
    print(
        pd.crosstab(fills["session"], fills["strategy"]).to_string()
    )

    print("\nTRADED QUANTITY BY SESSION AND STRATEGY")
    print(
        fills.pivot_table(
            index="session",
            columns="strategy",
            values="quantity",
            aggfunc="sum",
            fill_value=0,
        ).to_string()
    )

    print("\nFILL COUNTS BY SYMBOL AND STRATEGY")
    print(
        pd.crosstab(
            fills["symbol"],
            fills["strategy"],
        ).to_string()
    )


if __name__ == "__main__":
    main()