from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from pathlib import Path

import pandas as pd

from merit.data.lobster import (
    OrderExecuteEvent,
    read_messages,
)
from merit.data.lobster_orderbook import read_orderbooks
from merit.execution.queue_model import QueueModel
from merit.features.snapshot import (
    build_feature_snapshot_from_levels,
)
from merit.portfolio.portfolio import Portfolio
from merit.research.lobster_execution import (
    LobsterExecutionEngine,
)
from merit.research.m8_policy import M8Policy
from merit.risk.limits import RiskLimits, RiskManager
from merit.strategy.baseline import QuoteDecision
from merit.strategy.m8 import M8MarketMaker


BASE = Path("data/sample")
DATASET = BASE / "M8_replication_research_dataset.csv"

SYMBOLS = [
    "AAPL",
    "AMZN",
    "GOOG",
    "INTC",
    "MSFT",
]

TRADING_DATE = date(2012, 6, 21)

LEVELS = 10
ORDER_QUANTITY = 100
MAX_POSITION = 500

STARTING_CASH = Decimal("1000000")
QUEUE_AHEAD_FRACTION = 0.0
FEE_BPS = Decimal("0")

TRAIN_FRACTION = 0.60
VALIDATION_FRACTION = 0.20
PURGE_SECONDS = 5
EMBARGO_SECONDS = 5


@dataclass(frozen=True)
class ReplayFill:
    symbol: str
    event_index: int
    canonical_fill_id: str
    side: str
    quantity: int
    price: Decimal


def message_path(symbol: str) -> Path:
    return BASE / (
        f"{symbol}_2012-06-21_34200000_57600000"
        "_message_10.csv"
    )


def orderbook_path(symbol: str) -> Path:
    return BASE / (
        f"{symbol}_2012-06-21_34200000_57600000"
        "_orderbook_10.csv"
    )


def load_symbol(
    symbol: str,
):
    events = tuple(
        read_messages(
            message_path(symbol),
            symbol,
            TRADING_DATE,
        )
    )

    snapshots = tuple(
        read_orderbooks(
            orderbook_path(symbol),
            levels=LEVELS,
        )
    )

    aligned_count = min(
        len(events),
        len(snapshots),
    )

    return (
        events[:aligned_count],
        snapshots[:aligned_count],
    )


def load_frozen_test_start() -> tuple[
    pd.Timestamp,
    pd.Timestamp,
    int,
    int,
    int,
]:
    df = pd.read_csv(DATASET)

    required = {
        "timestamp",
        "fill_id",
        "symbol",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Dataset is missing required columns: "
            f"{sorted(missing)}"
        )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
    )

    df = (
        df.dropna(subset=["timestamp"])
        .sort_values(
            "timestamp",
            kind="stable",
        )
        .reset_index(drop=True)
    )

    n = len(df)

    train_end = int(
        n * TRAIN_FRACTION
    )

    validation_end = int(
        n * (
            TRAIN_FRACTION
            + VALIDATION_FRACTION
        )
    )

    train = df.iloc[:train_end].copy()
    validation = df.iloc[
        train_end:validation_end
    ].copy()
    test = df.iloc[
        validation_end:
    ].copy()

    if train.empty:
        raise ValueError(
            "Frozen train split is empty."
        )

    if validation.empty:
        raise ValueError(
            "Frozen validation split is empty."
        )

    if test.empty:
        raise ValueError(
            "Frozen test split is empty."
        )

    train_boundary = train[
        "timestamp"
    ].iloc[-1]

    validation_boundary = validation[
        "timestamp"
    ].iloc[-1]

    purge = pd.Timedelta(
        seconds=PURGE_SECONDS
    )

    embargo = pd.Timedelta(
        seconds=EMBARGO_SECONDS
    )

    validation = validation[
        validation["timestamp"]
        >= train_boundary + purge
    ].copy()

    test = test[
        test["timestamp"]
        >= validation_boundary + embargo
    ].copy()

    if validation.empty:
        raise ValueError(
            "Validation split became empty "
            "after purge."
        )

    if test.empty:
        raise ValueError(
            "Test split became empty "
            "after embargo."
        )

    return (
        test["timestamp"].iloc[0],
        validation["timestamp"].iloc[-1],
        len(train),
        len(validation),
        len(test),
    )


def make_engine(
    symbol: str,
    portfolio: Portfolio,
) -> LobsterExecutionEngine:
    return LobsterExecutionEngine(
        symbol=symbol,
        portfolio=portfolio,
        risk_manager=RiskManager(
            RiskLimits(
                max_order_quantity=ORDER_QUANTITY,
                max_position=MAX_POSITION,
            )
        ),
        queue_model=QueueModel(
            ahead_fraction=QUEUE_AHEAD_FRACTION,
        ),
        quantity=ORDER_QUANTITY,
        fee_bps=FEE_BPS,
    )


def build_canonical_test_rows(
    test_start: pd.Timestamp,
    symbol_data,
) -> dict[str, dict]:
    canonical = {}

    for symbol in SYMBOLS:
        events, snapshots = symbol_data[
            symbol
        ]

        for index in range(
            1,
            len(events),
        ):
            event = events[index]

            if event.timestamp < test_start:
                continue

            if not isinstance(
                event,
                OrderExecuteEvent,
            ):
                continue

            if event.execution_price is None:
                continue

            previous_bids, previous_asks = (
                snapshots[index - 1]
            )

            if (
                not previous_bids
                or not previous_asks
            ):
                continue

            if (
                event.execution_price
                == previous_bids[0][0]
            ):
                side = "BUY"
            elif (
                event.execution_price
                == previous_asks[0][0]
            ):
                side = "SELL"
            else:
                continue

            fill_id = (
                f"{event.order_id}-{index}"
            )

            canonical[fill_id] = {
                "symbol": symbol,
                "event_index": index,
                "timestamp": event.timestamp,
                "side": side,
                "execution_quantity": event.quantity,
                "price": event.execution_price,
            }

    return canonical


def run_replay(
    name: str,
    test_start: pd.Timestamp,
    symbol_data,
):
    portfolio = Portfolio(
        STARTING_CASH
    )

    engines = {
        symbol: make_engine(
            symbol,
            portfolio,
        )
        for symbol in SYMBOLS
    }

    m8 = (
        M8MarketMaker(
            quantity=ORDER_QUANTITY,
            policy=M8Policy(),
        )
        if name == "M8"
        else None
    )

    replay_fills: list[ReplayFill] = []

    per_symbol_fills = defaultdict(int)
    per_symbol_quantity = defaultdict(int)

    for symbol in SYMBOLS:
        events, snapshots = symbol_data[
            symbol
        ]

        engine = engines[symbol]

        for index in range(
            1,
            len(events),
        ):
            event = events[index]

            if event.timestamp < test_start:
                continue

            previous_bids, previous_asks = (
                snapshots[index - 1]
            )

            if (
                not previous_bids
                or not previous_asks
            ):
                continue

            features = (
                build_feature_snapshot_from_levels(
                    previous_bids,
                    previous_asks,
                )
            )

            bid_price = previous_bids[0][0]
            ask_price = previous_asks[0][0]

            if name == "BASELINE":
                decision = QuoteDecision(
                    bid_price=bid_price,
                    ask_price=ask_price,
                    quantity=ORDER_QUANTITY,
                )
            else:
                decision = m8.quote(
                    features,
                    bid_price,
                    ask_price,
                )

            engine.update_quotes(
                decision=decision,
                bid_visible_quantity=(
                    previous_bids[0][1]
                ),
                ask_visible_quantity=(
                    previous_asks[0][1]
                ),
                timestamp=(
                    events[index - 1].timestamp
                ),
            )

            if not isinstance(
                event,
                OrderExecuteEvent,
            ):
                portfolio.update_mark(
                    symbol,
                    (
                        previous_bids[0][0]
                        + previous_asks[0][0]
                    ) / Decimal("2"),
                )
                continue

            if event.execution_price is None:
                continue

            fill = engine.process_execution(
                timestamp=event.timestamp,
                execution_price=(
                    event.execution_price
                ),
                execution_quantity=(
                    event.quantity
                ),
            )

            if fill is not None:
                canonical_fill_id = (
                    f"{event.order_id}-{index}"
                )

                replay_fills.append(
                    ReplayFill(
                        symbol=symbol,
                        event_index=index,
                        canonical_fill_id=(
                            canonical_fill_id
                        ),
                        side=fill.side.value,
                        quantity=fill.quantity,
                        price=fill.price,
                    )
                )

                per_symbol_fills[
                    symbol
                ] += 1

                per_symbol_quantity[
                    symbol
                ] += fill.quantity

            portfolio.update_mark(
                symbol,
                (
                    previous_bids[0][0]
                    + previous_asks[0][0]
                ) / Decimal("2"),
            )

    for symbol in SYMBOLS:
        events, snapshots = symbol_data[
            symbol
        ]

        for index in range(
            len(events) - 1,
            -1,
            -1,
        ):
            bids, asks = snapshots[index]

            if not bids or not asks:
                continue

            portfolio.update_mark(
                symbol,
                (
                    bids[0][0]
                    + asks[0][0]
                ) / Decimal("2"),
            )

            break

    return (
        portfolio,
        replay_fills,
        per_symbol_fills,
        per_symbol_quantity,
    )


def print_portfolio(
    name: str,
    portfolio: Portfolio,
    per_symbol_fills,
    per_symbol_quantity,
):
    realized = portfolio.realized_pnl()
    unrealized = portfolio.unrealized_pnl()
    total = realized + unrealized

    print(f"\n{name}")
    print("-" * 90)
    print(
        f"Fills:           "
        f"{sum(per_symbol_fills.values()):,}"
    )
    print(
        f"Filled quantity: "
        f"{sum(per_symbol_quantity.values()):,}"
    )
    print(
        f"Realized P&L:    "
        f"{realized:,.2f}"
    )
    print(
        f"Unrealized P&L:  "
        f"{unrealized:,.2f}"
    )
    print(
        f"Total P&L:       "
        f"{total:,.2f}"
    )
    print(
        f"Final equity:    "
        f"{portfolio.total_equity():,.2f}"
    )

    print("\nBy symbol")

    for symbol in SYMBOLS:
        position = portfolio.position(
            symbol
        )

        print(
            f"{symbol:>5} | "
            f"fills={per_symbol_fills[symbol]:>6,} | "
            f"qty={per_symbol_quantity[symbol]:>8,} | "
            f"position={position.quantity:>6} | "
            f"realized="
            f"{position.realized_pnl:>12.2f}"
        )


def reconcile(
    label: str,
    replay_fills: list[ReplayFill],
    canonical: dict[str, dict],
):
    replay_ids = {
        fill.canonical_fill_id
        for fill in replay_fills
    }

    canonical_ids = set(
        canonical.keys()
    )

    matched = (
        replay_ids & canonical_ids
    )

    unmatched_replay = (
        replay_ids - canonical_ids
    )

    print(f"\n{label} event reconciliation")
    print("-" * 90)

    print(
        f"Replay fills:       "
        f"{len(replay_ids):,}"
    )
    print(
        f"Canonical fills:    "
        f"{len(canonical_ids):,}"
    )
    print(
        f"Exact event matches:"
        f" {len(matched):,}"
    )
    print(
        f"Unmatched replay:   "
        f"{len(unmatched_replay):,}"
    )

    if replay_ids:
        print(
            f"Replay -> canonical: "
            f"{len(matched) / len(replay_ids):.4%}"
        )
    else:
        print(
            "Replay -> canonical: 0.0000%"
        )

    return matched, unmatched_replay


def compare_quantities(
    label: str,
    replay_fills: list[ReplayFill],
    canonical: dict[str, dict],
):
    print(f"\n{label} quantity reconciliation")
    print("-" * 90)

    quantity_mismatches = []

    for fill in replay_fills:
        row = canonical.get(
            fill.canonical_fill_id
        )

        if row is None:
            continue

        expected_quantity = min(
            row["execution_quantity"],
            ORDER_QUANTITY,
        )

        if (
            fill.quantity
            > expected_quantity
        ):
            quantity_mismatches.append(
                (
                    fill.canonical_fill_id,
                    fill.quantity,
                    expected_quantity,
                )
            )

    print(
        f"Quantity mismatches: "
        f"{len(quantity_mismatches):,}"
    )

    if quantity_mismatches:
        print(
            "First mismatches:"
        )

        for item in quantity_mismatches[:10]:
            print(
                f"  {item[0]} | "
                f"replay={item[1]} | "
                f"canonical_cap={item[2]}"
            )

    return quantity_mismatches


def main():
    print(
        "\nM8 ECONOMIC CONSISTENCY AUDIT"
    )
    print("=" * 90)

    (
        test_start,
        validation_end,
        train_count,
        validation_count,
        test_count,
    ) = load_frozen_test_start()

    print("\nFrozen split")
    print("-" * 90)
    print(
        f"Train rows:       {train_count:,}"
    )
    print(
        f"Validation rows:  {validation_count:,}"
    )
    print(
        f"Test rows:        {test_count:,}"
    )
    print(
        f"Validation end:   {validation_end}"
    )
    print(
        f"Frozen test start: {test_start}"
    )

    symbol_data = {}

    for symbol in SYMBOLS:
        print(
            f"Loading {symbol}..."
        )

        symbol_data[symbol] = load_symbol(
            symbol
        )

    canonical = build_canonical_test_rows(
        test_start,
        symbol_data,
    )

    print("\nCanonical execution population")
    print("-" * 90)
    print(
        f"Canonical test execution fills: "
        f"{len(canonical):,}"
    )

    (
        baseline_portfolio,
        baseline_fills,
        baseline_symbol_fills,
        baseline_symbol_quantity,
    ) = run_replay(
        "BASELINE",
        test_start,
        symbol_data,
    )

    (
        m8_portfolio,
        m8_fills,
        m8_symbol_fills,
        m8_symbol_quantity,
    ) = run_replay(
        "M8",
        test_start,
        symbol_data,
    )

    baseline_matched, _ = reconcile(
        "BASELINE",
        baseline_fills,
        canonical,
    )

    m8_matched, _ = reconcile(
        "M8",
        m8_fills,
        canonical,
    )

    baseline_quantity_mismatches = (
        compare_quantities(
            "BASELINE",
            baseline_fills,
            canonical,
        )
    )

    m8_quantity_mismatches = (
        compare_quantities(
            "M8",
            m8_fills,
            canonical,
        )
    )

    print_portfolio(
        "BASELINE",
        baseline_portfolio,
        baseline_symbol_fills,
        baseline_symbol_quantity,
    )

    print_portfolio(
        "M8",
        m8_portfolio,
        m8_symbol_fills,
        m8_symbol_quantity,
    )

    baseline_pnl = (
        baseline_portfolio.realized_pnl()
        + baseline_portfolio.unrealized_pnl()
    )

    m8_pnl = (
        m8_portfolio.realized_pnl()
        + m8_portfolio.unrealized_pnl()
    )

    print("\nEconomic comparison")
    print("-" * 90)
    print(
        f"Baseline total P&L: "
        f"{baseline_pnl:,.2f}"
    )
    print(
        f"M8 total P&L:       "
        f"{m8_pnl:,.2f}"
    )
    print(
        f"M8 - baseline:      "
        f"{m8_pnl - baseline_pnl:+,.2f}"
    )

    print("\nAudit summary")
    print("-" * 90)

    baseline_reconciliation_rate = (
        len(baseline_matched)
        / len(baseline_fills)
        if baseline_fills
        else 0.0
    )

    m8_reconciliation_rate = (
        len(m8_matched)
        / len(m8_fills)
        if m8_fills
        else 0.0
    )

    print(
        f"Baseline event reconciliation: "
        f"{baseline_reconciliation_rate:.4%}"
    )
    print(
        f"M8 event reconciliation:       "
        f"{m8_reconciliation_rate:.4%}"
    )
    print(
        f"Baseline quantity mismatches:   "
        f"{len(baseline_quantity_mismatches):,}"
    )
    print(
        f"M8 quantity mismatches:         "
        f"{len(m8_quantity_mismatches):,}"
    )


if __name__ == "__main__":
    main()