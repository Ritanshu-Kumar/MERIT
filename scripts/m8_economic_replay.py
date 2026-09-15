from __future__ import annotations

from collections import defaultdict
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path

from merit.data.lobster import (
    OrderExecuteEvent,
    read_messages,
)
from merit.data.lobster_orderbook import read_orderbooks
from merit.execution.queue_model import QueueModel
from merit.features.snapshot import (
    build_feature_snapshot_from_levels,
)
from merit.portfolio.enums import OrderSide
from merit.portfolio.portfolio import Portfolio
from merit.research.lobster_execution import (
    LobsterExecutionEngine,
)
from merit.research.m8_policy import M8Policy
from merit.risk.limits import (
    RiskLimits,
    RiskManager,
)
from merit.strategy.baseline import QuoteDecision
from merit.strategy.m8 import M8MarketMaker


BASE = Path("data/sample")

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
EMBARGO_SECONDS = 5


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


def is_executable_candidate(
    event,
    bids,
    asks,
) -> bool:
    if not isinstance(
        event,
        OrderExecuteEvent,
    ):
        return False

    if event.execution_price is None:
        return False

    if not bids or not asks:
        return False

    return (
        event.execution_price == bids[0][0]
        or event.execution_price == asks[0][0]
    )


def find_test_start(
    symbol_data,
) -> object:
    timestamps = []

    for events, snapshots in symbol_data.values():
        for index in range(1, len(events)):
            previous_bids, previous_asks = snapshots[
                index - 1
            ]

            if is_executable_candidate(
                events[index],
                previous_bids,
                previous_asks,
            ):
                timestamps.append(
                    events[index].timestamp
                )

    timestamps.sort()

    if not timestamps:
        raise ValueError(
            "No eligible execution candidates found."
        )

    validation_end = int(
        len(timestamps)
        * (
            TRAIN_FRACTION
            + VALIDATION_FRACTION
        )
    )

    validation_boundary = timestamps[
        validation_end - 1
    ]

    return (
        validation_boundary
        + timedelta(
            seconds=EMBARGO_SECONDS
        )
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


def update_marks(
    portfolio: Portfolio,
    symbol: str,
    bids,
    asks,
) -> None:
    if not bids or not asks:
        return

    mid = (
        bids[0][0] + asks[0][0]
    ) / Decimal("2")

    portfolio.update_mark(
        symbol,
        mid,
    )


def run_strategy(
    name: str,
    symbol_data,
    test_start,
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

    fill_counts = defaultdict(int)
    fill_quantity = defaultdict(int)

    for symbol in SYMBOLS:
        events, snapshots = symbol_data[
            symbol
        ]

        engine = engines[symbol]

        for index in range(1, len(events)):
            event = events[index]

            if event.timestamp < test_start:
                continue

            previous_bids, previous_asks = snapshots[
                index - 1
            ]

            if not previous_bids or not previous_asks:
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
                bid_visible_quantity=previous_bids[0][1],
                ask_visible_quantity=previous_asks[0][1],
                timestamp=previous_events_timestamp(
                    events,
                    index,
                ),
            )

            if isinstance(
                event,
                OrderExecuteEvent,
            ):
                if event.execution_price is None:
                    continue

                fill = engine.process_execution(
                    timestamp=event.timestamp,
                    execution_price=event.execution_price,
                    execution_quantity=event.quantity,
                )

                if fill is not None:
                    fill_counts[symbol] += 1
                    fill_quantity[symbol] += (
                        fill.quantity
                    )

            update_marks(
                portfolio,
                symbol,
                previous_bids,
                previous_asks,
            )

    last_marks = {}

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

            if bids and asks:
                last_marks[symbol] = (
                    bids[0][0] + asks[0][0]
                ) / Decimal("2")
                break

    for symbol, price in last_marks.items():
        portfolio.update_mark(
            symbol,
            price,
        )

    return (
        portfolio,
        fill_counts,
        fill_quantity,
    )


def previous_events_timestamp(
    events,
    index,
):
    return events[index - 1].timestamp


def print_results(
    name: str,
    portfolio: Portfolio,
    fill_counts,
    fill_quantity,
) -> None:
    print(f"\n{name}")
    print("-" * 80)

    total_fills = sum(
        fill_counts.values()
    )

    total_quantity = sum(
        fill_quantity.values()
    )

    print(
        f"fills:          {total_fills:,}"
    )

    print(
        f"quantity:       {total_quantity:,}"
    )

    print(
        f"realized P&L:   "
        f"{portfolio.realized_pnl():,.2f}"
    )

    print(
        f"unrealized P&L: "
        f"{portfolio.unrealized_pnl():,.2f}"
    )

    print(
        f"total equity:   "
        f"{portfolio.total_equity():,.2f}"
    )

    print("\nBy symbol")

    for symbol in SYMBOLS:
        position = portfolio.position(
            symbol
        )

        print(
            f"{symbol:>5} | "
            f"fills={fill_counts[symbol]:>6,} | "
            f"qty={fill_quantity[symbol]:>8,} | "
            f"position={position.quantity:>6} | "
            f"realized="
            f"{position.realized_pnl:>12.2f}"
        )


def main() -> None:
    print("\nM8 ECONOMIC REPLAY")
    print("=" * 80)

    print(
        f"Queue ahead fraction: "
        f"{QUEUE_AHEAD_FRACTION}"
    )

    print(
        f"Fee: {FEE_BPS} bps"
    )

    print(
        f"Order quantity: "
        f"{ORDER_QUANTITY}"
    )

    symbol_data = {}

    for symbol in SYMBOLS:
        print(
            f"Loading {symbol}..."
        )

        symbol_data[symbol] = load_symbol(
            symbol
        )

    test_start = find_test_start(
        symbol_data
    )

    print(
        f"\nFrozen test start: "
        f"{test_start}"
    )

    baseline = run_strategy(
        "BASELINE",
        symbol_data,
        test_start,
    )

    m8 = run_strategy(
        "M8",
        symbol_data,
        test_start,
    )

    print_results(
        "BASELINE",
        *baseline,
    )

    print_results(
        "M8",
        *m8,
    )


if __name__ == "__main__":
    main()