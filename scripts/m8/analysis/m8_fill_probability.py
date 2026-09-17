from __future__ import annotations

import heapq
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path

from merit.data.lobster import (
    OrderExecuteEvent,
    read_messages,
)
from merit.data.lobster_orderbook import read_orderbooks
from merit.features.snapshot import (
    build_feature_snapshot_from_levels,
)
from merit.research.m8_policy import M8Policy


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

TRAIN_FRACTION = 0.60
VALIDATION_FRACTION = 0.20
EMBARGO_SECONDS = 5

@dataclass(frozen=True)
class Opportunity:
    timestamp: datetime
    symbol: str
    side: str
    quote_price: Decimal
    selected: bool
    filled: bool
    fill_quantity: int


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


def symbol_opportunities(
    symbol: str,
    policy: M8Policy,
):
    events = iter(
        read_messages(
            message_path(symbol),
            symbol,
            TRADING_DATE,
        )
    )

    snapshots = iter(
        read_orderbooks(
            orderbook_path(symbol),
            levels=LEVELS,
        )
    )

    try:
        previous_event = next(events)
        previous_snapshot = next(snapshots)
    except StopIteration:
        return

    while True:
        try:
            next_event = next(events)
            next_snapshot = next(snapshots)
        except StopIteration:
            return

        bids, asks = previous_snapshot

        if bids and asks:
            features = build_feature_snapshot_from_levels(
                bids,
                asks,
            )

            bid_price = bids[0][0]
            ask_price = asks[0][0]

            bid_selected = (
                features.relative_spread is not None
                and features.imbalance is not None
                and policy.select(
                    float(features.relative_spread),
                    float(features.imbalance),
                    "BUY",
                )
            )

            ask_selected = (
                features.relative_spread is not None
                and features.imbalance is not None
                and policy.select(
                    float(features.relative_spread),
                    float(features.imbalance),
                    "SELL",
                )
            )

            bid_filled = (
                isinstance(
                    next_event,
                    OrderExecuteEvent,
                )
                and next_event.execution_price is not None
                and next_event.execution_price == bid_price
            )

            ask_filled = (
                isinstance(
                    next_event,
                    OrderExecuteEvent,
                )
                and next_event.execution_price is not None
                and next_event.execution_price == ask_price
            )

            bid_quantity = (
                next_event.quantity
                if bid_filled
                else 0
            )

            ask_quantity = (
                next_event.quantity
                if ask_filled
                else 0
            )

            yield Opportunity(
                timestamp=previous_event.timestamp,
                symbol=symbol,
                side="BUY",
                quote_price=bid_price,
                selected=bid_selected,
                filled=bid_filled,
                fill_quantity=bid_quantity,
            )

            yield Opportunity(
                timestamp=previous_event.timestamp,
                symbol=symbol,
                side="SELL",
                quote_price=ask_price,
                selected=ask_selected,
                filled=ask_filled,
                fill_quantity=ask_quantity,
            )

        previous_event = next_event
        previous_snapshot = next_snapshot


def count_opportunities(
    policy: M8Policy,
) -> int:
    total = 0

    for symbol in SYMBOLS:
        count = sum(
            1
            for _ in symbol_opportunities(
                symbol,
                policy,
            )
        )

        print(
            f"{symbol}: {count:,} opportunities"
        )

        total += count

    return total


def merged_opportunities(
    policy: M8Policy,
):
    generators = {
        symbol: iter(
            symbol_opportunities(
                symbol,
                policy,
            )
        )
        for symbol in SYMBOLS
    }

    heap = []
    sequence = 0

    for symbol, generator in generators.items():
        try:
            opportunity = next(generator)
        except StopIteration:
            continue

        heapq.heappush(
            heap,
            (
                opportunity.timestamp,
                sequence,
                symbol,
                opportunity,
            ),
        )
        sequence += 1

    while heap:
        _, _, symbol, opportunity = heapq.heappop(heap)

        yield opportunity

        try:
            next_opportunity = next(
                generators[symbol]
            )
        except StopIteration:
            continue

        heapq.heappush(
            heap,
            (
                next_opportunity.timestamp,
                sequence,
                symbol,
                next_opportunity,
            ),
        )
        sequence += 1


def summarize(
    opportunities,
    label: str,
) -> dict:
    opportunities = list(opportunities)

    selected = [
        opportunity
        for opportunity in opportunities
        if opportunity.selected
    ]

    filled = [
        opportunity
        for opportunity in opportunities
        if opportunity.filled
    ]

    selected_filled = [
        opportunity
        for opportunity in selected
        if opportunity.filled
    ]

    return {
        "segment": label,
        "opportunities": len(opportunities),
        "selected": len(selected),
        "selected_rate": (
            len(selected) / len(opportunities)
            if opportunities
            else 0.0
        ),
        "baseline_filled": len(filled),
        "baseline_fill_probability": (
            len(filled) / len(opportunities)
            if opportunities
            else 0.0
        ),
        "m8_filled": len(selected_filled),
        "m8_fill_probability": (
            len(selected_filled) / len(selected)
            if selected
            else 0.0
        ),
        "selected_fill_share": (
            len(selected_filled) / len(filled)
            if filled
            else 0.0
        ),
        "baseline_quantity": sum(
            opportunity.fill_quantity
            for opportunity in filled
        ),
        "m8_quantity": sum(
            opportunity.fill_quantity
            for opportunity in selected_filled
        ),
    }


def main() -> None:
    policy = M8Policy()

    print("\nM8 FILL PROBABILITY")
    print("=" * 90)
    print(
        f"Frozen threshold: {policy.threshold:.8f}"
    )

    total = count_opportunities(policy)

    train_end = int(
        total * TRAIN_FRACTION
    )

    validation_end = int(
        total
        * (
            TRAIN_FRACTION
            + VALIDATION_FRACTION
        )
    )

    test_start_index = validation_end

    test_results = []

    validation_boundary = None

    for index, opportunity in enumerate(
        merged_opportunities(policy)
    ):
        if index == train_end - 1:
            pass

        if index == validation_end - 1:
            validation_boundary = (
                opportunity.timestamp
            )

        if index < test_start_index:
            continue

        if validation_boundary is None:
            continue

        test_start = (
            validation_boundary
            + timedelta(
                seconds=EMBARGO_SECONDS
            )
        )

        if opportunity.timestamp < test_start:
            continue

        test_results.append(opportunity)

    overall = summarize(
        test_results,
        "overall",
    )

    print("\nGlobal test result")
    print("-" * 90)

    for key, value in overall.items():
        print(
            f"{key}: {value}"
        )

    print("\nBy symbol")
    print("-" * 90)

    for symbol in SYMBOLS:
        symbol_results = [
            opportunity
            for opportunity in test_results
            if opportunity.symbol == symbol
        ]

        result = summarize(
            symbol_results,
            symbol,
        )

        print(
            f"{symbol:>5} | "
            f"opportunities={result['opportunities']:>8,} | "
            f"selected={result['selected']:>8,} | "
            f"selected_rate="
            f"{result['selected_rate']:.4f} | "
            f"baseline_fill="
            f"{result['baseline_fill_probability']:.6f} | "
            f"M8_fill="
            f"{result['m8_fill_probability']:.6f} | "
            f"selected_fill_share="
            f"{result['selected_fill_share']:.4f}"
        )

    print("\nBy side")
    print("-" * 90)

    for side in ["BUY", "SELL"]:
        side_results = [
            opportunity
            for opportunity in test_results
            if opportunity.side == side
        ]

        result = summarize(
            side_results,
            side,
        )

        print(
            f"{side:>5} | "
            f"opportunities={result['opportunities']:>8,} | "
            f"selected={result['selected']:>8,} | "
            f"selected_rate="
            f"{result['selected_rate']:.4f} | "
            f"baseline_fill="
            f"{result['baseline_fill_probability']:.6f} | "
            f"M8_fill="
            f"{result['m8_fill_probability']:.6f}"
        )


if __name__ == "__main__":
    main()