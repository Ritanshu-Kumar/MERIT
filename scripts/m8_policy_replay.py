from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path

from merit.data.lobster import OrderExecuteEvent, read_messages
from merit.data.lobster_orderbook import read_orderbooks
from merit.portfolio.enums import OrderSide
from merit.research.lobster_m8 import build_feature_snapshot_from_book
from merit.research.m8_policy import M8Policy
from merit.research.targets import (
    MidPricePoint,
    build_mid_price_index,
    calculate_fill_markout,
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

TRAIN_FRACTION = 0.60
VALIDATION_FRACTION = 0.20
EMBARGO_SECONDS = 5

HORIZON = timedelta(seconds=1)

THRESHOLD = 0.00957966

OUTPUT = "M8_policy_replay.csv"


@dataclass(frozen=True)
class Candidate:
    timestamp: object
    symbol: str
    side: OrderSide
    price: Decimal
    quantity: int
    mid_price: Decimal
    features: object
    markout_1s: Decimal
    initial_edge: Decimal
    post_fill_move_1s: Decimal


@dataclass(frozen=True)
class ReplayResult:
    symbol: str
    timestamp: object
    side: str
    price: Decimal
    quantity: int
    policy: str
    markout_1s: Decimal
    initial_edge: Decimal
    post_fill_move_1s: Decimal
    post_fill_return_bps: Decimal


def get_message_file(symbol: str) -> Path:
    return BASE / (
        f"{symbol}_2012-06-21_34200000_57600000"
        "_message_10.csv"
    )


def get_orderbook_file(symbol: str) -> Path:
    return BASE / (
        f"{symbol}_2012-06-21_34200000_57600000"
        "_orderbook_10.csv"
    )


def initial_edge(
    side: OrderSide,
    fill_price: Decimal,
    mid_price: Decimal,
) -> Decimal:
    if side == OrderSide.BUY:
        return mid_price - fill_price

    return fill_price - mid_price


def load_symbol(
    symbol: str,
) -> list[Candidate]:
    events = tuple(
        read_messages(
            get_message_file(symbol),
            symbol,
            TRADING_DATE,
        )
    )

    snapshots = tuple(
        read_orderbooks(
            get_orderbook_file(symbol),
            levels=LEVELS,
        )
    )

    aligned_count = min(
        len(events),
        len(snapshots),
    )

    events = events[:aligned_count]
    snapshots = snapshots[:aligned_count]

    mid_prices = []

    for event, snapshot in zip(
        events,
        snapshots,
    ):
        bids, asks = snapshot

        if not bids or not asks:
            continue

        mid_prices.append(
            MidPricePoint(
                timestamp=event.timestamp,
                mid_price=(
                    bids[0][0] + asks[0][0]
                ) / Decimal("2"),
            )
        )

    mid_price_index = build_mid_price_index(
        tuple(mid_prices)
    )

    candidates = []

    for index in range(1, aligned_count):
        event = events[index]

        if not isinstance(
            event,
            OrderExecuteEvent,
        ):
            continue

        if event.execution_price is None:
            continue

        bids, asks = snapshots[index - 1]

        if not bids or not asks:
            continue

        bid_price = bids[0][0]
        ask_price = asks[0][0]

        if event.execution_price == bid_price:
            side = OrderSide.BUY
        elif event.execution_price == ask_price:
            side = OrderSide.SELL
        else:
            continue

        features = build_feature_snapshot_from_book(
            bids,
            asks,
        )

        if features.mid_price is None:
            continue

        markout = calculate_fill_markout(
            fill_timestamp=event.timestamp,
            fill_price=event.execution_price,
            side=side,
            mid_price_index=mid_price_index,
            horizons=(HORIZON,),
        )

        markout_value = markout.markouts[0].value

        if markout_value is None:
            continue

        edge = initial_edge(
            side,
            event.execution_price,
            features.mid_price,
        )

        post_fill_move = markout_value - edge

        candidates.append(
            Candidate(
                timestamp=event.timestamp,
                symbol=symbol,
                side=side,
                price=event.execution_price,
                quantity=min(
                    event.quantity,
                    ORDER_QUANTITY,
                ),
                mid_price=features.mid_price,
                features=features,
                markout_1s=markout_value,
                initial_edge=edge,
                post_fill_move_1s=post_fill_move,
            )
        )

    return candidates


def build_global_test_set(
    candidates: list[Candidate],
) -> list[Candidate]:
    candidates = sorted(
        candidates,
        key=lambda candidate: candidate.timestamp,
    )

    n = len(candidates)

    validation_end = int(
        n * (
            TRAIN_FRACTION
            + VALIDATION_FRACTION
        )
    )

    validation_boundary = candidates[
        validation_end - 1
    ].timestamp

    test_start = (
        validation_boundary
        + timedelta(seconds=EMBARGO_SECONDS)
    )

    return [
        candidate
        for candidate in candidates[validation_end:]
        if candidate.timestamp >= test_start
    ]


def replay_candidate(
    candidate: Candidate,
    maker: M8MarketMaker,
) -> tuple[ReplayResult, ReplayResult]:
    baseline = QuoteDecision(
        bid_price=candidate.price
        if candidate.side == OrderSide.BUY
        else None,
        ask_price=candidate.price
        if candidate.side == OrderSide.SELL
        else None,
        quantity=ORDER_QUANTITY,
    )

    m8_decision = maker.quote(
        candidate.features,
        candidate.features.mid_price
        - candidate.features.spread / Decimal("2"),
        candidate.features.mid_price
        + candidate.features.spread / Decimal("2"),
    )

    baseline_price = (
        baseline.bid_price
        if candidate.side == OrderSide.BUY
        else baseline.ask_price
    )

    m8_price = (
        m8_decision.bid_price
        if candidate.side == OrderSide.BUY
        else m8_decision.ask_price
    )

    return (
        ReplayResult(
            symbol=candidate.symbol,
            timestamp=candidate.timestamp,
            side=candidate.side.value,
            price=candidate.price,
            quantity=candidate.quantity,
            policy="baseline",
            markout_1s=candidate.markout_1s,
            initial_edge=candidate.initial_edge,
            post_fill_move_1s=candidate.post_fill_move_1s,
            post_fill_return_bps=(
                candidate.post_fill_move_1s
                / candidate.price
                * Decimal("10000")
            ),
        ),
        ReplayResult(
            symbol=candidate.symbol,
            timestamp=candidate.timestamp,
            side=candidate.side.value,
            price=candidate.price,
            quantity=(
                candidate.quantity
                if m8_price == candidate.price
                else 0
            ),
            policy="m8",
            markout_1s=candidate.markout_1s,
            initial_edge=candidate.initial_edge,
            post_fill_move_1s=candidate.post_fill_move_1s,
            post_fill_return_bps=(
                candidate.post_fill_move_1s
                / candidate.price
                * Decimal("10000")
            ),
        ),
    )


def print_summary(
    results: list[ReplayResult],
    total_candidates: int,
) -> None:
    for policy in ["baseline", "m8"]:
        policy_results = [
            result
            for result in results
            if result.policy == policy
            and result.quantity > 0
        ]

        total_quantity = sum(
            result.quantity
            for result in policy_results
        )

        mean_return = (
            sum(
                result.post_fill_return_bps
                for result in policy_results
            )
            / len(policy_results)
            if policy_results
            else Decimal("NaN")
        )

        fill_rate = (
            len(policy_results)
            / total_candidates
            if total_candidates
            else 0.0
        )

        print(
            f"{policy:>8} | "
            f"fills={len(policy_results):>6,} | "
            f"quantity={total_quantity:>8,} | "
            f"fill_rate={fill_rate:.4f} | "
            f"post_fill={mean_return:+.6f} bps"
        )


def main() -> None:
    start = time.perf_counter()

    print("\nM8 POLICY REPLAY")
    print("=" * 80)
    print(
        f"Frozen threshold: {THRESHOLD:.8f}"
    )

    candidates = []

    for symbol in SYMBOLS:
        symbol_start = time.perf_counter()

        symbol_candidates = load_symbol(symbol)
        candidates.extend(symbol_candidates)

        print(
            f"{symbol}: "
            f"eligible={len(symbol_candidates):,} | "
            f"time="
            f"{time.perf_counter() - symbol_start:.2f}s"
        )

    test_candidates = build_global_test_set(
        candidates
    )

    maker = M8MarketMaker(
        quantity=ORDER_QUANTITY,
        policy=M8Policy(
            threshold=THRESHOLD
        ),
    )

    results = []

    for candidate in test_candidates:
        baseline, m8 = replay_candidate(
            candidate,
            maker,
        )

        results.append(baseline)

        if m8.quantity > 0:
            results.append(m8)

    print("\nGlobal test set")
    print("-" * 80)
    print(
        f"All valid candidates: "
        f"{len(candidates):,}"
    )
    print(
        f"Test candidates: "
        f"{len(test_candidates):,}"
    )

    print("\nSummary")
    print("-" * 80)

    print_summary(
        results,
        len(test_candidates),
    )

    fields = [
        "symbol",
        "timestamp",
        "side",
        "price",
        "quantity",
        "policy",
        "markout_1s",
        "initial_edge",
        "post_fill_move_1s",
        "post_fill_return_bps",
    ]

    with open(
        OUTPUT,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fields,
        )

        writer.writeheader()

        for result in results:
            writer.writerow(
                {
                    field: getattr(
                        result,
                        field,
                    )
                    for field in fields
                }
            )

    print(f"\nSaved: {OUTPUT}")
    print(
        f"Total time: "
        f"{time.perf_counter() - start:.2f}s"
    )


if __name__ == "__main__":
    main()