from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from pathlib import Path

import numpy as np
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

BOOTSTRAP_REPS = 5000
SEED = 20261001


@dataclass(frozen=True)
class ReplayEvent:
    timestamp: object
    symbol: str
    index: int


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


def load_symbol(symbol: str):
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


def load_frozen_test_start() -> pd.Timestamp:
    df = pd.read_csv(DATASET)

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
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

    train = df.iloc[
        :train_end
    ].copy()

    validation = df.iloc[
        train_end:validation_end
    ].copy()

    test = df.iloc[
        validation_end:
    ].copy()

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
            "Validation split is empty "
            "after purge."
        )

    if test.empty:
        raise ValueError(
            "Test split is empty "
            "after embargo."
        )

    return test["timestamp"].iloc[0]


def build_chronological_events(
    test_start: pd.Timestamp,
    symbol_data,
) -> list[ReplayEvent]:
    replay_events = []

    for symbol in SYMBOLS:
        events, _ = symbol_data[symbol]

        for index, event in enumerate(events):
            if index == 0:
                continue

            if event.timestamp < test_start:
                continue

            replay_events.append(
                ReplayEvent(
                    timestamp=event.timestamp,
                    symbol=symbol,
                    index=index,
                )
            )

    replay_events.sort(
        key=lambda item: (
            item.timestamp,
            item.symbol,
            item.index,
        )
    )

    return replay_events


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


def run_strategy(
    name: str,
    test_start: pd.Timestamp,
    symbol_data,
    chronological_events: list[ReplayEvent],
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

    equity_points = {}

    fills_by_symbol = {
        symbol: 0
        for symbol in SYMBOLS
    }

    quantity_by_symbol = {
        symbol: 0
        for symbol in SYMBOLS
    }

    first_minute = pd.Timestamp(
        test_start
    ).floor("1min")

    equity_points[first_minute - pd.Timedelta(minutes=1)] = float(
        STARTING_CASH
    )

    for replay_event in chronological_events:
        symbol = replay_event.symbol
        index = replay_event.index

        events, snapshots = symbol_data[
            symbol
        ]

        event = events[index]

        previous_bids, previous_asks = (
            snapshots[index - 1]
        )

        if (
            not previous_bids
            or not previous_asks
        ):
            continue

        engine = engines[symbol]

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
            timestamp=events[
                index - 1
            ].timestamp,
        )

        if (
            isinstance(
                event,
                OrderExecuteEvent,
            )
            and event.execution_price is not None
        ):
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
                fills_by_symbol[
                    symbol
                ] += 1

                quantity_by_symbol[
                    symbol
                ] += fill.quantity

        mid = (
            previous_bids[0][0]
            + previous_asks[0][0]
        ) / Decimal("2")

        portfolio.update_mark(
            symbol,
            mid,
        )

        minute = pd.Timestamp(
            event.timestamp
        ).floor("1min")

        equity_points[minute] = float(
            portfolio.total_equity()
        )

    final_marks = {}

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

            final_marks[symbol] = (
                bids[0][0]
                + asks[0][0]
            ) / Decimal("2")

            break

    for symbol, mark in final_marks.items():
        portfolio.update_mark(
            symbol,
            mark,
        )

    final_equity = float(
        portfolio.total_equity()
    )

    if chronological_events:
        last_minute = pd.Timestamp(
            chronological_events[-1].timestamp
        ).floor("1min")

        equity_points[last_minute] = (
            final_equity
        )

    return (
        portfolio,
        equity_points,
        fills_by_symbol,
        quantity_by_symbol,
    )


def complete_minute_equity_series(
    equity_points: dict[pd.Timestamp, float],
) -> pd.Series:
    if not equity_points:
        raise ValueError(
            "No equity observations."
        )

    series = (
        pd.Series(equity_points)
        .sort_index()
    )

    full_index = pd.date_range(
        start=series.index.min(),
        end=series.index.max(),
        freq="1min",
        tz="UTC",
    )

    return (
        series.reindex(full_index)
        .ffill()
        .bfill()
    )


def aggregate_pnl_blocks(
    minute_equity: pd.Series,
    block_minutes: int,
) -> pd.Series:
    minute_pnl = (
        minute_equity
        .diff()
        .dropna()
    )

    if block_minutes == 1:
        return minute_pnl

    block_labels = (
        minute_pnl.index.floor(
            f"{block_minutes}min"
        )
    )

    return (
        minute_pnl
        .groupby(block_labels)
        .sum()
    )


def bootstrap_difference(
    baseline_blocks: pd.Series,
    m8_blocks: pd.Series,
    repetitions: int,
    seed: int,
):
    common_index = (
        baseline_blocks.index.intersection(
            m8_blocks.index
        )
    )

    baseline = (
        baseline_blocks.loc[
            common_index
        ].to_numpy()
    )

    m8 = (
        m8_blocks.loc[
            common_index
        ].to_numpy()
    )

    differences = m8 - baseline

    observed = float(
        differences.sum()
    )

    rng = np.random.default_rng(
        seed
    )

    n = len(differences)

    if n == 0:
        raise ValueError(
            "No common P&L blocks."
        )

    bootstrap_totals = np.empty(
        repetitions,
        dtype=float,
    )

    for index in range(
        repetitions
    ):
        sample = rng.choice(
            differences,
            size=n,
            replace=True,
        )

        bootstrap_totals[index] = (
            sample.sum()
        )

    return {
        "observed": observed,
        "p025": float(
            np.quantile(
                bootstrap_totals,
                0.025,
            )
        ),
        "p50": float(
            np.quantile(
                bootstrap_totals,
                0.50,
            )
        ),
        "p975": float(
            np.quantile(
                bootstrap_totals,
                0.975,
            )
        ),
        "blocks": n,
    }


def main() -> None:
    print()
    print(
        "M8 ECONOMIC BLOCK BOOTSTRAP"
    )
    print("=" * 90)

    test_start = (
        load_frozen_test_start()
    )

    print(
        f"Frozen test start: "
        f"{test_start}"
    )

    symbol_data = {}

    for symbol in SYMBOLS:
        print(
            f"Loading {symbol}..."
        )

        symbol_data[symbol] = (
            load_symbol(symbol)
        )

    chronological_events = (
        build_chronological_events(
            test_start,
            symbol_data,
        )
    )

    print(
        f"Chronological events: "
        f"{len(chronological_events):,}"
    )

    (
        baseline_portfolio,
        baseline_equity,
        baseline_fills,
        baseline_quantity,
    ) = run_strategy(
        "BASELINE",
        test_start,
        symbol_data,
        chronological_events,
    )

    (
        m8_portfolio,
        m8_equity,
        m8_fills,
        m8_quantity,
    ) = run_strategy(
        "M8",
        test_start,
        symbol_data,
        chronological_events,
    )

    baseline_total = float(
        baseline_portfolio.total_equity()
        - STARTING_CASH
    )

    m8_total = float(
        m8_portfolio.total_equity()
        - STARTING_CASH
    )

    observed_difference = (
        m8_total
        - baseline_total
    )

    print()
    print(
        "Economic point estimates"
    )
    print("-" * 90)

    print(
        f"Baseline P&L: "
        f"{baseline_total:,.2f}"
    )

    print(
        f"M8 P&L:       "
        f"{m8_total:,.2f}"
    )

    print(
        f"M8 - baseline:"
        f" {observed_difference:+,.2f}"
    )

    print()
    print(
        "P&L by symbol"
    )
    print("-" * 90)

    for symbol in SYMBOLS:
        baseline_position = (
            baseline_portfolio.position(
                symbol
            )
        )

        m8_position = (
            m8_portfolio.position(
                symbol
            )
        )

        baseline_realized = (
            float(
                baseline_position.realized_pnl
            )
        )

        m8_realized = (
            float(
                m8_position.realized_pnl
            )
        )

        print(
            f"{symbol:>5} | "
            f"baseline="
            f"{baseline_realized:>12.2f} | "
            f"M8="
            f"{m8_realized:>12.2f} | "
            f"diff="
            f"{m8_realized - baseline_realized:>12.2f} | "
            f"baseline_fills="
            f"{baseline_fills[symbol]:>5} | "
            f"M8_fills="
            f"{m8_fills[symbol]:>5}"
        )

    baseline_minute = (
        complete_minute_equity_series(
            baseline_equity
        )
    )

    m8_minute = (
        complete_minute_equity_series(
            m8_equity
        )
    )

    common_start = max(
        baseline_minute.index.min(),
        m8_minute.index.min(),
    )

    common_end = min(
        baseline_minute.index.max(),
        m8_minute.index.max(),
    )

    baseline_minute = (
        baseline_minute.loc[
            common_start:common_end
        ]
    )

    m8_minute = (
        m8_minute.loc[
            common_start:common_end
        ]
    )

    reconstructed_difference = float(
        (
            m8_minute.iloc[-1]
            - m8_minute.iloc[0]
        )
        - (
            baseline_minute.iloc[-1]
            - baseline_minute.iloc[0]
        )
    )

    reconstruction_error = (
        reconstructed_difference
        - observed_difference
    )

    print()
    print(
        "P&L reconstruction check"
    )
    print("-" * 90)

    print(
        f"Replay M8 - baseline: "
        f"{observed_difference:+,.2f}"
    )

    print(
        f"Minute-series difference: "
        f"{reconstructed_difference:+,.2f}"
    )

    print(
        f"Reconstruction error: "
        f"{reconstruction_error:+,.6f}"
    )

    if abs(reconstruction_error) > 0.01:
        raise ValueError(
            "Minute P&L reconstruction "
            "does not reproduce the "
            "economic point estimate."
        )

    print(
        "PASS: minute P&L reproduces "
        "the audited point estimate."
    )

    print()
    print(
        "Economic block bootstrap"
    )
    print("-" * 90)

    for block_minutes in (
        1,
        5,
        10,
    ):
        baseline_blocks = (
            aggregate_pnl_blocks(
                baseline_minute,
                block_minutes,
            )
        )

        m8_blocks = (
            aggregate_pnl_blocks(
                m8_minute,
                block_minutes,
            )
        )

        result = bootstrap_difference(
            baseline_blocks,
            m8_blocks,
            BOOTSTRAP_REPS,
            SEED + block_minutes,
        )

        print(
            f"{block_minutes:>2} min | "
            f"observed="
            f"{result['observed']:+,.2f} | "
            f"p025="
            f"{result['p025']:+,.2f} | "
            f"p50="
            f"{result['p50']:+,.2f} | "
            f"p975="
            f"{result['p975']:+,.2f} | "
            f"blocks="
            f"{result['blocks']}"
        )


if __name__ == "__main__":
    main()