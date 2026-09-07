from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path

from merit.data.lobster import OrderExecuteEvent, read_messages
from merit.data.lobster_orderbook import read_orderbooks
from merit.execution.queue_model import QueueModel
from merit.features.snapshot import FeatureSnapshot
from merit.portfolio.enums import OrderSide
from merit.research.dataset import (
    ResearchObservation,
    build_research_observation,
)
from merit.research.fill_allocation import allocate_execution
from merit.research.historical_fills import HistoricalFill
from merit.research.opportunities import QuoteOpportunity
from merit.research.targets import MidPricePoint, calculate_fill_markout


HORIZONS = (
    timedelta(milliseconds=100),
    timedelta(seconds=1),
    timedelta(seconds=5),
)


@dataclass(frozen=True)
class LOBSTERState:
    timestamp: datetime
    symbol: str
    bids: tuple[tuple[Decimal, int], ...]
    asks: tuple[tuple[Decimal, int], ...]

    @property
    def best_bid(self) -> tuple[Decimal, int] | None:
        return self.bids[0] if self.bids else None

    @property
    def best_ask(self) -> tuple[Decimal, int] | None:
        return self.asks[0] if self.asks else None

    @property
    def mid_price(self) -> Decimal | None:
        if self.best_bid is None or self.best_ask is None:
            return None

        return (self.best_bid[0] + self.best_ask[0]) / Decimal("2")


def build_feature_snapshot_from_book(
    bids: tuple[tuple[Decimal, int], ...],
    asks: tuple[tuple[Decimal, int], ...],
) -> FeatureSnapshot:
    if not bids or not asks:
        raise ValueError("Feature snapshot requires a two-sided book.")

    bid_price, bid_size = bids[0]
    ask_price, ask_size = asks[0]

    mid_price = (bid_price + ask_price) / Decimal("2")
    spread = ask_price - bid_price
    relative_spread = (
        spread / mid_price if mid_price else Decimal("0")
    )

    total_size = bid_size + ask_size

    imbalance = (
        Decimal(bid_size - ask_size) / Decimal(total_size)
        if total_size
        else Decimal("0")
    )

    microprice = (
        (
            ask_price * Decimal(bid_size)
            + bid_price * Decimal(ask_size)
        )
        / Decimal(total_size)
        if total_size
        else mid_price
    )

    def imbalance_at_depth(levels: int) -> Decimal:
        bid_depth = sum(
            quantity for _, quantity in bids[:levels]
        )
        ask_depth = sum(
            quantity for _, quantity in asks[:levels]
        )

        total = bid_depth + ask_depth

        if total == 0:
            return Decimal("0")

        return Decimal(bid_depth - ask_depth) / Decimal(total)

    def weighted_imbalance(levels: int) -> Decimal:
        bid_weighted = Decimal("0")
        ask_weighted = Decimal("0")

        for index, (_, quantity) in enumerate(
            bids[:levels],
            start=1,
        ):
            bid_weighted += Decimal(quantity) / Decimal(index)

        for index, (_, quantity) in enumerate(
            asks[:levels],
            start=1,
        ):
            ask_weighted += Decimal(quantity) / Decimal(index)

        total = bid_weighted + ask_weighted

        if total == 0:
            return Decimal("0")

        return (bid_weighted - ask_weighted) / total

    return FeatureSnapshot(
        mid_price=mid_price,
        spread=spread,
        relative_spread=relative_spread,
        bid_size=bid_size,
        ask_size=ask_size,
        imbalance=imbalance,
        microprice=microprice,
        imbalance_l5=imbalance_at_depth(5),
        imbalance_l10=imbalance_at_depth(10),
        weighted_imbalance_l5=weighted_imbalance(5),
        weighted_imbalance_l10=weighted_imbalance(10),
    )


def _build_states(
    events: tuple,
    snapshots: tuple[
        tuple[tuple[Decimal, int], ...],
        tuple[tuple[Decimal, int], ...],
    ],
    symbol: str,
) -> tuple[LOBSTERState, ...]:
    aligned_count = min(len(events), len(snapshots))

    states: list[LOBSTERState] = []

    for index in range(aligned_count):
        bids, asks = snapshots[index]

        states.append(
            LOBSTERState(
                timestamp=events[index].timestamp,
                symbol=symbol,
                bids=bids,
                asks=asks,
            )
        )

    return tuple(states)


def _find_passive_side(
    state: LOBSTERState,
    execution_price: Decimal,
) -> str | None:
    if (
        state.best_bid is not None
        and execution_price == state.best_bid[0]
    ):
        return "BUY"

    if (
        state.best_ask is not None
        and execution_price == state.best_ask[0]
    ):
        return "SELL"

    return None


def build_lobster_research_dataset(
    message_path: str | Path,
    orderbook_path: str | Path,
    symbol: str,
    trading_date: date,
    queue_model: QueueModel,
    order_quantity: int,
    levels: int = 10,
    max_observations: int | None = None,
) -> tuple[ResearchObservation, ...]:
    events = tuple(
        read_messages(
            message_path,
            symbol,
            trading_date,
        )
    )

    snapshots = tuple(
        read_orderbooks(
            orderbook_path,
            levels=levels,
        )
    )

    if not events or not snapshots:
        return ()

    aligned_count = min(
        len(events),
        len(snapshots),
    )

    if aligned_count < 2:
        return ()

    states = _build_states(
        events[:aligned_count],
        snapshots[:aligned_count],
        symbol,
    )

    mid_prices = tuple(
        MidPricePoint(
            timestamp=state.timestamp,
            mid_price=state.mid_price,
        )
        for state in states
        if state.mid_price is not None
    )

    observations: list[ResearchObservation] = []

    for index in range(1, aligned_count):
        event = events[index]

        if not isinstance(event, OrderExecuteEvent):
            continue

        previous_state = states[index - 1]

        if (
            previous_state.best_bid is None
            or previous_state.best_ask is None
        ):
            continue

        if event.execution_price is None:
            continue

        side = _find_passive_side(
            previous_state,
            event.execution_price,
        )

        if side is None:
            continue

        feature_snapshot = build_feature_snapshot_from_book(
            previous_state.bids,
            previous_state.asks,
        )

        opportunity = QuoteOpportunity(
            timestamp=previous_state.timestamp,
            symbol=symbol,
            side=side,
            quote_price=event.execution_price,
            features=feature_snapshot,
        )

        visible_quantity = (
            previous_state.best_bid[1]
            if side == "BUY"
            else previous_state.best_ask[1]
        )

        queue_ahead = queue_model.estimate(visible_quantity)

        allocation = allocate_execution(
            quantity_ahead=queue_ahead.quantity_ahead,
            execution_quantity=event.quantity,
        )

        hypothetical_fill = min(
            allocation.hypothetical_fill,
            order_quantity,
        )

        if hypothetical_fill <= 0:
            continue

        execution_id = f"{event.order_id}-{index}"

        historical_fill = HistoricalFill(
            opportunity=opportunity,
            timestamp=event.timestamp,
            side=OrderSide(side),
            price=event.execution_price,
            quantity=hypothetical_fill,
            order_id=event.order_id,
            execution_id=execution_id,
        )

        markout = calculate_fill_markout(
            fill_timestamp=historical_fill.timestamp,
            fill_price=historical_fill.price,
            side=historical_fill.side,
            mid_prices=mid_prices,
            horizons=HORIZONS,
        )

        observation = build_research_observation(
            fill=historical_fill,
            features=feature_snapshot,
            markout=markout,
        )

        observations.append(observation)

        if (
            max_observations is not None
            and len(observations) >= max_observations
        ):
            break

    return tuple(observations)