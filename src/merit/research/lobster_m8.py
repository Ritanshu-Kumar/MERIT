from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path

from merit.data.lobster import OrderExecuteEvent, read_messages
from merit.data.lobster_orderbook import read_orderbooks
from merit.execution.queue_model import QueueModel
from merit.features.snapshot import (
    FeatureSnapshot,
    build_feature_snapshot_from_levels,
)
from merit.portfolio.enums import OrderSide
from merit.research.dataset import (
    ResearchObservation,
    build_research_observation,
)
from merit.research.fill_allocation import allocate_execution
from merit.research.historical_fills import HistoricalFill
from merit.research.opportunities import QuoteOpportunity
from merit.research.targets import (
    MidPriceIndex,
    MidPricePoint,
    build_mid_price_index,
    calculate_fill_markout,
)

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


def build_lobster_states(
    message_path: str | Path,
    orderbook_path: str | Path,
    symbol: str,
    trading_date: date,
    levels: int = 10,
) -> tuple[LOBSTERState, ...]:
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

    return _build_states(
        events[:aligned_count],
        snapshots[:aligned_count],
        symbol,
    )

def _build_states(
    events: tuple,
    snapshots: tuple[
        tuple[tuple[Decimal, int], ...],
        tuple[tuple[Decimal, int], ...],
    ],
    symbol: str,
) -> tuple[LOBSTERState, ...]:
    aligned_count = min(
        len(events),
        len(snapshots),
    )

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

    states = build_lobster_states(
        message_path=message_path,
        orderbook_path=orderbook_path,
        symbol=symbol,
        trading_date=trading_date,
        levels=levels,
    )

    if not events or not states:
        return ()

    aligned_count = len(states)

    mid_prices = tuple(
        MidPricePoint(
            timestamp=state.timestamp,
            mid_price=state.mid_price,
        )
        for state in states
        if state.mid_price is not None
    )

    mid_price_index: MidPriceIndex = build_mid_price_index(
        mid_prices
    )

    pending_fills: list[HistoricalFill] = []

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

        feature_snapshot = build_feature_snapshot_from_levels(
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

        queue_ahead = queue_model.estimate(
            visible_quantity
        )

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

        pending_fills.append(
            HistoricalFill(
                opportunity=opportunity,
                timestamp=event.timestamp,
                side=OrderSide(side),
                price=event.execution_price,
                quantity=hypothetical_fill,
                order_id=event.order_id,
                execution_id=execution_id,
            )
        )

    observations: list[ResearchObservation] = []

    for fill in pending_fills:
        markout = calculate_fill_markout(
            fill_timestamp=fill.timestamp,
            fill_price=fill.price,
            side=fill.side,
            mid_price_index=mid_price_index,
            horizons=HORIZONS,
        )

        observations.append(
            build_research_observation(
                fill=fill,
                features=fill.opportunity.features,
                markout=markout,
            )
        )

        if (
            max_observations is not None
            and len(observations) >= max_observations
        ):
            break

    return tuple(observations)