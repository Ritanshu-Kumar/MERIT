from collections.abc import Iterable, Iterator
from dataclasses import dataclass

from merit.book.l2_book import L2OrderBook
from merit.data.normalized import (
    NormalizedMarketEvent,
    OrderExecuteEvent,
)
from merit.execution.queue import QueuePosition
from merit.execution.queue_model import QueueModel
from merit.research.historical_fills import (
    HistoricalFill,
    detect_historical_fill,
)
from merit.research.opportunities import QuoteOpportunity


@dataclass
class ActiveOpportunity:
    opportunity: QuoteOpportunity
    queue: QueuePosition
    remaining_quantity: int


def replay_queue_aware_fills(
    opportunities: Iterable[QuoteOpportunity],
    events: Iterable[NormalizedMarketEvent],
    queue_model: QueueModel,
    order_quantity: int,
) -> Iterator[HistoricalFill]:
    if order_quantity <= 0:
        raise ValueError("order_quantity must be positive")

    opportunity_iter = iter(opportunities)
    pending: list[QuoteOpportunity] = []
    active: list[ActiveOpportunity] = []

    next_opportunity = next(opportunity_iter, None)
    book: L2OrderBook | None = None

    for event in events:
        if book is None:
            book = L2OrderBook(event.symbol)

        while (
            next_opportunity is not None
            and next_opportunity.timestamp < event.timestamp
        ):
            pending.append(next_opportunity)
            next_opportunity = next(opportunity_iter, None)

        if pending:
            for opportunity in pending:
                visible_quantity = _visible_quantity(opportunity)
                active.append(
                    ActiveOpportunity(
                        opportunity=opportunity,
                        queue=queue_model.estimate(visible_quantity),
                        remaining_quantity=order_quantity,
                    )
                )
            pending.clear()

        if isinstance(event, OrderExecuteEvent):
            updated_active: list[ActiveOpportunity] = []

            for candidate in active:
                if candidate.remaining_quantity <= 0:
                    continue

                if candidate.opportunity.symbol != event.symbol:
                    updated_active.append(candidate)
                    continue

                historical_fill = detect_historical_fill(
                    book=book,
                    opportunity=candidate.opportunity,
                    event=event,
                )

                if historical_fill is None:
                    updated_active.append(candidate)
                    continue

                allocation = _allocate(
                    candidate,
                    event.quantity,
                )

                if allocation > 0:
                    yield HistoricalFill(
                        opportunity=candidate.opportunity,
                        timestamp=historical_fill.timestamp,
                        side=historical_fill.side,
                        price=historical_fill.price,
                        quantity=allocation,
                        order_id=historical_fill.order_id,
                        execution_id=historical_fill.execution_id,
                    )

                if candidate.remaining_quantity > 0:
                    updated_active.append(candidate)

            active = updated_active

        book.apply(event)


def _visible_quantity(opportunity: QuoteOpportunity) -> int:
    if opportunity.side.value == "BUY":
        return opportunity.features.bid_size

    return opportunity.features.ask_size


def _allocate(
    candidate: ActiveOpportunity,
    execution_quantity: int,
) -> int:
    available = candidate.queue.consume(execution_quantity)

    fill_quantity = min(
        candidate.remaining_quantity,
        available,
    )

    candidate.remaining_quantity -= fill_quantity

    return fill_quantity