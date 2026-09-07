from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from merit.book.l2_book import L2OrderBook
from merit.data.normalized import (
    NormalizedMarketEvent,
    OrderExecuteEvent,
)
from merit.portfolio.enums import OrderSide
from merit.research.opportunities import QuoteOpportunity


@dataclass(frozen=True)
class HistoricalFill:
    opportunity: QuoteOpportunity
    timestamp: datetime
    side: OrderSide
    price: Decimal
    quantity: int
    order_id: int
    execution_id: str


def detect_historical_fill(
    book: L2OrderBook,
    opportunity: QuoteOpportunity,
    event: OrderExecuteEvent,
) -> HistoricalFill | None:
    if event.symbol != opportunity.symbol:
        return None

    if event.order_id == 0:
        return None

    order = book.order(event.order_id)

    if order is None:
        return None

    if order.side != opportunity.side:
        return None

    if order.price != opportunity.quote_price:
        return None

    if event.quantity <= 0:
        return None

    return HistoricalFill(
        opportunity=opportunity,
        timestamp=event.timestamp,
        side=order.side,
        price=order.price,
        quantity=event.quantity,
        order_id=event.order_id,
        execution_id=event.execution_id,
    )


def process_execution_event(
    book: L2OrderBook,
    opportunity: QuoteOpportunity,
    event: NormalizedMarketEvent,
) -> HistoricalFill | None:
    if not isinstance(event, OrderExecuteEvent):
        book.apply(event)
        return None

    historical_fill = detect_historical_fill(
        book=book,
        opportunity=opportunity,
        event=event,
    )

    book.apply(event)

    return historical_fill