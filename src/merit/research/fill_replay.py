from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from decimal import Decimal

from merit.data.normalized import NormalizedMarketEvent, TradeEvent
from merit.execution.simulator import ExecutionSimulator
from merit.portfolio.enums import OrderSide, OrderStatus, OrderType
from merit.portfolio.models import Fill, Order
from merit.research.opportunities import QuoteOpportunity


@dataclass(frozen=True)
class FillOpportunity:
    opportunity: QuoteOpportunity
    fill: Fill


def replay_opportunity(
    opportunity: QuoteOpportunity,
    future_events: Iterable[NormalizedMarketEvent],
    execution: ExecutionSimulator,
    quantity: int,
) -> Iterator[FillOpportunity]:
    if quantity <= 0:
        raise ValueError("quantity must be positive")

    order = Order(
        order_id=f"research-{opportunity.symbol}-{opportunity.timestamp.isoformat()}",
        symbol=opportunity.symbol,
        side=opportunity.side,
        quantity=quantity,
        order_type=OrderType.LIMIT,
        limit_price=opportunity.quote_price,
        status=OrderStatus.CREATED,
        created_at=opportunity.timestamp,
    )

    visible_quantity = _visible_quantity(opportunity)

    execution.submit(
        order,
        visible_quantity=visible_quantity,
    )

    for event in future_events:
        if not isinstance(event, TradeEvent):
            continue

        if event.symbol != opportunity.symbol:
            continue

        if event.timestamp <= opportunity.timestamp:
            continue

        result = execution.process_trade(event)

        for fill in result.fills:
            yield FillOpportunity(
                opportunity=opportunity,
                fill=fill,
            )


def _visible_quantity(opportunity: QuoteOpportunity) -> int:
    if opportunity.side == OrderSide.BUY:
        return opportunity.features.bid_size

    return opportunity.features.ask_size