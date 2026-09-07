from datetime import datetime, timedelta
from decimal import Decimal

import pytest

from merit.data.normalized import (
    MarketEventType,
    OrderAddEvent,
    OrderExecuteEvent,
)
from merit.execution.queue_model import QueueModel
from merit.features.snapshot import FeatureSnapshot
from merit.portfolio.enums import OrderSide
from merit.research.opportunities import QuoteOpportunity
from merit.research.queue_aware_replay import replay_queue_aware_fills


BASE = datetime(2012, 6, 21, 9, 30)


def make_features(
    bid_size: int = 100,
    ask_size: int = 100,
) -> FeatureSnapshot:
    return FeatureSnapshot(
        mid_price=Decimal("100.05"),
        spread=Decimal("0.10"),
        relative_spread=Decimal("0.001"),
        bid_size=bid_size,
        ask_size=ask_size,
        imbalance=Decimal("0"),
        microprice=Decimal("100.05"),
        imbalance_l5=Decimal("0"),
        imbalance_l10=Decimal("0"),
        weighted_imbalance_l5=Decimal("0"),
        weighted_imbalance_l10=Decimal("0"),
    )


def make_opportunity(
    timestamp: datetime = BASE,
    side: OrderSide = OrderSide.BUY,
    price: str = "100.00",
    bid_size: int = 100,
) -> QuoteOpportunity:
    return QuoteOpportunity(
        timestamp=timestamp,
        symbol="AAPL",
        side=side,
        quote_price=Decimal(price),
        features=make_features(bid_size=bid_size),
    )


def make_add(
    order_id: int,
    timestamp: datetime,
    price: str = "100.00",
    quantity: int = 100,
) -> OrderAddEvent:
    return OrderAddEvent(
        timestamp=timestamp,
        symbol="AAPL",
        event_type=MarketEventType.ADD,
        order_id=order_id,
        side=OrderSide.BUY,
        price=Decimal(price),
        quantity=quantity,
    )


def make_execute(
    quantity: int,
    timestamp: datetime,
    order_id: int = 1,
) -> OrderExecuteEvent:
    return OrderExecuteEvent(
        timestamp=timestamp,
        symbol="AAPL",
        event_type=MarketEventType.EXECUTE,
        order_id=order_id,
        quantity=quantity,
        execution_id=f"exec-{order_id}-{timestamp.isoformat()}",
        execution_price=Decimal("100.00"),
    )


def test_queue_must_be_consumed_before_hypothetical_fill() -> None:
    opportunity = make_opportunity(bid_size=100)

    events = (
        make_add(1, BASE),
        make_execute(40, BASE + timedelta(seconds=1)),
        make_execute(60, BASE + timedelta(seconds=2)),
    )

    fills = list(
        replay_queue_aware_fills(
            opportunities=(opportunity,),
            events=events,
            queue_model=QueueModel(ahead_fraction=1.0),
            order_quantity=100,
        )
    )

    assert fills == []


def test_hypothetical_order_quantity_caps_fill() -> None:
    opportunity = make_opportunity(bid_size=0)

    events = (
        make_add(1, BASE),
        make_execute(100, BASE + timedelta(seconds=1)),
    )

    fills = list(
        replay_queue_aware_fills(
            opportunities=(opportunity,),
            events=events,
            queue_model=QueueModel(ahead_fraction=0.0),
            order_quantity=25,
        )
    )

    assert len(fills) == 1
    assert fills[0].quantity == 25


def test_multiple_executions_can_fill_same_hypothetical_order() -> None:
    opportunity = make_opportunity(bid_size=0)

    events = (
        make_add(1, BASE),
        make_execute(30, BASE + timedelta(seconds=1)),
        make_execute(20, BASE + timedelta(seconds=2)),
    )

    fills = list(
        replay_queue_aware_fills(
            opportunities=(opportunity,),
            events=events,
            queue_model=QueueModel(ahead_fraction=0.0),
            order_quantity=100,
        )
    )

    assert [fill.quantity for fill in fills] == [30, 20]


def test_execution_at_different_price_does_not_fill() -> None:
    opportunity = make_opportunity(
        bid_size=0,
        price="99.90",
    )

    events = (
        make_add(
            1,
            BASE,
            price="100.00",
            quantity=100,
        ),
        make_execute(
            50,
            BASE + timedelta(seconds=1),
        ),
    )

    fills = list(
        replay_queue_aware_fills(
            opportunities=(opportunity,),
            events=events,
            queue_model=QueueModel(ahead_fraction=0.0),
            order_quantity=100,
        )
    )

    assert fills == []


def test_future_opportunity_is_not_activated_early() -> None:
    opportunity = make_opportunity(
        timestamp=BASE + timedelta(seconds=2),
        bid_size=0,
    )

    events = (
        make_add(1, BASE),
        make_execute(100, BASE + timedelta(seconds=1)),
    )

    fills = list(
        replay_queue_aware_fills(
            opportunities=(opportunity,),
            events=events,
            queue_model=QueueModel(ahead_fraction=0.0),
            order_quantity=100,
        )
    )

    assert fills == []


def test_invalid_order_quantity() -> None:
    with pytest.raises(ValueError):
        list(
            replay_queue_aware_fills(
                opportunities=(),
                events=(),
                queue_model=QueueModel(),
                order_quantity=0,
            )
        )