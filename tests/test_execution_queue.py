from datetime import datetime
from decimal import Decimal

from merit.data.normalized import MarketEventType, TradeEvent
from merit.execution.queue_model import QueueModel
from merit.execution.simulator import ExecutionSimulator
from merit.portfolio.enums import OrderSide, OrderStatus, OrderType
from merit.portfolio.models import Order


def make_order() -> Order:
    return Order(
        order_id=1,
        symbol="AAPL",
        side=OrderSide.BUY,
        quantity=100,
        order_type=OrderType.LIMIT,
        limit_price=Decimal("100.00"),
        status=OrderStatus.CREATED,
        created_at=datetime(2012, 6, 21, 9, 30),
    )


def make_trade(quantity: int) -> TradeEvent:
    return TradeEvent(
        timestamp=datetime(2012, 6, 21, 9, 31),
        event_type=MarketEventType.TRADE,
        symbol="AAPL",
        price=Decimal("99.99"),
        quantity=quantity,
    )


def test_queue_ahead_delays_fill() -> None:
    simulator = ExecutionSimulator(
        queue_model=QueueModel(ahead_fraction=1.0)
    )

    order = make_order()
    simulator.submit(order, visible_quantity=50)

    first = simulator.process_trade(make_trade(30))
    second = simulator.process_trade(make_trade(20))
    third = simulator.process_trade(make_trade(40))

    assert first.fills == ()
    assert second.fills == ()
    assert len(third.fills) == 1
    assert third.fills[0].quantity == 40


def test_no_queue_ahead_allows_fill() -> None:
    simulator = ExecutionSimulator(
        queue_model=QueueModel(ahead_fraction=1.0)
    )

    order = make_order()
    simulator.submit(order, visible_quantity=0)

    result = simulator.process_trade(make_trade(40))

    assert len(result.fills) == 1
    assert result.fills[0].quantity == 40


def test_queue_and_participation_both_apply() -> None:
    from merit.execution.participation import ParticipationModel

    simulator = ExecutionSimulator(
        participation_model=ParticipationModel(rate=0.5),
        queue_model=QueueModel(ahead_fraction=1.0),
    )

    order = make_order()
    simulator.submit(order, visible_quantity=50)

    result = simulator.process_trade(make_trade(200))

    assert len(result.fills) == 1
    assert result.fills[0].quantity == 50