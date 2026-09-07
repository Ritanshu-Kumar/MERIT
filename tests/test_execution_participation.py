from datetime import datetime
from decimal import Decimal

from merit.data.normalized import MarketEventType, TradeEvent
from merit.execution.participation import ParticipationModel
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


def test_participation_limits_fill() -> None:
    simulator = ExecutionSimulator(
        participation_model=ParticipationModel(rate=0.10)
    )

    order = make_order()
    simulator.submit(order)

    result = simulator.process_trade(make_trade(500))

    assert len(result.fills) == 1
    assert result.fills[0].quantity == 50
    assert order.filled_quantity == 50
    assert order.remaining_quantity == 50
    assert order.status == OrderStatus.PARTIALLY_FILLED


def test_order_size_still_limits_fill() -> None:
    simulator = ExecutionSimulator(
        participation_model=ParticipationModel(rate=0.50)
    )

    order = make_order()
    simulator.submit(order)

    result = simulator.process_trade(make_trade(500))

    assert len(result.fills) == 1
    assert result.fills[0].quantity == 100
    assert order.status == OrderStatus.FILLED


def test_zero_participation_fill_does_not_create_fill() -> None:
    simulator = ExecutionSimulator(
        participation_model=ParticipationModel(rate=0.01)
    )

    order = make_order()
    simulator.submit(order)

    result = simulator.process_trade(make_trade(50))

    assert result.fills == ()
    assert order.filled_quantity == 0
    assert order.status == OrderStatus.CREATED