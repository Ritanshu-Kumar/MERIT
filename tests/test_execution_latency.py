from datetime import datetime, timedelta
from decimal import Decimal

from merit.data.normalized import MarketEventType, TradeEvent
from merit.execution.latency import LatencyModel
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


def make_trade(timestamp: datetime) -> TradeEvent:
    return TradeEvent(
        timestamp=timestamp,
        event_type=MarketEventType.TRADE,
        symbol="AAPL",
        price=Decimal("99.99"),
        quantity=100,
    )


def test_trade_before_activation_does_not_fill() -> None:
    simulator = ExecutionSimulator(
        latency_model=LatencyModel(
            submit_latency=timedelta(milliseconds=500)
        )
    )

    order = make_order()
    simulator.submit(order)

    result = simulator.process_trade(
        make_trade(
            datetime(2012, 6, 21, 9, 30, 0, 499_999)
        )
    )

    assert result.fills == ()
    assert order.status == OrderStatus.CREATED


def test_trade_at_activation_can_fill() -> None:
    simulator = ExecutionSimulator(
        latency_model=LatencyModel(
            submit_latency=timedelta(milliseconds=500)
        )
    )

    order = make_order()
    simulator.submit(order)

    result = simulator.process_trade(
        make_trade(
            datetime(2012, 6, 21, 9, 30, 0, 500_000)
        )
    )

    assert len(result.fills) == 1
    assert result.fills[0].quantity == 100
    assert order.status == OrderStatus.FILLED