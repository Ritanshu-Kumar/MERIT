from datetime import datetime
from decimal import Decimal

import pytest

from merit.data.normalized import MarketEventType, TradeEvent
from merit.execution.simulator import ExecutionSimulator
from merit.portfolio.enums import OrderSide, OrderStatus, OrderType
from merit.portfolio.models import Order


def make_order(
    order_id: int,
    side: OrderSide,
    price: str,
    quantity: int,
) -> Order:
    return Order(
        order_id=order_id,
        symbol="AAPL",
        side=side,
        quantity=quantity,
        order_type=OrderType.LIMIT,
        limit_price=Decimal(price),
        status=OrderStatus.CREATED,
        created_at=datetime(2012, 6, 21),
    )


def make_trade(price: str, quantity: int) -> TradeEvent:
    return TradeEvent(
        timestamp=datetime(2012, 6, 21, 9, 30),
        event_type=MarketEventType.TRADE,
        symbol="AAPL",
        price=Decimal(price),
        quantity=quantity,
    )


def test_buy_limit_fills_when_trade_reaches_price() -> None:
    simulator = ExecutionSimulator()
    order = make_order(1, OrderSide.BUY, "100.00", 100)

    simulator.submit(order)

    result = simulator.process_trade(make_trade("99.99", 100))

    assert len(result.fills) == 1
    assert result.fills[0].quantity == 100
    assert result.fills[0].price == Decimal("100.00")
    assert order.status == OrderStatus.FILLED


def test_sell_limit_fills_when_trade_reaches_price() -> None:
    simulator = ExecutionSimulator()
    order = make_order(1, OrderSide.SELL, "100.00", 100)

    simulator.submit(order)

    result = simulator.process_trade(make_trade("100.01", 100))

    assert len(result.fills) == 1
    assert result.fills[0].quantity == 100
    assert result.fills[0].price == Decimal("100.00")
    assert order.status == OrderStatus.FILLED


def test_trade_not_crossing_order_does_not_fill() -> None:
    simulator = ExecutionSimulator()
    order = make_order(1, OrderSide.BUY, "100.00", 100)

    simulator.submit(order)

    result = simulator.process_trade(make_trade("100.01", 100))

    assert result.fills == ()
    assert order.status == OrderStatus.CREATED


def test_partial_fill() -> None:
    simulator = ExecutionSimulator()
    order = make_order(1, OrderSide.BUY, "100.00", 100)

    simulator.submit(order)

    result = simulator.process_trade(make_trade("99.99", 40))

    assert len(result.fills) == 1
    assert result.fills[0].quantity == 40
    assert order.filled_quantity == 40
    assert order.remaining_quantity == 60
    assert order.status == OrderStatus.PARTIALLY_FILLED


def test_order_fills_across_multiple_trades() -> None:
    simulator = ExecutionSimulator()
    order = make_order(1, OrderSide.BUY, "100.00", 100)

    simulator.submit(order)

    first = simulator.process_trade(make_trade("99.99", 40))
    second = simulator.process_trade(make_trade("100.00", 60))

    assert first.fills[0].quantity == 40
    assert second.fills[0].quantity == 60
    assert order.filled_quantity == 100
    assert order.remaining_quantity == 0
    assert order.status == OrderStatus.FILLED


def test_filled_order_does_not_fill_again() -> None:
    simulator = ExecutionSimulator()
    order = make_order(1, OrderSide.BUY, "100.00", 100)

    simulator.submit(order)

    simulator.process_trade(make_trade("99.99", 100))
    result = simulator.process_trade(make_trade("99.99", 100))

    assert result.fills == ()


def test_cancelled_order_does_not_fill() -> None:
    simulator = ExecutionSimulator()
    order = make_order(1, OrderSide.BUY, "100.00", 100)

    simulator.submit(order)
    simulator.cancel(1)

    result = simulator.process_trade(make_trade("99.99", 100))

    assert result.fills == ()
    assert order.status == OrderStatus.CANCELLED


def test_duplicate_submission_rejected() -> None:
    simulator = ExecutionSimulator()
    order = make_order(1, OrderSide.BUY, "100.00", 100)

    simulator.submit(order)

    with pytest.raises(ValueError):
        simulator.submit(order)


def test_unknown_cancel_rejected() -> None:
    simulator = ExecutionSimulator()

    with pytest.raises(KeyError):
        simulator.cancel(999)