from decimal import Decimal

import pytest

from merit.portfolio.enums import OrderSide, OrderStatus, OrderType
from merit.portfolio.models import Order


def test_create_limit_order() -> None:
    order = Order(
        order_id="ORD-001",
        symbol="AAPL",
        side=OrderSide.BUY,
        quantity=100,
        order_type=OrderType.LIMIT,
        limit_price=Decimal("50.10"),
    )

    assert order.status == OrderStatus.CREATED
    assert order.quantity == 100
    assert order.filled_quantity == 0
    assert order.remaining_quantity == 100
    assert order.limit_price == Decimal("50.10")


def test_partial_fill_remaining_quantity() -> None:
    order = Order(
        order_id="ORD-002",
        symbol="AAPL",
        side=OrderSide.BUY,
        quantity=100,
        order_type=OrderType.LIMIT,
        limit_price=Decimal("50.10"),
        filled_quantity=40,
    )

    assert order.remaining_quantity == 60
    assert not order.is_fully_filled


def test_order_rejects_invalid_quantity() -> None:
    with pytest.raises(ValueError):
        Order(
            order_id="ORD-003",
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=0,
            order_type=OrderType.LIMIT,
            limit_price=Decimal("50.10"),
        )