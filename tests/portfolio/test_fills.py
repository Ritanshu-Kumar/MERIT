from datetime import UTC, datetime
from decimal import Decimal

import pytest

from merit.portfolio.enums import OrderSide
from merit.portfolio.models import Fill


def test_fill_notional() -> None:
    fill = Fill(
        fill_id="FILL-001",
        order_id="ORD-001",
        timestamp=datetime.now(UTC),
        symbol="AAPL",
        side=OrderSide.BUY,
        quantity=100,
        price=Decimal("50.10"),
        fee=Decimal("0.50"),
    )

    assert fill.notional == Decimal("5010.00")


def test_fill_rejects_negative_fee() -> None:
    with pytest.raises(ValueError):
        Fill(
            fill_id="FILL-002",
            order_id="ORD-001",
            timestamp=datetime.now(UTC),
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            price=Decimal("50.10"),
            fee=Decimal("-1.00"),
        )