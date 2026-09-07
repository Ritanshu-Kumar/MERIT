from datetime import UTC, datetime
from decimal import Decimal

import pytest

from merit.data.normalized import (
    MarketEventType,
    OrderAddEvent,
    OrderCancelEvent,
    OrderDeleteEvent,
    OrderExecuteEvent,
    OrderReplaceEvent,
    SystemEvent,
    TradeEvent,
)


TIMESTAMP = datetime(2026, 1, 1, 14, 30, tzinfo=UTC)


def test_order_add_event() -> None:
    event = OrderAddEvent(
        timestamp=TIMESTAMP,
        symbol="AAPL",
        event_type=MarketEventType.ADD,
        order_id=1001,
        side="BUY",
        price=Decimal("150.25"),
        quantity=100,
    )

    assert event.order_id == 1001
    assert event.side == "BUY"
    assert event.price == Decimal("150.25")
    assert event.quantity == 100


def test_order_execute_event() -> None:
    event = OrderExecuteEvent(
        timestamp=TIMESTAMP,
        symbol="AAPL",
        event_type=MarketEventType.EXECUTE,
        order_id=1001,
        quantity=50,
        execution_id="E001",
    )

    assert event.order_id == 1001
    assert event.quantity == 50
    assert event.execution_id == "E001"


def test_cancel_event() -> None:
    event = OrderCancelEvent(
        timestamp=TIMESTAMP,
        symbol="AAPL",
        event_type=MarketEventType.CANCEL,
        order_id=1001,
        quantity=25,
    )

    assert event.quantity == 25


def test_delete_event() -> None:
    event = OrderDeleteEvent(
        timestamp=TIMESTAMP,
        symbol="AAPL",
        event_type=MarketEventType.DELETE,
        order_id=1001,
    )

    assert event.order_id == 1001


def test_replace_event() -> None:
    event = OrderReplaceEvent(
        timestamp=TIMESTAMP,
        symbol="AAPL",
        event_type=MarketEventType.REPLACE,
        old_order_id=1001,
        new_order_id=2001,
        quantity=150,
        price=Decimal("150.30"),
    )

    assert event.old_order_id == 1001
    assert event.new_order_id == 2001
    assert event.quantity == 150


def test_trade_event() -> None:
    event = TradeEvent(
        timestamp=TIMESTAMP,
        symbol="AAPL",
        event_type=MarketEventType.TRADE,
        price=Decimal("150.50"),
        quantity=75,
    )

    assert event.price == Decimal("150.50")
    assert event.quantity == 75


def test_system_event() -> None:
    event = SystemEvent(
        timestamp=TIMESTAMP,
        symbol="AAPL",
        event_type=MarketEventType.SYSTEM,
        code="START",
    )

    assert event.code == "START"


def test_order_add_rejects_invalid_side() -> None:
    with pytest.raises(ValueError):
        OrderAddEvent(
            timestamp=TIMESTAMP,
            symbol="AAPL",
            event_type=MarketEventType.ADD,
            order_id=1001,
            side="INVALID",
            price=Decimal("150.25"),
            quantity=100,
        )


def test_order_execute_rejects_invalid_quantity() -> None:
    with pytest.raises(ValueError):
        OrderExecuteEvent(
            timestamp=TIMESTAMP,
            symbol="AAPL",
            event_type=MarketEventType.EXECUTE,
            order_id=1001,
            quantity=0,
            execution_id="E001",
        )