from datetime import UTC, datetime
from decimal import Decimal

import pytest

from merit.simulator.events import (
    BookUpdateEvent,
    EventType,
    TradeEvent,
)


def test_trade_event() -> None:
    event = TradeEvent(
        timestamp=datetime.now(UTC),
        symbol="AAPL",
        event_type=EventType.TRADE,
        price=Decimal("50.10"),
        quantity=100,
    )

    assert event.symbol == "AAPL"
    assert event.event_type == EventType.TRADE
    assert event.price == Decimal("50.10")
    assert event.quantity == 100


def test_book_update_event() -> None:
    event = BookUpdateEvent(
        timestamp=datetime.now(UTC),
        symbol="AAPL",
        event_type=EventType.BOOK_UPDATE,
        side="BID",
        price=Decimal("50.00"),
        quantity=Decimal("100"),
    )

    assert event.side == "BID"
    assert event.price == Decimal("50.00")
    assert event.quantity == Decimal("100")


def test_trade_event_rejects_invalid_price() -> None:
    with pytest.raises(ValueError):
        TradeEvent(
            timestamp=datetime.now(UTC),
            symbol="AAPL",
            event_type=EventType.TRADE,
            price=Decimal("0"),
            quantity=100,
        )


def test_book_update_rejects_invalid_side() -> None:
    with pytest.raises(ValueError):
        BookUpdateEvent(
            timestamp=datetime.now(UTC),
            symbol="AAPL",
            event_type=EventType.BOOK_UPDATE,
            side="INVALID",
            price=Decimal("50.00"),
            quantity=Decimal("100"),
        )


def test_book_update_allows_zero_quantity() -> None:
    event = BookUpdateEvent(
        timestamp=datetime.now(UTC),
        symbol="AAPL",
        event_type=EventType.BOOK_UPDATE,
        side="ASK",
        price=Decimal("50.10"),
        quantity=Decimal("0"),
    )

    assert event.quantity == Decimal("0")