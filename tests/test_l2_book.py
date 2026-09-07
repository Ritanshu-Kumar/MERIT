from datetime import UTC, datetime
from decimal import Decimal

import pytest

from merit.book.l2_book import L2OrderBook
from merit.data.normalized import (
    MarketEventType,
    OrderAddEvent,
    OrderCancelEvent,
    OrderDeleteEvent,
    OrderExecuteEvent,
    OrderReplaceEvent,
)


TIMESTAMP = datetime(2026, 1, 2, tzinfo=UTC)


def add(
    order_id: int,
    side: str,
    quantity: int,
    price: str,
) -> OrderAddEvent:
    return OrderAddEvent(
        timestamp=TIMESTAMP,
        symbol="AAPL",
        event_type=MarketEventType.ADD,
        order_id=order_id,
        side=side,
        price=Decimal(price),
        quantity=quantity,
    )


def cancel(order_id: int, quantity: int) -> OrderCancelEvent:
    return OrderCancelEvent(
        timestamp=TIMESTAMP,
        symbol="AAPL",
        event_type=MarketEventType.CANCEL,
        order_id=order_id,
        quantity=quantity,
    )


def delete(order_id: int) -> OrderDeleteEvent:
    return OrderDeleteEvent(
        timestamp=TIMESTAMP,
        symbol="AAPL",
        event_type=MarketEventType.DELETE,
        order_id=order_id,
    )


def execute(
    order_id: int,
    quantity: int,
) -> OrderExecuteEvent:
    return OrderExecuteEvent(
        timestamp=TIMESTAMP,
        symbol="AAPL",
        event_type=MarketEventType.EXECUTE,
        order_id=order_id,
        quantity=quantity,
        execution_id=f"EXEC-{order_id}",
    )


def replace(
    old_order_id: int,
    new_order_id: int,
    quantity: int,
    price: str,
) -> OrderReplaceEvent:
    return OrderReplaceEvent(
        timestamp=TIMESTAMP,
        symbol="AAPL",
        event_type=MarketEventType.REPLACE,
        old_order_id=old_order_id,
        new_order_id=new_order_id,
        quantity=quantity,
        price=Decimal(price),
    )


def test_add_order() -> None:
    book = L2OrderBook("AAPL")

    book.apply(add(1, "BUY", 100, "150.00"))

    assert book.best_bid() == (Decimal("150.00"), 100)
    assert book.order_count() == 1


def test_multiple_orders_aggregate_at_same_price() -> None:
    book = L2OrderBook("AAPL")

    book.apply(add(1, "BUY", 100, "150.00"))
    book.apply(add(2, "BUY", 50, "150.00"))

    assert book.best_bid() == (Decimal("150.00"), 150)


def test_best_bid_and_ask() -> None:
    book = L2OrderBook("AAPL")

    book.apply(add(1, "BUY", 100, "150.00"))
    book.apply(add(2, "BUY", 50, "149.90"))
    book.apply(add(3, "SELL", 80, "150.10"))
    book.apply(add(4, "SELL", 40, "150.20"))

    assert book.best_bid() == (Decimal("150.00"), 100)
    assert book.best_ask() == (Decimal("150.10"), 80)


def test_mid_price() -> None:
    book = L2OrderBook("AAPL")

    book.apply(add(1, "BUY", 100, "150.00"))
    book.apply(add(2, "SELL", 100, "150.10"))

    assert book.mid_price() == Decimal("150.05")


def test_spread() -> None:
    book = L2OrderBook("AAPL")

    book.apply(add(1, "BUY", 100, "150.00"))
    book.apply(add(2, "SELL", 100, "150.10"))

    assert book.spread() == Decimal("0.10")


def test_partial_cancel_updates_level() -> None:
    book = L2OrderBook("AAPL")

    book.apply(add(1, "BUY", 100, "150.00"))
    book.apply(cancel(1, 40))

    assert book.best_bid() == (Decimal("150.00"), 60)
    assert book.order(1).quantity == 60


def test_full_cancel_removes_order() -> None:
    book = L2OrderBook("AAPL")

    book.apply(add(1, "BUY", 100, "150.00"))
    book.apply(cancel(1, 100))

    assert book.best_bid() is None
    assert book.order(1) is None


def test_delete_removes_full_order_quantity() -> None:
    book = L2OrderBook("AAPL")

    book.apply(add(1, "BUY", 100, "150.00"))
    book.apply(delete(1))

    assert book.best_bid() is None
    assert book.order_count() == 0


def test_execution_reduces_quantity() -> None:
    book = L2OrderBook("AAPL")

    book.apply(add(1, "SELL", 100, "150.10"))
    book.apply(execute(1, 30))

    assert book.best_ask() == (Decimal("150.10"), 70)


def test_full_execution_removes_order() -> None:
    book = L2OrderBook("AAPL")

    book.apply(add(1, "SELL", 100, "150.10"))
    book.apply(execute(1, 100))

    assert book.best_ask() is None
    assert book.order_count() == 0


def test_replace_moves_order() -> None:
    book = L2OrderBook("AAPL")

    book.apply(add(1, "BUY", 100, "150.00"))
    book.apply(replace(1, 2, 80, "149.90"))

    assert book.best_bid() == (Decimal("149.90"), 80)
    assert book.order(1) is None
    assert book.order(2).price == Decimal("149.90")
    assert book.order(2).quantity == 80


def test_depth_returns_sorted_levels() -> None:
    book = L2OrderBook("AAPL")

    book.apply(add(1, "BUY", 100, "150.00"))
    book.apply(add(2, "BUY", 50, "149.90"))
    book.apply(add(3, "BUY", 80, "149.80"))

    book.apply(add(4, "SELL", 60, "150.10"))
    book.apply(add(5, "SELL", 40, "150.20"))

    bids, asks = book.depth(2)

    assert bids == (
        (Decimal("150.00"), 100),
        (Decimal("149.90"), 50),
    )

    assert asks == (
        (Decimal("150.10"), 60),
        (Decimal("150.20"), 40),
    )


def test_unknown_order_rejected() -> None:
    book = L2OrderBook("AAPL")

    with pytest.raises(ValueError, match="Unknown order"):
        book.apply(delete(999))


def test_cross_symbol_event_rejected() -> None:
    book = L2OrderBook("AAPL")

    event = OrderAddEvent(
        timestamp=TIMESTAMP,
        symbol="MSFT",
        event_type=MarketEventType.ADD,
        order_id=1,
        side="BUY",
        price=Decimal("500.00"),
        quantity=100,
    )

    with pytest.raises(ValueError, match="does not match"):
        book.apply(event)


def test_negative_level_quantity_is_impossible() -> None:
    book = L2OrderBook("AAPL")

    book.apply(add(1, "BUY", 100, "150.00"))

    with pytest.raises(ValueError):
        book.apply(cancel(1, 101))