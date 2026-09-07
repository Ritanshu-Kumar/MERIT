from decimal import Decimal
from datetime import datetime

from merit.book.l2_book import L2OrderBook
from merit.data.normalized import MarketEventType, OrderAddEvent
from merit.features.depth import depth_weighted_imbalance, level_imbalance
from merit.portfolio.enums import OrderSide


def add_order(
    book: L2OrderBook,
    order_id: int,
    side: OrderSide,
    price: str,
    quantity: int,
) -> None:
    book.apply(
        OrderAddEvent(
            timestamp=datetime(2012, 6, 21),
            event_type=MarketEventType.ADD,
            symbol="AAPL",
            order_id=order_id,
            side=side,
            quantity=quantity,
            price=Decimal(price),
        )
    )


def test_level_imbalance() -> None:
    book = L2OrderBook("AAPL")

    add_order(book, 1, OrderSide.BUY, "100.00", 200)
    add_order(book, 2, OrderSide.BUY, "99.90", 100)
    add_order(book, 3, OrderSide.SELL, "100.10", 100)
    add_order(book, 4, OrderSide.SELL, "100.20", 100)

    assert level_imbalance(book, 2) == Decimal("0.2")


def test_level_imbalance_respects_requested_depth() -> None:
    book = L2OrderBook("AAPL")

    add_order(book, 1, OrderSide.BUY, "100.00", 200)
    add_order(book, 2, OrderSide.BUY, "99.90", 100)
    add_order(book, 3, OrderSide.SELL, "100.10", 100)
    add_order(book, 4, OrderSide.SELL, "100.20", 500)

    assert level_imbalance(book, 1) == Decimal("1") / Decimal("3")


def test_depth_weighted_imbalance() -> None:
    book = L2OrderBook("AAPL")

    add_order(book, 1, OrderSide.BUY, "100.00", 200)
    add_order(book, 2, OrderSide.BUY, "99.90", 100)
    add_order(book, 3, OrderSide.SELL, "100.10", 100)
    add_order(book, 4, OrderSide.SELL, "100.20", 100)

    expected = Decimal("1") / Decimal("4")

    assert depth_weighted_imbalance(book, 2) == expected


def test_empty_book() -> None:
    book = L2OrderBook("AAPL")

    assert level_imbalance(book, 5) is None
    assert depth_weighted_imbalance(book, 5) is None


def test_invalid_levels() -> None:
    book = L2OrderBook("AAPL")

    try:
        level_imbalance(book, 0)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError")

    try:
        depth_weighted_imbalance(book, 0)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError")