from decimal import Decimal
from datetime import datetime

from merit.book.l2_book import L2OrderBook
from merit.data.normalized import MarketEventType, OrderAddEvent
from merit.features.microstructure import calculate_features
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


def test_empty_book() -> None:
    book = L2OrderBook("AAPL")

    features = calculate_features(book)

    assert features.mid_price is None
    assert features.spread is None
    assert features.relative_spread is None
    assert features.bid_size == 0
    assert features.ask_size == 0
    assert features.imbalance is None
    assert features.microprice is None


def test_balanced_book() -> None:
    book = L2OrderBook("AAPL")

    add_order(book, 1, OrderSide.BUY, "100.00", 100)
    add_order(book, 2, OrderSide.SELL, "100.10", 100)

    features = calculate_features(book)

    assert features.best_bid == Decimal("100.00")
    assert features.best_ask == Decimal("100.10")
    assert features.mid_price == Decimal("100.05")
    assert features.spread == Decimal("0.10")
    assert features.bid_size == 100
    assert features.ask_size == 100
    assert features.imbalance == Decimal("0")
    assert features.microprice == Decimal("100.05")


def test_bid_heavy_book() -> None:
    book = L2OrderBook("AAPL")

    add_order(book, 1, OrderSide.BUY, "100.00", 300)
    add_order(book, 2, OrderSide.SELL, "100.10", 100)

    features = calculate_features(book)

    assert features.imbalance == Decimal("0.5")
    assert features.microprice == Decimal("100.075")


def test_ask_heavy_book() -> None:
    book = L2OrderBook("AAPL")

    add_order(book, 1, OrderSide.BUY, "100.00", 100)
    add_order(book, 2, OrderSide.SELL, "100.10", 300)

    features = calculate_features(book)

    assert features.imbalance == Decimal("-0.5")
    assert features.microprice == Decimal("100.025")


def test_relative_spread() -> None:
    book = L2OrderBook("AAPL")

    add_order(book, 1, OrderSide.BUY, "100.00", 100)
    add_order(book, 2, OrderSide.SELL, "100.10", 100)

    features = calculate_features(book)

    assert features.relative_spread == Decimal("0.10") / Decimal("100.05")