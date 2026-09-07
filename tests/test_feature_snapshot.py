from decimal import Decimal
from datetime import datetime

from merit.book.l2_book import L2OrderBook
from merit.data.normalized import MarketEventType, OrderAddEvent
from merit.features.snapshot import build_feature_snapshot
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


def test_feature_snapshot() -> None:
    book = L2OrderBook("AAPL")

    add_order(book, 1, OrderSide.BUY, "100.00", 200)
    add_order(book, 2, OrderSide.BUY, "99.90", 100)
    add_order(book, 3, OrderSide.BUY, "99.80", 100)
    add_order(book, 4, OrderSide.SELL, "100.10", 100)
    add_order(book, 5, OrderSide.SELL, "100.20", 100)
    add_order(book, 6, OrderSide.SELL, "100.30", 100)

    snapshot = build_feature_snapshot(book)

    assert snapshot.mid_price == Decimal("100.05")
    assert snapshot.spread == Decimal("0.10")
    assert snapshot.bid_size == 200
    assert snapshot.ask_size == 100
    assert snapshot.imbalance == Decimal("1") / Decimal("3")
    assert snapshot.microprice == Decimal("100.0666666666666666666666667")

    assert snapshot.imbalance_l5 == Decimal("1") / Decimal("7")
    assert snapshot.imbalance_l10 == snapshot.imbalance_l5

    assert snapshot.weighted_imbalance_l5 is not None
    assert snapshot.weighted_imbalance_l10 == snapshot.weighted_imbalance_l5


def test_empty_snapshot() -> None:
    book = L2OrderBook("AAPL")

    snapshot = build_feature_snapshot(book)

    assert snapshot.mid_price is None
    assert snapshot.spread is None
    assert snapshot.imbalance is None
    assert snapshot.microprice is None
    assert snapshot.imbalance_l5 is None
    assert snapshot.imbalance_l10 is None
    assert snapshot.weighted_imbalance_l5 is None
    assert snapshot.weighted_imbalance_l10 is None