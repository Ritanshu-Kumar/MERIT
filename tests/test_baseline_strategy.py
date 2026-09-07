from datetime import datetime
from decimal import Decimal

import pytest

from merit.book.l2_book import L2OrderBook
from merit.data.normalized import MarketEventType, OrderAddEvent
from merit.portfolio.enums import OrderSide
from merit.strategy.baseline import BaselineMarketMaker


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


def test_quotes_best_bid_and_ask() -> None:
    book = L2OrderBook("AAPL")

    add_order(book, 1, OrderSide.BUY, "100.00", 200)
    add_order(book, 2, OrderSide.BUY, "99.90", 100)
    add_order(book, 3, OrderSide.SELL, "100.10", 150)
    add_order(book, 4, OrderSide.SELL, "100.20", 100)

    strategy = BaselineMarketMaker(quantity=50)

    decision = strategy.quote(book)

    assert decision.bid_price == Decimal("100.00")
    assert decision.ask_price == Decimal("100.10")
    assert decision.quantity == 50


def test_empty_book_returns_no_quotes() -> None:
    book = L2OrderBook("AAPL")

    strategy = BaselineMarketMaker()

    decision = strategy.quote(book)

    assert decision.bid_price is None
    assert decision.ask_price is None
    assert decision.quantity == 100


def test_one_sided_book() -> None:
    book = L2OrderBook("AAPL")

    add_order(book, 1, OrderSide.BUY, "100.00", 100)

    strategy = BaselineMarketMaker(quantity=25)

    decision = strategy.quote(book)

    assert decision.bid_price == Decimal("100.00")
    assert decision.ask_price is None
    assert decision.quantity == 25


def test_invalid_quantity() -> None:
    with pytest.raises(ValueError):
        BaselineMarketMaker(quantity=0)

    with pytest.raises(ValueError):
        BaselineMarketMaker(quantity=-10)