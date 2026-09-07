from datetime import datetime
from decimal import Decimal

from merit.book.l2_book import L2OrderBook
from merit.data.normalized import (
    MarketEventType,
    OrderAddEvent,
    OrderExecuteEvent,
)
from merit.features.snapshot import FeatureSnapshot
from merit.portfolio.enums import OrderSide
from merit.research.historical_fills import (
    detect_historical_fill,
    process_execution_event,
)
from merit.research.opportunities import QuoteOpportunity


TIMESTAMP = datetime(2012, 6, 21, 9, 30)


def make_opportunity(side: OrderSide, price: str) -> QuoteOpportunity:
    return QuoteOpportunity(
        timestamp=TIMESTAMP,
        symbol="AAPL",
        side=side,
        quote_price=Decimal(price),
        features=FeatureSnapshot(
            mid_price=Decimal("100.05"),
            spread=Decimal("0.10"),
            relative_spread=Decimal("0.001"),
            bid_size=100,
            ask_size=100,
            imbalance=Decimal("0"),
            microprice=Decimal("100.05"),
            imbalance_l5=Decimal("0"),
            imbalance_l10=Decimal("0"),
            weighted_imbalance_l5=Decimal("0"),
            weighted_imbalance_l10=Decimal("0"),
        ),
    )


def make_add(
    order_id: int,
    side: OrderSide,
    price: str,
    quantity: int,
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


def make_execute(
    order_id: int,
    quantity: int,
) -> OrderExecuteEvent:
    return OrderExecuteEvent(
        timestamp=TIMESTAMP.replace(second=31),
        symbol="AAPL",
        event_type=MarketEventType.EXECUTE,
        order_id=order_id,
        quantity=quantity,
        execution_id="exec-1",
        execution_price=Decimal("100.00"),
    )


def test_matching_execution_creates_historical_fill() -> None:
    book = L2OrderBook("AAPL")
    book.apply(
        make_add(
            1,
            OrderSide.BUY,
            "100.00",
            100,
        )
    )

    opportunity = make_opportunity(
        OrderSide.BUY,
        "100.00",
    )

    result = detect_historical_fill(
        book,
        opportunity,
        make_execute(1, 40),
    )

    assert result is not None
    assert result.order_id == 1
    assert result.execution_id == "exec-1"
    assert result.side == OrderSide.BUY
    assert result.price == Decimal("100.00")
    assert result.quantity == 40


def test_different_side_does_not_match() -> None:
    book = L2OrderBook("AAPL")
    book.apply(
        make_add(
            1,
            OrderSide.BUY,
            "100.00",
            100,
        )
    )

    opportunity = make_opportunity(
        OrderSide.SELL,
        "100.00",
    )

    result = detect_historical_fill(
        book,
        opportunity,
        make_execute(1, 40),
    )

    assert result is None


def test_different_price_does_not_match() -> None:
    book = L2OrderBook("AAPL")
    book.apply(
        make_add(
            1,
            OrderSide.BUY,
            "100.00",
            100,
        )
    )

    opportunity = make_opportunity(
        OrderSide.BUY,
        "99.90",
    )

    result = detect_historical_fill(
        book,
        opportunity,
        make_execute(1, 40),
    )

    assert result is None


def test_unknown_order_does_not_match() -> None:
    book = L2OrderBook("AAPL")

    opportunity = make_opportunity(
        OrderSide.BUY,
        "100.00",
    )

    result = detect_historical_fill(
        book,
        opportunity,
        make_execute(999, 40),
    )

    assert result is None


def test_non_execution_event_is_applied_without_fill() -> None:
    book = L2OrderBook("AAPL")

    opportunity = make_opportunity(
        OrderSide.BUY,
        "100.00",
    )

    event = make_add(
        1,
        OrderSide.BUY,
        "100.00",
        100,
    )

    result = process_execution_event(
        book,
        opportunity,
        event,
    )

    assert result is None
    assert book.order(1) is not None


def test_execution_is_applied_after_detection() -> None:
    book = L2OrderBook("AAPL")

    book.apply(
        make_add(
            1,
            OrderSide.BUY,
            "100.00",
            100,
        )
    )

    opportunity = make_opportunity(
        OrderSide.BUY,
        "100.00",
    )

    result = process_execution_event(
        book,
        opportunity,
        make_execute(1, 40),
    )

    assert result is not None
    assert result.quantity == 40
    assert book.order(1).quantity == 60