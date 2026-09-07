from datetime import datetime
from decimal import Decimal

from merit.data.normalized import MarketEventType, OrderAddEvent
from merit.portfolio.enums import OrderSide
from merit.research.replay import build_quote_opportunities


def make_event(
    order_id: int,
    side: OrderSide,
    price: str,
    quantity: int,
    timestamp: datetime,
) -> OrderAddEvent:
    return OrderAddEvent(
        timestamp=timestamp,
        symbol="AAPL",
        event_type=MarketEventType.ADD,
        order_id=order_id,
        side=side,
        price=Decimal(price),
        quantity=quantity,
    )


def test_builds_opportunities_from_available_quotes() -> None:
    timestamp = datetime(2012, 6, 21, 9, 30)

    events = (
        make_event(
            1,
            OrderSide.BUY,
            "100.00",
            200,
            timestamp,
        ),
        make_event(
            2,
            OrderSide.SELL,
            "100.10",
            100,
            timestamp,
        ),
    )

    opportunities = list(
        build_quote_opportunities(events, "AAPL")
    )

    assert len(opportunities) == 3

    first = opportunities[0]
    second = opportunities[1]
    third = opportunities[2]

    assert first.side == OrderSide.BUY
    assert first.quote_price == Decimal("100.00")
    assert first.features.mid_price is None

    assert second.side == OrderSide.BUY
    assert second.quote_price == Decimal("100.00")

    assert third.side == OrderSide.SELL
    assert third.quote_price == Decimal("100.10")


def test_features_are_captured_after_two_sided_book_exists() -> None:
    timestamp = datetime(2012, 6, 21, 9, 30)

    events = (
        make_event(
            1,
            OrderSide.BUY,
            "100.00",
            200,
            timestamp,
        ),
        make_event(
            2,
            OrderSide.SELL,
            "100.10",
            100,
            timestamp,
        ),
    )

    opportunities = list(
        build_quote_opportunities(events, "AAPL")
    )

    buy_opportunity = opportunities[1]
    sell_opportunity = opportunities[2]

    assert buy_opportunity.features.mid_price == Decimal("100.05")
    assert sell_opportunity.features.mid_price == Decimal("100.05")

    assert buy_opportunity.features.bid_size == 200
    assert buy_opportunity.features.ask_size == 100

    assert sell_opportunity.features.bid_size == 200
    assert sell_opportunity.features.ask_size == 100


def test_non_book_events_are_ignored() -> None:
    opportunities = list(
        build_quote_opportunities((), "AAPL")
    )

    assert opportunities == []