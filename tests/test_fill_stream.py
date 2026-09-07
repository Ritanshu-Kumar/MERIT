from datetime import datetime, timedelta
from decimal import Decimal

from merit.data.normalized import (
    MarketEventType,
    OrderAddEvent,
    OrderExecuteEvent,
)
from merit.features.snapshot import FeatureSnapshot
from merit.portfolio.enums import OrderSide
from merit.research.fill_stream import replay_historical_fills
from merit.research.opportunities import QuoteOpportunity


BASE_TIME = datetime(2012, 6, 21, 9, 30)


def make_features() -> FeatureSnapshot:
    return FeatureSnapshot(
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
    )


def make_add(
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


def make_execute(
    order_id: int,
    quantity: int,
    timestamp: datetime,
) -> OrderExecuteEvent:
    return OrderExecuteEvent(
        timestamp=timestamp,
        symbol="AAPL",
        event_type=MarketEventType.EXECUTE,
        order_id=order_id,
        quantity=quantity,
        execution_id=f"exec-{order_id}-{quantity}",
        execution_price=Decimal("100.00"),
    )


def make_opportunity(
    timestamp: datetime,
    side: OrderSide,
    price: str,
) -> QuoteOpportunity:
    return QuoteOpportunity(
        timestamp=timestamp,
        symbol="AAPL",
        side=side,
        quote_price=Decimal(price),
        features=make_features(),
    )


def test_replay_detects_historical_fill() -> None:
    t0 = BASE_TIME
    t1 = BASE_TIME + timedelta(seconds=1)

    opportunities = (
        make_opportunity(t0, OrderSide.BUY, "100.00"),
    )

    events = (
        make_add(1, OrderSide.BUY, "100.00", 100, t0),
        make_execute(1, 40, t1),
    )

    fills = list(
        replay_historical_fills(
            opportunities=opportunities,
            events=events,
        )
    )

    assert len(fills) == 1
    assert fills[0].quantity == 40
    assert fills[0].order_id == 1


def test_execution_before_opportunity_is_not_used() -> None:
    t0 = BASE_TIME
    t1 = BASE_TIME + timedelta(seconds=1)

    opportunities = (
        make_opportunity(t1, OrderSide.BUY, "100.00"),
    )

    events = (
        make_add(1, OrderSide.BUY, "100.00", 100, t0),
        make_execute(1, 40, t0),
    )

    fills = list(
        replay_historical_fills(
            opportunities=opportunities,
            events=events,
        )
    )

    assert fills == []


def test_different_price_does_not_fill() -> None:
    t0 = BASE_TIME
    t1 = BASE_TIME + timedelta(seconds=1)

    opportunities = (
        make_opportunity(t0, OrderSide.BUY, "99.90"),
    )

    events = (
        make_add(1, OrderSide.BUY, "100.00", 100, t0),
        make_execute(1, 40, t1),
    )

    fills = list(
        replay_historical_fills(
            opportunities=opportunities,
            events=events,
        )
    )

    assert fills == []


def test_multiple_executions_are_detected() -> None:
    t0 = BASE_TIME
    t1 = BASE_TIME + timedelta(seconds=1)
    t2 = BASE_TIME + timedelta(seconds=2)

    opportunities = (
        make_opportunity(t0, OrderSide.BUY, "100.00"),
    )

    events = (
        make_add(1, OrderSide.BUY, "100.00", 100, t0),
        make_execute(1, 30, t1),
        make_execute(1, 20, t2),
    )

    fills = list(
        replay_historical_fills(
            opportunities=opportunities,
            events=events,
        )
    )

    assert [fill.quantity for fill in fills] == [30, 20]
def test_opportunities_can_be_generated_lazily() -> None:
    t0 = BASE_TIME
    t1 = BASE_TIME + timedelta(seconds=1)

    consumed = []

    def opportunities():
        consumed.append("first")
        yield make_opportunity(
            t0,
            OrderSide.BUY,
            "100.00",
        )

        consumed.append("second")
        yield make_opportunity(
            t1,
            OrderSide.SELL,
            "100.10",
        )

    events = (
        make_add(
            1,
            OrderSide.BUY,
            "100.00",
            100,
            t0,
        ),
    )

    list(
        replay_historical_fills(
            opportunities=opportunities(),
            events=events,
        )
    )

    assert consumed == ["first", "second"]

def test_multiple_opportunities_same_timestamp_remain_active() -> None:
    t0 = BASE_TIME
    t1 = BASE_TIME + timedelta(seconds=1)

    opportunities = (
        make_opportunity(t0, OrderSide.BUY, "100.00"),
        make_opportunity(t0, OrderSide.BUY, "100.00"),
    )

    events = (
        make_add(
            1,
            OrderSide.BUY,
            "100.00",
            100,
            t0,
        ),
        make_execute(
            1,
            40,
            t1,
        ),
    )

    fills = list(
        replay_historical_fills(
            opportunities=opportunities,
            events=events,
        )
    )

    assert len(fills) == 2
    assert [fill.quantity for fill in fills] == [40, 40]