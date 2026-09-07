from datetime import datetime
from decimal import Decimal

from merit.data.normalized import MarketEventType, OrderAddEvent
from merit.portfolio.enums import OrderSide
from merit.research.market_replay import replay_market


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


def test_replay_records_market_states() -> None:
    t1 = datetime(2012, 6, 21, 9, 30, 0)
    t2 = datetime(2012, 6, 21, 9, 30, 1)

    events = (
        make_event(1, OrderSide.BUY, "100.00", 200, t1),
        make_event(2, OrderSide.SELL, "100.10", 100, t2),
    )

    states = list(replay_market(events, "AAPL"))

    assert len(states) == 2

    assert states[0].timestamp == t1
    assert states[0].mid_price is None

    assert states[1].timestamp == t2
    assert states[1].mid_price == Decimal("100.05")
    assert states[1].features.bid_size == 200
    assert states[1].features.ask_size == 100


def test_replay_preserves_event_order() -> None:
    timestamps = (
        datetime(2012, 6, 21, 9, 30, 0),
        datetime(2012, 6, 21, 9, 30, 1),
        datetime(2012, 6, 21, 9, 30, 2),
    )

    events = tuple(
        make_event(
            order_id=index + 1,
            side=OrderSide.BUY,
            price=str(100 - index * 0.1),
            quantity=100,
            timestamp=timestamp,
        )
        for index, timestamp in enumerate(timestamps)
    )

    states = list(replay_market(events, "AAPL"))

    assert [state.timestamp for state in states] == list(timestamps)


def test_other_symbols_are_ignored() -> None:
    event = OrderAddEvent(
        timestamp=datetime(2012, 6, 21, 9, 30),
        symbol="MSFT",
        event_type=MarketEventType.ADD,
        order_id=1,
        side=OrderSide.BUY,
        price=Decimal("100.00"),
        quantity=100,
    )

    assert list(replay_market((event,), "AAPL")) == []