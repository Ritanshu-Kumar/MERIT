from datetime import datetime, timedelta
from decimal import Decimal

from merit.data.normalized import (
    MarketEventType,
    OrderAddEvent,
    OrderExecuteEvent,
)
from merit.features.snapshot import build_feature_snapshot
from merit.portfolio.enums import OrderSide
from merit.research.dataset_builder import build_research_dataset
from merit.research.fill_stream import replay_historical_fills
from merit.research.opportunities import QuoteOpportunity
from merit.research.targets import MidPricePoint
from merit.book.l2_book import L2OrderBook


BASE = datetime(2012, 6, 21, 9, 30)


def test_end_to_end_research_pipeline() -> None:
    t0 = BASE
    t1 = BASE + timedelta(milliseconds=100)
    t2 = BASE + timedelta(seconds=1)
    t3 = BASE + timedelta(seconds=5)

    events = (
        OrderAddEvent(
            timestamp=t0,
            symbol="AAPL",
            event_type=MarketEventType.ADD,
            order_id=1,
            side=OrderSide.BUY,
            price=Decimal("100.00"),
            quantity=100,
        ),
        OrderAddEvent(
            timestamp=t0,
            symbol="AAPL",
            event_type=MarketEventType.ADD,
            order_id=2,
            side=OrderSide.SELL,
            price=Decimal("100.10"),
            quantity=100,
        ),
        OrderExecuteEvent(
            timestamp=t1,
            symbol="AAPL",
            event_type=MarketEventType.EXECUTE,
            order_id=1,
            quantity=100,
            execution_id="exec-1",
            execution_price=Decimal("100.00"),
        ),
    )

    book = L2OrderBook("AAPL")

    book.apply(events[0])
    book.apply(events[1])

    features = build_feature_snapshot(book)

    opportunity = QuoteOpportunity(
        timestamp=t0,
        symbol="AAPL",
        side=OrderSide.BUY,
        quote_price=Decimal("100.00"),
        features=features,
    )

    fills = list(
        replay_historical_fills(
            opportunities=(opportunity,),
            events=events,
        )
    )

    assert len(fills) == 1

    historical_fill = fills[0]

    mid_prices = (
        MidPricePoint(
            t1 + timedelta(milliseconds=100),
            Decimal("100.01"),
        ),
        MidPricePoint(
            t1 + timedelta(seconds=1),
            Decimal("100.04"),
        ),
        MidPricePoint(
            t1 + timedelta(seconds=5),
            Decimal("99.95"),
        ),
    )

    observations = build_research_dataset(
        fills=(historical_fill,),
        mid_prices=mid_prices,
    )

    assert len(observations) == 1

    observation = observations[0]

    assert observation.fill_id == "exec-1"
    assert observation.symbol == "AAPL"
    assert observation.side == "BUY"
    assert observation.quantity == 100
    assert observation.fill_price == Decimal("100.00")

    assert observation.mid_price == Decimal("100.05")
    assert observation.spread == Decimal("0.10")

    assert observation.markout_100ms == Decimal("0.01")
    assert observation.markout_1s == Decimal("0.04")
    assert observation.markout_5s == Decimal("-0.05")