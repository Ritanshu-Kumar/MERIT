from datetime import datetime
from decimal import Decimal

import pytest

from merit.data.normalized import MarketEventType, TradeEvent
from merit.execution.simulator import ExecutionSimulator
from merit.features.snapshot import FeatureSnapshot
from merit.portfolio.enums import OrderSide
from merit.research.fill_replay import replay_opportunity
from merit.research.opportunities import QuoteOpportunity


TIMESTAMP = datetime(2012, 6, 21, 9, 30)


def make_features() -> FeatureSnapshot:
    return FeatureSnapshot(
        mid_price=Decimal("100.05"),
        spread=Decimal("0.10"),
        relative_spread=Decimal("0.10") / Decimal("100.05"),
        bid_size=0,
        ask_size=0,
        imbalance=Decimal("0"),
        microprice=Decimal("100.05"),
        imbalance_l5=Decimal("0"),
        imbalance_l10=Decimal("0"),
        weighted_imbalance_l5=Decimal("0"),
        weighted_imbalance_l10=Decimal("0"),
    )


def make_opportunity(
    side: OrderSide,
    quote_price: str,
) -> QuoteOpportunity:
    return QuoteOpportunity(
        timestamp=TIMESTAMP,
        symbol="AAPL",
        side=side,
        quote_price=Decimal(quote_price),
        features=make_features(),
    )


def make_trade(
    timestamp: datetime,
    price: str,
    quantity: int,
) -> TradeEvent:
    return TradeEvent(
        timestamp=timestamp,
        symbol="AAPL",
        event_type=MarketEventType.TRADE,
        price=Decimal(price),
        quantity=quantity,
    )


def test_future_trade_can_fill_opportunity() -> None:
    opportunity = make_opportunity(OrderSide.BUY, "100.00")

    execution = ExecutionSimulator()

    fills = list(
        replay_opportunity(
            opportunity=opportunity,
            future_events=(
                make_trade(
                    TIMESTAMP.replace(second=31),
                    "99.99",
                    100,
                ),
            ),
            execution=execution,
            quantity=100,
        )
    )

    assert len(fills) == 1
    assert fills[0].fill.symbol == "AAPL"
    assert fills[0].fill.side == OrderSide.BUY
    assert fills[0].fill.quantity == 100
    assert fills[0].fill.price == Decimal("100.00")


def test_event_at_opportunity_time_is_not_used() -> None:
    opportunity = make_opportunity(OrderSide.BUY, "100.00")

    execution = ExecutionSimulator()

    fills = list(
        replay_opportunity(
            opportunity=opportunity,
            future_events=(
                make_trade(
                    TIMESTAMP,
                    "99.99",
                    100,
                ),
            ),
            execution=execution,
            quantity=100,
        )
    )

    assert fills == []


def test_non_matching_symbol_is_ignored() -> None:
    opportunity = make_opportunity(OrderSide.BUY, "100.00")

    execution = ExecutionSimulator()

    event = TradeEvent(
        timestamp=TIMESTAMP.replace(second=31),
        symbol="MSFT",
        event_type=MarketEventType.TRADE,
        price=Decimal("99.99"),
        quantity=100,
    )

    fills = list(
        replay_opportunity(
            opportunity=opportunity,
            future_events=(event,),
            execution=execution,
            quantity=100,
        )
    )

    assert fills == []


def test_zero_quantity_rejected() -> None:
    opportunity = make_opportunity(OrderSide.BUY, "100.00")

    with pytest.raises(ValueError):
        list(
            replay_opportunity(
                opportunity=opportunity,
                future_events=(),
                execution=ExecutionSimulator(),
                quantity=0,
            )
        )