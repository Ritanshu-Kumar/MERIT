from datetime import datetime, timedelta
from decimal import Decimal

from merit.features.snapshot import FeatureSnapshot
from merit.portfolio.enums import OrderSide
from merit.research.dataset_builder import build_research_dataset
from merit.research.historical_fills import HistoricalFill
from merit.research.opportunities import QuoteOpportunity
from merit.research.targets import MidPricePoint


FILL_TIME = datetime(2012, 6, 21, 9, 30)


def make_opportunity() -> QuoteOpportunity:
    return QuoteOpportunity(
        timestamp=FILL_TIME,
        symbol="AAPL",
        side=OrderSide.BUY,
        quote_price=Decimal("100.00"),
        features=FeatureSnapshot(
            mid_price=Decimal("100.05"),
            spread=Decimal("0.10"),
            relative_spread=Decimal("0.001"),
            bid_size=200,
            ask_size=100,
            imbalance=Decimal("1") / Decimal("3"),
            microprice=Decimal("100.0666666666666666666666667"),
            imbalance_l5=Decimal("1") / Decimal("7"),
            imbalance_l10=Decimal("1") / Decimal("7"),
            weighted_imbalance_l5=Decimal("1") / Decimal("4"),
            weighted_imbalance_l10=Decimal("1") / Decimal("4"),
        ),
    )


def make_fill() -> HistoricalFill:
    return HistoricalFill(
        opportunity=make_opportunity(),
        timestamp=FILL_TIME,
        side=OrderSide.BUY,
        price=Decimal("100.00"),
        quantity=50,
        order_id=123,
        execution_id="exec-1",
    )


def make_mid_prices() -> tuple[MidPricePoint, ...]:
    return (
        MidPricePoint(
            timestamp=FILL_TIME + timedelta(milliseconds=100),
            mid_price=Decimal("100.01"),
        ),
        MidPricePoint(
            timestamp=FILL_TIME + timedelta(seconds=1),
            mid_price=Decimal("100.03"),
        ),
        MidPricePoint(
            timestamp=FILL_TIME + timedelta(seconds=5),
            mid_price=Decimal("99.95"),
        ),
    )


def test_build_dataset() -> None:
    observations = build_research_dataset(
        fills=(make_fill(),),
        mid_prices=make_mid_prices(),
    )

    assert len(observations) == 1

    observation = observations[0]

    assert observation.fill_id == "exec-1"
    assert observation.order_id == "123"
    assert observation.symbol == "AAPL"
    assert observation.side == "BUY"
    assert observation.quantity == 50
    assert observation.fill_price == Decimal("100.00")

    assert observation.mid_price == Decimal("100.05")
    assert observation.imbalance == Decimal("1") / Decimal("3")

    assert observation.markout_100ms == Decimal("0.01")
    assert observation.markout_1s == Decimal("0.03")
    assert observation.markout_5s == Decimal("-0.05")


def test_multiple_fills_produce_multiple_observations() -> None:
    first = make_fill()

    second = HistoricalFill(
        opportunity=make_opportunity(),
        timestamp=FILL_TIME + timedelta(seconds=1),
        side=OrderSide.BUY,
        price=Decimal("100.00"),
        quantity=25,
        order_id=124,
        execution_id="exec-2",
    )

    observations = build_research_dataset(
        fills=(first, second),
        mid_prices=make_mid_prices(),
    )

    assert len(observations) == 2
    assert observations[0].fill_id == "exec-1"
    assert observations[1].fill_id == "exec-2"


def test_empty_dataset() -> None:
    observations = build_research_dataset(
        fills=(),
        mid_prices=make_mid_prices(),
    )

    assert observations == ()