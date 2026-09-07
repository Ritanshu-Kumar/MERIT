from datetime import datetime, timedelta
from decimal import Decimal

from merit.features.snapshot import FeatureSnapshot
from merit.portfolio.enums import OrderSide
from merit.portfolio.models import Fill
from merit.research.dataset import build_research_observation
from merit.research.targets import (
    FillMarkout,
    Markout,
    MarkoutStatus,
)


def make_fill() -> Fill:
    return Fill(
        fill_id="fill-1",
        order_id="order-1",
        timestamp=datetime(2012, 6, 21, 9, 30),
        symbol="AAPL",
        side=OrderSide.BUY,
        quantity=100,
        price=Decimal("100.00"),
    )


def make_features() -> FeatureSnapshot:
    return FeatureSnapshot(
        mid_price=Decimal("100.05"),
        spread=Decimal("0.10"),
        relative_spread=Decimal("0.10") / Decimal("100.05"),
        bid_size=200,
        ask_size=100,
        imbalance=Decimal("1") / Decimal("3"),
        microprice=Decimal("100.0666666666666666666666667"),
        imbalance_l5=Decimal("1") / Decimal("7"),
        imbalance_l10=Decimal("1") / Decimal("7"),
        weighted_imbalance_l5=Decimal("1") / Decimal("4"),
        weighted_imbalance_l10=Decimal("1") / Decimal("4"),
    )


def make_markout() -> FillMarkout:
    return FillMarkout(
        fill_timestamp=datetime(2012, 6, 21, 9, 30),
        fill_price=Decimal("100.00"),
        side=OrderSide.BUY,
        markouts=(
            Markout(
                horizon=timedelta(milliseconds=100),
                value=Decimal("0.01"),
                status=MarkoutStatus.AVAILABLE,
            ),
            Markout(
                horizon=timedelta(seconds=1),
                value=Decimal("-0.02"),
                status=MarkoutStatus.AVAILABLE,
            ),
            Markout(
                horizon=timedelta(seconds=5),
                value=Decimal("0.05"),
                status=MarkoutStatus.AVAILABLE,
            ),
        ),
    )


def test_build_research_observation() -> None:
    observation = build_research_observation(
        fill=make_fill(),
        features=make_features(),
        markout=make_markout(),
    )

    assert observation.fill_id == "fill-1"
    assert observation.order_id == "order-1"
    assert observation.symbol == "AAPL"
    assert observation.side == "BUY"
    assert observation.quantity == 100
    assert observation.fill_price == Decimal("100.00")

    assert observation.mid_price == Decimal("100.05")
    assert observation.spread == Decimal("0.10")
    assert observation.bid_size == 200
    assert observation.ask_size == 100

    assert observation.markout_100ms == Decimal("0.01")
    assert observation.markout_1s == Decimal("-0.02")
    assert observation.markout_5s == Decimal("0.05")


def test_unavailable_markout_is_none() -> None:
    markout = FillMarkout(
        fill_timestamp=datetime(2012, 6, 21, 9, 30),
        fill_price=Decimal("100.00"),
        side=OrderSide.SELL,
        markouts=(
            Markout(
                horizon=timedelta(milliseconds=100),
                value=Decimal("0.01"),
                status=MarkoutStatus.AVAILABLE,
            ),
            Markout(
                horizon=timedelta(seconds=1),
                value=None,
                status=MarkoutStatus.UNAVAILABLE,
            ),
            Markout(
                horizon=timedelta(seconds=5),
                value=None,
                status=MarkoutStatus.UNAVAILABLE,
            ),
        ),
    )

    observation = build_research_observation(
        fill=make_fill(),
        features=make_features(),
        markout=markout,
    )

    assert observation.markout_100ms == Decimal("0.01")
    assert observation.markout_1s is None
    assert observation.markout_5s is None