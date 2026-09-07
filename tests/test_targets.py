from datetime import datetime, timedelta
from decimal import Decimal

import pytest

from merit.portfolio.enums import OrderSide
from merit.research.targets import (
    MarkoutStatus,
    MidPricePoint,
    calculate_fill_markout,
)


FILL_TIME = datetime(2012, 6, 21, 9, 30)


def test_buy_markout_is_positive_when_mid_rises() -> None:
    points = (
        MidPricePoint(
            timestamp=FILL_TIME + timedelta(milliseconds=150),
            mid_price=Decimal("100.05"),
        ),
    )

    result = calculate_fill_markout(
        fill_timestamp=FILL_TIME,
        fill_price=Decimal("100.00"),
        side=OrderSide.BUY,
        mid_prices=points,
        horizons=(timedelta(milliseconds=100),),
    )

    assert result.markouts[0].value == Decimal("0.05")
    assert result.markouts[0].status == MarkoutStatus.AVAILABLE


def test_buy_markout_is_negative_when_mid_falls() -> None:
    points = (
        MidPricePoint(
            timestamp=FILL_TIME + timedelta(seconds=1),
            mid_price=Decimal("99.90"),
        ),
    )

    result = calculate_fill_markout(
        fill_timestamp=FILL_TIME,
        fill_price=Decimal("100.00"),
        side=OrderSide.BUY,
        mid_prices=points,
        horizons=(timedelta(seconds=1),),
    )

    assert result.markouts[0].value == Decimal("-0.10")


def test_sell_markout_is_positive_when_mid_falls() -> None:
    points = (
        MidPricePoint(
            timestamp=FILL_TIME + timedelta(seconds=1),
            mid_price=Decimal("99.90"),
        ),
    )

    result = calculate_fill_markout(
        fill_timestamp=FILL_TIME,
        fill_price=Decimal("100.00"),
        side=OrderSide.SELL,
        mid_prices=points,
        horizons=(timedelta(seconds=1),),
    )

    assert result.markouts[0].value == Decimal("0.10")


def test_uses_first_mid_at_or_after_horizon() -> None:
    points = (
        MidPricePoint(
            timestamp=FILL_TIME + timedelta(milliseconds=90),
            mid_price=Decimal("100.01"),
        ),
        MidPricePoint(
            timestamp=FILL_TIME + timedelta(milliseconds=120),
            mid_price=Decimal("100.03"),
        ),
    )

    result = calculate_fill_markout(
        fill_timestamp=FILL_TIME,
        fill_price=Decimal("100.00"),
        side=OrderSide.BUY,
        mid_prices=points,
        horizons=(timedelta(milliseconds=100),),
    )

    assert result.markouts[0].value == Decimal("0.03")


def test_multiple_horizons() -> None:
    points = (
        MidPricePoint(
            timestamp=FILL_TIME + timedelta(milliseconds=100),
            mid_price=Decimal("100.01"),
        ),
        MidPricePoint(
            timestamp=FILL_TIME + timedelta(seconds=1),
            mid_price=Decimal("100.04"),
        ),
        MidPricePoint(
            timestamp=FILL_TIME + timedelta(seconds=5),
            mid_price=Decimal("99.95"),
        ),
    )

    result = calculate_fill_markout(
        fill_timestamp=FILL_TIME,
        fill_price=Decimal("100.00"),
        side=OrderSide.BUY,
        mid_prices=points,
        horizons=(
            timedelta(milliseconds=100),
            timedelta(seconds=1),
            timedelta(seconds=5),
        ),
    )

    assert [markout.value for markout in result.markouts] == [
        Decimal("0.01"),
        Decimal("0.04"),
        Decimal("-0.05"),
    ]


def test_unavailable_future_mid() -> None:
    points = (
        MidPricePoint(
            timestamp=FILL_TIME + timedelta(milliseconds=100),
            mid_price=Decimal("100.01"),
        ),
    )

    result = calculate_fill_markout(
        fill_timestamp=FILL_TIME,
        fill_price=Decimal("100.00"),
        side=OrderSide.BUY,
        mid_prices=points,
        horizons=(timedelta(seconds=1),),
    )

    assert result.markouts[0].value is None
    assert result.markouts[0].status == MarkoutStatus.UNAVAILABLE


def test_empty_horizons_rejected() -> None:
    with pytest.raises(ValueError):
        calculate_fill_markout(
            fill_timestamp=FILL_TIME,
            fill_price=Decimal("100.00"),
            side=OrderSide.BUY,
            mid_prices=(),
            horizons=(),
        )


def test_non_positive_horizon_rejected() -> None:
    with pytest.raises(ValueError):
        calculate_fill_markout(
            fill_timestamp=FILL_TIME,
            fill_price=Decimal("100.00"),
            side=OrderSide.BUY,
            mid_prices=(),
            horizons=(timedelta(0),),
        )


def test_unsorted_mid_prices_rejected() -> None:
    points = (
        MidPricePoint(
            timestamp=FILL_TIME + timedelta(seconds=1),
            mid_price=Decimal("100.02"),
        ),
        MidPricePoint(
            timestamp=FILL_TIME,
            mid_price=Decimal("100.00"),
        ),
    )

    with pytest.raises(ValueError):
        calculate_fill_markout(
            fill_timestamp=FILL_TIME,
            fill_price=Decimal("100.00"),
            side=OrderSide.BUY,
            mid_prices=points,
            horizons=(timedelta(seconds=1),),
        )