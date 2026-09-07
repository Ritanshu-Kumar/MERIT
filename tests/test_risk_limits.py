import pytest

from merit.portfolio.enums import OrderSide
from merit.risk.limits import RiskLimits, RiskManager


def test_order_within_limits_is_approved() -> None:
    manager = RiskManager(
        RiskLimits(
            max_order_quantity=100,
            max_position=500,
        )
    )

    result = manager.check_order(
        side=OrderSide.BUY,
        quantity=100,
        current_position=0,
    )

    assert result.approved
    assert result.quantity == 100
    assert result.reason is None


def test_order_size_limit() -> None:
    manager = RiskManager(
        RiskLimits(
            max_order_quantity=100,
            max_position=500,
        )
    )

    result = manager.check_order(
        side=OrderSide.BUY,
        quantity=101,
        current_position=0,
    )

    assert not result.approved
    assert result.quantity == 0
    assert result.reason == "order quantity exceeds limit"


def test_long_position_limit() -> None:
    manager = RiskManager(
        RiskLimits(
            max_order_quantity=100,
            max_position=500,
        )
    )

    result = manager.check_order(
        side=OrderSide.BUY,
        quantity=100,
        current_position=450,
    )

    assert not result.approved
    assert result.reason == "position limit exceeded"


def test_short_position_limit() -> None:
    manager = RiskManager(
        RiskLimits(
            max_order_quantity=100,
            max_position=500,
        )
    )

    result = manager.check_order(
        side=OrderSide.SELL,
        quantity=100,
        current_position=-450,
    )

    assert not result.approved
    assert result.reason == "position limit exceeded"


def test_sell_from_long_position() -> None:
    manager = RiskManager(
        RiskLimits(
            max_order_quantity=100,
            max_position=500,
        )
    )

    result = manager.check_order(
        side=OrderSide.SELL,
        quantity=100,
        current_position=450,
    )

    assert result.approved
    assert result.quantity == 100


def test_invalid_quantity() -> None:
    manager = RiskManager(
        RiskLimits(
            max_order_quantity=100,
            max_position=500,
        )
    )

    result = manager.check_order(
        side=OrderSide.BUY,
        quantity=0,
        current_position=0,
    )

    assert not result.approved
    assert result.reason == "quantity must be positive"


def test_invalid_limits() -> None:
    with pytest.raises(ValueError):
        RiskLimits(max_order_quantity=0, max_position=500)

    with pytest.raises(ValueError):
        RiskLimits(max_order_quantity=100, max_position=0)