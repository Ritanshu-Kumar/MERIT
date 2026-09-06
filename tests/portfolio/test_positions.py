from decimal import Decimal

from merit.portfolio.models import Position


def test_long_position_unrealized_pnl() -> None:
    position = Position(
        symbol="AAPL",
        quantity=100,
        average_entry_price=Decimal("50.00"),
    )

    assert position.market_side == "LONG"
    assert position.unrealized_pnl(Decimal("52.00")) == Decimal("200.00")


def test_short_position_unrealized_pnl() -> None:
    position = Position(
        symbol="AAPL",
        quantity=-100,
        average_entry_price=Decimal("50.00"),
    )

    assert position.market_side == "SHORT"
    assert position.unrealized_pnl(Decimal("48.00")) == Decimal("200.00")


def test_flat_position() -> None:
    position = Position(symbol="AAPL")

    assert position.market_side == "FLAT"
    assert position.unrealized_pnl(Decimal("50.00")) == Decimal(0)