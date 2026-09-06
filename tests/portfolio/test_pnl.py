from datetime import UTC, datetime
from decimal import Decimal

from merit.portfolio.accounting import apply_fill
from merit.portfolio.enums import OrderSide
from merit.portfolio.models import Fill, Position


def make_fill(
    fill_id: str,
    side: OrderSide,
    quantity: int,
    price: str,
    fee: str = "0",
) -> Fill:
    return Fill(
        fill_id=fill_id,
        order_id=f"ORD-{fill_id}",
        timestamp=datetime.now(UTC),
        symbol="AAPL",
        side=side,
        quantity=quantity,
        price=Decimal(price),
        fee=Decimal(fee),
    )


def test_buy_opens_long_position() -> None:
    position = Position(symbol="AAPL")

    result = apply_fill(
        position,
        make_fill("F1", OrderSide.BUY, 100, "50.00"),
    )

    assert position.quantity == 100
    assert position.average_entry_price == Decimal("50.00")
    assert result.realized_pnl == Decimal(0)


def test_multiple_buys_calculate_weighted_average() -> None:
    position = Position(symbol="AAPL")

    apply_fill(
        position,
        make_fill("F1", OrderSide.BUY, 100, "50.00"),
    )

    apply_fill(
        position,
        make_fill("F2", OrderSide.BUY, 100, "52.00"),
    )

    assert position.quantity == 200
    assert position.average_entry_price == Decimal("51.00")
    assert position.realized_pnl == Decimal(0)


def test_partial_long_close_realizes_pnl() -> None:
    position = Position(
        symbol="AAPL",
        quantity=100,
        average_entry_price=Decimal("50.00"),
    )

    result = apply_fill(
        position,
        make_fill("F1", OrderSide.SELL, 40, "55.00"),
    )

    assert position.quantity == 60
    assert position.average_entry_price == Decimal("50.00")
    assert result.realized_pnl == Decimal("200.00")


def test_full_long_close_realizes_pnl() -> None:
    position = Position(
        symbol="AAPL",
        quantity=100,
        average_entry_price=Decimal("50.00"),
    )

    result = apply_fill(
        position,
        make_fill("F1", OrderSide.SELL, 100, "55.00"),
    )

    assert position.quantity == 0
    assert position.average_entry_price == Decimal(0)
    assert result.realized_pnl == Decimal("500.00")


def test_buying_back_short_realizes_pnl() -> None:
    position = Position(
        symbol="AAPL",
        quantity=-100,
        average_entry_price=Decimal("50.00"),
    )

    result = apply_fill(
        position,
        make_fill("F1", OrderSide.BUY, 40, "45.00"),
    )

    assert position.quantity == -60
    assert position.average_entry_price == Decimal("50.00")
    assert result.realized_pnl == Decimal("200.00")


def test_full_short_close_realizes_pnl() -> None:
    position = Position(
        symbol="AAPL",
        quantity=-100,
        average_entry_price=Decimal("50.00"),
    )

    result = apply_fill(
        position,
        make_fill("F1", OrderSide.BUY, 100, "45.00"),
    )

    assert position.quantity == 0
    assert position.average_entry_price == Decimal(0)
    assert result.realized_pnl == Decimal("500.00")


def test_long_to_short_transition() -> None:
    position = Position(
        symbol="AAPL",
        quantity=100,
        average_entry_price=Decimal("50.00"),
    )

    result = apply_fill(
        position,
        make_fill("F1", OrderSide.SELL, 150, "55.00"),
    )

    assert result.realized_pnl == Decimal("500.00")
    assert position.quantity == -50
    assert position.average_entry_price == Decimal("55.00")


def test_short_to_long_transition() -> None:
    position = Position(
        symbol="AAPL",
        quantity=-100,
        average_entry_price=Decimal("50.00"),
    )

    result = apply_fill(
        position,
        make_fill("F1", OrderSide.BUY, 150, "45.00"),
    )

    assert result.realized_pnl == Decimal("500.00")
    assert position.quantity == 50
    assert position.average_entry_price == Decimal("45.00")


def test_fill_fee_is_preserved_separately() -> None:
    position = Position(symbol="AAPL")

    result = apply_fill(
        position,
        make_fill(
            "F1",
            OrderSide.BUY,
            100,
            "50.00",
            fee="2.50",
        ),
    )

    assert result.realized_pnl == Decimal(0)
    assert result.fee == Decimal("2.50")