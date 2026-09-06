from datetime import UTC, datetime
from decimal import Decimal

import pytest

from merit.portfolio.enums import OrderSide
from merit.portfolio.models import Fill
from merit.portfolio.portfolio import Portfolio


def make_fill(
    fill_id: str,
    side: OrderSide,
    quantity: int,
    price: str,
    fee: str = "0",
    symbol: str = "AAPL",
) -> Fill:
    return Fill(
        fill_id=fill_id,
        order_id=f"ORD-{fill_id}",
        timestamp=datetime.now(UTC),
        symbol=symbol,
        side=side,
        quantity=quantity,
        price=Decimal(price),
        fee=Decimal(fee),
    )


def test_portfolio_starts_with_cash() -> None:
    portfolio = Portfolio(Decimal("100000"))

    assert portfolio.cash == Decimal("100000")
    assert portfolio.total_equity() == Decimal("100000")


def test_buy_updates_cash_and_position() -> None:
    portfolio = Portfolio(Decimal("100000"))

    portfolio.process_fill(
        make_fill(
            "F1",
            OrderSide.BUY,
            100,
            "50.00",
        )
    )

    assert portfolio.cash == Decimal("95000")
    assert portfolio.position("AAPL").quantity == 100


def test_sell_updates_cash_and_closes_position() -> None:
    portfolio = Portfolio(Decimal("100000"))

    portfolio.process_fill(
        make_fill("F1", OrderSide.BUY, 100, "50.00")
    )

    portfolio.process_fill(
        make_fill("F2", OrderSide.SELL, 100, "55.00")
    )

    assert portfolio.cash == Decimal("100500")
    assert portfolio.position("AAPL").quantity == 0
    assert portfolio.realized_pnl() == Decimal("500")


def test_fees_reduce_cash() -> None:
    portfolio = Portfolio(Decimal("100000"))

    portfolio.process_fill(
        make_fill(
            "F1",
            OrderSide.BUY,
            100,
            "50.00",
            fee="10.00",
        )
    )

    assert portfolio.cash == Decimal("94990")


def test_unrealized_pnl() -> None:
    portfolio = Portfolio(Decimal("100000"))

    portfolio.process_fill(
        make_fill("F1", OrderSide.BUY, 100, "50.00")
    )

    portfolio.update_mark("AAPL", Decimal("55.00"))

    assert portfolio.unrealized_pnl() == Decimal("500")
    assert portfolio.market_value() == Decimal("5500")
    assert portfolio.total_equity() == Decimal("100500")


def test_realized_and_unrealized_pnl_are_separate() -> None:
    portfolio = Portfolio(Decimal("100000"))

    portfolio.process_fill(
        make_fill("F1", OrderSide.BUY, 100, "50.00")
    )

    portfolio.process_fill(
        make_fill("F2", OrderSide.SELL, 40, "55.00")
    )

    portfolio.update_mark("AAPL", Decimal("53.00"))

    assert portfolio.realized_pnl() == Decimal("200")
    assert portfolio.unrealized_pnl() == Decimal("180")


def test_multiple_symbols_are_independent() -> None:
    portfolio = Portfolio(Decimal("100000"))

    portfolio.process_fill(
        make_fill(
            "F1",
            OrderSide.BUY,
            100,
            "50.00",
            symbol="AAPL",
        )
    )

    portfolio.process_fill(
        make_fill(
            "F2",
            OrderSide.BUY,
            50,
            "100.00",
            symbol="MSFT",
        )
    )

    assert portfolio.position("AAPL").quantity == 100
    assert portfolio.position("MSFT").quantity == 50


def test_multiple_symbols_market_value() -> None:
    portfolio = Portfolio(Decimal("100000"))

    portfolio.process_fill(
        make_fill(
            "F1",
            OrderSide.BUY,
            100,
            "50.00",
            symbol="AAPL",
        )
    )

    portfolio.process_fill(
        make_fill(
            "F2",
            OrderSide.BUY,
            50,
            "100.00",
            symbol="MSFT",
        )
    )

    portfolio.update_mark("AAPL", Decimal("55.00"))
    portfolio.update_mark("MSFT", Decimal("105.00"))

    assert portfolio.market_value() == Decimal("10750")
    assert portfolio.total_equity() == Decimal("100750")


def test_missing_mark_price_fails_unrealized_calculation() -> None:
    portfolio = Portfolio(Decimal("100000"))

    portfolio.process_fill(
        make_fill("F1", OrderSide.BUY, 100, "50.00")
    )

    with pytest.raises(ValueError, match="No mark price"):
        portfolio.unrealized_pnl()

def test_ledger_and_portfolio_realized_pnl_reconcile() -> None:
    portfolio = Portfolio(Decimal("100000"))

    portfolio.process_fill(
        make_fill("F1", OrderSide.BUY, 100, "50.00")
    )

    portfolio.process_fill(
        make_fill("F2", OrderSide.SELL, 100, "55.00")
    )

    assert portfolio.realized_pnl() == portfolio.ledger.total_realized_pnl()

def test_end_to_end_multi_symbol_scenario() -> None:
    portfolio = Portfolio(Decimal("100000"))

    portfolio.process_fill(
        make_fill("F1", OrderSide.BUY, 100, "50.00", symbol="AAPL")
    )
    portfolio.process_fill(
        make_fill("F2", OrderSide.BUY, 50, "52.00", symbol="AAPL")
    )
    portfolio.process_fill(
        make_fill("F3", OrderSide.SELL, 75, "55.00", symbol="AAPL")
    )

    portfolio.process_fill(
        make_fill("F4", OrderSide.BUY, 20, "100.00", symbol="MSFT")
    )
    portfolio.process_fill(
        make_fill("F5", OrderSide.SELL, 10, "105.00", symbol="MSFT")
    )

    portfolio.update_mark("AAPL", Decimal("56.00"))
    portfolio.update_mark("MSFT", Decimal("107.00"))

    aapl = portfolio.position("AAPL")
    msft = portfolio.position("MSFT")

    assert aapl.quantity == 75

    expected_aapl_average = Decimal("50.66666667")
    assert aapl.average_entry_price == expected_aapl_average

    assert msft.quantity == 10
    assert msft.average_entry_price == Decimal("100.00")

    assert aapl.realized_pnl == Decimal("325.00")
    assert msft.realized_pnl == Decimal("50.00")

    assert portfolio.realized_pnl() == Decimal("375.00")
    assert portfolio.unrealized_pnl() == Decimal("470.00")
    assert portfolio.ledger.buy_quantity() == 170
    assert portfolio.ledger.sell_quantity() == 85