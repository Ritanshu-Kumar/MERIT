from datetime import UTC, datetime
from decimal import Decimal

import pytest

from merit.portfolio.enums import OrderSide
from merit.portfolio.ledger import LedgerEntry, TradeLedger
from merit.portfolio.models import Fill


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


def make_entry(
    fill_id: str,
    side: OrderSide,
    quantity: int,
    price: str,
    fee: str = "0",
    realized_pnl: str = "0",
    position_after: int = 0,
    cash_after: str = "100000",
    symbol: str = "AAPL",
) -> LedgerEntry:
    fill = make_fill(
        fill_id=fill_id,
        side=side,
        quantity=quantity,
        price=price,
        fee=fee,
        symbol=symbol,
    )

    return LedgerEntry.from_fill(
        fill=fill,
        realized_pnl=Decimal(realized_pnl),
        position_after=position_after,
        cash_after=Decimal(cash_after),
    )


def test_ledger_records_entry() -> None:
    ledger = TradeLedger()

    entry = make_entry(
        fill_id="F1",
        side=OrderSide.BUY,
        quantity=100,
        price="50.00",
        fee="1.00",
        position_after=100,
        cash_after="94999.00",
    )

    ledger.record(entry)

    assert len(ledger.entries) == 1
    assert ledger.entries[0] == entry


def test_ledger_entry_is_immutable() -> None:
    entry = make_entry(
        fill_id="F1",
        side=OrderSide.BUY,
        quantity=100,
        price="50.00",
    )

    with pytest.raises(AttributeError):
        entry.quantity = 200  # type: ignore[misc]


def test_duplicate_fill_is_rejected() -> None:
    ledger = TradeLedger()

    entry = make_entry(
        fill_id="F1",
        side=OrderSide.BUY,
        quantity=100,
        price="50.00",
    )

    ledger.record(entry)

    with pytest.raises(ValueError, match="Duplicate fill_id"):
        ledger.record(entry)


def test_entries_are_returned_as_tuple() -> None:
    ledger = TradeLedger()

    ledger.record(
        make_entry(
            fill_id="F1",
            side=OrderSide.BUY,
            quantity=100,
            price="50.00",
        )
    )

    assert isinstance(ledger.entries, tuple)


def test_filter_entries_by_symbol() -> None:
    ledger = TradeLedger()

    ledger.record(
        make_entry(
            fill_id="F1",
            side=OrderSide.BUY,
            quantity=100,
            price="50.00",
            symbol="AAPL",
        )
    )

    ledger.record(
        make_entry(
            fill_id="F2",
            side=OrderSide.BUY,
            quantity=50,
            price="200.00",
            symbol="MSFT",
        )
    )

    aapl_entries = ledger.entries_for_symbol("AAPL")

    assert len(aapl_entries) == 1
    assert aapl_entries[0].symbol == "AAPL"


def test_total_fees() -> None:
    ledger = TradeLedger()

    ledger.record(
        make_entry(
            fill_id="F1",
            side=OrderSide.BUY,
            quantity=100,
            price="50.00",
            fee="1.25",
        )
    )

    ledger.record(
        make_entry(
            fill_id="F2",
            side=OrderSide.SELL,
            quantity=100,
            price="55.00",
            fee="1.50",
            realized_pnl="500.00",
        )
    )

    assert ledger.total_fees() == Decimal("2.75")


def test_total_realized_pnl() -> None:
    ledger = TradeLedger()

    ledger.record(
        make_entry(
            fill_id="F1",
            side=OrderSide.SELL,
            quantity=100,
            price="55.00",
            realized_pnl="500.00",
        )
    )

    ledger.record(
        make_entry(
            fill_id="F2",
            side=OrderSide.SELL,
            quantity=50,
            price="60.00",
            realized_pnl="250.00",
        )
    )

    assert ledger.total_realized_pnl() == Decimal("750.00")


def test_net_pnl_includes_fees() -> None:
    ledger = TradeLedger()

    ledger.record(
        make_entry(
            fill_id="F1",
            side=OrderSide.SELL,
            quantity=100,
            price="55.00",
            fee="2.00",
            realized_pnl="500.00",
        )
    )

    ledger.record(
        make_entry(
            fill_id="F2",
            side=OrderSide.SELL,
            quantity=50,
            price="60.00",
            fee="1.00",
            realized_pnl="250.00",
        )
    )

    assert ledger.total_net_pnl_contribution() == Decimal("747.00")


def test_buy_and_sell_quantities() -> None:
    ledger = TradeLedger()

    ledger.record(
        make_entry(
            fill_id="F1",
            side=OrderSide.BUY,
            quantity=100,
            price="50.00",
        )
    )

    ledger.record(
        make_entry(
            fill_id="F2",
            side=OrderSide.SELL,
            quantity=40,
            price="55.00",
        )
    )

    ledger.record(
        make_entry(
            fill_id="F3",
            side=OrderSide.BUY,
            quantity=20,
            price="49.00",
        )
    )

    assert ledger.buy_quantity() == 120
    assert ledger.sell_quantity() == 40


def test_symbol_specific_quantities() -> None:
    ledger = TradeLedger()

    ledger.record(
        make_entry(
            fill_id="F1",
            side=OrderSide.BUY,
            quantity=100,
            price="50.00",
            symbol="AAPL",
        )
    )

    ledger.record(
        make_entry(
            fill_id="F2",
            side=OrderSide.BUY,
            quantity=50,
            price="200.00",
            symbol="MSFT",
        )
    )

    assert ledger.buy_quantity("AAPL") == 100
    assert ledger.buy_quantity("MSFT") == 50