from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from .enums import OrderSide
from .models import Fill


@dataclass(frozen=True)
class LedgerEntry:
    """
    Immutable record of one executed fill.

    This is the audit record used to reconstruct trading activity.
    """

    timestamp: datetime
    fill_id: str
    order_id: str
    symbol: str
    side: OrderSide
    quantity: int
    price: Decimal
    notional: Decimal
    fee: Decimal
    realized_pnl: Decimal
    position_after: int
    cash_after: Decimal

    @classmethod
    def from_fill(
        cls,
        fill: Fill,
        realized_pnl: Decimal,
        position_after: int,
        cash_after: Decimal,
    ) -> "LedgerEntry":
        """Create an immutable ledger entry from an executed fill."""

        return cls(
            timestamp=fill.timestamp,
            fill_id=fill.fill_id,
            order_id=fill.order_id,
            symbol=fill.symbol,
            side=fill.side,
            quantity=fill.quantity,
            price=fill.price,
            notional=fill.notional,
            fee=fill.fee,
            realized_pnl=realized_pnl,
            position_after=position_after,
            cash_after=cash_after,
        )

    @property
    def net_pnl_contribution(self) -> Decimal:
        """
        Realized P&L contribution after the fee associated with this fill.
        """
        return self.realized_pnl - self.fee


class TradeLedger:
    """
    Append-only collection of executed trade records.

    The ledger does not perform portfolio accounting itself.
    It records the result produced by the accounting layer.
    """

    def __init__(self) -> None:
        self._entries: list[LedgerEntry] = []

    def record(self, entry: LedgerEntry) -> None:
        """
        Record a ledger entry.

        Duplicate fill IDs are rejected because one execution must only
        appear once in the audit trail.
        """
        if any(existing.fill_id == entry.fill_id for existing in self._entries):
            raise ValueError(f"Duplicate fill_id: {entry.fill_id}")

        self._entries.append(entry)

    @property
    def entries(self) -> tuple[LedgerEntry, ...]:
        """Return an immutable view of all ledger entries."""
        return tuple(self._entries)

    def entries_for_symbol(self, symbol: str) -> tuple[LedgerEntry, ...]:
        """Return all ledger entries for one symbol."""
        return tuple(
            entry
            for entry in self._entries
            if entry.symbol == symbol
        )

    def total_fees(self) -> Decimal:
        """Return total fees across all recorded fills."""
        return sum(
            (entry.fee for entry in self._entries),
            Decimal(0),
        )

    def total_realized_pnl(self) -> Decimal:
        """Return gross realized P&L across all recorded fills."""
        return sum(
            (entry.realized_pnl for entry in self._entries),
            Decimal(0),
        )

    def total_net_pnl_contribution(self) -> Decimal:
        """Return realized P&L after recorded fees."""
        return sum(
            (entry.net_pnl_contribution for entry in self._entries),
            Decimal(0),
        )

    def buy_quantity(self, symbol: str | None = None) -> int:
        """Return total bought quantity, optionally filtered by symbol."""
        entries = self._entries

        if symbol is not None:
            entries = [
                entry for entry in entries
                if entry.symbol == symbol
            ]

        return sum(
            entry.quantity
            for entry in entries
            if entry.side == OrderSide.BUY
        )

    def sell_quantity(self, symbol: str | None = None) -> int:
        """Return total sold quantity, optionally filtered by symbol."""
        entries = self._entries

        if symbol is not None:
            entries = [
                entry for entry in entries
                if entry.symbol == symbol
            ]

        return sum(
            entry.quantity
            for entry in entries
            if entry.side == OrderSide.SELL
        )

    def clear(self) -> None:
        """Remove all entries. Intended for isolated test/setup usage."""
        self._entries.clear()