from decimal import Decimal

from .accounting import apply_fill
from .enums import OrderSide
from .ledger import LedgerEntry, TradeLedger
from .models import Fill, Position


class Portfolio:
    def __init__(self, starting_cash: Decimal) -> None:
        if starting_cash < 0:
            raise ValueError("starting_cash cannot be negative")

        self.cash = starting_cash
        self.positions: dict[str, Position] = {}
        self.ledger = TradeLedger()
        self._mark_prices: dict[str, Decimal] = {}

    def process_fill(self, fill: Fill) -> LedgerEntry:
        position = self.positions.setdefault(
            fill.symbol,
            Position(symbol=fill.symbol),
        )

        result = apply_fill(position, fill)

        if fill.side == OrderSide.BUY:
            self.cash -= fill.notional + fill.fee
        else:
            self.cash += fill.notional - fill.fee

        entry = LedgerEntry.from_fill(
            fill=fill,
            realized_pnl=result.realized_pnl,
            position_after=position.quantity,
            cash_after=self.cash,
        )

        self.ledger.record(entry)
        return entry

    def update_mark(self, symbol: str, price: Decimal) -> None:
        if price <= 0:
            raise ValueError("mark price must be greater than zero")

        self._mark_prices[symbol] = price

    def unrealized_pnl(self) -> Decimal:
      total = Decimal(0)

      for symbol, position in self.positions.items():
          if position.quantity == 0:
              continue

          mark_price = self._mark_prices.get(symbol)
          if mark_price is None:
              raise ValueError(f"No mark price available for {symbol}")

          total += position.unrealized_pnl(mark_price)

      return total.quantize(Decimal("0.01"))

    def realized_pnl(self) -> Decimal:
        return sum(
            (position.realized_pnl for position in self.positions.values()),
            Decimal(0),
        )

    def market_value(self) -> Decimal:
      total = Decimal(0)

      for symbol, position in self.positions.items():
          if position.quantity == 0:
              continue

          mark_price = self._mark_prices.get(symbol)
          if mark_price is None:
              raise ValueError(f"No mark price available for {symbol}")

          total += Decimal(position.quantity) * mark_price

      return total.quantize(Decimal("0.01"))

    def total_equity(self) -> Decimal:
        return self.cash + self.market_value()

    def position(self, symbol: str) -> Position:
        return self.positions.setdefault(
            symbol,
            Position(symbol=symbol),
        )