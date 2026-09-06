from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_EVEN

from .enums import OrderSide
from .models import Fill, Position


PRICE_QUANTUM = Decimal("0.00000001")
MONEY_QUANTUM = Decimal("0.01")


def _quantize_price(value: Decimal) -> Decimal:
    return value.quantize(PRICE_QUANTUM, rounding=ROUND_HALF_EVEN)


def _quantize_money(value: Decimal) -> Decimal:
    return value.quantize(MONEY_QUANTUM, rounding=ROUND_HALF_EVEN)


@dataclass(frozen=True)
class AccountingResult:
    realized_pnl: Decimal
    fee: Decimal
    position: Position


def apply_fill(position: Position, fill: Fill) -> AccountingResult:
    if position.symbol != fill.symbol:
        raise ValueError(
            f"Position symbol {position.symbol!r} does not match "
            f"fill symbol {fill.symbol!r}"
        )

    if fill.side == OrderSide.BUY:
        realized_pnl = _apply_buy(position, fill.quantity, fill.price)
    elif fill.side == OrderSide.SELL:
        realized_pnl = _apply_sell(position, fill.quantity, fill.price)
    else:
        raise ValueError(f"Unsupported order side: {fill.side}")

    position.realized_pnl += realized_pnl

    return AccountingResult(
        realized_pnl=realized_pnl,
        fee=fill.fee,
        position=position,
    )


def _apply_buy(
    position: Position,
    quantity: int,
    price: Decimal,
) -> Decimal:
    if position.quantity >= 0:
        old_quantity = position.quantity
        new_quantity = old_quantity + quantity

        if old_quantity == 0:
            position.average_entry_price = price
        else:
            position.average_entry_price = _quantize_price(
                (
                    Decimal(old_quantity) * position.average_entry_price
                    + Decimal(quantity) * price
                )
                / Decimal(new_quantity)
            )

        position.quantity = new_quantity
        return Decimal(0)

    short_quantity = -position.quantity

    if quantity < short_quantity:
        realized_pnl = _quantize_money(
            Decimal(quantity)
            * (position.average_entry_price - price)
        )

        position.quantity += quantity
        return realized_pnl

    if quantity == short_quantity:
        realized_pnl = _quantize_money(
            Decimal(short_quantity)
            * (position.average_entry_price - price)
        )

        position.quantity = 0
        position.average_entry_price = Decimal(0)
        return realized_pnl

    realized_pnl = _quantize_money(
        Decimal(short_quantity)
        * (position.average_entry_price - price)
    )

    position.quantity = quantity - short_quantity
    position.average_entry_price = price

    return realized_pnl


def _apply_sell(
    position: Position,
    quantity: int,
    price: Decimal,
) -> Decimal:
    if position.quantity <= 0:
        old_short_quantity = -position.quantity
        new_short_quantity = old_short_quantity + quantity

        if old_short_quantity == 0:
            position.average_entry_price = price
        else:
            position.average_entry_price = _quantize_price(
                (
                    Decimal(old_short_quantity) * position.average_entry_price
                    + Decimal(quantity) * price
                )
                / Decimal(new_short_quantity)
            )

        position.quantity = -new_short_quantity
        return Decimal(0)

    long_quantity = position.quantity

    if quantity < long_quantity:
        realized_pnl = _quantize_money(
            Decimal(quantity)
            * (price - position.average_entry_price)
        )

        position.quantity -= quantity
        return realized_pnl

    if quantity == long_quantity:
        realized_pnl = _quantize_money(
            Decimal(long_quantity)
            * (price - position.average_entry_price)
        )

        position.quantity = 0
        position.average_entry_price = Decimal(0)
        return realized_pnl

    realized_pnl = _quantize_money(
        Decimal(long_quantity)
        * (price - position.average_entry_price)
    )

    position.quantity = -(quantity - long_quantity)
    position.average_entry_price = price

    return realized_pnl