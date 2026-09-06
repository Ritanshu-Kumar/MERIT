from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal

from .enums import OrderSide, OrderStatus, OrderType


def utc_now() -> datetime:
    return datetime.now(UTC)


@dataclass
class Order:

    order_id: str
    symbol: str
    side: OrderSide
    quantity: int
    order_type: OrderType

    limit_price: Decimal | None = None
    status: OrderStatus = OrderStatus.CREATED
    filled_quantity: int = 0
    created_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        if not self.order_id:
            raise ValueError("order_id cannot be empty")

        if not self.symbol:
            raise ValueError("symbol cannot be empty")

        if self.quantity <= 0:
            raise ValueError("quantity must be greater than zero")

        if self.filled_quantity < 0:
            raise ValueError("filled_quantity cannot be negative")

        if self.filled_quantity > self.quantity:
            raise ValueError("filled_quantity cannot exceed quantity")

        if self.order_type == OrderType.LIMIT:
            if self.limit_price is None:
                raise ValueError("limit orders require limit_price")

            if self.limit_price <= 0:
                raise ValueError("limit_price must be greater than zero")

        if self.order_type == OrderType.MARKET and self.limit_price is not None:
            raise ValueError("market orders cannot have a limit_price")

    @property
    def remaining_quantity(self) -> int:
        return self.quantity - self.filled_quantity

    @property
    def is_fully_filled(self) -> bool:
        return self.filled_quantity == self.quantity


@dataclass(frozen=True)
class Fill:

    fill_id: str
    order_id: str
    timestamp: datetime
    symbol: str
    side: OrderSide
    quantity: int
    price: Decimal
    fee: Decimal = Decimal(0)  # Fee associated with the fill, if any

    def __post_init__(self) -> None:
        if not self.fill_id:
            raise ValueError("fill_id cannot be empty")

        if not self.order_id:
            raise ValueError("order_id cannot be empty")

        if not self.symbol:
            raise ValueError("symbol cannot be empty")

        if self.quantity <= 0:
            raise ValueError("fill quantity must be greater than zero")

        if self.price <= 0:
            raise ValueError("fill price must be greater than zero")

        if self.fee < 0:
            raise ValueError("fee cannot be negative")

    @property
    def notional(self) -> Decimal:
        """Gross value of the fill."""
        return Decimal(self.quantity) * self.price


@dataclass
class Position:
    """
    Current position for one symbol.

    quantity:
        > 0  -> long
        < 0  -> short
        = 0  -> flat

    average_entry_price stores the average price of the currently
    open position. Realized P&L is accumulated as positions are closed.
    """

    symbol: str
    quantity: int = 0
    average_entry_price: Decimal = Decimal(0)
    realized_pnl: Decimal = Decimal(0)

    def __post_init__(self) -> None:
        if not self.symbol:
            raise ValueError("symbol cannot be empty")

        if self.quantity == 0:
            self.average_entry_price = Decimal(0)

        if self.average_entry_price < 0:
            raise ValueError("average_entry_price cannot be negative")

    def unrealized_pnl(self, mark_price: Decimal) -> Decimal:
        if mark_price <= 0:
            raise ValueError("mark_price must be greater than zero")

        if self.quantity == 0:
            return Decimal(0)

        return (
            Decimal(self.quantity)
            * (mark_price - self.average_entry_price)
        )

    @property
    def market_side(self) -> str:
        if self.quantity > 0:
            return "LONG"
        if self.quantity < 0:
            return "SHORT"
        return "FLAT"