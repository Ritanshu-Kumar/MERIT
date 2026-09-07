from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum


class MarketEventType(str, Enum):
    ADD = "ADD"
    EXECUTE = "EXECUTE"
    CANCEL = "CANCEL"
    DELETE = "DELETE"
    REPLACE = "REPLACE"
    TRADE = "TRADE"
    SYSTEM = "SYSTEM"


@dataclass(frozen=True)
class NormalizedMarketEvent:
    timestamp: datetime
    symbol: str
    event_type: MarketEventType


@dataclass(frozen=True)
class OrderAddEvent(NormalizedMarketEvent):
    order_id: int
    side: str
    price: Decimal
    quantity: int

    def __post_init__(self) -> None:
        if self.event_type != MarketEventType.ADD:
            raise ValueError("OrderAddEvent must use ADD event type")
        if not self.symbol:
            raise ValueError("symbol cannot be empty")
        if self.side not in {"BUY", "SELL"}:
            raise ValueError("side must be BUY or SELL")
        if self.order_id <= 0:
            raise ValueError("order_id must be positive")
        if self.price <= 0:
            raise ValueError("price must be positive")
        if self.quantity <= 0:
            raise ValueError("quantity must be positive")


@dataclass(frozen=True)
class OrderExecuteEvent(NormalizedMarketEvent):
    order_id: int
    quantity: int
    execution_id: str
    execution_price: Decimal | None = None

    def __post_init__(self) -> None:
        if self.event_type != MarketEventType.EXECUTE:
            raise ValueError("OrderExecuteEvent must use EXECUTE event type")
        if self.order_id <= 0:
            raise ValueError("order_id must be positive")
        if self.quantity <= 0:
            raise ValueError("quantity must be positive")
        if not self.execution_id:
            raise ValueError("execution_id cannot be empty")


@dataclass(frozen=True)
class OrderCancelEvent(NormalizedMarketEvent):
    order_id: int
    quantity: int

    def __post_init__(self) -> None:
        if self.event_type != MarketEventType.CANCEL:
            raise ValueError("OrderCancelEvent must use CANCEL event type")
        if self.order_id <= 0:
            raise ValueError("order_id must be positive")
        if self.quantity <= 0:
            raise ValueError("quantity must be positive")


@dataclass(frozen=True)
class OrderDeleteEvent(NormalizedMarketEvent):
    order_id: int

    def __post_init__(self) -> None:
        if self.event_type != MarketEventType.DELETE:
            raise ValueError("OrderDeleteEvent must use DELETE event type")
        if self.order_id <= 0:
            raise ValueError("order_id must be positive")


@dataclass(frozen=True)
class OrderReplaceEvent(NormalizedMarketEvent):
    old_order_id: int
    new_order_id: int
    quantity: int
    price: Decimal

    def __post_init__(self) -> None:
        if self.event_type != MarketEventType.REPLACE:
            raise ValueError("OrderReplaceEvent must use REPLACE event type")
        if self.old_order_id <= 0 or self.new_order_id <= 0:
            raise ValueError("order IDs must be positive")
        if self.quantity <= 0:
            raise ValueError("quantity must be positive")
        if self.price <= 0:
            raise ValueError("price must be positive")


@dataclass(frozen=True)
class TradeEvent(NormalizedMarketEvent):
    price: Decimal
    quantity: int

    def __post_init__(self) -> None:
        if self.event_type != MarketEventType.TRADE:
            raise ValueError("TradeEvent must use TRADE event type")
        if self.price <= 0:
            raise ValueError("price must be positive")
        if self.quantity <= 0:
            raise ValueError("quantity must be positive")


@dataclass(frozen=True)
class SystemEvent(NormalizedMarketEvent):
    code: str

    def __post_init__(self) -> None:
        if self.event_type != MarketEventType.SYSTEM:
            raise ValueError("SystemEvent must use SYSTEM event type")
        if not self.code:
            raise ValueError("code cannot be empty")