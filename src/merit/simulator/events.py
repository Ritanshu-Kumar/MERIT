from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum


class EventType(str, Enum):
    MARKET = "MARKET"
    TRADE = "TRADE"
    BOOK_UPDATE = "BOOK_UPDATE"
    ORDER = "ORDER"
    FILL = "FILL"


@dataclass(frozen=True)
class MarketEvent:
    timestamp: datetime
    symbol: str
    event_type: EventType

    def __post_init__(self) -> None:
        if not self.symbol:
            raise ValueError("symbol cannot be empty")


@dataclass(frozen=True)
class TradeEvent(MarketEvent):
    price: Decimal
    quantity: int

    def __post_init__(self) -> None:
        super().__post_init__()

        if self.event_type != EventType.TRADE:
            raise ValueError("TradeEvent must have TRADE event_type")

        if self.price <= 0:
            raise ValueError("price must be greater than zero")

        if self.quantity <= 0:
            raise ValueError("quantity must be greater than zero")


@dataclass(frozen=True)
class BookUpdateEvent(MarketEvent):
    side: str
    price: Decimal
    quantity: Decimal

    def __post_init__(self) -> None:
        super().__post_init__()

        if self.event_type != EventType.BOOK_UPDATE:
            raise ValueError("BookUpdateEvent must have BOOK_UPDATE event_type")

        if self.side not in {"BID", "ASK"}:
            raise ValueError("side must be BID or ASK")

        if self.price <= 0:
            raise ValueError("price must be greater than zero")

        if self.quantity < 0:
            raise ValueError("quantity cannot be negative")