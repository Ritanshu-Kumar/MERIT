from dataclasses import dataclass
from datetime import date, datetime, timedelta
from decimal import Decimal
import struct

from merit.data.normalized import (
    MarketEventType,
    OrderAddEvent,
    OrderCancelEvent,
    OrderDeleteEvent,
    OrderExecuteEvent,
    OrderReplaceEvent,
    TradeEvent,
)


PRICE_SCALE = Decimal("10000")


class ITCHParseError(ValueError):
    pass


@dataclass
class OrderState:
    symbol: str
    side: str
    price: Decimal
    remaining_quantity: int


def _timestamp(value: int, trading_date: date) -> datetime:
    start = datetime.combine(
        trading_date,
        datetime.min.time(),
    )

    seconds, nanoseconds = divmod(value, 1_000_000_000)

    return start + timedelta(
        seconds=seconds,
        microseconds=nanoseconds // 1_000,
    )


def _price(value: int) -> Decimal:
    return Decimal(value) / PRICE_SCALE


def _read_timestamp(message: bytes) -> int:
    return int.from_bytes(message[5:11], "big")


def _read_order_id(message: bytes, offset: int) -> int:
    return struct.unpack_from(">Q", message, offset)[0]


def _read_uint32(message: bytes, offset: int) -> int:
    return struct.unpack_from(">I", message, offset)[0]


def _read_symbol(message: bytes) -> str:
    return message[24:32].decode("ascii").strip()


class ITCHDecoder:
    def __init__(self, trading_date: date) -> None:
        self.trading_date = trading_date
        self.orders: dict[int, OrderState] = {}

    def decode(
        self,
        message: bytes,
    ) -> (
        OrderAddEvent
        | OrderExecuteEvent
        | OrderCancelEvent
        | OrderDeleteEvent
        | OrderReplaceEvent
        | TradeEvent
        | None
    ):
        if not message:
            raise ITCHParseError("Empty message")

        message_type = chr(message[0])

        if message_type in {"A", "F"}:
            return self._decode_add(message)

        if message_type == "E":
            return self._decode_execute(message)

        if message_type == "C":
            return self._decode_execute_with_price(message)

        if message_type == "X":
            return self._decode_cancel(message)

        if message_type == "D":
            return self._decode_delete(message)

        if message_type == "U":
            return self._decode_replace(message)

        if message_type == "P":
            return self._decode_trade(message)

        return None

    def _decode_add(self, message: bytes) -> OrderAddEvent:
        expected_length = 36 if message[0:1] == b"A" else 40

        if len(message) != expected_length:
            raise ITCHParseError(
                f"Unexpected Add Order length: {len(message)}"
            )

        timestamp = _read_timestamp(message)
        order_id = _read_order_id(message, 11)
        side_code = chr(message[19])
        quantity = _read_uint32(message, 20)
        symbol = _read_symbol(message)
        price = _price(_read_uint32(message, 32))

        if side_code not in {"B", "S"}:
            raise ITCHParseError(f"Invalid side: {side_code!r}")

        side = "BUY" if side_code == "B" else "SELL"

        self.orders[order_id] = OrderState(
            symbol=symbol,
            side=side,
            price=price,
            remaining_quantity=quantity,
        )

        return OrderAddEvent(
            timestamp=_timestamp(timestamp, self.trading_date),
            symbol=symbol,
            event_type=MarketEventType.ADD,
            order_id=order_id,
            side=side,
            price=price,
            quantity=quantity,
        )

    def _decode_execute(self, message: bytes) -> OrderExecuteEvent:
        if len(message) != 31:
            raise ITCHParseError(
                f"Unexpected Order Executed length: {len(message)}"
            )

        return self._apply_execution(
            message=message,
            execution_price=None,
        )

    def _decode_execute_with_price(
        self,
        message: bytes,
    ) -> OrderExecuteEvent:
        if len(message) != 36:
            raise ITCHParseError(
                f"Unexpected Order Executed With Price length: {len(message)}"
            )

        execution_price = _price(_read_uint32(message, 32))

        return self._apply_execution(
            message=message,
            execution_price=execution_price,
        )

    def _apply_execution(
        self,
        message: bytes,
        execution_price: Decimal | None,
    ) -> OrderExecuteEvent:
        timestamp = _read_timestamp(message)
        order_id = _read_order_id(message, 11)
        quantity = _read_uint32(message, 19)
        match_number = _read_order_id(message, 23)

        order = self.orders.get(order_id)

        if order is None:
            raise ITCHParseError(
                f"Unknown order reference: {order_id}"
            )

        if quantity > order.remaining_quantity:
            raise ITCHParseError(
                f"Execution exceeds remaining quantity for order {order_id}"
            )

        order.remaining_quantity -= quantity

        symbol = order.symbol

        if order.remaining_quantity == 0:
            del self.orders[order_id]

        return OrderExecuteEvent(
            timestamp=_timestamp(timestamp, self.trading_date),
            symbol=symbol,
            event_type=MarketEventType.EXECUTE,
            order_id=order_id,
            quantity=quantity,
            execution_id=str(match_number),
            execution_price=execution_price,
        )

    def _decode_cancel(self, message: bytes) -> OrderCancelEvent:
        if len(message) != 23:
            raise ITCHParseError(
                f"Unexpected Order Cancel length: {len(message)}"
            )

        timestamp = _read_timestamp(message)
        order_id = _read_order_id(message, 11)
        quantity = _read_uint32(message, 19)

        order = self.orders.get(order_id)

        if order is None:
            raise ITCHParseError(
                f"Unknown order reference: {order_id}"
            )

        if quantity > order.remaining_quantity:
            raise ITCHParseError(
                f"Cancellation exceeds remaining quantity for order {order_id}"
            )

        order.remaining_quantity -= quantity

        if order.remaining_quantity == 0:
            del self.orders[order_id]

        return OrderCancelEvent(
            timestamp=_timestamp(timestamp, self.trading_date),
            symbol=order.symbol,
            event_type=MarketEventType.CANCEL,
            order_id=order_id,
            quantity=quantity,
        )

    def _decode_delete(self, message: bytes) -> OrderDeleteEvent:
        if len(message) != 19:
            raise ITCHParseError(
                f"Unexpected Order Delete length: {len(message)}"
            )

        timestamp = _read_timestamp(message)
        order_id = _read_order_id(message, 11)

        order = self.orders.pop(order_id, None)

        if order is None:
            raise ITCHParseError(
                f"Unknown order reference: {order_id}"
            )

        return OrderDeleteEvent(
            timestamp=_timestamp(timestamp, self.trading_date),
            symbol=order.symbol,
            event_type=MarketEventType.DELETE,
            order_id=order_id,
        )

    def _decode_replace(self, message: bytes) -> OrderReplaceEvent:
        if len(message) != 35:
            raise ITCHParseError(
                f"Unexpected Order Replace length: {len(message)}"
            )

        timestamp = _read_timestamp(message)
        old_order_id = _read_order_id(message, 11)
        new_order_id = _read_order_id(message, 19)
        quantity = _read_uint32(message, 27)
        price = _price(_read_uint32(message, 31))

        old_order = self.orders.pop(old_order_id, None)

        if old_order is None:
            raise ITCHParseError(
                f"Unknown order reference: {old_order_id}"
            )

        self.orders[new_order_id] = OrderState(
            symbol=old_order.symbol,
            side=old_order.side,
            price=price,
            remaining_quantity=quantity,
        )

        return OrderReplaceEvent(
            timestamp=_timestamp(timestamp, self.trading_date),
            symbol=old_order.symbol,
            event_type=MarketEventType.REPLACE,
            old_order_id=old_order_id,
            new_order_id=new_order_id,
            quantity=quantity,
            price=price,
        )

    def _decode_trade(self, message: bytes) -> TradeEvent:
        if len(message) != 44:
            raise ITCHParseError(
                f"Unexpected Trade length: {len(message)}"
            )

        timestamp = _read_timestamp(message)
        quantity = _read_uint32(message, 20)
        symbol = _read_symbol(message)
        price = _price(_read_uint32(message, 32))

        return TradeEvent(
            timestamp=_timestamp(timestamp, self.trading_date),
            symbol=symbol,
            event_type=MarketEventType.TRADE,
            price=price,
            quantity=quantity,
        )