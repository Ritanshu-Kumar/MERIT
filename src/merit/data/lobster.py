import csv
from collections.abc import Iterator
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path

from merit.data.normalized import (
    MarketEventType,
    OrderAddEvent,
    OrderCancelEvent,
    OrderDeleteEvent,
    OrderExecuteEvent,
)


PRICE_SCALE = Decimal("10000")
NANOSECONDS = Decimal("1000000000")


class LOBSTERParseError(ValueError):
    pass


def _parse_timestamp(value: str, trading_date: date) -> datetime:
    seconds = Decimal(value)
    whole_seconds = int(seconds)
    nanoseconds = int(
        (seconds - Decimal(whole_seconds)) * NANOSECONDS
    )

    return datetime.combine(
        trading_date,
        datetime.min.time(),
        tzinfo=UTC,
    ) + timedelta(
        seconds=whole_seconds,
        microseconds=nanoseconds // 1000,
    )


def _parse_price(value: str) -> Decimal:
    return Decimal(value) / PRICE_SCALE


def _parse_row(row: list[str]) -> tuple[str, int, int, int, Decimal, int]:
    if len(row) != 6:
        raise LOBSTERParseError(
            f"Expected 6 columns, got {len(row)}"
        )

    try:
        timestamp = row[0]
        event_type = int(row[1])
        order_id = int(row[2])
        quantity = int(row[3])
        price = _parse_price(row[4])
        direction = int(row[5])
    except (ValueError, ArithmeticError) as exc:
        raise LOBSTERParseError(
            f"Invalid message row: {row}"
        ) from exc

    if order_id <= 0:
        raise LOBSTERParseError("order_id must be positive")

    if quantity <= 0:
        raise LOBSTERParseError("quantity must be positive")

    if price <= 0:
        raise LOBSTERParseError("price must be positive")

    if direction not in {-1, 1}:
        raise LOBSTERParseError(
            f"Invalid direction: {direction}"
        )

    return (
        timestamp,
        event_type,
        order_id,
        quantity,
        price,
        direction,
    )


def read_messages(
    path: str | Path,
    symbol: str,
    trading_date: date,
) -> Iterator[
    OrderAddEvent
    | OrderCancelEvent
    | OrderDeleteEvent
    | OrderExecuteEvent
]:
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(path)

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        reader = csv.reader(file)

        for sequence, row in enumerate(reader):
            if not row:
                continue

            (
                timestamp_text,
                event_type,
                order_id,
                quantity,
                price,
                direction,
            ) = _parse_row(row)

            timestamp = _parse_timestamp(
                timestamp_text,
                trading_date,
            )

            side = "BUY" if direction == 1 else "SELL"

            if event_type == 1:
                yield OrderAddEvent(
                    timestamp=timestamp,
                    symbol=symbol,
                    event_type=MarketEventType.ADD,
                    order_id=order_id,
                    side=side,
                    price=price,
                    quantity=quantity,
                )

            elif event_type == 2:
                yield OrderCancelEvent(
                    timestamp=timestamp,
                    symbol=symbol,
                    event_type=MarketEventType.CANCEL,
                    order_id=order_id,
                    quantity=quantity,
                )

            elif event_type == 3:
                yield OrderDeleteEvent(
                    timestamp=timestamp,
                    symbol=symbol,
                    event_type=MarketEventType.DELETE,
                    order_id=order_id,
                )

            elif event_type == 4:
                yield OrderExecuteEvent(
                    timestamp=timestamp,
                    symbol=symbol,
                    event_type=MarketEventType.EXECUTE,
                    order_id=order_id,
                    quantity=quantity,
                    execution_id=f"{order_id}-{sequence}",
                )

            elif event_type in {5, 6, 7}:
                continue

            else:
                raise LOBSTERParseError(
                    f"Unsupported LOBSTER event type: {event_type}"
                )