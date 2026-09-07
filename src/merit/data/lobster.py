import csv
from collections.abc import Iterator
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

from merit.data.normalized import (
    MarketEventType,
    OrderAddEvent,
    OrderCancelEvent,
    OrderDeleteEvent,
    OrderExecuteEvent,
)


class LOBSTERParseError(ValueError):
    pass


def _parse_timestamp(value: str, trading_date: date) -> datetime:
    try:
        seconds = Decimal(value)
    except Exception as exc:
        raise LOBSTERParseError(
            f"Invalid timestamp: {value}"
        ) from exc

    whole_seconds = int(seconds)
    fractional_seconds = seconds - Decimal(whole_seconds)

    microseconds = int(
        fractional_seconds * Decimal("1000000")
    )

    return datetime.combine(
        trading_date,
        datetime.min.time(),
        tzinfo=timezone.utc,
    ) + timedelta(
        seconds=whole_seconds,
        microseconds=microseconds,
    )


def _parse_price(value: str) -> Decimal:
    try:
        return Decimal(value) / Decimal("10000")
    except Exception as exc:
        raise LOBSTERParseError(
            f"Invalid price: {value}"
        ) from exc


def _parse_row(
    row: list[str],
    symbol: str,
    trading_date: date,
):
    if len(row) != 6:
        raise LOBSTERParseError(
            f"Expected 6 columns, got {len(row)}"
        )

    try:
        event_type = int(row[1])
    except Exception as exc:
        raise LOBSTERParseError(
            f"Invalid event type: {row[1]}"
        ) from exc

    if event_type in {5, 6, 7}:
        return None

    timestamp = _parse_timestamp(
        row[0],
        trading_date,
    )

    try:
        order_id = int(row[2])
        quantity = int(row[3])
        direction = int(row[5])
    except Exception as exc:
        raise LOBSTERParseError(
            "Invalid integer field"
        ) from exc

    if order_id <= 0:
        raise LOBSTERParseError("order_id must be positive")

    if quantity <= 0:
        raise LOBSTERParseError("quantity must be positive")

    if direction not in {-1, 1}:
        raise LOBSTERParseError(
            "direction must be either 1 or -1"
        )

    price = _parse_price(row[4])

    if event_type == 1:
        side = "BUY" if direction == 1 else "SELL"

        return OrderAddEvent(
            timestamp=timestamp,
            symbol=symbol,
            event_type=MarketEventType.ADD,
            order_id=order_id,
            side=side,
            price=price,
            quantity=quantity,
        )

    if event_type == 2:
        return OrderCancelEvent(
            timestamp=timestamp,
            symbol=symbol,
            event_type=MarketEventType.CANCEL,
            order_id=order_id,
            quantity=quantity,
        )

    if event_type == 3:
        return OrderDeleteEvent(
            timestamp=timestamp,
            symbol=symbol,
            event_type=MarketEventType.DELETE,
            order_id=order_id,
        )

    if event_type == 4:
        return OrderExecuteEvent(
            timestamp=timestamp,
            symbol=symbol,
            event_type=MarketEventType.EXECUTE,
            order_id=order_id,
            quantity=quantity,
            execution_id=f"{order_id}-0",
            execution_price=price,
        )

    raise LOBSTERParseError(
        f"Unknown LOBSTER event type: {event_type}"
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

    if not path.exists():
        raise FileNotFoundError(path)

    with path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:
        reader = csv.reader(file)

        for row_number, row in enumerate(
            reader,
            start=1,
        ):
            try:
                event = _parse_row(
                    row,
                    symbol,
                    trading_date,
                )
            except LOBSTERParseError as exc:
                raise LOBSTERParseError(
                    f"Row {row_number}: {exc}"
                ) from exc

            if event is None:
                continue

            yield event