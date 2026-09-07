from datetime import date
from decimal import Decimal

import pytest

from merit.data.nasdaq.itch import ITCHDecoder, ITCHParseError


def encode_timestamp(nanoseconds: int) -> bytes:
    return nanoseconds.to_bytes(6, "big")


def add_message(
    order_id: int,
    side: str,
    quantity: int,
    symbol: str,
    price: int,
    timestamp: int = 1_000_000,
) -> bytes:
    return (
        b"A"
        + b"\x00\x01"
        + b"\x00\x01"
        + encode_timestamp(timestamp)
        + order_id.to_bytes(8, "big")
        + side.encode()
        + quantity.to_bytes(4, "big")
        + symbol.ljust(8).encode()
        + price.to_bytes(4, "big")
    )


def execute_message(
    order_id: int,
    quantity: int,
    match_number: int,
    timestamp: int = 2_000_000,
) -> bytes:
    return (
        b"E"
        + b"\x00\x01"
        + b"\x00\x01"
        + encode_timestamp(timestamp)
        + order_id.to_bytes(8, "big")
        + quantity.to_bytes(4, "big")
        + match_number.to_bytes(8, "big")
    )


def cancel_message(
    order_id: int,
    quantity: int,
    timestamp: int = 3_000_000,
) -> bytes:
    return (
        b"X"
        + b"\x00\x01"
        + b"\x00\x01"
        + encode_timestamp(timestamp)
        + order_id.to_bytes(8, "big")
        + quantity.to_bytes(4, "big")
    )


def delete_message(
    order_id: int,
    timestamp: int = 4_000_000,
) -> bytes:
    return (
        b"D"
        + b"\x00\x01"
        + b"\x00\x01"
        + encode_timestamp(timestamp)
        + order_id.to_bytes(8, "big")
    )


def replace_message(
    old_order_id: int,
    new_order_id: int,
    quantity: int,
    price: int,
    timestamp: int = 5_000_000,
) -> bytes:
    return (
        b"U"
        + b"\x00\x01"
        + b"\x00\x01"
        + encode_timestamp(timestamp)
        + old_order_id.to_bytes(8, "big")
        + new_order_id.to_bytes(8, "big")
        + quantity.to_bytes(4, "big")
        + price.to_bytes(4, "big")
    )


def test_add_order_stores_order_state() -> None:
    decoder = ITCHDecoder(date(2026, 1, 2))

    event = decoder.decode(
        add_message(1001, "B", 100, "AAPL", 1_500_000)
    )

    assert event.symbol == "AAPL"
    assert event.order_id == 1001
    assert event.side == "BUY"
    assert event.quantity == 100
    assert event.price == Decimal("150.00")


def test_execute_resolves_symbol_from_order_state() -> None:
    decoder = ITCHDecoder(date(2026, 1, 2))

    decoder.decode(
        add_message(1001, "B", 100, "AAPL", 1_500_000)
    )

    event = decoder.decode(
        execute_message(1001, 40, 5001)
    )

    assert event.symbol == "AAPL"
    assert event.order_id == 1001
    assert event.quantity == 40
    assert event.execution_id == "5001"
    assert decoder.orders[1001].remaining_quantity == 60


def test_fully_executed_order_is_removed() -> None:
    decoder = ITCHDecoder(date(2026, 1, 2))

    decoder.decode(
        add_message(1001, "B", 100, "AAPL", 1_500_000)
    )

    decoder.decode(
        execute_message(1001, 100, 5001)
    )

    assert 1001 not in decoder.orders


def test_cancel_updates_remaining_quantity() -> None:
    decoder = ITCHDecoder(date(2026, 1, 2))

    decoder.decode(
        add_message(1001, "S", 100, "AAPL", 1_500_000)
    )

    decoder.decode(
        cancel_message(1001, 30)
    )

    assert decoder.orders[1001].remaining_quantity == 70


def test_delete_removes_order() -> None:
    decoder = ITCHDecoder(date(2026, 1, 2))

    decoder.decode(
        add_message(1001, "S", 100, "AAPL", 1_500_000)
    )

    event = decoder.decode(delete_message(1001))

    assert event.symbol == "AAPL"
    assert 1001 not in decoder.orders


def test_replace_preserves_symbol_and_side() -> None:
    decoder = ITCHDecoder(date(2026, 1, 2))

    decoder.decode(
        add_message(1001, "B", 100, "AAPL", 1_500_000)
    )

    event = decoder.decode(
        replace_message(1001, 2001, 150, 1_510_000)
    )

    assert event.symbol == "AAPL"
    assert event.new_order_id == 2001
    assert decoder.orders[2001].symbol == "AAPL"
    assert decoder.orders[2001].side == "BUY"
    assert decoder.orders[2001].remaining_quantity == 150
    assert decoder.orders[2001].price == Decimal("151.00")
    assert 1001 not in decoder.orders


def test_unknown_order_reference_fails() -> None:
    decoder = ITCHDecoder(date(2026, 1, 2))

    with pytest.raises(ITCHParseError, match="Unknown order reference"):
        decoder.decode(execute_message(9999, 10, 5001))


def test_execution_cannot_exceed_remaining_quantity() -> None:
    decoder = ITCHDecoder(date(2026, 1, 2))

    decoder.decode(
        add_message(1001, "B", 100, "AAPL", 1_500_000)
    )

    with pytest.raises(
        ITCHParseError,
        match="Execution exceeds remaining quantity",
    ):
        decoder.decode(execute_message(1001, 101, 5001))

def test_timestamp_conversion_preserves_microseconds() -> None:
    decoder = ITCHDecoder(date(2026, 1, 2))

    event = decoder.decode(
        add_message(
            1001,
            "B",
            100,
            "AAPL",
            1_500_000,
            timestamp=12_345_678_901,
        )
    )

    assert event.timestamp.hour == 0
    assert event.timestamp.minute == 0
    assert event.timestamp.second == 12
    assert event.timestamp.microsecond == 345_678