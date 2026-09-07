from datetime import UTC, date, datetime

from merit.data.normalized import (
    MarketEventType,
    StockDirectoryEvent,
    SystemEvent,
    TradingActionEvent,
)
from merit.data.nasdaq.itch import ITCHDecoder


def encode_timestamp(nanoseconds: int) -> bytes:
    return nanoseconds.to_bytes(6, "big")


def system_message(code: str) -> bytes:
    return (
        b"S"
        + b"\x00\x00"
        + b"\x00\x01"
        + encode_timestamp(1_000_000)
        + code.encode()
    )


def stock_directory_message() -> bytes:
    return (
        b"R"
        + (10).to_bytes(2, "big")
        + (1).to_bytes(2, "big")
        + encode_timestamp(2_000_000)
        + b"AAPL    "
        + b"Q"
        + b"N"
        + (100).to_bytes(4, "big")
        + b"N"
        + b"C"
        + b"  "
        + b"P"
        + b"N"
        + b"1"
        + b"Y"
        + b"N"
        + b"\x00\x00\x00\x01"
        + b"N"
    )


def trading_action_message() -> bytes:
    return (
        b"H"
        + (10).to_bytes(2, "big")
        + (1).to_bytes(2, "big")
        + encode_timestamp(3_000_000)
        + b"AAPL    "
        + b"T"
        + b" "
        + b"    "
    )


def test_system_event() -> None:
    decoder = ITCHDecoder(date(2026, 1, 2))

    event = decoder.decode(system_message("O"))

    assert isinstance(event, SystemEvent)
    assert event.event_type == MarketEventType.SYSTEM
    assert event.code == "O"


def test_stock_directory() -> None:
    decoder = ITCHDecoder(date(2026, 1, 2))

    event = decoder.decode(stock_directory_message())

    assert isinstance(event, StockDirectoryEvent)
    assert event.symbol == "AAPL"
    assert event.stock_locate == 10
    assert event.market_category == "Q"
    assert event.financial_status == "N"
    assert event.round_lot_size == 100


def test_trading_action() -> None:
    decoder = ITCHDecoder(date(2026, 1, 2))

    event = decoder.decode(trading_action_message())

    assert isinstance(event, TradingActionEvent)
    assert event.symbol == "AAPL"
    assert event.stock_locate == 10
    assert event.trading_state == "T"
    assert event.reason == ""