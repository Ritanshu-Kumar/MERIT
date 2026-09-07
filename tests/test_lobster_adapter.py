from datetime import UTC, date, datetime
from decimal import Decimal

import pytest

from merit.data.lobster import LOBSTERParseError, read_messages
from merit.data.normalized import (
    MarketEventType,
    OrderAddEvent,
    OrderCancelEvent,
    OrderDeleteEvent,
    OrderExecuteEvent,
)


TRADING_DATE = date(2012, 6, 21)


def write_messages(path, rows: list[str]) -> None:
    path.write_text(
        "\n".join(rows) + "\n",
        encoding="utf-8",
    )


def test_add_order() -> None:
    path = "tests/data/nonexistent.csv"
    with pytest.raises(FileNotFoundError):
        list(
            read_messages(
                path,
                "AAPL",
                TRADING_DATE,
            )
        )


def test_read_add_message(tmp_path) -> None:
    path = tmp_path / "messages.csv"

    write_messages(
        path,
        [
            "34200.004241176,1,16113575,18,5853300,1",
        ],
    )

    events = list(
        read_messages(
            path,
            "AAPL",
            TRADING_DATE,
        )
    )

    assert len(events) == 1

    event = events[0]

    assert isinstance(event, OrderAddEvent)
    assert event.symbol == "AAPL"
    assert event.order_id == 16113575
    assert event.quantity == 18
    assert event.price == Decimal("585.33")
    assert event.side == "BUY"
    assert event.event_type == MarketEventType.ADD


def test_read_sell_message(tmp_path) -> None:
    path = tmp_path / "messages.csv"

    write_messages(
        path,
        [
            "34200.025551909,1,16120456,18,5859100,-1",
        ],
    )

    events = list(
        read_messages(
            path,
            "AAPL",
            TRADING_DATE,
        )
    )

    event = events[0]

    assert isinstance(event, OrderAddEvent)
    assert event.side == "SELL"
    assert event.price == Decimal("585.91")


def test_read_cancel_message(tmp_path) -> None:
    path = tmp_path / "messages.csv"

    write_messages(
        path,
        [
            "34200.100000000,2,16113575,5,5853300,1",
        ],
    )

    event = list(
        read_messages(
            path,
            "AAPL",
            TRADING_DATE,
        )
    )[0]

    assert isinstance(event, OrderCancelEvent)
    assert event.order_id == 16113575
    assert event.quantity == 5


def test_read_delete_message(tmp_path) -> None:
    path = tmp_path / "messages.csv"

    write_messages(
        path,
        [
            "34200.100000000,3,16113575,18,5853300,1",
        ],
    )

    event = list(
        read_messages(
            path,
            "AAPL",
            TRADING_DATE,
        )
    )[0]

    assert isinstance(event, OrderDeleteEvent)
    assert event.order_id == 16113575


def test_read_execute_message(tmp_path) -> None:
    path = tmp_path / "messages.csv"

    write_messages(
        path,
        [
            "34200.100000000,4,16113575,10,5853300,1",
        ],
    )

    event = list(
        read_messages(
            path,
            "AAPL",
            TRADING_DATE,
        )
    )[0]

    assert isinstance(event, OrderExecuteEvent)
    assert event.order_id == 16113575
    assert event.quantity == 10
    assert event.execution_id == "16113575-0"


def test_timestamp_conversion(tmp_path) -> None:
    path = tmp_path / "messages.csv"

    write_messages(
        path,
        [
            "34200.004241176,1,16113575,18,5853300,1",
        ],
    )

    event = list(
        read_messages(
            path,
            "AAPL",
            TRADING_DATE,
        )
    )[0]

    assert event.timestamp == datetime(
        2012,
        6,
        21,
        9,
        30,
        0,
        4241,
        tzinfo=UTC,
    )


def test_invalid_column_count(tmp_path) -> None:
    path = tmp_path / "messages.csv"

    write_messages(
        path,
        [
            "34200.0,1,123",
        ],
    )

    with pytest.raises(LOBSTERParseError):
        list(
            read_messages(
                path,
                "AAPL",
                TRADING_DATE,
            )
        )


def test_invalid_direction(tmp_path) -> None:
    path = tmp_path / "messages.csv"

    write_messages(
        path,
        [
            "34200.0,1,123,10,5853300,2",
        ],
    )

    with pytest.raises(LOBSTERParseError):
        list(
            read_messages(
                path,
                "AAPL",
                TRADING_DATE,
            )
        )