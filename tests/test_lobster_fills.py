from datetime import date
from decimal import Decimal

from merit.data.lobster_orderbook import read_orderbooks
from merit.data.normalized import (
    MarketEventType,
    OrderExecuteEvent,
)
from merit.execution.queue_model import QueueModel
from merit.research.lobster_fills import replay_lobster_fills


TRADING_DATE = date(2012, 6, 21)


def test_realistic_execution_candidate_pipeline(tmp_path) -> None:
    message_path = tmp_path / "messages.csv"
    book_path = tmp_path / "orderbook.csv"

    message_path.write_text(
        "34200.000000,1,1,100,1000000,1\n"
        "34200.100000,4,1,50,1000000,1\n",
        encoding="utf-8",
    )

    book_path.write_text(
        "1001000,100,1000000,100\n"
        "1001000,100,1000000,50\n",
        encoding="utf-8",
    )

    candidates = list(
        replay_lobster_fills(
            message_path=str(message_path),
            orderbook_path=str(book_path),
            symbol="AAPL",
            trading_date=TRADING_DATE,
            queue_model=QueueModel(ahead_fraction=0.0),
            order_quantity=100,
            levels=1,
        )
    )

    assert len(candidates) == 1

    candidate = candidates[0]

    assert candidate.message_index == 2
    assert candidate.fill.quantity == 50
    assert candidate.fill.price == Decimal("100.00")
    assert candidate.fill.order_id == 1
    assert candidate.fill.execution_id == "1-0"


def test_non_execution_events_are_ignored(tmp_path) -> None:
    message_path = tmp_path / "messages.csv"
    book_path = tmp_path / "orderbook.csv"

    message_path.write_text(
        "34200.000000,1,1,100,1000000,1\n",
        encoding="utf-8",
    )

    book_path.write_text(
        "1001000,100,1000000,100\n",
        encoding="utf-8",
    )

    candidates = list(
        replay_lobster_fills(
            message_path=str(message_path),
            orderbook_path=str(book_path),
            symbol="AAPL",
            trading_date=TRADING_DATE,
            queue_model=QueueModel(),
            order_quantity=100,
            levels=1,
        )
    )

    assert candidates == []


def test_order_quantity_caps_fill(tmp_path) -> None:
    message_path = tmp_path / "messages.csv"
    book_path = tmp_path / "orderbook.csv"

    message_path.write_text(
        "34200.000000,1,1,100,1000000,1\n"
        "34200.100000,4,1,80,1000000,1\n",
        encoding="utf-8",
    )

    book_path.write_text(
        "1001000,100,1000000,100\n"
        "1001000,100,1000000,80\n",
        encoding="utf-8",
    )

    candidates = list(
        replay_lobster_fills(
            message_path=str(message_path),
            orderbook_path=str(book_path),
            symbol="AAPL",
            trading_date=TRADING_DATE,
            queue_model=QueueModel(ahead_fraction=0.0),
            order_quantity=25,
            levels=1,
        )
    )

    assert len(candidates) == 1
    assert candidates[0].fill.quantity == 25