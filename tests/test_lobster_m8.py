from datetime import date
from decimal import Decimal

from merit.execution.queue_model import QueueModel
from merit.research.lobster_m8 import (
    build_lobster_research_dataset,
)


TRADING_DATE = date(2012, 6, 21)


def test_causal_execution_generates_research_observation(
    tmp_path,
) -> None:
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

    observations = build_lobster_research_dataset(
        message_path=str(message_path),
        orderbook_path=str(book_path),
        symbol="AAPL",
        trading_date=TRADING_DATE,
        queue_model=QueueModel(ahead_fraction=0.0),
        order_quantity=100,
        levels=1,
    )

    assert len(observations) == 1

    observation = observations[0]

    assert observation.symbol == "AAPL"
    assert observation.side == "BUY"
    assert observation.fill_price == Decimal("100.00")
    assert observation.quantity == 50
    assert observation.mid_price == Decimal("100.05")


def test_queue_can_prevent_hypothetical_fill(
    tmp_path,
) -> None:
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

    observations = build_lobster_research_dataset(
        message_path=str(message_path),
        orderbook_path=str(book_path),
        symbol="AAPL",
        trading_date=TRADING_DATE,
        queue_model=QueueModel(ahead_fraction=1.0),
        order_quantity=100,
        levels=1,
    )

    assert observations == ()


def test_execution_not_at_previous_best_quote_is_ignored(
    tmp_path,
) -> None:
    message_path = tmp_path / "messages.csv"
    book_path = tmp_path / "orderbook.csv"

    message_path.write_text(
        "34200.000000,1,1,100,1000000,1\n"
        "34200.100000,4,1,50,999900,1\n",
        encoding="utf-8",
    )

    book_path.write_text(
        "1001000,100,1000000,100\n"
        "1001000,100,1000000,50\n",
        encoding="utf-8",
    )

    observations = build_lobster_research_dataset(
        message_path=str(message_path),
        orderbook_path=str(book_path),
        symbol="AAPL",
        trading_date=TRADING_DATE,
        queue_model=QueueModel(ahead_fraction=0.0),
        order_quantity=100,
        levels=1,
    )

    assert observations == ()

def test_lobster_m8_execution_ids_are_unique_for_multiple_executions(
    tmp_path,
) -> None:
    message_path = tmp_path / "messages.csv"
    orderbook_path = tmp_path / "orderbook.csv"

    message_path.write_text(
        "0.100000000,1,100,10,58533,1\n"
        "0.200000000,4,100,5,58533,1\n"
        "0.300000000,4,100,3,58533,1\n",
        encoding="utf-8",
    )

    orderbook_path.write_text(
        "58534,100,58533,100,"
        "58535,100,58532,100,"
        "58536,100,58531,100,"
        "58537,100,58530,100,"
        "58538,100,58529,100,"
        "58539,100,58528,100,"
        "58540,100,58527,100,"
        "58541,100,58526,100,"
        "58542,100,58525,100,"
        "58543,100,58524,100\n"
        "58534,100,58533,90,"
        "58535,100,58532,100,"
        "58536,100,58531,100,"
        "58537,100,58530,100,"
        "58538,100,58529,100,"
        "58539,100,58528,100,"
        "58540,100,58527,100,"
        "58541,100,58526,100,"
        "58542,100,58525,100,"
        "58543,100,58524,100\n"
        "58534,100,58533,87,"
        "58535,100,58532,100,"
        "58536,100,58531,100,"
        "58537,100,58530,100,"
        "58538,100,58529,100,"
        "58539,100,58528,100,"
        "58540,100,58527,100,"
        "58541,100,58526,100,"
        "58542,100,58525,100,"
        "58543,100,58524,100\n",
        encoding="utf-8",
    )

    observations = build_lobster_research_dataset(
        message_path=message_path,
        orderbook_path=orderbook_path,
        symbol="AAPL",
        trading_date=date(2012, 6, 21),
        queue_model=QueueModel(ahead_fraction=0.0),
        order_quantity=100,
        levels=10,
        max_observations=None,
    )

    fill_ids = [observation.fill_id for observation in observations]

    assert len(fill_ids) == len(set(fill_ids))