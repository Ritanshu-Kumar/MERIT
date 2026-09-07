from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from merit.simulator.events import EventType, TradeEvent
from merit.simulator.queue import EventQueue


def make_event(seconds: int, quantity: int) -> TradeEvent:
    return TradeEvent(
        timestamp=datetime(2026, 1, 1, tzinfo=UTC) + timedelta(seconds=seconds),
        symbol="AAPL",
        event_type=EventType.TRADE,
        price=Decimal("50.00"),
        quantity=quantity,
    )


def test_queue_orders_events_by_timestamp() -> None:
    queue = EventQueue()

    later = make_event(2, 200)
    earlier = make_event(1, 100)

    queue.push(later)
    queue.push(earlier)

    assert queue.pop() == earlier
    assert queue.pop() == later


def test_equal_timestamps_preserve_insertion_order() -> None:
    queue = EventQueue()

    first = make_event(1, 100)
    second = make_event(1, 200)
    third = make_event(1, 300)

    queue.push(first)
    queue.push(second)
    queue.push(third)

    assert queue.pop() == first
    assert queue.pop() == second
    assert queue.pop() == third


def test_peek_does_not_remove_event() -> None:
    queue = EventQueue()

    event = make_event(1, 100)
    queue.push(event)

    assert queue.peek() == event
    assert len(queue) == 1


def test_empty_queue() -> None:
    queue = EventQueue()

    assert queue.is_empty()
    assert len(queue) == 0

    with pytest.raises(IndexError):
        queue.pop()

    with pytest.raises(IndexError):
        queue.peek()


def test_queue_length() -> None:
    queue = EventQueue()

    queue.push(make_event(1, 100))
    queue.push(make_event(2, 200))

    assert len(queue) == 2

    queue.pop()

    assert len(queue) == 1