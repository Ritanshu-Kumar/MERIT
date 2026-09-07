import pytest

from merit.execution.queue import QueuePosition


def test_queue_starts_inactive() -> None:
    queue = QueuePosition(quantity_ahead=100)

    assert not queue.active


def test_queue_becomes_active_after_ahead_is_consumed() -> None:
    queue = QueuePosition(quantity_ahead=100)

    remaining = queue.consume(100)

    assert remaining == 0
    assert queue.quantity_ahead == 0
    assert queue.active


def test_partial_queue_consumption() -> None:
    queue = QueuePosition(quantity_ahead=100)

    remaining = queue.consume(40)

    assert remaining == 0
    assert queue.quantity_ahead == 60
    assert not queue.active


def test_excess_trade_passes_through_queue() -> None:
    queue = QueuePosition(quantity_ahead=60)

    remaining = queue.consume(100)

    assert remaining == 40
    assert queue.quantity_ahead == 0
    assert queue.active


def test_zero_queue_is_active() -> None:
    queue = QueuePosition(quantity_ahead=0)

    assert queue.active
    assert queue.consume(50) == 50


def test_negative_initial_queue_rejected() -> None:
    with pytest.raises(ValueError):
        QueuePosition(quantity_ahead=-1)


def test_negative_consumption_rejected() -> None:
    queue = QueuePosition(quantity_ahead=100)

    with pytest.raises(ValueError):
        queue.consume(-1)