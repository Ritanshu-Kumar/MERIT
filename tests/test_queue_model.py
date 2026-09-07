import pytest

from merit.execution.queue_model import QueueModel


def test_full_visible_quantity_ahead() -> None:
    model = QueueModel(ahead_fraction=1.0)

    queue = model.estimate(500)

    assert queue.quantity_ahead == 500
    assert not queue.active


def test_partial_visible_quantity_ahead() -> None:
    model = QueueModel(ahead_fraction=0.5)

    queue = model.estimate(500)

    assert queue.quantity_ahead == 250


def test_zero_visible_quantity() -> None:
    model = QueueModel(ahead_fraction=1.0)

    queue = model.estimate(0)

    assert queue.quantity_ahead == 0
    assert queue.active


def test_invalid_fraction() -> None:
    with pytest.raises(ValueError):
        QueueModel(ahead_fraction=-0.1)


def test_invalid_visible_quantity() -> None:
    model = QueueModel()

    with pytest.raises(ValueError):
        model.estimate(-1)