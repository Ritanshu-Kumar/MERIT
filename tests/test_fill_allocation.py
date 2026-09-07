import pytest

from merit.research.fill_allocation import allocate_execution


def test_execution_consumes_queue_before_our_order() -> None:
    result = allocate_execution(
        quantity_ahead=100,
        execution_quantity=40,
    )

    assert result.quantity_ahead_before == 100
    assert result.hypothetical_fill == 0
    assert result.quantity_ahead_after == 60


def test_execution_reaches_our_order_after_queue_is_consumed() -> None:
    result = allocate_execution(
        quantity_ahead=40,
        execution_quantity=100,
    )

    assert result.quantity_ahead_before == 40
    assert result.hypothetical_fill == 60
    assert result.quantity_ahead_after == 0


def test_exact_queue_consumption() -> None:
    result = allocate_execution(
        quantity_ahead=50,
        execution_quantity=50,
    )

    assert result.hypothetical_fill == 0
    assert result.quantity_ahead_after == 0


def test_zero_queue() -> None:
    result = allocate_execution(
        quantity_ahead=0,
        execution_quantity=25,
    )

    assert result.hypothetical_fill == 25
    assert result.quantity_ahead_after == 0


def test_invalid_queue() -> None:
    with pytest.raises(ValueError):
        allocate_execution(
            quantity_ahead=-1,
            execution_quantity=10,
        )


def test_invalid_execution() -> None:
    with pytest.raises(ValueError):
        allocate_execution(
            quantity_ahead=10,
            execution_quantity=0,
        )