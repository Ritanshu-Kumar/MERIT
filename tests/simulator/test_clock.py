from datetime import UTC, datetime, timedelta

import pytest

from merit.simulator.clock import SimulationClock


START = datetime(2026, 1, 1, 9, 30, tzinfo=UTC)


def test_clock_starts_at_given_time() -> None:
    clock = SimulationClock(START)

    assert clock.current_time == START


def test_clock_advances_forward() -> None:
    clock = SimulationClock(START)
    new_time = START + timedelta(seconds=5)

    clock.advance_to(new_time)

    assert clock.current_time == new_time


def test_clock_cannot_move_backwards() -> None:
    clock = SimulationClock(START)

    with pytest.raises(ValueError, match="cannot move backwards"):
        clock.advance_to(START - timedelta(seconds=1))


def test_clock_can_stay_at_same_time() -> None:
    clock = SimulationClock(START)

    clock.advance_to(START)

    assert clock.current_time == START


def test_clock_reset() -> None:
    clock = SimulationClock(START)

    new_start = START + timedelta(minutes=10)
    clock.reset(new_start)

    assert clock.current_time == new_start