import pytest

from scripts.m9.build_fill_hazard import advance_sample_grid
from scripts.m9.build_fill_hazard import (
    advance_sample_grid,
    filter_decisions_to_asof_snapshot,
)


def test_delayed_event_preserves_fixed_grid() -> None:
    due, next_due = advance_sample_grid(
        next_due_ns=1_000,
        current_timestamp_ns=3_501,
        interval_ns=1_000,
    )
    assert due == [1_000, 2_000, 3_000]
    assert next_due == 4_000


def test_event_at_due_time_is_processed_before_decision() -> None:
    due, next_due = advance_sample_grid(
        next_due_ns=1_000,
        current_timestamp_ns=1_000,
        interval_ns=1_000,
    )
    assert due == []
    assert next_due == 1_000


def test_event_before_due_does_not_advance_grid() -> None:
    due, next_due = advance_sample_grid(
        next_due_ns=2_000,
        current_timestamp_ns=1_500,
        interval_ns=1_000,
    )
    assert due == []
    assert next_due == 2_000


def test_non_positive_interval_is_rejected() -> None:
    with pytest.raises(ValueError, match="interval_ns must be positive"):
        advance_sample_grid(
            next_due_ns=1_000,
            current_timestamp_ns=2_000,
            interval_ns=0,
        )

def test_backfilled_decisions_before_snapshot_are_dropped():
    due_times = [1000, 2000, 3000]

    assert filter_decisions_to_asof_snapshot(
        due_times, 2500
    ) == [3000]


def test_decision_equal_to_snapshot_timestamp_is_retained():
    due_times = [1000, 2000, 3000]

    assert filter_decisions_to_asof_snapshot(
        due_times, 2000
    ) == [2000, 3000]


def test_no_decisions_are_dropped_when_snapshot_is_older():
    due_times = [1000, 2000, 3000]

    assert filter_decisions_to_asof_snapshot(
        due_times, 500
    ) == due_times