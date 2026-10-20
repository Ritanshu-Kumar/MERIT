from __future__ import annotations

from scripts.m9.build_fill_hazard import (
    Candidate,
    HazardEngine,
    OUTPUT_COLUMNS,
    candidate_to_row,
)


def make_filled_candidate() -> Candidate:
    return Candidate(
        candidate_id=1,
        timestamp_ns=0,
        symbol="AAPL",
        stock_locate=1,
        quote_side="BUY",
        quote_price=100_0000,
        quote_qty=100,
        mid=100.0,
        spread=0.01,
        relative_spread=0.0001,
        imbalance=0.0,
        delta_imbalance=0.0,
        delta_microprice=0.0,
        delta_spread=0.0,
        queue_ahead_initial=0,
        level_epoch=0,
        start_execution_volume=0,
        queue_ahead_current=0,
        filled_qty=100,
        expiry_ns=1_000_000_000,
        adverse_threshold=99.99,
        status="FILL",
        resolved_timestamp_ns=1_000,
        resolution_mid=100.0,
        resolution_mid_timestamp_ns=1_000,
        fill_timestamp_ns=1_000,
    )


def test_markout_timestamps_record_actual_observation_events() -> None:
    engine = HazardEngine()
    candidate = make_filled_candidate()
    engine.pending_fill_marks[1].append(candidate)

    # The first observed event after each horizon deliberately overshoots it.
    observations = [
        (1_000 + 15_000_000, "10ms", 100.01),
        (1_000 + 110_000_000, "100ms", 100.02),
        (1_000 + 520_000_000, "500ms", 100.03),
        (1_000 + 1_200_000_000, "1s", 100.04),
    ]

    for timestamp_ns, _horizon, midpoint in observations:
        engine.resolve_post_fill_markouts(
            stock_locate=1,
            timestamp_ns=timestamp_ns,
            current_mid=midpoint,
        )

    for timestamp_ns, horizon, midpoint in observations:
        assert candidate.future_mid_timestamps[horizon] == timestamp_ns
        assert candidate.future_mids[horizon] == midpoint

    row = candidate_to_row(candidate)
    for horizon in ("10ms", "100ms", "500ms", "1s"):
        column = f"future_mid_{horizon}_timestamp_ns"
        assert column in OUTPUT_COLUMNS
        assert row[column] == candidate.future_mid_timestamps[horizon]


def test_unresolved_markout_timestamps_remain_missing() -> None:
    candidate = make_filled_candidate()

    assert candidate.future_mid_timestamps == {
        "10ms": None,
        "100ms": None,
        "500ms": None,
        "1s": None,
    }

    row = candidate_to_row(candidate)
    for horizon in ("10ms", "100ms", "500ms", "1s"):
        assert row[f"future_mid_{horizon}_timestamp_ns"] is None
