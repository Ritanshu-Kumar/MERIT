from __future__ import annotations

import csv

from scripts.m9.build_fill_hazard import (
    OUTPUT_COLUMNS,
    Candidate,
    HazardEngine,
    candidate_to_row,
    finalize_cluster_metadata,
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
    )


def test_markouts_use_last_midpoint_at_or_before_each_horizon() -> None:
    engine = HazardEngine()
    candidate = make_filled_candidate()
    engine._resolve_fill(candidate, timestamp_ns=1_000, resolution_mid=100.0)

    assert all(
        midpoint == 100.0
        for midpoint in candidate.future_mids.values()
    )
    assert all(
        timestamp_ns == 1_000
        for timestamp_ns in candidate.future_mid_timestamps.values()
    )

    expected = {}
    for index, (horizon, horizon_ns) in enumerate(
        {
            "10ms": 10_000_000,
            "100ms": 100_000_000,
            "500ms": 500_000_000,
            "1s": 1_000_000_000,
            "5s": 5_000_000_000,
        }.items(),
        start=1,
    ):
        target_ns = 1_000 + horizon_ns
        midpoint = 100.0 + index / 100
        expected[horizon] = (target_ns, midpoint)

        engine.resolve_post_fill_markouts(
            stock_locate=1,
            timestamp_ns=target_ns,
            current_mid=midpoint,
        )
        engine.resolve_post_fill_markouts(
            stock_locate=1,
            timestamp_ns=target_ns + 1,
            current_mid=midpoint + 1,
        )

    for horizon, (timestamp_ns, midpoint) in expected.items():
        assert candidate.future_mid_timestamps[horizon] == timestamp_ns
        assert candidate.future_mids[horizon] == midpoint
        assert candidate.markout_resolved[horizon]

    assert engine.pop_completed() == [candidate]


def test_eof_finalizes_reached_horizons_without_fabricating_future_marks() -> None:
    engine = HazardEngine()
    candidate = make_filled_candidate()
    engine._resolve_fill(candidate, timestamp_ns=1_000, resolution_mid=100.0)

    last_timestamp_ns = 1_000 + 10_000_000
    engine.resolve_post_fill_markouts(
        stock_locate=1,
        timestamp_ns=last_timestamp_ns,
        current_mid=100.01,
    )
    engine.finalize_pending_markouts(last_timestamp_ns)

    assert candidate.markout_resolved["10ms"]
    assert not candidate.markout_resolved["100ms"]
    assert candidate.future_mid_timestamps["10ms"] == last_timestamp_ns
    assert candidate.future_mid_timestamps["100ms"] == last_timestamp_ns
    assert candidate.future_mids["100ms"] == 100.01
    assert engine.pop_completed() == []
    assert list(engine.pending_fill_marks[1]) == [candidate]

    horizon_engine = HazardEngine()
    horizon_candidate = make_filled_candidate()
    horizon_engine._resolve_fill(
        horizon_candidate,
        timestamp_ns=1_000,
        resolution_mid=100.0,
    )
    last_timestamp_ns = 1_000 + 5_000_000_000
    horizon_engine.resolve_post_fill_markouts(
        stock_locate=1,
        timestamp_ns=last_timestamp_ns,
        current_mid=100.05,
    )
    horizon_engine.finalize_pending_markouts(last_timestamp_ns)

    assert all(horizon_candidate.markout_resolved.values())
    assert horizon_candidate.future_mid_timestamps["5s"] == last_timestamp_ns
    assert horizon_engine.pop_completed() == [horizon_candidate]


def test_unresolved_markout_timestamps_remain_at_fill_midpoint() -> None:
    engine = HazardEngine()
    candidate = make_filled_candidate()
    engine._resolve_fill(candidate, timestamp_ns=1_000, resolution_mid=100.0)

    for horizon in ("10ms", "100ms", "500ms", "1s", "5s"):
        assert candidate.future_mid_timestamps[horizon] == 1_000
        assert candidate.future_mids[horizon] == 100.0

    row = candidate_to_row(candidate)
    for horizon in ("10ms", "100ms", "500ms", "1s", "5s"):
        column = f"future_mid_{horizon}_timestamp_ns"
        assert column in OUTPUT_COLUMNS
        assert row[column] == candidate.future_mid_timestamps[horizon]


def test_finalize_cluster_metadata_counts_rows_present_in_csv(tmp_path) -> None:
    output_path = tmp_path / "candidates.csv"
    rows = [
        {"candidate_id": "1", "event_id": "50", "cluster_id": "AAPL:100"},
        {"candidate_id": "2", "event_id": "50", "cluster_id": "AAPL:100"},
        {"candidate_id": "3", "event_id": "51", "cluster_id": "MSFT:100"},
    ]
    with output_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    finalize_cluster_metadata(output_path)

    with output_path.open("r", newline="", encoding="utf-8") as source:
        result = list(csv.DictReader(source))
    assert [row["cluster_size"] for row in result] == ["2", "2", "1"]
