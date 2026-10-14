
import pytest

from scripts.m8.build.itch_build_m8_dataset import (
    HORIZONS,
    PendingFill,
    resolve_pending,
)


def make_fill(timestamp: int = 1_000) -> PendingFill:
    return PendingFill(
        timestamp=timestamp,
        symbol="AAPL",
        order_id=1,
        side="B",
        execution_price=1_000_000,
        quantity=10,
        pre_mid=100.0,
        initial_edge=0.0,
        score=0.01,
        selected=True,
        relative_spread=0.01,
        imbalance=0.5,
        bid_size=100,
        ask_size=50,
        spread=100,
        horizons={
            name: timestamp + delta
            for name, delta in HORIZONS.items()
        },
    )


def test_resolve_pending_does_not_use_mid_before_horizon() -> None:
    fill = make_fill()
    pending = [fill]
    rows = []

    resolve_pending(
        pending,
        timestamp=fill.horizons["100ms"] - 1,
        symbol="AAPL",
        mid=100.01,
        rows=rows,
    )

    assert "mid_100ms" not in fill.__dict__
    assert pending == [fill]
    assert rows == []


def test_resolve_pending_uses_mid_at_horizon() -> None:
    fill = make_fill()
    pending = [fill]
    rows = []

    resolve_pending(
        pending,
        timestamp=fill.horizons["100ms"],
        symbol="AAPL",
        mid=100.02,
        rows=rows,
    )

    assert fill.__dict__["mid_100ms"] == pytest.approx(100.02)
    assert "mid_1s" not in fill.__dict__
    assert pending == [fill]
    assert rows == []
