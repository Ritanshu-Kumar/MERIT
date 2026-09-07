from datetime import UTC, datetime, timedelta
from decimal import Decimal

from merit.simulator.engine import SimulationEngine
from merit.simulator.events import EventType, TradeEvent


START = datetime(2026, 1, 1, 9, 30, tzinfo=UTC)


def make_event(seconds: int, quantity: int) -> TradeEvent:
    return TradeEvent(
        timestamp=START + timedelta(seconds=seconds),
        symbol="AAPL",
        event_type=EventType.TRADE,
        price=Decimal("50.00"),
        quantity=quantity,
    )


def test_engine_processes_events_in_time_order() -> None:
    processed: list[int] = []

    def handler(event: TradeEvent) -> None:
        processed.append(event.quantity)

    engine = SimulationEngine(START, handler)

    engine.schedule(make_event(2, 200))
    engine.schedule(make_event(1, 100))
    engine.schedule(make_event(3, 300))

    engine.run()

    assert processed == [100, 200, 300]


def test_engine_advances_clock_with_events() -> None:
    timestamps = []

    def handler(event: TradeEvent) -> None:
        timestamps.append(engine.clock.current_time)

    engine = SimulationEngine(START, handler)

    first = make_event(1, 100)
    second = make_event(5, 500)

    engine.schedule(first)
    engine.schedule(second)

    engine.run()

    assert timestamps == [first.timestamp, second.timestamp]
    assert engine.clock.current_time == second.timestamp


def test_engine_handles_equal_timestamps_deterministically() -> None:
    processed: list[int] = []

    def handler(event: TradeEvent) -> None:
        processed.append(event.quantity)

    engine = SimulationEngine(START, handler)

    engine.schedule(make_event(1, 100))
    engine.schedule(make_event(1, 200))
    engine.schedule(make_event(1, 300))

    engine.run()

    assert processed == [100, 200, 300]


def test_empty_engine_does_nothing() -> None:
    processed: list[int] = []

    def handler(event: TradeEvent) -> None:
        processed.append(event.quantity)

    engine = SimulationEngine(START, handler)

    engine.run()

    assert processed == []
    assert engine.clock.current_time == START

def test_replay_is_deterministic() -> None:
    events = [
        make_event(1, 100),
        make_event(2, 200),
        make_event(3, 300),
    ]

    def run_replay() -> tuple[list[int], object]:
        processed: list[int] = []

        def handler(event: TradeEvent) -> None:
            processed.append(event.quantity)

        engine = SimulationEngine(START, handler)

        for event in events:
            engine.schedule(event)

        engine.run()

        return processed, engine.clock.current_time

    first_result = run_replay()
    second_result = run_replay()

    assert first_result == second_result

def test_replay_produces_identical_event_sequence() -> None:
    events = [
        make_event(3, 300),
        make_event(1, 100),
        make_event(2, 200),
    ]

    def run_replay() -> list[tuple[object, int]]:
        processed: list[tuple[object, int]] = []

        def handler(event: TradeEvent) -> None:
            processed.append(
                (engine.clock.current_time, event.quantity)
            )

        engine = SimulationEngine(START, handler)

        for event in events:
            engine.schedule(event)

        engine.run()

        return processed

    assert run_replay() == run_replay()