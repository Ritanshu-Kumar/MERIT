from scripts.m9 import build_state_transitions as m9
from scripts.m9.build_fill_hazard import (
    HazardEngine,
    candidate_to_row,
)


STOCK_LOCATE = 14
QUOTE_PRICE = 1_000_000
QUOTE_QTY = 100


def make_state(bid_size: int = 100) -> m9.BookState:
    return m9.BookState(
        mid=100.0,
        spread=0.02,
        relative_spread=0.0002,
        bid_size=bid_size,
        ask_size=120,
        imbalance=0.0,
        imbalance_l5=0.0,
        imbalance_l10=0.0,
        microprice=100.0,
    )


def create_buy_candidate(
    engine: HazardEngine,
    activation_queue_ahead: int,
    displayed_bid_size: int = 100,
):
    key = engine.level_key(
        STOCK_LOCATE,
        "B",
        QUOTE_PRICE,
    )

    level = engine.level(key)

    engine.create_candidate(
        timestamp_ns=1_000_000_000,
        symbol="AAPL",
        stock_locate=STOCK_LOCATE,
        quote_side="BUY",
        quote_price=QUOTE_PRICE,
        quote_qty=QUOTE_QTY,
        state=make_state(displayed_bid_size),
        delta_imbalance=0.0,
        delta_microprice=0.0,
        delta_spread=0.0,
        pre_level_epoch=level.epoch,
        source_event_id=123,
        activation_queue_ahead=activation_queue_ahead,
    )

    return engine.candidates[engine.next_id - 1]


def execute(
    engine: HazardEngine,
    shares: int,
    message_index: int,
) -> None:
    engine.register_execution(
        stock_locate=STOCK_LOCATE,
        resting_side="B",
        price=QUOTE_PRICE,
        shares=shares,
        timestamp_ns=2_000_000_000 + message_index,
        current_mid=100.0,
        message_index=message_index,
    )


def test_fill_requires_queue_and_order_quantity() -> None:
    engine = HazardEngine()

    candidate = create_buy_candidate(
        engine,
        activation_queue_ahead=100,
        displayed_bid_size=300,
    )

    # The initial feature snapshot and activation queue are distinct.
    assert candidate.queue_ahead_initial == 300
    assert candidate.queue_ahead_at_activation == 100
    assert candidate.queue_ahead_current == 100

    # 199 shares cannot consume 100 ahead + our 100-share order.
    execute(engine, shares=199, message_index=201)
    assert candidate.status == "ACTIVE"
    assert candidate.filled_qty == 0

    # The 200th eligible share reaches the fill threshold.
    execute(engine, shares=1, message_index=202)

    assert candidate.status == "FILL"
    assert candidate.filled_qty == QUOTE_QTY
    assert candidate.fill_resolution_message_index == 202

    row = candidate_to_row(candidate)
    assert row["queue_ahead_initial"] == 300
    assert row["queue_ahead_at_activation"] == 100


def test_pre_candidate_execution_is_not_counted_twice() -> None:
    engine = HazardEngine()

    # This execution occurs before the hypothetical order is created.
    execute(engine, shares=50, message_index=100)

    candidate = create_buy_candidate(
        engine,
        activation_queue_ahead=100,
        displayed_bid_size=100,
    )

    # The earlier 50 shares are the baseline, not new volume
    # available to fill this candidate.
    assert candidate.start_execution_volume == 50
    assert candidate.required_execution_volume == 250

    execute(engine, shares=199, message_index=101)
    assert candidate.status == "ACTIVE"

    execute(engine, shares=1, message_index=102)
    assert candidate.status == "FILL"
    assert candidate.fill_resolution_message_index == 102

def test_cancellation_of_order_ahead_reduces_fill_threshold() -> None:
    engine = HazardEngine()

    candidate = create_buy_candidate(
        engine,
        activation_queue_ahead=100,
        displayed_bid_size=100,
    )

    assert candidate.required_execution_volume == 200

    engine.register_cancellation(
        stock_locate=STOCK_LOCATE,
        resting_side="B",
        price=QUOTE_PRICE,
        cancelled_shares=40,
        order_added_message_index=100,
    )

    assert candidate.queue_ahead_current == 60
    assert candidate.required_execution_volume == 160

    execute(engine, shares=159, message_index=201)
    assert candidate.status == "ACTIVE"

    execute(engine, shares=1, message_index=202)
    assert candidate.status == "FILL"


def test_cancellation_of_later_order_does_not_reduce_queue() -> None:
    engine = HazardEngine()

    candidate = create_buy_candidate(
        engine,
        activation_queue_ahead=100,
        displayed_bid_size=100,
    )

    engine.register_cancellation(
        stock_locate=STOCK_LOCATE,
        resting_side="B",
        price=QUOTE_PRICE,
        cancelled_shares=80,
        order_added_message_index=200,
    )

    assert candidate.queue_ahead_current == 100
    assert candidate.required_execution_volume == 200

    execute(engine, shares=199, message_index=201)
    assert candidate.status == "ACTIVE"

    execute(engine, shares=1, message_index=202)
    assert candidate.status == "FILL"


def test_order_added_by_activation_event_is_not_ahead() -> None:
    engine = HazardEngine()
    candidate = create_buy_candidate(
        engine,
        activation_queue_ahead=100,
        displayed_bid_size=100,
    )
    engine.register_cancellation(
        stock_locate=STOCK_LOCATE,
        resting_side="B",
        price=QUOTE_PRICE,
        cancelled_shares=40,
        order_added_message_index=123,
    )
    assert candidate.queue_ahead_current == 100
    assert candidate.required_execution_volume == 200


def test_cancellation_after_partial_queue_consumption() -> None:
    engine = HazardEngine()

    candidate = create_buy_candidate(
        engine,
        activation_queue_ahead=100,
        displayed_bid_size=100,
    )

    # First, 40 shares execute against the queue.
    execute(engine, shares=40, message_index=201)

    assert candidate.status == "ACTIVE"
    assert candidate.required_execution_volume == 200

    # Then 40 remaining shares ahead are cancelled.
    engine.register_cancellation(
        stock_locate=STOCK_LOCATE,
        resting_side="B",
        price=QUOTE_PRICE,
        cancelled_shares=40,
        order_added_message_index=100,
    )

    assert candidate.queue_ahead_current == 60
    assert candidate.required_execution_volume == 160

    # Total eligible execution volume reaches 159, not 160.
    execute(engine, shares=119, message_index=202)
    assert candidate.status == "ACTIVE"

    # The final share reaches the adjusted threshold.
    execute(engine, shares=1, message_index=203)

    assert candidate.status == "FILL"
    assert candidate.filled_qty == QUOTE_QTY
    assert candidate.fill_resolution_message_index == 203