from datetime import datetime
from decimal import Decimal

from merit.execution.queue_model import QueueModel
from merit.portfolio.enums import OrderSide
from merit.portfolio.portfolio import Portfolio
from merit.research.lobster_execution import LobsterExecutionEngine
from merit.risk.limits import RiskLimits, RiskManager
from merit.strategy.baseline import QuoteDecision


TIMESTAMP = datetime(2012, 6, 21, 9, 30)


def make_engine(
    queue_fraction: float = 0.0,
) -> LobsterExecutionEngine:
    return LobsterExecutionEngine(
        symbol="AAPL",
        portfolio=Portfolio(
            Decimal("100000")
        ),
        risk_manager=RiskManager(
            RiskLimits(
                max_order_quantity=100,
                max_position=500,
            )
        ),
        queue_model=QueueModel(
            ahead_fraction=queue_fraction,
        ),
        quantity=100,
    )


def test_create_and_fill() -> None:
    engine = make_engine()

    engine.update_quotes(
        QuoteDecision(
            bid_price=Decimal("100.00"),
            ask_price=Decimal("100.10"),
            quantity=100,
        ),
        bid_visible_quantity=100,
        ask_visible_quantity=100,
        timestamp=TIMESTAMP,
    )

    fill = engine.process_execution(
        TIMESTAMP,
        Decimal("100.00"),
        40,
    )

    assert fill is not None
    assert fill.side == OrderSide.BUY
    assert fill.quantity == 40
    assert (
        engine.active_order(OrderSide.BUY)
        is not None
    )
    assert (
        engine.active_order(
            OrderSide.BUY
        ).remaining_quantity
        == 60
    )


def test_queue_can_block_fill() -> None:
    engine = make_engine(
        queue_fraction=1.0,
    )

    engine.update_quotes(
        QuoteDecision(
            bid_price=Decimal("100.00"),
            ask_price=None,
            quantity=100,
        ),
        bid_visible_quantity=100,
        ask_visible_quantity=0,
        timestamp=TIMESTAMP,
    )

    fill = engine.process_execution(
        TIMESTAMP,
        Decimal("100.00"),
        50,
    )

    assert fill is None

    assert (
        engine.active_order(
            OrderSide.BUY
        ).queue.quantity_ahead
        == 50
    )


def test_cancel_removes_active_quote() -> None:
    engine = make_engine()

    engine.update_quotes(
        QuoteDecision(
            bid_price=Decimal("100.00"),
            ask_price=None,
            quantity=100,
        ),
        bid_visible_quantity=100,
        ask_visible_quantity=0,
        timestamp=TIMESTAMP,
    )

    engine.update_quotes(
        QuoteDecision(
            bid_price=None,
            ask_price=None,
            quantity=100,
        ),
        bid_visible_quantity=0,
        ask_visible_quantity=0,
        timestamp=TIMESTAMP,
    )

    assert (
        engine.active_order(
            OrderSide.BUY
        )
        is None
    )


def test_position_limit_blocks_new_quote() -> None:
    portfolio = Portfolio(
        Decimal("100000")
    )

    engine = LobsterExecutionEngine(
        symbol="AAPL",
        portfolio=portfolio,
        risk_manager=RiskManager(
            RiskLimits(
                max_order_quantity=100,
                max_position=100,
            )
        ),
        queue_model=QueueModel(
            ahead_fraction=0.0,
        ),
        quantity=100,
    )

    engine.update_quotes(
        QuoteDecision(
            bid_price=Decimal("100.00"),
            ask_price=None,
            quantity=100,
        ),
        bid_visible_quantity=100,
        ask_visible_quantity=0,
        timestamp=TIMESTAMP,
    )

    engine.process_execution(
        TIMESTAMP,
        Decimal("100.00"),
        100,
    )

    engine.update_quotes(
        QuoteDecision(
            bid_price=Decimal("100.00"),
            ask_price=None,
            quantity=100,
        ),
        bid_visible_quantity=100,
        ask_visible_quantity=0,
        timestamp=TIMESTAMP,
    )

    assert (
        engine.active_order(
            OrderSide.BUY
        )
        is None
    )