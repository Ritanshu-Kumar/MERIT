from decimal import Decimal

from merit.portfolio.enums import OrderSide
from merit.strategy.baseline import QuoteDecision
from merit.strategy.quote_lifecycle import (
    ActiveQuote,
    QuoteActionType,
    QuoteLifecycle,
)


def test_create_quotes_when_no_active_orders() -> None:
    lifecycle = QuoteLifecycle()

    decision = QuoteDecision(
        bid_price=Decimal("100.00"),
        ask_price=Decimal("100.10"),
        quantity=100,
    )

    actions = lifecycle.evaluate(
        decision,
        active_bid=None,
        active_ask=None,
    )

    assert len(actions) == 2

    assert actions[0].action == QuoteActionType.CREATE
    assert actions[0].side == OrderSide.BUY
    assert actions[0].price == Decimal("100.00")
    assert actions[0].quantity == 100

    assert actions[1].action == QuoteActionType.CREATE
    assert actions[1].side == OrderSide.SELL
    assert actions[1].price == Decimal("100.10")
    assert actions[1].quantity == 100


def test_keep_unchanged_quotes() -> None:
    lifecycle = QuoteLifecycle()

    decision = QuoteDecision(
        bid_price=Decimal("100.00"),
        ask_price=Decimal("100.10"),
        quantity=100,
    )

    actions = lifecycle.evaluate(
        decision,
        active_bid=ActiveQuote(
            order_id=1,
            side=OrderSide.BUY,
            price=Decimal("100.00"),
            quantity=100,
        ),
        active_ask=ActiveQuote(
            order_id=2,
            side=OrderSide.SELL,
            price=Decimal("100.10"),
            quantity=100,
        ),
    )

    assert len(actions) == 2
    assert all(action.action == QuoteActionType.KEEP for action in actions)


def test_replace_quote_when_price_changes() -> None:
    lifecycle = QuoteLifecycle()

    decision = QuoteDecision(
        bid_price=Decimal("100.01"),
        ask_price=Decimal("100.10"),
        quantity=100,
    )

    actions = lifecycle.evaluate(
        decision,
        active_bid=ActiveQuote(
            order_id=1,
            side=OrderSide.BUY,
            price=Decimal("100.00"),
            quantity=100,
        ),
        active_ask=ActiveQuote(
            order_id=2,
            side=OrderSide.SELL,
            price=Decimal("100.10"),
            quantity=100,
        ),
    )

    assert actions[0].action == QuoteActionType.REPLACE
    assert actions[0].order_id == 1
    assert actions[0].price == Decimal("100.01")

    assert actions[1].action == QuoteActionType.KEEP


def test_replace_quote_when_quantity_changes() -> None:
    lifecycle = QuoteLifecycle()

    decision = QuoteDecision(
        bid_price=Decimal("100.00"),
        ask_price=Decimal("100.10"),
        quantity=50,
    )

    actions = lifecycle.evaluate(
        decision,
        active_bid=ActiveQuote(
            order_id=1,
            side=OrderSide.BUY,
            price=Decimal("100.00"),
            quantity=100,
        ),
        active_ask=ActiveQuote(
            order_id=2,
            side=OrderSide.SELL,
            price=Decimal("100.10"),
            quantity=100,
        ),
    )

    assert actions[0].action == QuoteActionType.REPLACE
    assert actions[1].action == QuoteActionType.REPLACE


def test_cancel_when_desired_quote_disappears() -> None:
    lifecycle = QuoteLifecycle()

    decision = QuoteDecision(
        bid_price=None,
        ask_price=Decimal("100.10"),
        quantity=100,
    )

    actions = lifecycle.evaluate(
        decision,
        active_bid=ActiveQuote(
            order_id=1,
            side=OrderSide.BUY,
            price=Decimal("100.00"),
            quantity=100,
        ),
        active_ask=None,
    )

    assert len(actions) == 2
    assert actions[0].action == QuoteActionType.CANCEL
    assert actions[0].order_id == 1
    assert actions[1].action == QuoteActionType.CREATE