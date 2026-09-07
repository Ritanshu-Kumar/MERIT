from dataclasses import dataclass
from decimal import Decimal
from enum import Enum

from merit.portfolio.enums import OrderSide
from merit.strategy.baseline import QuoteDecision


class QuoteActionType(str, Enum):
    KEEP = "KEEP"
    CANCEL = "CANCEL"
    CREATE = "CREATE"
    REPLACE = "REPLACE"


@dataclass(frozen=True)
class ActiveQuote:
    order_id: int
    side: OrderSide
    price: Decimal
    quantity: int


@dataclass(frozen=True)
class QuoteAction:
    action: QuoteActionType
    side: OrderSide
    order_id: int | None
    price: Decimal | None
    quantity: int | None


class QuoteLifecycle:
    def evaluate(
        self,
        decision: QuoteDecision,
        active_bid: ActiveQuote | None,
        active_ask: ActiveQuote | None,
    ) -> tuple[QuoteAction, ...]:
        actions: list[QuoteAction] = []

        actions.append(
            self._evaluate_side(
                side=OrderSide.BUY,
                desired_price=decision.bid_price,
                desired_quantity=decision.quantity,
                active=active_bid,
            )
        )

        actions.append(
            self._evaluate_side(
                side=OrderSide.SELL,
                desired_price=decision.ask_price,
                desired_quantity=decision.quantity,
                active=active_ask,
            )
        )

        return tuple(action for action in actions if action is not None)

    def _evaluate_side(
        self,
        side: OrderSide,
        desired_price: Decimal | None,
        desired_quantity: int,
        active: ActiveQuote | None,
    ) -> QuoteAction | None:
        if desired_price is None:
            if active is None:
                return None

            return QuoteAction(
                action=QuoteActionType.CANCEL,
                side=side,
                order_id=active.order_id,
                price=None,
                quantity=None,
            )

        if active is None:
            return QuoteAction(
                action=QuoteActionType.CREATE,
                side=side,
                order_id=None,
                price=desired_price,
                quantity=desired_quantity,
            )

        if (
            active.price == desired_price
            and active.quantity == desired_quantity
        ):
            return QuoteAction(
                action=QuoteActionType.KEEP,
                side=side,
                order_id=active.order_id,
                price=active.price,
                quantity=active.quantity,
            )

        return QuoteAction(
            action=QuoteActionType.REPLACE,
            side=side,
            order_id=active.order_id,
            price=desired_price,
            quantity=desired_quantity,
        )