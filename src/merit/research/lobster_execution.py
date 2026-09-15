from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from merit.execution.queue import QueuePosition
from merit.execution.queue_model import QueueModel
from merit.portfolio.enums import OrderSide
from merit.portfolio.models import Fill
from merit.portfolio.portfolio import Portfolio
from merit.risk.limits import RiskManager
from merit.strategy.baseline import QuoteDecision
from merit.strategy.quote_lifecycle import (
    ActiveQuote,
    QuoteActionType,
    QuoteLifecycle,
)


@dataclass
class ActiveOrder:
    order_id: str
    side: OrderSide
    price: Decimal
    quantity: int
    remaining_quantity: int
    queue: QueuePosition


@dataclass(frozen=True)
class ExecutionReplayFill:
    fill_id: str
    order_id: str
    timestamp: datetime
    symbol: str
    side: OrderSide
    quantity: int
    price: Decimal


class LobsterExecutionEngine:
    def __init__(
        self,
        symbol: str,
        portfolio: Portfolio,
        risk_manager: RiskManager,
        queue_model: QueueModel,
        quantity: int,
        fee_bps: Decimal = Decimal("0"),
    ) -> None:
        if not symbol:
            raise ValueError("symbol cannot be empty")

        if quantity <= 0:
            raise ValueError("quantity must be positive")

        if fee_bps < 0:
            raise ValueError("fee_bps cannot be negative")

        self.symbol = symbol
        self.portfolio = portfolio
        self.risk_manager = risk_manager
        self.queue_model = queue_model
        self.quantity = quantity
        self.fee_bps = fee_bps

        self._active: dict[
            OrderSide,
            ActiveOrder | None,
        ] = {
            OrderSide.BUY: None,
            OrderSide.SELL: None,
        }

        self._next_order_id = 1
        self._next_fill_id = 1

    def update_quotes(
        self,
        decision: QuoteDecision,
        bid_visible_quantity: int,
        ask_visible_quantity: int,
        timestamp: datetime,
    ) -> None:
        lifecycle = QuoteLifecycle()

        active_bid = self._active_quote(OrderSide.BUY)
        active_ask = self._active_quote(OrderSide.SELL)

        actions = lifecycle.evaluate(
            decision,
            active_bid,
            active_ask,
        )

        for action in actions:
            if action.action == QuoteActionType.CANCEL:
                self._cancel(action.side)

            elif action.action == QuoteActionType.CREATE:
                visible_quantity = (
                    bid_visible_quantity
                    if action.side == OrderSide.BUY
                    else ask_visible_quantity
                )

                self._create(
                    side=action.side,
                    price=action.price,
                    quantity=action.quantity,
                    visible_quantity=visible_quantity,
                    timestamp=timestamp,
                )

            elif action.action == QuoteActionType.REPLACE:
                self._cancel(action.side)

                visible_quantity = (
                    bid_visible_quantity
                    if action.side == OrderSide.BUY
                    else ask_visible_quantity
                )

                self._create(
                    side=action.side,
                    price=action.price,
                    quantity=action.quantity,
                    visible_quantity=visible_quantity,
                    timestamp=timestamp,
                )

    def process_execution(
        self,
        timestamp: datetime,
        execution_price: Decimal,
        execution_quantity: int,
    ) -> ExecutionReplayFill | None:
        if execution_quantity <= 0:
            return None

        side = self._execution_side(
            execution_price,
        )

        if side is None:
            return None

        active = self._active[side]

        if active is None:
            return None

        if active.price != execution_price:
            return None

        available = active.queue.consume(
            execution_quantity,
        )

        if available <= 0:
            return None

        quantity = min(
            active.remaining_quantity,
            available,
        )

        if quantity <= 0:
            return None

        fee = (
            Decimal(quantity)
            * active.price
            * self.fee_bps
            / Decimal("10000")
        )

        fill_id = f"{self.symbol}-{self._next_fill_id}"

        fill = Fill(
            fill_id=fill_id,
            order_id=active.order_id,
            timestamp=timestamp,
            symbol=self.symbol,
            side=active.side,
            quantity=quantity,
            price=active.price,
            fee=fee,
        )

        self.portfolio.process_fill(fill)

        result = ExecutionReplayFill(
            fill_id=fill_id,
            order_id=active.order_id,
            timestamp=timestamp,
            symbol=self.symbol,
            side=active.side,
            quantity=quantity,
            price=active.price,
        )

        self._next_fill_id += 1
        active.remaining_quantity -= quantity

        if active.remaining_quantity == 0:
            self._active[side] = None

        return result

    def mark_to_market(
        self,
        price: Decimal,
    ) -> None:
        self.portfolio.update_mark(
            self.symbol,
            price,
        )

    def active_order(
        self,
        side: OrderSide,
    ) -> ActiveOrder | None:
        return self._active[side]

    def _active_quote(
        self,
        side: OrderSide,
    ) -> ActiveQuote | None:
        active = self._active[side]

        if active is None:
            return None

        return ActiveQuote(
            order_id=int(active.order_id),
            side=active.side,
            price=active.price,
            quantity=active.quantity,
        )

    def _create(
        self,
        side: OrderSide,
        price: Decimal | None,
        quantity: int | None,
        visible_quantity: int,
        timestamp: datetime,
    ) -> None:
        if price is None or quantity is None:
            return

        position = self.portfolio.position(
            self.symbol,
        )

        risk = self.risk_manager.check_order(
            side=side,
            quantity=quantity,
            current_position=position.quantity,
        )

        if not risk.approved:
            return

        order_id = str(
            self._next_order_id
        )

        self._next_order_id += 1

        self._active[side] = ActiveOrder(
            order_id=order_id,
            side=side,
            price=price,
            quantity=risk.quantity,
            remaining_quantity=risk.quantity,
            queue=self.queue_model.estimate(
                visible_quantity
            ),
        )

    def _cancel(
        self,
        side: OrderSide,
    ) -> None:
        self._active[side] = None

    def _execution_side(
        self,
        execution_price: Decimal,
    ) -> OrderSide | None:
        for side in (
            OrderSide.BUY,
            OrderSide.SELL,
        ):
            active = self._active[side]

            if (
                active is not None
                and active.price == execution_price
            ):
                return side

        return None