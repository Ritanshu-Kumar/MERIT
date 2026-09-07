from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from merit.data.normalized import TradeEvent
from merit.execution import queue_model
from merit.execution import queue_model
from merit.portfolio.enums import OrderSide, OrderStatus, OrderType
from merit.portfolio.models import Fill, Order
from merit.execution.latency import LatencyModel
from merit.execution.participation import ParticipationModel
from merit.execution.queue import QueuePosition
from merit.execution.queue_model import QueueModel

@dataclass(frozen=True)
class ExecutionResult:
    fills: tuple[Fill, ...]


class ExecutionSimulator:
    def __init__(
        self,
        latency_model: LatencyModel | None = None,
        participation_model: ParticipationModel | None = None,
        queue_model: QueueModel | None = None,
    ) -> None:
        self._orders: dict[int, Order] = {}
        self._activation_times: dict[int, datetime] = {}
        self._queue_positions: dict[int, QueuePosition] = {}
        self._next_fill_id = 1
        self._latency_model = latency_model
        self._participation_model = participation_model
        self._queue_model = queue_model

    def submit(
        self,
        order: Order,
        visible_quantity: int = 0,
    ) -> None:
        if order.order_type != OrderType.LIMIT:
            raise ValueError("ExecutionSimulator only supports limit orders")

        if order.status.is_terminal:
            raise ValueError("cannot submit a terminal order")

        if order.order_id in self._orders:
            raise ValueError(f"order {order.order_id} already submitted")

        self._orders[order.order_id] = order

        if self._latency_model is not None:
            self._activation_times[order.order_id] = (
                self._latency_model.activation_time(order.created_at)
            )
        else:
            self._activation_times[order.order_id] = order.created_at

        if self._queue_model is not None:
            self._queue_positions[order.order_id] = (
                self._queue_model.estimate(visible_quantity)
            )
        else:
            self._queue_positions[order.order_id] = QueuePosition(0)

    def cancel(self, order_id: int) -> None:
        order = self._orders.get(order_id)

        if order is None:
            raise KeyError(f"unknown order: {order_id}")

        order.status = OrderStatus.CANCELLED

    def process_trade(self, trade: TradeEvent) -> ExecutionResult:
        fills: list[Fill] = []

        for order in self._orders.values():
            if order.symbol != trade.symbol:
                continue

            if order.status.is_terminal:
                continue

            if order.remaining_quantity <= 0:
                continue

            activation_time = self._activation_times[order.order_id]

            if trade.timestamp < activation_time:
                continue

            if not self._crosses(order, trade.price):
                continue

            available_quantity = (
                self._participation_model.available_quantity(trade.quantity)
                if self._participation_model is not None
                else trade.quantity
            )

            queue = self._queue_positions[order.order_id]
            available_after_queue = queue.consume(available_quantity)

            if available_after_queue <= 0:
                continue

            quantity = min(
                order.remaining_quantity,
                available_after_queue,
            )

            fill = Fill(
                fill_id=self._next_fill_id,
                order_id=order.order_id,
                timestamp=trade.timestamp,
                symbol=order.symbol,
                side=order.side,
                quantity=quantity,
                price=order.limit_price,
                fee=Decimal("0"),
            )

            self._next_fill_id += 1
            order.filled_quantity += quantity

            if order.is_fully_filled:
                order.status = OrderStatus.FILLED
            else:
                order.status = OrderStatus.PARTIALLY_FILLED

            fills.append(fill)

        return ExecutionResult(fills=tuple(fills))

    @staticmethod
    def _crosses(order: Order, trade_price: Decimal) -> bool:
        if order.limit_price is None:
            return False

        if order.side == OrderSide.BUY:
            return trade_price <= order.limit_price

        return trade_price >= order.limit_price