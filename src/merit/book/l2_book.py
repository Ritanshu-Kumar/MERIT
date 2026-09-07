from dataclasses import dataclass
from decimal import Decimal

from merit.data.normalized import (
    MarketEventType,
    OrderAddEvent,
    OrderCancelEvent,
    OrderDeleteEvent,
    OrderExecuteEvent,
    OrderReplaceEvent,
)

from merit.portfolio.enums import OrderSide


@dataclass
class BookOrder:
    order_id: int
    side: str
    price: Decimal
    quantity: int


class L2OrderBook:
    def __init__(self, symbol: str) -> None:
        if not symbol:
            raise ValueError("symbol cannot be empty")

        self.symbol = symbol
        self._orders: dict[int, BookOrder] = {}
        self._bids: dict[Decimal, int] = {}
        self._asks: dict[Decimal, int] = {}

    def apply(
        self,
        event: (
            OrderAddEvent
            | OrderCancelEvent
            | OrderDeleteEvent
            | OrderExecuteEvent
            | OrderReplaceEvent
        ),
    ) -> None:
        if event.symbol != self.symbol:
            raise ValueError(
                f"Event symbol {event.symbol!r} does not match "
                f"book symbol {self.symbol!r}"
            )

        if event.event_type == MarketEventType.ADD:
            self._add(event)
        elif event.event_type == MarketEventType.CANCEL:
            self._cancel(event)
        elif event.event_type == MarketEventType.DELETE:
            self._delete(event)
        elif event.event_type == MarketEventType.EXECUTE:
            self._execute(event)
        elif event.event_type == MarketEventType.REPLACE:
            self._replace(event)
        else:
            raise ValueError(
                f"Unsupported event type: {event.event_type}"
            )

    def best_bid(self) -> tuple[Decimal, int] | None:
        if not self._bids:
            return None

        price = max(self._bids)
        return price, self._bids[price]

    def best_ask(self) -> tuple[Decimal, int] | None:
        if not self._asks:
            return None

        price = min(self._asks)
        return price, self._asks[price]

    def mid_price(self) -> Decimal | None:
        bid = self.best_bid()
        ask = self.best_ask()

        if bid is None or ask is None:
            return None

        return (bid[0] + ask[0]) / Decimal(2)

    def spread(self) -> Decimal | None:
        bid = self.best_bid()
        ask = self.best_ask()

        if bid is None or ask is None:
            return None

        return ask[0] - bid[0]

    def depth(
        self,
        levels: int,
    ) -> tuple[
        tuple[tuple[Decimal, int], ...],
        tuple[tuple[Decimal, int], ...],
    ]:
        if levels <= 0:
            raise ValueError("levels must be greater than zero")

        bids = tuple(
            sorted(
                self._bids.items(),
                key=lambda item: item[0],
                reverse=True,
            )[:levels]
        )

        asks = tuple(
            sorted(
                self._asks.items(),
                key=lambda item: item[0],
            )[:levels]
        )

        return bids, asks

    def order(self, order_id: int) -> BookOrder | None:
        return self._orders.get(order_id)

    def order_count(self) -> int:
        return len(self._orders)

    def _add(self, event: OrderAddEvent) -> None:
        if event.order_id in self._orders:
            raise ValueError(
                f"Order already exists: {event.order_id}"
            )

        order = BookOrder(
            order_id=event.order_id,
            side=event.side,
            price=event.price,
            quantity=event.quantity,
        )

        self._orders[event.order_id] = order
        self._update_level(order, order.quantity)

    def _cancel(self, event: OrderCancelEvent) -> None:
        order = self._get_order(event.order_id)

        if event.quantity > order.quantity:
            raise ValueError(
                f"Cancellation exceeds order quantity: {event.order_id}"
            )

        self._update_level(order, -event.quantity)
        order.quantity -= event.quantity

        if order.quantity == 0:
            del self._orders[order.order_id]

    def _delete(self, event: OrderDeleteEvent) -> None:
        order = self._get_order(event.order_id)

        self._update_level(order, -order.quantity)
        del self._orders[order.order_id]

    def _execute(self, event: OrderExecuteEvent) -> None:
        order = self._get_order(event.order_id)

        if event.quantity > order.quantity:
            raise ValueError(
                f"Execution exceeds order quantity: {event.order_id}"
            )

        self._update_level(order, -event.quantity)
        order.quantity -= event.quantity

        if order.quantity == 0:
            del self._orders[order.order_id]

    def _replace(self, event: OrderReplaceEvent) -> None:
        old_order = self._get_order(event.old_order_id)

        self._update_level(old_order, -old_order.quantity)
        del self._orders[old_order.order_id]

        new_order = BookOrder(
            order_id=event.new_order_id,
            side=old_order.side,
            price=event.price,
            quantity=event.quantity,
        )

        self._orders[new_order.order_id] = new_order
        self._update_level(new_order, new_order.quantity)

    def _get_order(self, order_id: int) -> BookOrder:
        order = self._orders.get(order_id)

        if order is None:
            raise ValueError(
                f"Unknown order: {order_id}"
            )

        return order

    def _update_level(
        self,
        order: BookOrder,
        quantity_delta: int,
    ) -> None:
        levels = self._bids if order.side == "BUY" else self._asks

        new_quantity = levels.get(order.price, 0) + quantity_delta

        if new_quantity < 0:
            raise ValueError(
                f"Negative quantity at price level {order.price}"
            )

        if new_quantity == 0:
            levels.pop(order.price, None)
        else:
            levels[order.price] = new_quantity

    def load_snapshot(
        self,
        bids: tuple[tuple[Decimal, int], ...],
        asks: tuple[tuple[Decimal, int], ...],
    ) -> None:
        self._orders.clear()
        self._bids.clear()
        self._asks.clear()

        for level_index, (price, quantity) in enumerate(bids):
            if quantity <= 0:
                continue

            order_id = f"__snapshot_bid_{level_index}"
            self._orders[order_id] = BookOrder(
                order_id=order_id,
                side=OrderSide.BUY,
                price=price,
                quantity=quantity,
            )
            self._bids[price] = quantity

        for level_index, (price, quantity) in enumerate(asks):
            if quantity <= 0:
                continue

            order_id = f"__snapshot_ask_{level_index}"
            self._orders[order_id] = BookOrder(
                order_id=order_id,
                side=OrderSide.SELL,
                price=price,
                quantity=quantity,
            )
            self._asks[price] = quantity