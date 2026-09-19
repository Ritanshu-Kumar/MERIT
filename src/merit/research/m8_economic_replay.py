from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from pathlib import Path

from merit.data.lobster import OrderExecuteEvent, read_messages
from merit.data.lobster_orderbook import read_orderbooks
from merit.execution.queue import QueuePosition
from merit.execution.queue_model import QueueModel
from merit.portfolio.enums import OrderSide
from merit.portfolio.models import Fill
from merit.portfolio.portfolio import Portfolio
from merit.research.lobster_m8 import build_feature_snapshot_from_book
from merit.risk.limits import RiskManager


@dataclass
class ActiveOrder:
    side: OrderSide
    price: Decimal
    quantity: int
    queue: QueuePosition


@dataclass(frozen=True)
class EconomicResult:
    name: str
    fills: int
    quantity: int
    realized_pnl: Decimal
    unrealized_pnl: Decimal
    total_equity: Decimal


def run_symbol(
    message_path: str | Path,
    orderbook_path: str | Path,
    symbol: str,
    trading_date: date,
    maker,
    queue_model: QueueModel,
    risk_manager: RiskManager,
    starting_cash: Decimal,
) -> EconomicResult:
    events = tuple(
        read_messages(
            message_path,
            symbol,
            trading_date,
        )
    )

    snapshots = tuple(
        read_orderbooks(
            orderbook_path,
            levels=10,
        )
    )

    aligned_count = min(
        len(events),
        len(snapshots),
    )

    portfolio = Portfolio(
        starting_cash=starting_cash,
    )

    active: dict[OrderSide, ActiveOrder | None] = {
        OrderSide.BUY: None,
        OrderSide.SELL: None,
    }

    order_id = 1
    fill_id = 1

    for index in range(aligned_count):
        event = events[index]
        bids, asks = snapshots[index]

        if not bids or not asks:
            continue

        features = build_feature_snapshot_from_book(
            bids,
            asks,
        )

        bid_price = bids[0][0]
        ask_price = asks[0][0]

        decision = maker.quote(
            features,
            bid_price,
            ask_price,
        )

        desired = {
            OrderSide.BUY: decision.bid_price,
            OrderSide.SELL: decision.ask_price,
        }

        for side in (OrderSide.BUY, OrderSide.SELL):
            current = active[side]
            target_price = desired[side]

            if target_price is None:
                active[side] = None
                continue

            if (
                current is not None
                and current.price == target_price
                and current.quantity > 0
            ):
                continue

            position = portfolio.position(symbol)

            risk = risk_manager.check_order(
                side=side,
                quantity=decision.quantity,
                current_position=position.quantity,
            )

            if not risk.approved:
                active[side] = None
                continue

            visible_quantity = (
                bids[0][1]
                if side == OrderSide.BUY
                else asks[0][1]
            )

            active[side] = ActiveOrder(
                side=side,
                price=target_price,
                quantity=risk.quantity,
                queue=queue_model.estimate(
                    visible_quantity
                ),
            )

            order_id += 1

        if isinstance(event, OrderExecuteEvent):
            if event.execution_price is None:
                continue

            if event.execution_price == bid_price:
                side = OrderSide.BUY
            elif event.execution_price == ask_price:
                side = OrderSide.SELL
            else:
                continue

            resting = active[side]

            if resting is None:
                continue

            if resting.price != event.execution_price:
                continue

            available = resting.queue.consume(
                event.quantity
            )

            if available <= 0:
                continue

            quantity = min(
                resting.quantity,
                available,
            )

            if quantity <= 0:
                continue

            fill = Fill(
                fill_id=str(fill_id),
                order_id=str(order_id),
                timestamp=event.timestamp,
                symbol=symbol,
                side=side,
                quantity=quantity,
                price=resting.price,
                fee=Decimal(0),
            )

            portfolio.process_fill(fill)

            fill_id += 1
            resting.quantity -= quantity

            if resting.quantity <= 0:
                active[side] = None

        portfolio.update_mark(
            symbol,
            features.mid_price,
        )

    return EconomicResult(
        name=maker.__class__.__name__,
        fills=len(portfolio.ledger.entries),
        quantity=sum(
            entry.quantity
            for entry in portfolio.ledger.entries
        ),
        realized_pnl=portfolio.realized_pnl(),
        unrealized_pnl=portfolio.unrealized_pnl(),
        total_equity=portfolio.total_equity(),
    )