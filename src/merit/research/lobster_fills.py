from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date

from merit.data.lobster import read_messages
from merit.data.lobster_orderbook import read_orderbooks
from merit.data.normalized import OrderExecuteEvent
from merit.execution.queue_model import QueueModel
from merit.research.historical_fills import HistoricalFill
from merit.research.lobster_replay import build_feature_snapshot_from_book
from merit.research.opportunities import QuoteOpportunity


@dataclass(frozen=True)
class LOBSTERFillCandidate:
    message_index: int
    fill: HistoricalFill
    queue_ahead: int


def replay_lobster_fills(
    message_path: str,
    orderbook_path: str,
    symbol: str,
    trading_date: date,
    queue_model: QueueModel,
    order_quantity: int,
    levels: int = 10,
) -> Iterator[LOBSTERFillCandidate]:
    if order_quantity <= 0:
        raise ValueError("order_quantity must be positive")

    messages = read_messages(
        message_path,
        symbol,
        trading_date,
    )

    snapshots = read_orderbooks(
        orderbook_path,
        levels=levels,
    )

    for message_index, (event, snapshot) in enumerate(
        zip(messages, snapshots),
        start=1,
    ):
        bids, asks = snapshot

        features = build_feature_snapshot_from_book(
            bids=bids,
            asks=asks,
        )

        if features.mid_price is None:
            continue

        if not isinstance(event, OrderExecuteEvent):
            continue

        executed_order_price = event.execution_price

        if executed_order_price is None:
            continue

        best_bid = bids[0][0] if bids else None
        best_ask = asks[0][0] if asks else None

        if best_bid is None or best_ask is None:
            continue

        side = _execution_side(
            executed_price=executed_order_price,
            best_bid=best_bid,
            best_ask=best_ask,
        )

        if side is None:
            continue

        quote_price = (
            best_bid
            if side.value == "BUY"
            else best_ask
        )

        opportunity = QuoteOpportunity(
            timestamp=event.timestamp,
            symbol=symbol,
            side=side,
            quote_price=quote_price,
            features=features,
        )

        queue_ahead = (
            features.bid_size
            if side.value == "BUY"
            else features.ask_size
        )

        queue = queue_model.estimate(queue_ahead)

        available = queue.consume(event.quantity)

        fill_quantity = min(
            order_quantity,
            available,
        )

        if fill_quantity <= 0:
            continue

        historical_fill = HistoricalFill(
            opportunity=opportunity,
            timestamp=event.timestamp,
            side=side,
            price=executed_order_price,
            quantity=fill_quantity,
            order_id=event.order_id,
            execution_id=event.execution_id,
        )

        yield LOBSTERFillCandidate(
            message_index=message_index,
            fill=historical_fill,
            queue_ahead=queue_ahead,
        )


def _execution_side(
    executed_price,
    best_bid,
    best_ask,
):
    from merit.portfolio.enums import OrderSide

    if executed_price <= best_bid:
        return OrderSide.BUY

    if executed_price >= best_ask:
        return OrderSide.SELL

    return None