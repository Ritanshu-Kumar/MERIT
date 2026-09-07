from collections.abc import Iterable, Iterator

from merit.book.l2_book import L2OrderBook
from merit.data.normalized import (
    NormalizedMarketEvent,
    OrderAddEvent,
    OrderCancelEvent,
    OrderDeleteEvent,
    OrderExecuteEvent,
    OrderReplaceEvent,
)
from merit.features.snapshot import build_feature_snapshot
from merit.portfolio.enums import OrderSide
from merit.research.opportunities import QuoteOpportunity


BookEvent = (
    OrderAddEvent
    | OrderCancelEvent
    | OrderDeleteEvent
    | OrderExecuteEvent
    | OrderReplaceEvent
)


def build_quote_opportunities(
    events: Iterable[NormalizedMarketEvent],
    symbol: str,
) -> Iterator[QuoteOpportunity]:
    book = L2OrderBook(symbol)

    for event in events:
        if not isinstance(event, BookEvent):
            continue

        book.apply(event)

        features = build_feature_snapshot(book)

        best_bid = book.best_bid()
        best_ask = book.best_ask()

        if best_bid is not None:
            yield QuoteOpportunity(
                timestamp=event.timestamp,
                symbol=symbol,
                side=OrderSide.BUY,
                quote_price=best_bid[0],
                features=features,
            )

        if best_ask is not None:
            yield QuoteOpportunity(
                timestamp=event.timestamp,
                symbol=symbol,
                side=OrderSide.SELL,
                quote_price=best_ask[0],
                features=features,
            )