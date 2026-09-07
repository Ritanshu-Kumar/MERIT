from collections import deque
from collections.abc import Iterable, Iterator

from merit.book.l2_book import L2OrderBook
from merit.data.normalized import NormalizedMarketEvent, OrderExecuteEvent
from merit.research.historical_fills import (
    HistoricalFill,
    detect_historical_fill,
)
from merit.research.opportunities import QuoteOpportunity


def replay_historical_fills(
    opportunities: Iterable[QuoteOpportunity],
    events: Iterable[NormalizedMarketEvent],
) -> Iterator[HistoricalFill]:
    opportunity_iter = iter(opportunities)
    pending: deque[QuoteOpportunity] = deque()
    active: list[QuoteOpportunity] = []

    next_opportunity = next(opportunity_iter, None)

    book: L2OrderBook | None = None

    for event in events:
        if book is None:
            book = L2OrderBook(event.symbol)

        while (
            next_opportunity is not None
            and next_opportunity.timestamp <= event.timestamp
        ):
            pending.append(next_opportunity)
            next_opportunity = next(opportunity_iter, None)

        while pending:
            active.append(pending.popleft())

        if isinstance(event, OrderExecuteEvent):
            remaining: list[QuoteOpportunity] = []

            for opportunity in active:
                if opportunity.symbol != event.symbol:
                    remaining.append(opportunity)
                    continue

                fill = detect_historical_fill(
                    book=book,
                    opportunity=opportunity,
                    event=event,
                )

                if fill is not None:
                    yield fill

                remaining.append(opportunity)

            active = remaining

        book.apply(event)