from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from merit.book.l2_book import L2OrderBook
from merit.data.normalized import NormalizedMarketEvent
from merit.features.snapshot import FeatureSnapshot
from merit.features.snapshot import build_feature_snapshot


@dataclass(frozen=True)
class MarketState:
    timestamp: datetime
    symbol: str
    mid_price: Decimal | None
    features: FeatureSnapshot


def replay_market(
    events: Iterable[NormalizedMarketEvent],
    symbol: str,
) -> Iterator[MarketState]:
    book = L2OrderBook(symbol)

    for event in events:
        if event.symbol != symbol:
            continue

        book.apply(event)

        features = build_feature_snapshot(book)

        yield MarketState(
            timestamp=event.timestamp,
            symbol=symbol,
            mid_price=features.mid_price,
            features=features,
        )