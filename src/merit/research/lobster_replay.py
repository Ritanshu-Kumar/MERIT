from collections.abc import Iterable, Iterator
from datetime import datetime
from decimal import Decimal

from merit.data.normalized import NormalizedMarketEvent
from merit.data.lobster_orderbook import read_orderbooks
from merit.features.snapshot import FeatureSnapshot
from merit.research.market_replay import MarketState


def build_feature_snapshot_from_book(
    bids: tuple[tuple[Decimal, int], ...],
    asks: tuple[tuple[Decimal, int], ...],
) -> FeatureSnapshot:
    best_bid = bids[0] if bids else None
    best_ask = asks[0] if asks else None

    bid_size = best_bid[1] if best_bid is not None else 0
    ask_size = best_ask[1] if best_ask is not None else 0

    if best_bid is None or best_ask is None:
        return FeatureSnapshot(
            mid_price=None,
            spread=None,
            relative_spread=None,
            bid_size=bid_size,
            ask_size=ask_size,
            imbalance=None,
            microprice=None,
            imbalance_l5=_level_imbalance(bids, asks, 5),
            imbalance_l10=_level_imbalance(bids, asks, 10),
            weighted_imbalance_l5=_weighted_imbalance(bids, asks, 5),
            weighted_imbalance_l10=_weighted_imbalance(bids, asks, 10),
        )

    mid_price = (best_bid[0] + best_ask[0]) / Decimal("2")
    spread = best_ask[0] - best_bid[0]

    total_size = bid_size + ask_size

    imbalance = (
        Decimal(bid_size - ask_size) / Decimal(total_size)
        if total_size > 0
        else None
    )

    microprice = (
        (
            best_ask[0] * Decimal(bid_size)
            + best_bid[0] * Decimal(ask_size)
        )
        / Decimal(total_size)
        if total_size > 0
        else None
    )

    return FeatureSnapshot(
        mid_price=mid_price,
        spread=spread,
        relative_spread=spread / mid_price,
        bid_size=bid_size,
        ask_size=ask_size,
        imbalance=imbalance,
        microprice=microprice,
        imbalance_l5=_level_imbalance(bids, asks, 5),
        imbalance_l10=_level_imbalance(bids, asks, 10),
        weighted_imbalance_l5=_weighted_imbalance(bids, asks, 5),
        weighted_imbalance_l10=_weighted_imbalance(bids, asks, 10),
    )


def replay_lobster_snapshots(
    events: Iterable[NormalizedMarketEvent],
    orderbook_path: str,
    levels: int = 10,
) -> Iterator[MarketState]:
    snapshots = read_orderbooks(orderbook_path, levels=levels)

    for event, snapshot in zip(events, snapshots):
        bids, asks = snapshot

        yield MarketState(
            timestamp=event.timestamp,
            symbol=event.symbol,
            mid_price=(
                (bids[0][0] + asks[0][0]) / Decimal("2")
                if bids and asks
                else None
            ),
            features=build_feature_snapshot_from_book(
                bids=bids,
                asks=asks,
            ),
        )


def _level_imbalance(
    bids: tuple[tuple[Decimal, int], ...],
    asks: tuple[tuple[Decimal, int], ...],
    levels: int,
) -> Decimal | None:
    bid_volume = sum(quantity for _, quantity in bids[:levels])
    ask_volume = sum(quantity for _, quantity in asks[:levels])

    total_volume = bid_volume + ask_volume

    if total_volume == 0:
        return None

    return Decimal(bid_volume - ask_volume) / Decimal(total_volume)


def _weighted_imbalance(
    bids: tuple[tuple[Decimal, int], ...],
    asks: tuple[tuple[Decimal, int], ...],
    levels: int,
) -> Decimal | None:
    bid_weighted = sum(
        Decimal(quantity) / Decimal(index)
        for index, (_, quantity) in enumerate(bids[:levels], start=1)
    )

    ask_weighted = sum(
        Decimal(quantity) / Decimal(index)
        for index, (_, quantity) in enumerate(asks[:levels], start=1)
    )

    total_weighted = bid_weighted + ask_weighted

    if total_weighted == 0:
        return None

    return (bid_weighted - ask_weighted) / total_weighted