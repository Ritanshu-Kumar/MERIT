from dataclasses import dataclass
from decimal import Decimal

from merit.book.l2_book import L2OrderBook
from merit.features.depth import (
    depth_weighted_imbalance,
    level_imbalance,
)
from merit.features.microstructure import calculate_features


@dataclass(frozen=True)
class FeatureSnapshot:
    mid_price: Decimal | None
    spread: Decimal | None
    relative_spread: Decimal | None
    bid_size: int
    ask_size: int
    imbalance: Decimal | None
    microprice: Decimal | None
    imbalance_l5: Decimal | None
    imbalance_l10: Decimal | None
    weighted_imbalance_l5: Decimal | None
    weighted_imbalance_l10: Decimal | None


def build_feature_snapshot(book: L2OrderBook) -> FeatureSnapshot:
    features = calculate_features(book)

    return FeatureSnapshot(
        mid_price=features.mid_price,
        spread=features.spread,
        relative_spread=features.relative_spread,
        bid_size=features.bid_size,
        ask_size=features.ask_size,
        imbalance=features.imbalance,
        microprice=features.microprice,
        imbalance_l5=level_imbalance(book, 5),
        imbalance_l10=level_imbalance(book, 10),
        weighted_imbalance_l5=depth_weighted_imbalance(book, 5),
        weighted_imbalance_l10=depth_weighted_imbalance(book, 10),
    )

def build_feature_snapshot_from_levels(
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
            imbalance_l5=_level_imbalance(
                bids,
                asks,
                5,
            ),
            imbalance_l10=_level_imbalance(
                bids,
                asks,
                10,
            ),
            weighted_imbalance_l5=_weighted_imbalance(
                bids,
                asks,
                5,
            ),
            weighted_imbalance_l10=_weighted_imbalance(
                bids,
                asks,
                10,
            ),
        )

    mid_price = (
        best_bid[0] + best_ask[0]
    ) / Decimal("2")

    spread = best_ask[0] - best_bid[0]

    relative_spread = spread / mid_price

    total_size = bid_size + ask_size

    imbalance = (
        Decimal(bid_size - ask_size)
        / Decimal(total_size)
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
        relative_spread=relative_spread,
        bid_size=bid_size,
        ask_size=ask_size,
        imbalance=imbalance,
        microprice=microprice,
        imbalance_l5=_level_imbalance(
            bids,
            asks,
            5,
        ),
        imbalance_l10=_level_imbalance(
            bids,
            asks,
            10,
        ),
        weighted_imbalance_l5=_weighted_imbalance(
            bids,
            asks,
            5,
        ),
        weighted_imbalance_l10=_weighted_imbalance(
            bids,
            asks,
            10,
        ),
    )


def _level_imbalance(
    bids: tuple[tuple[Decimal, int], ...],
    asks: tuple[tuple[Decimal, int], ...],
    levels: int,
) -> Decimal | None:
    bid_volume = sum(
        quantity
        for _, quantity in bids[:levels]
    )

    ask_volume = sum(
        quantity
        for _, quantity in asks[:levels]
    )

    total = bid_volume + ask_volume

    if total == 0:
        return None

    return Decimal(
        bid_volume - ask_volume
    ) / Decimal(total)


def _weighted_imbalance(
    bids: tuple[tuple[Decimal, int], ...],
    asks: tuple[tuple[Decimal, int], ...],
    levels: int,
) -> Decimal | None:
    bid_weighted = sum(
        Decimal(quantity) / Decimal(index)
        for index, (_, quantity)
        in enumerate(
            bids[:levels],
            start=1,
        )
    )

    ask_weighted = sum(
        Decimal(quantity) / Decimal(index)
        for index, (_, quantity)
        in enumerate(
            asks[:levels],
            start=1,
        )
    )

    total = bid_weighted + ask_weighted

    if total == 0:
        return None

    return (
        bid_weighted - ask_weighted
    ) / Decimal(total)