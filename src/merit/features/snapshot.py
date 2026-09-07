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