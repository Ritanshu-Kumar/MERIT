from dataclasses import dataclass
from datetime import timedelta
from decimal import Decimal

from merit.features.snapshot import FeatureSnapshot
from merit.portfolio.models import Fill
from merit.research.historical_fills import HistoricalFill
from merit.research.targets import FillMarkout


@dataclass(frozen=True)
class ResearchObservation:
    fill_id: str
    order_id: str
    timestamp: object
    symbol: str
    side: str
    quantity: int
    fill_price: Decimal

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

    markout_100ms: Decimal | None
    markout_1s: Decimal | None
    markout_5s: Decimal | None


def build_research_observation(
    fill: Fill | HistoricalFill,
    features: FeatureSnapshot,
    markout: FillMarkout,
) -> ResearchObservation:
    horizons = {
        timedelta(milliseconds=100): None,
        timedelta(seconds=1): None,
        timedelta(seconds=5): None,
    }

    for item in markout.markouts:
        if item.horizon in horizons:
            horizons[item.horizon] = item.value

    if isinstance(fill, HistoricalFill):
        fill_id = fill.execution_id
        symbol = fill.opportunity.symbol
    else:
        fill_id = fill.fill_id
        symbol = fill.symbol

    return ResearchObservation(
        fill_id=fill_id,
        order_id=str(fill.order_id),
        timestamp=fill.timestamp,
        symbol=symbol,
        side=fill.side.value,
        quantity=fill.quantity,
        fill_price=fill.price,
        mid_price=features.mid_price,
        spread=features.spread,
        relative_spread=features.relative_spread,
        bid_size=features.bid_size,
        ask_size=features.ask_size,
        imbalance=features.imbalance,
        microprice=features.microprice,
        imbalance_l5=features.imbalance_l5,
        imbalance_l10=features.imbalance_l10,
        weighted_imbalance_l5=features.weighted_imbalance_l5,
        weighted_imbalance_l10=features.weighted_imbalance_l10,
        markout_100ms=horizons[timedelta(milliseconds=100)],
        markout_1s=horizons[timedelta(seconds=1)],
        markout_5s=horizons[timedelta(seconds=5)],
    )