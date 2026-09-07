from collections.abc import Iterable
from datetime import timedelta

from merit.research.dataset import ResearchObservation
from merit.research.historical_fills import HistoricalFill
from merit.research.targets import MidPricePoint, calculate_fill_markout


HORIZONS = (
    timedelta(milliseconds=100),
    timedelta(seconds=1),
    timedelta(seconds=5),
)


def build_research_dataset(
    fills: Iterable[HistoricalFill],
    mid_prices: tuple[MidPricePoint, ...],
) -> tuple[ResearchObservation, ...]:
    observations: list[ResearchObservation] = []

    for historical_fill in fills:
        markout = calculate_fill_markout(
            fill_timestamp=historical_fill.timestamp,
            fill_price=historical_fill.price,
            side=historical_fill.side,
            mid_prices=mid_prices,
            horizons=HORIZONS,
        )

        features = historical_fill.opportunity.features

        observations.append(
            ResearchObservation(
                fill_id=historical_fill.execution_id,
                order_id=str(historical_fill.order_id),
                timestamp=historical_fill.timestamp,
                symbol=historical_fill.opportunity.symbol,
                side=historical_fill.side.value,
                quantity=historical_fill.quantity,
                fill_price=historical_fill.price,
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
                markout_100ms=markout.markouts[0].value,
                markout_1s=markout.markouts[1].value,
                markout_5s=markout.markouts[2].value,
            )
        )

    return tuple(observations)