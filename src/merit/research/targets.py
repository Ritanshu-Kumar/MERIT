from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum

from merit.portfolio.enums import OrderSide


class MarkoutStatus(str, Enum):
    AVAILABLE = "AVAILABLE"
    UNAVAILABLE = "UNAVAILABLE"


@dataclass(frozen=True)
class MidPricePoint:
    timestamp: datetime
    mid_price: Decimal


@dataclass(frozen=True)
class Markout:
    horizon: timedelta
    value: Decimal | None
    status: MarkoutStatus


@dataclass(frozen=True)
class FillMarkout:
    fill_timestamp: datetime
    fill_price: Decimal
    side: OrderSide
    markouts: tuple[Markout, ...]


@dataclass(frozen=True)
class MidPriceIndex:
    timestamps: tuple[datetime, ...]
    prices: tuple[Decimal, ...]


def build_mid_price_index(
    mid_prices: tuple[MidPricePoint, ...],
) -> MidPriceIndex:
    timestamps = tuple(
        point.timestamp
        for point in mid_prices
    )

    if any(
        current < previous
        for previous, current in zip(
            timestamps,
            timestamps[1:],
        )
    ):
        raise ValueError(
            "mid_prices must be sorted by timestamp"
        )

    prices = tuple(
        point.mid_price
        for point in mid_prices
    )

    return MidPriceIndex(
        timestamps=timestamps,
        prices=prices,
    )


def _future_mid_price(
    index: MidPriceIndex,
    target_timestamp: datetime,
) -> Decimal | None:
    position = bisect_left(
        index.timestamps,
        target_timestamp,
    )

    if position >= len(index.prices):
        return None

    return index.prices[position]


def calculate_fill_markout(
    fill_timestamp: datetime,
    fill_price: Decimal,
    side: OrderSide,
    horizons: tuple[timedelta, ...],
    mid_price_index: MidPriceIndex | None = None,
    mid_prices: tuple[MidPricePoint, ...] | None = None,
) -> FillMarkout:
    if not horizons:
        raise ValueError("horizons cannot be empty")

    if any(
        horizon <= timedelta(0)
        for horizon in horizons
    ):
        raise ValueError("horizons must be positive")

    if mid_price_index is None:
        if mid_prices is None:
            raise ValueError(
                "Either mid_price_index or mid_prices is required"
            )

        mid_price_index = build_mid_price_index(
            mid_prices
        )

    markouts: list[Markout] = []

    for horizon in horizons:
        future_mid = _future_mid_price(
            mid_price_index,
            fill_timestamp + horizon,
        )

        if future_mid is None:
            markouts.append(
                Markout(
                    horizon=horizon,
                    value=None,
                    status=MarkoutStatus.UNAVAILABLE,
                )
            )
            continue

        if side == OrderSide.BUY:
            value = future_mid - fill_price
        else:
            value = fill_price - future_mid

        markouts.append(
            Markout(
                horizon=horizon,
                value=value,
                status=MarkoutStatus.AVAILABLE,
            )
        )

    return FillMarkout(
        fill_timestamp=fill_timestamp,
        fill_price=fill_price,
        side=side,
        markouts=tuple(markouts),
    )