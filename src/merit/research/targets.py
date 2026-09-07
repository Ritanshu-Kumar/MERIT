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


def _future_mid_price(
    points: tuple[MidPricePoint, ...],
    target_time: datetime,
) -> Decimal | None:
    for point in points:
        if point.timestamp >= target_time:
            return point.mid_price

    return None


def calculate_fill_markout(
    fill_timestamp: datetime,
    fill_price: Decimal,
    side: OrderSide,
    mid_prices: tuple[MidPricePoint, ...],
    horizons: tuple[timedelta, ...],
) -> FillMarkout:
    if not horizons:
        raise ValueError("horizons cannot be empty")

    if any(horizon <= timedelta(0) for horizon in horizons):
        raise ValueError("horizons must be positive")

    if any(
        points.timestamp < points_prev.timestamp
        for points_prev, points in zip(mid_prices, mid_prices[1:])
    ):
        raise ValueError("mid_prices must be sorted by timestamp")

    markouts: list[Markout] = []

    for horizon in horizons:
        future_mid = _future_mid_price(
            mid_prices,
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