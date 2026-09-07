from dataclasses import dataclass
from decimal import Decimal

from merit.book.l2_book import L2OrderBook


@dataclass(frozen=True)
class MicrostructureFeatures:
    mid_price: Decimal | None
    spread: Decimal | None
    relative_spread: Decimal | None
    best_bid: Decimal | None
    best_ask: Decimal | None
    bid_size: int
    ask_size: int
    imbalance: Decimal | None
    microprice: Decimal | None


def calculate_features(book: L2OrderBook) -> MicrostructureFeatures:
    best_bid_level = book.best_bid()
    best_ask_level = book.best_ask()

    best_bid = best_bid_level[0] if best_bid_level is not None else None
    best_ask = best_ask_level[0] if best_ask_level is not None else None

    bid_levels, ask_levels = book.depth(1)

    bid_size = bid_levels[0][1] if bid_levels else 0
    ask_size = ask_levels[0][1] if ask_levels else 0

    if best_bid is None or best_ask is None:
        return MicrostructureFeatures(
            mid_price=None,
            spread=None,
            relative_spread=None,
            best_bid=best_bid,
            best_ask=best_ask,
            bid_size=bid_size,
            ask_size=ask_size,
            imbalance=None,
            microprice=None,
        )

    mid_price = (best_bid + best_ask) / Decimal("2")
    spread = best_ask - best_bid
    relative_spread = spread / mid_price

    total_size = bid_size + ask_size

    imbalance = (
        Decimal(bid_size - ask_size) / Decimal(total_size)
        if total_size > 0
        else None
    )

    microprice = (
        (
            best_ask * Decimal(bid_size)
            + best_bid * Decimal(ask_size)
        )
        / Decimal(total_size)
        if total_size > 0
        else None
    )

    return MicrostructureFeatures(
        mid_price=mid_price,
        spread=spread,
        relative_spread=relative_spread,
        best_bid=best_bid,
        best_ask=best_ask,
        bid_size=bid_size,
        ask_size=ask_size,
        imbalance=imbalance,
        microprice=microprice,
    )