from decimal import Decimal

from merit.book.l2_book import L2OrderBook


def level_imbalance(
    book: L2OrderBook,
    levels: int,
) -> Decimal | None:
    if levels <= 0:
        raise ValueError("levels must be positive")

    bids, asks = book.depth(levels)

    bid_volume = sum(quantity for _, quantity in bids)
    ask_volume = sum(quantity for _, quantity in asks)

    total_volume = bid_volume + ask_volume

    if total_volume == 0:
        return None

    return Decimal(bid_volume - ask_volume) / Decimal(total_volume)


def depth_weighted_imbalance(
    book: L2OrderBook,
    levels: int,
) -> Decimal | None:
    if levels <= 0:
        raise ValueError("levels must be positive")

    bids, asks = book.depth(levels)

    bid_weighted = sum(
        Decimal(quantity) / Decimal(index)
        for index, (_, quantity) in enumerate(bids, start=1)
    )

    ask_weighted = sum(
        Decimal(quantity) / Decimal(index)
        for index, (_, quantity) in enumerate(asks, start=1)
    )

    total_weighted = bid_weighted + ask_weighted

    if total_weighted == 0:
        return None

    return (bid_weighted - ask_weighted) / total_weighted