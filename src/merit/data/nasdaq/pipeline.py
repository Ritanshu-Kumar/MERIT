from collections.abc import Iterator
from datetime import date

from merit.data.normalized import (
    OrderAddEvent,
    OrderCancelEvent,
    OrderDeleteEvent,
    OrderExecuteEvent,
    OrderReplaceEvent,
    TradeEvent,
)
from merit.data.nasdaq.itch import ITCHDecoder
from merit.data.nasdaq.itch_reader import read_binaryfile


DecodedEvent = (
    OrderAddEvent
    | OrderExecuteEvent
    | OrderCancelEvent
    | OrderDeleteEvent
    | OrderReplaceEvent
    | TradeEvent
)


def replay_itch_file(
    path: str,
    trading_date: date,
) -> Iterator[DecodedEvent]:
    decoder = ITCHDecoder(trading_date)

    for message in read_binaryfile(path):
        event = decoder.decode(message)

        if event is not None:
            yield event