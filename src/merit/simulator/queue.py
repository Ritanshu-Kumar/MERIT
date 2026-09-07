from dataclasses import dataclass, field
import heapq
from itertools import count

from .events import MarketEvent


@dataclass(order=True)
class _QueueItem:
    timestamp: object
    sequence: int
    event: MarketEvent = field(compare=False)


class EventQueue:
    def __init__(self) -> None:
        self._queue: list[_QueueItem] = []
        self._sequence = count()

    def push(self, event: MarketEvent) -> None:
        heapq.heappush(
            self._queue,
            _QueueItem(
                timestamp=event.timestamp,
                sequence=next(self._sequence),
                event=event,
            ),
        )

    def pop(self) -> MarketEvent:
        if not self._queue:
            raise IndexError("Cannot pop from an empty event queue")

        return heapq.heappop(self._queue).event

    def peek(self) -> MarketEvent:
        if not self._queue:
            raise IndexError("Cannot peek at an empty event queue")

        return self._queue[0].event

    def __len__(self) -> int:
        return len(self._queue)

    def is_empty(self) -> bool:
        return not self._queue