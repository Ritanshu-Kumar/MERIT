from collections.abc import Callable
from typing import TypeAlias

from .clock import SimulationClock
from .events import MarketEvent
from .queue import EventQueue


EventHandler: TypeAlias = Callable[[MarketEvent], None]


class SimulationEngine:
    def __init__(
        self,
        start_time,
        handler: EventHandler,
    ) -> None:
        self.clock = SimulationClock(start_time)
        self.queue = EventQueue()
        self.handler = handler

    def schedule(self, event: MarketEvent) -> None:
        self.queue.push(event)

    def run(self) -> None:
        while not self.queue.is_empty():
            event = self.queue.pop()
            self.clock.advance_to(event.timestamp)
            self.handler(event)