from dataclasses import dataclass

from merit.execution.queue import QueuePosition


@dataclass(frozen=True)
class QueueModel:
    ahead_fraction: float = 1.0

    def __post_init__(self) -> None:
        if self.ahead_fraction < 0:
            raise ValueError("ahead_fraction cannot be negative")

    def estimate(self, visible_quantity: int) -> QueuePosition:
        if visible_quantity < 0:
            raise ValueError("visible_quantity cannot be negative")

        quantity_ahead = int(visible_quantity * self.ahead_fraction)

        return QueuePosition(quantity_ahead=quantity_ahead)