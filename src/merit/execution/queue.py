from dataclasses import dataclass


@dataclass
class QueuePosition:
    quantity_ahead: int

    def __post_init__(self) -> None:
        if self.quantity_ahead < 0:
            raise ValueError("quantity_ahead cannot be negative")

    @property
    def active(self) -> bool:
        return self.quantity_ahead == 0

    def consume(self, quantity: int) -> int:
        if quantity < 0:
            raise ValueError("quantity cannot be negative")

        consumed = min(self.quantity_ahead, quantity)
        self.quantity_ahead -= consumed
        return quantity - consumed