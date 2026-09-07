from dataclasses import dataclass


@dataclass(frozen=True)
class ParticipationModel:
    rate: float

    def __post_init__(self) -> None:
        if not 0 < self.rate <= 1:
            raise ValueError("rate must be greater than 0 and at most 1")

    def available_quantity(self, trade_quantity: int) -> int:
        if trade_quantity < 0:
            raise ValueError("trade_quantity cannot be negative")

        return int(trade_quantity * self.rate)