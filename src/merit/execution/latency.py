from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass(frozen=True)
class LatencyModel:
    submit_latency: timedelta

    def __post_init__(self) -> None:
        if self.submit_latency < timedelta(0):
            raise ValueError("submit_latency cannot be negative")

    def activation_time(self, created_at: datetime) -> datetime:
        return created_at + self.submit_latency