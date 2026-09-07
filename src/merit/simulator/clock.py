from datetime import datetime


class SimulationClock:
    def __init__(self, start_time: datetime) -> None:
        self._current_time = start_time

    @property
    def current_time(self) -> datetime:
        return self._current_time

    def advance_to(self, timestamp: datetime) -> None:
        if timestamp < self._current_time:
            raise ValueError("Simulation time cannot move backwards")

        self._current_time = timestamp

    def reset(self, start_time: datetime) -> None:
        self._current_time = start_time