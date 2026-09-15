from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class M8Policy:
    threshold: float = 0.00957966

    def score(
        self,
        relative_spread: float,
        imbalance: float,
        side: str,
    ) -> float:
        signed_imbalance = (
            imbalance
            if side == "BUY"
            else -imbalance
        )

        relative_spread_pct = relative_spread * 100.0

        return (
            relative_spread_pct
            * signed_imbalance
        )

    def select(
        self,
        relative_spread: float,
        imbalance: float,
        side: str,
    ) -> bool:
        return (
            self.score(
                relative_spread,
                imbalance,
                side,
            )
            >= self.threshold
        )