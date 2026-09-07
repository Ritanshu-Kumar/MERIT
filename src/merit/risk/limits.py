from dataclasses import dataclass

from merit.portfolio.enums import OrderSide


@dataclass(frozen=True)
class RiskLimits:
    max_order_quantity: int
    max_position: int

    def __post_init__(self) -> None:
        if self.max_order_quantity <= 0:
            raise ValueError("max_order_quantity must be positive")

        if self.max_position <= 0:
            raise ValueError("max_position must be positive")


@dataclass(frozen=True)
class RiskResult:
    approved: bool
    quantity: int
    reason: str | None = None


class RiskManager:
    def __init__(self, limits: RiskLimits) -> None:
        self.limits = limits

    def check_order(
        self,
        side: OrderSide,
        quantity: int,
        current_position: int,
    ) -> RiskResult:
        if quantity <= 0:
            return RiskResult(
                approved=False,
                quantity=0,
                reason="quantity must be positive",
            )

        if quantity > self.limits.max_order_quantity:
            return RiskResult(
                approved=False,
                quantity=0,
                reason="order quantity exceeds limit",
            )

        position_delta = (
            quantity if side == OrderSide.BUY else -quantity
        )

        resulting_position = current_position + position_delta

        if abs(resulting_position) > self.limits.max_position:
            return RiskResult(
                approved=False,
                quantity=0,
                reason="position limit exceeded",
            )

        return RiskResult(
            approved=True,
            quantity=quantity,
        )