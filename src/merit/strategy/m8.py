from merit.features.snapshot import FeatureSnapshot
from merit.research.m8_policy import M8Policy
from merit.strategy.baseline import QuoteDecision


class M8MarketMaker:
    def __init__(
        self,
        quantity: int = 100,
        policy: M8Policy | None = None,
    ) -> None:
        if quantity <= 0:
            raise ValueError("quantity must be positive")

        self.quantity = quantity
        self.policy = policy or M8Policy()

    def quote(
        self,
        features: FeatureSnapshot,
        bid_price,
        ask_price,
    ) -> QuoteDecision:
        if (
            features.relative_spread is None
            or features.imbalance is None
            or bid_price is None
            or ask_price is None
        ):
            return QuoteDecision(
                bid_price=None,
                ask_price=None,
                quantity=self.quantity,
            )

        imbalance = float(features.imbalance)
        relative_spread = float(features.relative_spread)

        allow_bid = self.policy.select(
            relative_spread,
            imbalance,
            "BUY",
        )

        allow_ask = self.policy.select(
            relative_spread,
            imbalance,
            "SELL",
        )

        return QuoteDecision(
            bid_price=bid_price if allow_bid else None,
            ask_price=ask_price if allow_ask else None,
            quantity=self.quantity,
        )