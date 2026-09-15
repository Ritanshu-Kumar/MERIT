from decimal import Decimal

from merit.features.snapshot import FeatureSnapshot
from merit.research.m8_policy import M8Policy
from merit.strategy.m8 import M8MarketMaker


def make_features(
    relative_spread: str,
    imbalance: str,
) -> FeatureSnapshot:
    return FeatureSnapshot(
        mid_price=Decimal("100"),
        spread=Decimal(relative_spread) * Decimal("100"),
        relative_spread=Decimal(relative_spread),
        bid_size=100,
        ask_size=100,
        imbalance=Decimal(imbalance),
        microprice=Decimal("100"),
        imbalance_l5=None,
        imbalance_l10=None,
        weighted_imbalance_l5=None,
        weighted_imbalance_l10=None,
    )


def test_quotes_side_selectively() -> None:
    features = make_features("0.01", "0.5")

    strategy = M8MarketMaker(
        policy=M8Policy(threshold=0.0)
    )

    decision = strategy.quote(
        features,
        Decimal("100.00"),
        Decimal("101.00"),
    )

    assert decision.bid_price == Decimal("100.00")
    assert decision.ask_price is None


def test_rejects_quotes_below_threshold() -> None:
    features = make_features("0.0002", "0.2")

    strategy = M8MarketMaker(
        policy=M8Policy(threshold=0.01)
    )

    decision = strategy.quote(
        features,
        Decimal("100.00"),
        Decimal("101.00"),
    )

    assert decision.bid_price is None
    assert decision.ask_price is None