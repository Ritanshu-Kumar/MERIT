from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from merit.features.snapshot import FeatureSnapshot
from merit.portfolio.enums import OrderSide


@dataclass(frozen=True)
class QuoteOpportunity:
    timestamp: datetime
    symbol: str
    side: OrderSide
    quote_price: Decimal
    features: FeatureSnapshot