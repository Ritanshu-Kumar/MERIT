from dataclasses import dataclass
from decimal import Decimal

from merit.book.l2_book import L2OrderBook


@dataclass(frozen=True)
class QuoteDecision:
    bid_price: Decimal | None
    ask_price: Decimal | None
    quantity: int


class BaselineMarketMaker:
    def __init__(self, quantity: int = 100) -> None:
        if quantity <= 0:
            raise ValueError("quantity must be positive")

        self.quantity = quantity

    def quote(self, book: L2OrderBook) -> QuoteDecision:
        best_bid = book.best_bid()
        best_ask = book.best_ask()

        return QuoteDecision(
            bid_price=best_bid[0] if best_bid is not None else None,
            ask_price=best_ask[0] if best_ask is not None else None,
            quantity=self.quantity,
        )