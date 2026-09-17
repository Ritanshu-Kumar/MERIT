import csv
from datetime import date
from pathlib import Path

from merit.data.lobster import read_messages
from merit.research.lobster_replay import replay_lobster_snapshots
from merit.research.opportunities import QuoteOpportunity
from merit.portfolio.enums import OrderSide


BASE = Path("data/sample")

MESSAGE_FILE = (
    BASE / "AAPL_2012-06-21_34200000_57600000_message_10.csv"
)
BOOK_FILE = (
    BASE / "AAPL_2012-06-21_34200000_57600000_orderbook_10.csv"
)
OUTPUT_FILE = BASE / "AAPL_m8_opportunities.csv"

SYMBOL = "AAPL"
TRADING_DATE = date(2012, 6, 21)


def build_opportunities():
    events = read_messages(
        MESSAGE_FILE,
        SYMBOL,
        TRADING_DATE,
    )

    states = replay_lobster_snapshots(
        events=events,
        orderbook_path=str(BOOK_FILE),
        levels=10,
    )

    opportunities: list[QuoteOpportunity] = []

    for state in states:
        features = state.features

        if features.mid_price is None:
            continue

        best_bid = _best_bid_from_features(
            state.mid_price,
            features.spread,
        )
        best_ask = _best_ask_from_features(
            state.mid_price,
            features.spread,
        )

        if best_bid is not None:
            opportunities.append(
                QuoteOpportunity(
                    timestamp=state.timestamp,
                    symbol=SYMBOL,
                    side=OrderSide.BUY,
                    quote_price=best_bid,
                    features=features,
                )
            )

        if best_ask is not None:
            opportunities.append(
                QuoteOpportunity(
                    timestamp=state.timestamp,
                    symbol=SYMBOL,
                    side=OrderSide.SELL,
                    quote_price=best_ask,
                    features=features,
                )
            )

    return opportunities


def _best_bid_from_features(mid_price, spread):
    if spread is None:
        return None

    return mid_price - spread / 2


def _best_ask_from_features(mid_price, spread):
    if spread is None:
        return None

    return mid_price + spread / 2


def write_opportunities(
    path: Path,
    opportunities: list[QuoteOpportunity],
) -> None:
    if not opportunities:
        raise ValueError("No opportunities generated")

    fieldnames = [
        "timestamp",
        "symbol",
        "side",
        "quote_price",
        "mid_price",
        "spread",
        "relative_spread",
        "bid_size",
        "ask_size",
        "imbalance",
        "microprice",
        "imbalance_l5",
        "imbalance_l10",
        "weighted_imbalance_l5",
        "weighted_imbalance_l10",
    ]

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()

        for opportunity in opportunities:
            features = opportunity.features

            writer.writerow(
                {
                    "timestamp": opportunity.timestamp.isoformat(),
                    "symbol": opportunity.symbol,
                    "side": opportunity.side.value,
                    "quote_price": opportunity.quote_price,
                    "mid_price": features.mid_price,
                    "spread": features.spread,
                    "relative_spread": features.relative_spread,
                    "bid_size": features.bid_size,
                    "ask_size": features.ask_size,
                    "imbalance": features.imbalance,
                    "microprice": features.microprice,
                    "imbalance_l5": features.imbalance_l5,
                    "imbalance_l10": features.imbalance_l10,
                    "weighted_imbalance_l5": (
                        features.weighted_imbalance_l5
                    ),
                    "weighted_imbalance_l10": (
                        features.weighted_imbalance_l10
                    ),
                }
            )


def main() -> None:
    if not MESSAGE_FILE.exists():
        raise FileNotFoundError(
            f"Missing message file: {MESSAGE_FILE}"
        )

    if not BOOK_FILE.exists():
        raise FileNotFoundError(
            f"Missing orderbook file: {BOOK_FILE}"
        )

    opportunities = build_opportunities()

    print(
        f"Generated {len(opportunities):,} "
        "quote opportunities."
    )

    write_opportunities(
        OUTPUT_FILE,
        opportunities,
    )

    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()