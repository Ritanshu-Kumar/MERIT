from __future__ import annotations

import argparse
import csv
import gzip
import heapq
from dataclasses import dataclass
from pathlib import Path


ITCH_PATH = Path(
    "data/itch/07302019.NASDAQ_ITCH50.gz"
)

FILL_OUTPUT = Path(
    "research/itch_economic_fills.csv"
)

EQUITY_OUTPUT = Path(
    "research/itch_economic_minute_equity.csv"
)

TARGETS = {
    14: "AAPL",
    386: "AMZN",
    3459: "GOOG",
    4184: "INTC",
    5289: "MSFT",
}

ORDER_QUANTITY = 100
MAX_POSITION = 500
M8_THRESHOLD = 0.00957966

STARTING_CASH_DOLLARS = 1_000_000
STARTING_CASH_UNITS = (
    STARTING_CASH_DOLLARS * 10_000
)


@dataclass
class Order:
    order_id: int
    side: str
    quantity: int
    price: int


class Book:
    def __init__(self) -> None:
        self.orders: dict[int, Order] = {}
        self.bids: dict[int, int] = {}
        self.asks: dict[int, int] = {}
        self.bid_heap: list[int] = []
        self.ask_heap: list[int] = []

    def levels(self, side: str) -> dict[int, int]:
        return (
            self.bids
            if side == "B"
            else self.asks
        )

    def add_level(
        self,
        side: str,
        price: int,
        quantity: int,
    ) -> None:
        levels = self.levels(side)

        if levels.get(price, 0) == 0:
            if side == "B":
                heapq.heappush(
                    self.bid_heap,
                    -price,
                )
            else:
                heapq.heappush(
                    self.ask_heap,
                    price,
                )

        levels[price] = (
            levels.get(price, 0)
            + quantity
        )

    def remove_level(
        self,
        side: str,
        price: int,
        quantity: int,
    ) -> None:
        levels = self.levels(side)
        current = levels.get(price, 0)

        if quantity >= current:
            levels.pop(price, None)
        else:
            levels[price] = current - quantity

    def best_bid(self) -> int | None:
        while self.bid_heap:
            price = -self.bid_heap[0]

            if self.bids.get(price, 0) > 0:
                return price

            heapq.heappop(self.bid_heap)

        return None

    def best_ask(self) -> int | None:
        while self.ask_heap:
            price = self.ask_heap[0]

            if self.asks.get(price, 0) > 0:
                return price

            heapq.heappop(self.ask_heap)

        return None

    def mid_units(self) -> float | None:
        bid = self.best_bid()
        ask = self.best_ask()

        if bid is None or ask is None:
            return None

        return (bid + ask) / 2.0

    def bid_size(self) -> int:
        price = self.best_bid()
        return 0 if price is None else self.bids[price]

    def ask_size(self) -> int:
        price = self.best_ask()
        return 0 if price is None else self.asks[price]

    def add(
        self,
        order_id: int,
        side: str,
        quantity: int,
        price: int,
    ) -> None:
        if order_id in self.orders:
            return

        self.orders[order_id] = Order(
            order_id=order_id,
            side=side,
            quantity=quantity,
            price=price,
        )

        self.add_level(
            side,
            price,
            quantity,
        )

    def execute(
        self,
        order_id: int,
        quantity: int,
    ) -> None:
        order = self.orders.get(order_id)

        if order is None:
            return

        quantity = min(
            quantity,
            order.quantity,
        )

        self.remove_level(
            order.side,
            order.price,
            quantity,
        )

        order.quantity -= quantity

        if order.quantity == 0:
            del self.orders[order_id]

    def delete(
        self,
        order_id: int,
    ) -> None:
        order = self.orders.pop(
            order_id,
            None,
        )

        if order is None:
            return

        self.remove_level(
            order.side,
            order.price,
            order.quantity,
        )

    def replace(
        self,
        old_order_id: int,
        new_order_id: int,
        quantity: int,
        price: int,
    ) -> None:
        order = self.orders.pop(
            old_order_id,
            None,
        )

        if order is None:
            return

        self.remove_level(
            order.side,
            order.price,
            order.quantity,
        )

        self.add(
            new_order_id,
            order.side,
            quantity,
            price,
        )


@dataclass
class Quote:
    price: int
    remaining: int


@dataclass
class StrategyState:
    name: str
    cash_units: int = STARTING_CASH_UNITS
    position: dict[str, int] | None = None
    fills: int = 0
    traded_quantity: int = 0
    max_abs_inventory: int = 0
    inventory_abs_time: float = 0.0
    inventory_time: float = 0.0
    last_timestamp_ns: int | None = None
    last_mid: dict[str, float] | None = None
    quotes: dict[
        str,
        dict[str, Quote | None],
    ] | None = None

    def __post_init__(self) -> None:
        self.position = {}
        self.last_mid = {}
        self.quotes = {}

        for symbol in TARGETS.values():
            self.position[symbol] = 0
            self.quotes[symbol] = {
                "B": None,
                "S": None,
            }

    def update_exposure_time(
        self,
        timestamp_ns: int,
    ) -> None:
        if self.last_timestamp_ns is None:
            self.last_timestamp_ns = timestamp_ns
            return

        dt_seconds = (
            timestamp_ns
            - self.last_timestamp_ns
        ) / 1_000_000_000.0

        if dt_seconds < 0:
            dt_seconds = 0.0

        for symbol in TARGETS.values():
            self.inventory_abs_time += (
                abs(self.position[symbol])
                * dt_seconds
            )

            self.inventory_time += dt_seconds

        self.last_timestamp_ns = timestamp_ns

    def equity_units(self) -> float:
        total = float(self.cash_units)

        for symbol in TARGETS.values():
            mid = self.last_mid.get(symbol)

            if mid is not None:
                total += (
                    self.position[symbol]
                    * mid
                )

        return total


def read_uint(
    payload: bytes,
    start: int,
    length: int,
) -> int:
    return int.from_bytes(
        payload[start:start + length],
        "big",
    )


def iter_messages(path: Path):
    with gzip.open(path, "rb") as fh:
        buffer = bytearray()

        while True:
            chunk = fh.read(
                4 * 1024 * 1024
            )

            if not chunk:
                break

            buffer.extend(chunk)

            while True:
                if len(buffer) < 2:
                    break

                length = int.from_bytes(
                    buffer[:2],
                    "big",
                )

                if len(buffer) < length + 2:
                    break

                payload = bytes(
                    buffer[
                        2:2 + length
                    ]
                )

                del buffer[
                    :2 + length
                ]

                yield payload


def calculate_score(
    book: Book,
    quote_side: str,
) -> float | None:
    bid = book.best_bid()
    ask = book.best_ask()

    if bid is None or ask is None:
        return None

    bid_size = book.bid_size()
    ask_size = book.ask_size()

    total = bid_size + ask_size

    if total <= 0:
        return None

    imbalance = (
        bid_size - ask_size
    ) / total

    mid_units = (bid + ask) / 2.0
    spread_units = ask - bid

    relative_spread = (
        spread_units
        / mid_units
    )

    signed_imbalance = (
        imbalance
        if quote_side == "BUY"
        else -imbalance
    )

    return (
        relative_spread
        * 100.0
        * signed_imbalance
    )


def update_quotes(
    strategy: StrategyState,
    symbol: str,
    book: Book,
) -> None:
    bid = book.best_bid()
    ask = book.best_ask()

    if bid is None or ask is None:
        strategy.quotes[symbol]["B"] = None
        strategy.quotes[symbol]["S"] = None
        return

    if strategy.name == "BASELINE":
        allow_bid = True
        allow_ask = True
    else:
        bid_score = calculate_score(
            book,
            "BUY",
        )

        ask_score = calculate_score(
            book,
            "SELL",
        )

        allow_bid = (
            bid_score is not None
            and bid_score >= M8_THRESHOLD
        )

        allow_ask = (
            ask_score is not None
            and ask_score >= M8_THRESHOLD
        )

    current_bid = (
        strategy.quotes[symbol]["B"]
    )

    if allow_bid:
        if (
            current_bid is None
            or current_bid.price != bid
        ):
            strategy.quotes[symbol]["B"] = Quote(
                price=bid,
                remaining=ORDER_QUANTITY,
            )
    else:
        strategy.quotes[symbol]["B"] = None

    current_ask = (
        strategy.quotes[symbol]["S"]
    )

    if allow_ask:
        if (
            current_ask is None
            or current_ask.price != ask
        ):
            strategy.quotes[symbol]["S"] = Quote(
                price=ask,
                remaining=ORDER_QUANTITY,
            )
    else:
        strategy.quotes[symbol]["S"] = None


def apply_fill(
    strategy: StrategyState,
    symbol: str,
    side: str,
    quantity: int,
    price: int,
) -> bool:
    current_position = strategy.position[symbol]

    delta = (
        quantity
        if side == "BUY"
        else -quantity
    )

    new_position = (
        current_position
        + delta
    )

    if abs(new_position) > MAX_POSITION:
        return False

    notional_units = (
        price * quantity
    )

    if side == "BUY":
        strategy.cash_units -= (
            notional_units
        )
    else:
        strategy.cash_units += (
            notional_units
        )

    strategy.position[symbol] = (
        new_position
    )

    strategy.fills += 1
    strategy.traded_quantity += quantity
    strategy.max_abs_inventory = max(
        strategy.max_abs_inventory,
        abs(new_position),
    )

    return True


def process_execution(
    strategy: StrategyState,
    symbol: str,
    resting_side: str,
    execution_price: int,
    quantity: int,
) -> tuple[
    bool,
    str | None,
    int,
]:
    if resting_side == "S":
        quote = (
            strategy.quotes[symbol]["B"]
        )
        fill_side = "BUY"
        quote_key = "B"
    else:
        quote = (
            strategy.quotes[symbol]["S"]
        )
        fill_side = "SELL"
        quote_key = "S"

    if quote is None:
        return False, None, 0

    if quote.price != execution_price:
        return False, None, 0

    if quote.remaining <= 0:
        return False, None, 0

    fill_quantity = min(
        quantity,
        quote.remaining,
        ORDER_QUANTITY,
    )

    if fill_quantity <= 0:
        return False, None, 0

    if not apply_fill(
        strategy,
        symbol,
        fill_side,
        fill_quantity,
        execution_price,
    ):
        return False, None, 0

    quote.remaining -= fill_quantity

    if quote.remaining == 0:
        strategy.quotes[symbol][
            quote_key
        ] = None

    return (
        True,
        fill_side,
        fill_quantity,
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--path",
        default=str(ITCH_PATH),
    )

    parser.add_argument(
        "--max-messages",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--fill-output",
        default=str(FILL_OUTPUT),
    )

    parser.add_argument(
        "--equity-output",
        default=str(EQUITY_OUTPUT),
    )

    args = parser.parse_args()

    books = {
        locate: Book()
        for locate in TARGETS
    }

    baseline = StrategyState(
        name="BASELINE"
    )

    m8 = StrategyState(
        name="M8"
    )

    fill_path = Path(
        args.fill_output
    )

    equity_path = Path(
        args.equity_output
    )

    fill_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    equity_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fill_file = fill_path.open(
        "w",
        newline="",
        encoding="utf-8",
    )

    equity_file = equity_path.open(
        "w",
        newline="",
        encoding="utf-8",
    )

    fill_writer = csv.writer(
        fill_file
    )

    equity_writer = csv.writer(
        equity_file
    )

    fill_writer.writerow(
        [
            "timestamp_ns",
            "symbol",
            "strategy",
            "resting_side",
            "passive_side",
            "quote_price",
            "execution_price",
            "quantity",
        ]
    )

    equity_writer.writerow(
        [
            "timestamp_ns",
            "strategy",
            "equity",
        ]
    )

    total_messages = 0
    target_messages = 0
    execution_messages = 0
    simulated_fills = 0

    fill_audit: list[dict] = []

    last_equity_minute = None

    try:
        for payload in iter_messages(
            Path(args.path)
        ):
            if (
                args.max_messages is not None
                and total_messages
                >= args.max_messages
            ):
                break

            total_messages += 1

            if not payload:
                continue

            message_type = chr(payload[0])

            if message_type == "S":
                continue

            if len(payload) < 11:
                continue

            locate = read_uint(
                payload,
                1,
                2,
            )

            if locate not in books:
                continue

            target_messages += 1

            symbol = TARGETS[locate]

            timestamp_ns = read_uint(
                payload,
                5,
                6,
            )

            book = books[locate]

            baseline.update_exposure_time(
                timestamp_ns
            )

            m8.update_exposure_time(
                timestamp_ns
            )

            # Quote using state immediately
            # before the current ITCH message.
            update_quotes(
                baseline,
                symbol,
                book,
            )

            update_quotes(
                m8,
                symbol,
                book,
            )

            pre_event_mid = book.mid_units()

            if pre_event_mid is not None:
                baseline.last_mid[symbol] = (
                    pre_event_mid
                )
                m8.last_mid[symbol] = (
                    pre_event_mid
                )

            if message_type in ("E", "C"):
                order_id = read_uint(
                    payload,
                    11,
                    8,
                )

                quantity = read_uint(
                    payload,
                    19,
                    4,
                )

                order = book.orders.get(
                    order_id
                )

                if order is not None:
                    execution_messages += 1

                    execution_price = (
                        order.price
                    )

                    if message_type == "C":
                        execution_price = read_uint(
                            payload,
                            32,
                            4,
                        )

                    for strategy in (
                        baseline,
                        m8,
                    ):
                        (
                            matched,
                            passive_side,
                            fill_quantity,
                        ) = process_execution(
                            strategy,
                            symbol,
                            order.side,
                            execution_price,
                            quantity,
                        )

                        if not matched:
                            continue

                        simulated_fills += 1

                        fill_price = (
                            execution_price
                            / 10000.0
                        )

                        fill_writer.writerow(
                            [
                                timestamp_ns,
                                symbol,
                                strategy.name,
                                order.side,
                                passive_side,
                                fill_price,
                                fill_price,
                                fill_quantity,
                            ]
                        )

                        if len(fill_audit) < 10:
                            fill_audit.append(
                                {
                                    "timestamp_ns": (
                                        timestamp_ns
                                    ),
                                    "symbol": symbol,
                                    "strategy": (
                                        strategy.name
                                    ),
                                    "resting_side": (
                                        order.side
                                    ),
                                    "passive_side": (
                                        passive_side
                                    ),
                                    "quote_price": (
                                        fill_price
                                    ),
                                    "execution_price": (
                                        fill_price
                                    ),
                                    "quantity": (
                                        fill_quantity
                                    ),
                                }
                            )

            if message_type == "A":
                order_id = read_uint(
                    payload,
                    11,
                    8,
                )

                side = chr(
                    payload[19]
                )

                quantity = read_uint(
                    payload,
                    20,
                    4,
                )

                price = read_uint(
                    payload,
                    32,
                    4,
                )

                book.add(
                    order_id,
                    side,
                    quantity,
                    price,
                )

            elif message_type == "F":
                order_id = read_uint(
                    payload,
                    11,
                    8,
                )

                side = chr(
                    payload[19]
                )

                quantity = read_uint(
                    payload,
                    20,
                    4,
                )

                price = read_uint(
                    payload,
                    32,
                    4,
                )

                book.add(
                    order_id,
                    side,
                    quantity,
                    price,
                )

            elif message_type == "X":
                order_id = read_uint(
                    payload,
                    11,
                    8,
                )

                quantity = read_uint(
                    payload,
                    19,
                    4,
                )

                book.execute(
                    order_id,
                    quantity,
                )

            elif message_type == "D":
                order_id = read_uint(
                    payload,
                    11,
                    8,
                )

                book.delete(
                    order_id
                )

            elif message_type == "U":
                old_order_id = read_uint(
                    payload,
                    11,
                    8,
                )

                new_order_id = read_uint(
                    payload,
                    19,
                    8,
                )

                quantity = read_uint(
                    payload,
                    27,
                    4,
                )

                price = read_uint(
                    payload,
                    31,
                    4,
                )

                book.replace(
                    old_order_id,
                    new_order_id,
                    quantity,
                    price,
                )

            minute = (
                timestamp_ns
                // 60_000_000_000
            )

            if (
                last_equity_minute is None
                or minute != last_equity_minute
            ):
                equity_writer.writerow(
                    [
                        timestamp_ns,
                        "BASELINE",
                        baseline.equity_units()
                        / 10000.0,
                    ]
                )

                equity_writer.writerow(
                    [
                        timestamp_ns,
                        "M8",
                        m8.equity_units()
                        / 10000.0,
                    ]
                )

                last_equity_minute = minute

            if (
                total_messages
                % 25_000_000
                == 0
            ):
                print(
                    f"processed="
                    f"{total_messages:,} "
                    f"target="
                    f"{target_messages:,} "
                    f"executions="
                    f"{execution_messages:,} "
                    f"fills="
                    f"{simulated_fills:,}"
                )

    finally:
        fill_file.close()
        equity_file.close()

    print()
    print("=== FIRST 10 FILLS AUDIT ===")

    for fill in fill_audit:
        print(
            f"{fill['symbol']:>5} "
            f"{fill['strategy']:>8} "
            f"resting={fill['resting_side']} "
            f"passive={fill['passive_side']} "
            f"quote={fill['quote_price']:.4f} "
            f"exec={fill['execution_price']:.4f} "
            f"qty={fill['quantity']}"
        )

    print()
    print("=== ITCH ECONOMIC REPLAY ===")
    print(
        f"messages: "
        f"{total_messages:,}"
    )
    print(
        f"target messages: "
        f"{target_messages:,}"
    )
    print(
        f"execution messages: "
        f"{execution_messages:,}"
    )
    print(
        f"simulated fills: "
        f"{simulated_fills:,}"
    )

    for strategy in (
        baseline,
        m8,
    ):
        final_equity = (
            strategy.equity_units()
            / 10000.0
        )

        pnl = (
            final_equity
            - STARTING_CASH_DOLLARS
        )

        avg_abs_inventory = (
            strategy.inventory_abs_time
            / strategy.inventory_time
            if strategy.inventory_time > 0
            else 0.0
        )

        print()
        print(
            f"--- {strategy.name} ---"
        )
        print(
            f"P&L: "
            f"{pnl:+,.2f}"
        )
        print(
            f"final equity: "
            f"{final_equity:,.2f}"
        )
        print(
            f"fills: "
            f"{strategy.fills:,}"
        )
        print(
            f"traded quantity: "
            f"{strategy.traded_quantity:,}"
        )
        print(
            f"max abs inventory: "
            f"{strategy.max_abs_inventory}"
        )
        print(
            "time-weighted avg abs inventory: "
            f"{avg_abs_inventory:.2f}"
        )

    print()
    print(
        "M8 - baseline P&L: "
        f"{(
            m8.equity_units()
            - baseline.equity_units()
        ) / 10000.0:+,.2f}"
    )

    print()
    print(
        f"fills written to: "
        f"{fill_path}"
    )

    print(
        f"minute equity written to: "
        f"{equity_path}"
    )


if __name__ == "__main__":
    main()