from __future__ import annotations

import argparse
import gzip
import heapq
from collections import Counter, defaultdict
from dataclasses import dataclass


ITCH_PATH = "data/itch/07302019.NASDAQ_ITCH50.gz"

TARGETS = {
    14: "AAPL",
    386: "AMZN",
    3459: "GOOG",
    4184: "INTC",
    5289: "MSFT",
}


@dataclass
class Order:
    stock_locate: int
    order_id: int
    side: str
    shares: int
    price: int


class Book:
    def __init__(self, symbol: str) -> None:
        self.symbol = symbol

        self.orders: dict[int, Order] = {}
        self.bids: dict[int, int] = defaultdict(int)
        self.asks: dict[int, int] = defaultdict(int)

        self.bid_heap: list[int] = []
        self.ask_heap: list[int] = []

        self.adds = 0
        self.fills = 0
        self.cancels = 0
        self.deletes = 0
        self.replaces = 0

        self.unknown_order_events = 0
        self.duplicate_order_ids = 0
        self.over_execution = 0
        self.negative_depth = 0
        self.self_crossed = 0

        self.min_spread = None
        self.max_spread = None
        self.execution_snapshots = 0

    def _levels(self, side: str) -> dict[int, int]:
        return self.bids if side == "B" else self.asks

    def _heap(self, side: str) -> list[int]:
        return self.bid_heap if side == "B" else self.ask_heap

    def _add_level(self, side: str, price: int, shares: int) -> None:
        levels = self._levels(side)

        was_empty = levels.get(price, 0) == 0
        levels[price] += shares

        if was_empty:
            if side == "B":
                heapq.heappush(self.bid_heap, -price)
            else:
                heapq.heappush(self.ask_heap, price)

    def _remove_level(self, side: str, price: int, shares: int) -> None:
        levels = self._levels(side)

        current = levels.get(price, 0)

        if shares > current:
            self.over_execution += 1
            shares = current

        new_value = current - shares

        if new_value < 0:
            self.negative_depth += 1
            new_value = 0

        if new_value == 0:
            levels.pop(price, None)
        else:
            levels[price] = new_value

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

    def best_bid_size(self) -> int:
        price = self.best_bid()
        return 0 if price is None else self.bids[price]

    def best_ask_size(self) -> int:
        price = self.best_ask()
        return 0 if price is None else self.asks[price]

    def mid(self) -> float | None:
        bid = self.best_bid()
        ask = self.best_ask()

        if bid is None or ask is None:
            return None

        return (bid + ask) / 2.0

    def spread(self) -> int | None:
        bid = self.best_bid()
        ask = self.best_ask()

        if bid is None or ask is None:
            return None

        spread = ask - bid

        if spread <= 0:
            self_crossed += 1

        if self.min_spread is None or spread < self.min_spread:
            self.min_spread = spread

        if self.max_spread is None or spread > self.max_spread:
            self.max_spread = spread

        return spread

    def top_levels(self, side: str, depth: int = 10) -> list[tuple[int, int]]:
        levels = self.bids if side == "B" else self.asks

        if side == "B":
            prices = sorted(levels.keys(), reverse=True)[:depth]
        else:
            prices = sorted(levels.keys())[:depth]

        return [(price, levels[price]) for price in prices]

    def add_order(
        self,
        order_id: int,
        side: str,
        shares: int,
        price: int,
    ) -> None:
        if order_id in self.orders:
            self.duplicate_order_ids += 1
            return

        order = Order(
            stock_locate=0,
            order_id=order_id,
            side=side,
            shares=shares,
            price=price,
        )

        self.orders[order_id] = order
        self._add_level(side, price, shares)
        self.adds += 1

    def execute_order(
        self,
        order_id: int,
        executed_shares: int,
    ) -> bool:
        order = self.orders.get(order_id)

        if order is None:
            self.unknown_order_events += 1
            return False

        if executed_shares > order.shares:
            self.over_execution += 1
            executed_shares = order.shares

        self._remove_level(
            order.side,
            order.price,
            executed_shares,
        )

        order.shares -= executed_shares
        self.fills += 1

        if order.shares == 0:
            del self.orders[order_id]

        return True

    def cancel_order(
        self,
        order_id: int,
        cancelled_shares: int,
    ) -> bool:
        order = self.orders.get(order_id)

        if order is None:
            self.unknown_order_events += 1
            return False

        if cancelled_shares > order.shares:
            self.over_execution += 1
            cancelled_shares = order.shares

        self._remove_level(
            order.side,
            order.price,
            cancelled_shares,
        )

        order.shares -= cancelled_shares
        self.cancels += 1

        if order.shares == 0:
            del self.orders[order_id]

        return True

    def delete_order(self, order_id: int) -> bool:
        order = self.orders.pop(order_id, None)

        if order is None:
            self.unknown_order_events += 1
            return False

        self._remove_level(
            order.side,
            order.price,
            order.shares,
        )

        self.deletes += 1
        return True

    def replace_order(
        self,
        old_order_id: int,
        new_order_id: int,
        new_shares: int,
        new_price: int,
    ) -> bool:
        old_order = self.orders.get(old_order_id)

        if old_order is None:
            self.unknown_order_events += 1
            return False

        if new_order_id in self.orders:
            self.duplicate_order_ids += 1
            return False

        self._remove_level(
            old_order.side,
            old_order.price,
            old_order.shares,
        )

        del self.orders[old_order_id]

        new_order = Order(
            stock_locate=old_order.stock_locate,
            order_id=new_order_id,
            side=old_order.side,
            shares=new_shares,
            price=new_price,
        )

        self.orders[new_order_id] = new_order
        self._add_level(
            new_order.side,
            new_order.price,
            new_order.shares,
        )

        self.replaces += 1
        return True

    def snapshot(self) -> dict:
        bid = self.best_bid()
        ask = self.best_ask()
        spread = self.spread()

        return {
            "symbol": self.symbol,
            "best_bid": None if bid is None else bid / 10000.0,
            "best_ask": None if ask is None else ask / 10000.0,
            "bid_size": self.best_bid_size(),
            "ask_size": self.best_ask_size(),
            "mid": None if self.mid() is None else self.mid() / 10000.0,
            "spread": None if spread is None else spread / 10000.0,
            "bid_levels": len(self.bids),
            "ask_levels": len(self.asks),
            "orders": len(self.orders),
        }


def read_uint(payload: bytes, start: int, length: int) -> int:
    return int.from_bytes(payload[start:start + length], "big")


def read_text(payload: bytes, start: int, length: int) -> str:
    return payload[start:start + length].decode("ascii").strip()


def iter_messages(path: str):
    with gzip.open(path, "rb") as fh:
        buffer = bytearray()

        while True:
            chunk = fh.read(4 * 1024 * 1024)

            if not chunk:
                break

            buffer.extend(chunk)

            while True:
                if len(buffer) < 2:
                    break

                message_length = int.from_bytes(buffer[:2], "big")

                if len(buffer) < message_length + 2:
                    break

                payload = bytes(buffer[2:2 + message_length])
                del buffer[:2 + message_length]

                yield payload


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        default=ITCH_PATH,
    )
    parser.add_argument(
        "--sample-executions",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--max-messages",
        type=int,
        default=None,
        help="Stop after processing this many messages (useful for smoke checks).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    books = {
        stock_locate: Book(symbol)
        for stock_locate, symbol in TARGETS.items()
    }

    total_messages = 0
    target_messages = 0

    message_types = Counter()
    target_types = Counter()

    sampled_execution_snapshots = []

    last_timestamp = {
        stock_locate: None
        for stock_locate in TARGETS
    }

    for payload in iter_messages(args.path):
        if args.max_messages is not None and total_messages >= args.max_messages:
            break

        total_messages += 1

        if not payload:
            continue

        msg_type = chr(payload[0])
        message_types[msg_type] += 1

        if msg_type == "S":
            continue

        if len(payload) < 11:
            continue

        stock_locate = read_uint(payload, 1, 2)

        if stock_locate not in books:
            continue

        target_messages += 1
        target_types[msg_type] += 1

        book = books[stock_locate]

        timestamp = read_uint(payload, 5, 6)
        last_timestamp[stock_locate] = timestamp

        # Add Order
        if msg_type == "A":
            order_id = read_uint(payload, 11, 8)
            side = chr(payload[19])
            shares = read_uint(payload, 20, 4)
            price = read_uint(payload, 32, 4)

            book.add_order(
                order_id,
                side,
                shares,
                price,
            )

        # Add Order with MPID Attribution
        elif msg_type == "F":
            order_id = read_uint(payload, 11, 8)
            side = chr(payload[19])
            shares = read_uint(payload, 20, 4)
            price = read_uint(payload, 32, 4)

            book.add_order(
                order_id,
                side,
                shares,
                price,
            )

        # Order Executed
        elif msg_type == "E":
            order_id = read_uint(payload, 11, 8)
            shares = read_uint(payload, 19, 4)

            # Capture the state immediately BEFORE the execution.
            if len(sampled_execution_snapshots) < args.sample_executions:
                order = book.orders.get(order_id)

                if order is not None:
                    sampled_execution_snapshots.append(
                        {
                            "timestamp": timestamp,
                            "symbol": book.symbol,
                            "type": msg_type,
                            "order_id": order_id,
                            "side": order.side,
                            "execution_price": order.price / 10000.0,
                            "executed_shares": shares,
                            "book": book.snapshot(),
                            "top10_bid": book.top_levels("B"),
                            "top10_ask": book.top_levels("S"),
                        }
                    )

            book.execute_order(
                order_id,
                shares,
            )

        # Order Executed with Price
        elif msg_type == "C":
            order_id = read_uint(payload, 11, 8)
            shares = read_uint(payload, 19, 4)
            execution_price = read_uint(payload, 32, 4)

            if len(sampled_execution_snapshots) < args.sample_executions:
                order = book.orders.get(order_id)

                if order is not None:
                    sampled_execution_snapshots.append(
                        {
                            "timestamp": timestamp,
                            "symbol": book.symbol,
                            "type": msg_type,
                            "order_id": order_id,
                            "side": order.side,
                            "execution_price": execution_price / 10000.0,
                            "resting_price": order.price / 10000.0,
                            "executed_shares": shares,
                            "book": book.snapshot(),
                            "top10_bid": book.top_levels("B"),
                            "top10_ask": book.top_levels("S"),
                        }
                    )

            book.execute_order(
                order_id,
                shares,
            )

        # Order Cancel
        elif msg_type == "X":
            order_id = read_uint(payload, 11, 8)
            shares = read_uint(payload, 19, 4)

            book.cancel_order(
                order_id,
                shares,
            )

        # Order Delete
        elif msg_type == "D":
            order_id = read_uint(payload, 11, 8)

            book.delete_order(order_id)

        # Order Replace
        elif msg_type == "U":
            old_order_id = read_uint(payload, 11, 8)
            new_order_id = read_uint(payload, 19, 8)
            new_shares = read_uint(payload, 27, 4)
            new_price = read_uint(payload, 31, 4)

            book.replace_order(
                old_order_id,
                new_order_id,
                new_shares,
                new_price,
            )

        # Every relevant event gets a basic crossed-book check.
        bid = book.best_bid()
        ask = book.best_ask()

        if bid is not None and ask is not None and bid >= ask:
            book.self_crossed += 1

        if total_messages % 10_000_000 == 0:
            print(
                f"processed={total_messages:,} "
                f"target={target_messages:,}"
            )

    print("\n=== ITCH FULL RECONSTRUCTION ===")
    print(f"Total messages:   {total_messages:,}")
    print(f"Target messages:  {target_messages:,}")

    print("\n=== MESSAGE TYPES ===")
    for msg_type, count in sorted(
        message_types.items(),
        key=lambda x: (-x[1], x[0]),
    ):
        print(f"{msg_type}: {count:,}")

    print("\n=== TARGET MESSAGE TYPES ===")
    for msg_type, count in sorted(
        target_types.items(),
        key=lambda x: (-x[1], x[0]),
    ):
        print(f"{msg_type}: {count:,}")

    print("\n=== BOOK VALIDATION ===")

    for stock_locate, book in books.items():
        symbol = book.symbol

        bid = book.best_bid()
        ask = book.best_ask()

        print(f"\n{symbol}")
        print(f"  active orders:       {len(book.orders):,}")
        print(f"  active bid levels:   {len(book.bids):,}")
        print(f"  active ask levels:   {len(book.asks):,}")
        print(f"  adds:                {book.adds:,}")
        print(f"  executions:          {book.fills:,}")
        print(f"  cancels:             {book.cancels:,}")
        print(f"  deletes:             {book.deletes:,}")
        print(f"  replaces:            {book.replaces:,}")
        print(f"  unknown order refs:  {book.unknown_order_events:,}")
        print(f"  duplicate order ids: {book.duplicate_order_ids:,}")
        print(f"  over-executions:     {book.over_execution:,}")
        print(f"  negative depth:      {book.negative_depth:,}")
        print(f"  crossed book:        {book.self_crossed:,}")

        if bid is not None:
            print(f"  final best bid:      {bid / 10000:.4f}")

        if ask is not None:
            print(f"  final best ask:      {ask / 10000:.4f}")

        if bid is not None and ask is not None:
            print(f"  final midpoint:      {(bid + ask) / 20000:.4f}")

    print("\n=== SAMPLE EXECUTION SNAPSHOTS ===")

    for snapshot in sampled_execution_snapshots[:20]:
        print(
            snapshot["symbol"],
            snapshot["type"],
            f"order={snapshot['order_id']}",
            f"side={snapshot['side']}",
            f"price={snapshot['execution_price']:.4f}",
            f"shares={snapshot['executed_shares']}",
            f"mid={snapshot['book']['mid']}",
            f"spread={snapshot['book']['spread']}",
        )


if __name__ == "__main__":
    main()