from __future__ import annotations

import argparse
import csv
import gzip
import heapq
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


ITCH_PATH = Path("data/itch/07302019.NASDAQ_ITCH50.gz")

OUTPUT_PATH = Path(
    "data/itch/M8_ITCH_2019-07-30_events.csv"
)

TARGETS = {
    14: "AAPL",
    386: "AMZN",
    3459: "GOOG",
    4184: "INTC",
    5289: "MSFT",
}

HORIZONS = {
    "100ms": 100_000_000,
    "1s": 1_000_000_000,
    "5s": 5_000_000_000,
}

M8_THRESHOLD = 0.00957966


@dataclass
class Order:
    order_id: int
    side: str
    shares: int
    price: int


@dataclass
class PendingFill:
    timestamp: int
    symbol: str
    order_id: int
    side: str
    execution_price: int
    quantity: int
    pre_mid: float | None
    initial_edge: float | None
    score: float
    selected: bool
    relative_spread: float | None
    imbalance: float | None
    bid_size: int
    ask_size: int
    spread: int
    horizons: dict[str, int]


class Book:
    def __init__(self, symbol: str) -> None:
        self.symbol = symbol

        self.orders: dict[int, Order] = {}

        self.bids: dict[int, int] = defaultdict(int)
        self.asks: dict[int, int] = defaultdict(int)

        self.bid_heap: list[int] = []
        self.ask_heap: list[int] = []

    def _levels(self, side: str) -> dict[int, int]:
        return self.bids if side == "B" else self.asks

    def _add_level(
        self,
        side: str,
        price: int,
        quantity: int,
    ) -> None:
        levels = self._levels(side)

        was_empty = levels.get(price, 0) == 0
        levels[price] += quantity

        if was_empty:
            if side == "B":
                heapq.heappush(self.bid_heap, -price)
            else:
                heapq.heappush(self.ask_heap, price)

    def _remove_level(
        self,
        side: str,
        price: int,
        quantity: int,
    ) -> None:
        levels = self._levels(side)
        current = levels.get(price, 0)

        quantity = min(quantity, current)
        remaining = current - quantity

        if remaining == 0:
            levels.pop(price, None)
        else:
            levels[price] = remaining

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

        return (bid + ask) / 20000.0

    def spread(self) -> int | None:
        bid = self.best_bid()
        ask = self.best_ask()

        if bid is None or ask is None:
            return None

        return ask - bid

    def top_levels(
        self,
        side: str,
        depth: int = 10,
    ) -> list[tuple[int, int]]:
        levels = self._levels(side)

        if side == "B":
            prices = sorted(
                levels.keys(),
                reverse=True,
            )[:depth]
        else:
            prices = sorted(
                levels.keys()
            )[:depth]

        return [
            (price, levels[price])
            for price in prices
        ]

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
            shares=quantity,
            price=price,
        )

        self._add_level(
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
            order.shares,
        )

        self._remove_level(
            order.side,
            order.price,
            quantity,
        )

        order.shares -= quantity

        if order.shares == 0:
            del self.orders[order_id]

    def cancel(
        self,
        order_id: int,
        quantity: int,
    ) -> None:
        order = self.orders.get(order_id)

        if order is None:
            return

        quantity = min(
            quantity,
            order.shares,
        )

        self._remove_level(
            order.side,
            order.price,
            quantity,
        )

        order.shares -= quantity

        if order.shares == 0:
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

        self._remove_level(
            order.side,
            order.price,
            order.shares,
        )

    def replace(
        self,
        old_order_id: int,
        new_order_id: int,
        quantity: int,
        price: int,
    ) -> None:
        old = self.orders.get(
            old_order_id
        )

        if old is None:
            return

        side = old.side

        self._remove_level(
            side,
            old.price,
            old.shares,
        )

        del self.orders[old_order_id]

        self.orders[new_order_id] = Order(
            order_id=new_order_id,
            side=side,
            shares=quantity,
            price=price,
        )

        self._add_level(
            side,
            price,
            quantity,
        )


def read_uint(
    payload: bytes,
    start: int,
    length: int,
) -> int:
    return int.from_bytes(
        payload[
            start:start + length
        ],
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


def calculate_features(
    book: Book,
) -> dict:
    bid = book.best_bid()
    ask = book.best_ask()

    bid_size = book.best_bid_size()
    ask_size = book.best_ask_size()

    spread = book.spread()
    mid = book.mid()

    if (
        bid is None
        or ask is None
        or spread is None
        or mid is None
    ):
        return {
            "mid": None,
            "spread": None,
            "relative_spread": None,
            "imbalance": None,
            "bid_size": bid_size,
            "ask_size": ask_size,
        }

    total = bid_size + ask_size

    imbalance = (
        0.0
        if total == 0
        else (bid_size - ask_size) / total
    )

    relative_spread = (
        spread / 10000.0
    ) / mid

    return {
        "mid": mid,
        "spread": spread,
        "relative_spread": relative_spread,
        "imbalance": imbalance,
        "bid_size": bid_size,
        "ask_size": ask_size,
    }


def m8_score(
    relative_spread: float | None,
    imbalance: float | None,
    side: str,
) -> float:
    if (
        relative_spread is None
        or imbalance is None
    ):
        return float("-inf")

    signed_imbalance = (
        imbalance
        if side == "B"
        else -imbalance
    )

    relative_spread_pct = (
        relative_spread * 100.0
    )

    return (
        relative_spread_pct
        * signed_imbalance
    )


def initial_edge(
    side: str,
    fill_price: float,
    mid: float | None,
) -> float | None:
    if mid is None:
        return None

    if side == "B":
        return mid - fill_price

    return fill_price - mid


def resolve_pending(
    pending: list[PendingFill],
    timestamp: int,
    symbol: str,
    mid: float | None,
    rows: list[dict],
) -> None:
    if mid is None:
        return

    remaining = []

    for fill in pending:
        if fill.symbol != symbol:
            remaining.append(fill)
            continue

        future_columns = {}

        unresolved = False

        for horizon_name, horizon_ts in fill.horizons.items():
            column_name = (
                f"mid_{horizon_name}"
            )

            if (
                horizon_ts <= timestamp
                and column_name not in fill.__dict__
            ):
                future_columns[column_name] = (
                    mid
                )
            else:
                unresolved = True

        if future_columns:
            fill.__dict__.update(
                future_columns
            )

        required = [
            f"mid_{name}"
            for name in HORIZONS
        ]

        if all(
            key in fill.__dict__
            for key in required
        ):
            row = fill_to_row(fill)
            rows.append(row)
        else:
            remaining.append(fill)

    pending[:] = remaining


def fill_to_row(
    fill: PendingFill,
) -> dict:
    row = {
        "timestamp_ns": fill.timestamp,
        "symbol": fill.symbol,
        "order_id": fill.order_id,
        "side": fill.side,
        "execution_price": (
            fill.execution_price
            / 10000.0
        ),
        "quantity": fill.quantity,
        "pre_mid": fill.pre_mid,
        "initial_edge": fill.initial_edge,
        "score": fill.score,
        "selected": int(
            fill.selected
        ),
        "relative_spread": (
            fill.relative_spread
        ),
        "imbalance": fill.imbalance,
        "bid_size": fill.bid_size,
        "ask_size": fill.ask_size,
        "spread_ticks": (
            fill.spread / 10000.0
        ),
    }

    for horizon_name in HORIZONS:
        future_mid = fill.__dict__[
            f"mid_{horizon_name}"
        ]

        if fill.pre_mid is None:
            row[
                f"post_fill_move_{horizon_name}"
            ] = None

        elif fill.side == "B":
            row[
                f"post_fill_move_{horizon_name}"
            ] = (
                future_mid
                - fill.pre_mid
            )

        else:
            row[
                f"post_fill_move_{horizon_name}"
            ] = (
                fill.pre_mid
                - future_mid
            )

        if fill.execution_price != 0:
            row[
                f"return_bps_{horizon_name}"
            ] = (
                row[
                    f"post_fill_move_{horizon_name}"
                ]
                / (
                    fill.execution_price
                    / 10000.0
                )
                * 10000.0
            )

        row[
            f"future_mid_{horizon_name}"
        ] = future_mid

    return row


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--path",
        default=str(ITCH_PATH),
    )

    parser.add_argument(
        "--output",
        default=str(OUTPUT_PATH),
    )

    parser.add_argument(
        "--max-messages",
        type=int,
        default=None,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    books = {
        locate: Book(symbol)
        for locate, symbol in TARGETS.items()
    }

    pending = {
        locate: []
        for locate in TARGETS
    }

    rows: list[dict] = []

    total_messages = 0
    execution_events = 0

    for payload in iter_messages(
        Path(args.path)
    ):
        total_messages += 1

        if (
            args.max_messages is not None
            and total_messages > args.max_messages
        ):
            break

        if not payload:
            continue

        msg_type = chr(payload[0])

        if msg_type == "S":
            continue

        if len(payload) < 11:
            continue

        locate = read_uint(
            payload,
            1,
            2,
        )

        book = books.get(locate)

        if book is None:
            continue

        timestamp = read_uint(
            payload,
            5,
            6,
        )

        # Resolve pending fills against
        # the current pre-event midpoint.
        resolve_pending(
            pending[locate],
            timestamp,
            book.symbol,
            book.mid(),
            rows,
        )

        if msg_type == "A":
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

        elif msg_type == "F":
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

        elif msg_type in ("E", "C"):
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

            if order is None:
                continue

            features = calculate_features(
                book
            )

            if msg_type == "C":
                execution_price = read_uint(
                    payload,
                    32,
                    4,
                )
            else:
                execution_price = order.price

            score = m8_score(
                features["relative_spread"],
                features["imbalance"],
                order.side,
            )

            selected = (
                score
                >= M8_THRESHOLD
            )

            pre_mid = features["mid"]

            fill_price = (
                execution_price
                / 10000.0
            )

            edge = initial_edge(
                order.side,
                fill_price,
                pre_mid,
            )

            fill = PendingFill(
                timestamp=timestamp,
                symbol=book.symbol,
                order_id=order_id,
                side=order.side,
                execution_price=execution_price,
                quantity=quantity,
                pre_mid=pre_mid,
                initial_edge=edge,
                score=score,
                selected=selected,
                relative_spread=(
                    features[
                        "relative_spread"
                    ]
                ),
                imbalance=(
                    features[
                        "imbalance"
                    ]
                ),
                bid_size=features[
                    "bid_size"
                ],
                ask_size=features[
                    "ask_size"
                ],
                spread=(
                    features["spread"]
                    or 0
                ),
                horizons={
                    name: (
                        timestamp
                        + delta
                    )
                    for name, delta
                    in HORIZONS.items()
                },
            )

            pending[locate].append(
                fill
            )

            execution_events += 1

            book.execute(
                order_id,
                quantity,
            )

        elif msg_type == "X":
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

            book.cancel(
                order_id,
                quantity,
            )

        elif msg_type == "D":
            order_id = read_uint(
                payload,
                11,
                8,
            )

            book.delete(
                order_id
            )

        elif msg_type == "U":
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

        if (
            total_messages
            % 10_000_000
            == 0
        ):
            print(
                f"processed={total_messages:,} "
                f"executions={execution_events:,} "
                f"rows={len(rows):,}"
            )

    output = Path(
        args.output
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if rows:
        fieldnames = list(
            rows[0].keys()
        )

        with output.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=fieldnames,
            )

            writer.writeheader()
            writer.writerows(rows)

    print()
    print(
        "=== ITCH M8 DATASET ==="
    )
    print(
        f"messages:   {total_messages:,}"
    )
    print(
        f"executions: {execution_events:,}"
    )
    print(
        f"completed:  {len(rows):,}"
    )
    print(
        f"output:     {output}"
    )


if __name__ == "__main__":
    main()