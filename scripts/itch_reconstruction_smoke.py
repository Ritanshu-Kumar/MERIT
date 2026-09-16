from __future__ import annotations

import argparse
import gzip
from collections import Counter, defaultdict
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path


PATH = Path(
    "data/itch/07302019.NASDAQ_ITCH50.gz"
)

TARGET_LOCATES = {
    "AAPL": 14,
    "AMZN": 386,
    "GOOG": 3459,
    "INTC": 4184,
    "MSFT": 5289,
}

PRICE_SCALE = Decimal("10000")


@dataclass
class Order:
    order_id: int
    side: str
    shares: int
    price: Decimal
    symbol: str


def u16(data: bytes, start: int) -> int:
    return int.from_bytes(
        data[start:start + 2],
        byteorder="big",
        signed=False,
    )


def u32(data: bytes, start: int) -> int:
    return int.from_bytes(
        data[start:start + 4],
        byteorder="big",
        signed=False,
    )


def u48(data: bytes, start: int) -> int:
    return int.from_bytes(
        data[start:start + 6],
        byteorder="big",
        signed=False,
    )


def u64(data: bytes, start: int) -> int:
    return int.from_bytes(
        data[start:start + 8],
        byteorder="big",
        signed=False,
    )


def symbol_value(data: bytes) -> str:
    return data.decode(
        "ascii",
        errors="ignore",
    ).strip()


def price_value(raw: int) -> Decimal:
    return Decimal(raw) / PRICE_SCALE


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--messages",
        type=int,
        default=5_000_000,
        help="Maximum ITCH messages to scan.",
    )

    args = parser.parse_args()

    locate_to_symbol = {
        locate: symbol
        for symbol, locate
        in TARGET_LOCATES.items()
    }

    orders: dict[int, Order] = {}

    level_sizes = {
        symbol: {
            "B": defaultdict(int),
            "S": defaultdict(int),
        }
        for symbol in TARGET_LOCATES
    }

    message_counts = Counter()

    target_message_counts = Counter()

    invariant_failures = Counter()

    total_messages = 0
    target_messages = 0
    add_messages = 0
    execution_messages = 0
    cancel_messages = 0
    replace_messages = 0

    first_timestamp = None
    last_timestamp = None

    with gzip.open(
        PATH,
        "rb",
    ) as file:

        while total_messages < args.messages:
            header = file.read(2)

            if not header:
                break

            if len(header) != 2:
                raise ValueError(
                    "Truncated message length."
                )

            length = u16(
                header,
                0,
            )

            payload = file.read(length)

            if len(payload) != length:
                raise ValueError(
                    "Truncated ITCH message."
                )

            if not payload:
                continue

            message_type = chr(payload[0])

            message_counts[
                message_type
            ] += 1

            total_messages += 1

            if len(payload) >= 11:
                timestamp = u48(
                    payload,
                    5,
                )

                if first_timestamp is None:
                    first_timestamp = timestamp

                last_timestamp = timestamp

            if message_type not in {
                "A",
                "F",
                "E",
                "C",
                "X",
                "D",
                "U",
            }:
                continue

            if len(payload) < 3:
                continue

            stock_locate = u16(
                payload,
                1,
            )

            symbol = locate_to_symbol.get(
                stock_locate
            )

            if symbol is None:
                continue

            target_messages += 1
            target_message_counts[
                message_type
            ] += 1

            if message_type in {"A", "F"}:
                if len(payload) < 36:
                    invariant_failures[
                        "short_add"
                    ] += 1
                    continue

                order_id = u64(
                    payload,
                    11,
                )

                side = chr(
                    payload[19]
                )

                shares = u32(
                    payload,
                    20,
                )

                stock = symbol_value(
                    payload[24:32]
                )

                price = price_value(
                    u32(
                        payload,
                        32,
                    )
                )

                if stock != symbol:
                    invariant_failures[
                        "symbol_mismatch"
                    ] += 1

                if side not in {"B", "S"}:
                    invariant_failures[
                        "invalid_side"
                    ] += 1
                    continue

                if shares <= 0:
                    invariant_failures[
                        "nonpositive_add"
                    ] += 1
                    continue

                if price <= 0:
                    invariant_failures[
                        "nonpositive_price"
                    ] += 1
                    continue

                if order_id in orders:
                    invariant_failures[
                        "duplicate_order_id"
                    ] += 1

                    old = orders.pop(
                        order_id
                    )

                    level_sizes[
                        old.symbol
                    ][old.side][
                        old.price
                    ] -= old.shares

                orders[order_id] = Order(
                    order_id=order_id,
                    side=side,
                    shares=shares,
                    price=price,
                    symbol=symbol,
                )

                level_sizes[
                    symbol
                ][side][price] += shares

                add_messages += 1

            elif message_type in {
                "E",
                "C",
            }:
                if len(payload) < 27:
                    invariant_failures[
                        "short_execution"
                    ] += 1
                    continue

                order_id = u64(
                    payload,
                    11,
                )

                executed = u32(
                    payload,
                    19,
                )

                order = orders.get(
                    order_id
                )

                if order is None:
                    invariant_failures[
                        "execution_unknown_order"
                    ] += 1
                    continue

                if executed <= 0:
                    invariant_failures[
                        "nonpositive_execution"
                    ] += 1
                    continue

                if executed > order.shares:
                    invariant_failures[
                        "over_execution"
                    ] += 1

                    executed = order.shares

                level_sizes[
                    symbol
                ][order.side][
                    order.price
                ] -= executed

                order.shares -= executed

                if order.shares == 0:
                    del orders[
                        order_id
                    ]

                execution_messages += 1

            elif message_type == "X":
                if len(payload) < 23:
                    invariant_failures[
                        "short_cancel"
                    ] += 1
                    continue

                order_id = u64(
                    payload,
                    11,
                )

                canceled = u32(
                    payload,
                    19,
                )

                order = orders.get(
                    order_id
                )

                if order is None:
                    invariant_failures[
                        "cancel_unknown_order"
                    ] += 1
                    continue

                if (
                    canceled <= 0
                    or canceled
                    > order.shares
                ):
                    invariant_failures[
                        "invalid_cancel"
                    ] += 1

                    canceled = min(
                        max(
                            canceled,
                            0,
                        ),
                        order.shares,
                    )

                level_sizes[
                    symbol
                ][order.side][
                    order.price
                ] -= canceled

                order.shares -= canceled

                if order.shares == 0:
                    del orders[
                        order_id
                    ]

                cancel_messages += 1

            elif message_type == "D":
                if len(payload) < 19:
                    invariant_failures[
                        "short_delete"
                    ] += 1
                    continue

                order_id = u64(
                    payload,
                    11,
                )

                order = orders.get(
                    order_id
                )

                if order is None:
                    invariant_failures[
                        "delete_unknown_order"
                    ] += 1
                    continue

                level_sizes[
                    symbol
                ][order.side][
                    order.price
                ] -= order.shares

                del orders[
                    order_id
                ]

                cancel_messages += 1

            elif message_type == "U":
                if len(payload) < 35:
                    invariant_failures[
                        "short_replace"
                    ] += 1
                    continue

                old_id = u64(
                    payload,
                    11,
                )

                new_id = u64(
                    payload,
                    19,
                )

                new_shares = u32(
                    payload,
                    27,
                )

                new_price = price_value(
                    u32(
                        payload,
                        31,
                    )
                )

                old_order = orders.get(
                    old_id
                )

                if old_order is None:
                    invariant_failures[
                        "replace_unknown_order"
                    ] += 1
                    continue

                level_sizes[
                    symbol
                ][old_order.side][
                    old_order.price
                ] -= old_order.shares

                del orders[
                    old_id
                ]

                if new_shares <= 0:
                    invariant_failures[
                        "replace_nonpositive_shares"
                    ] += 1
                    continue

                if new_price <= 0:
                    invariant_failures[
                        "replace_nonpositive_price"
                    ] += 1
                    continue

                if new_id in orders:
                    invariant_failures[
                        "replace_duplicate_new_id"
                    ] += 1

                    duplicate = orders.pop(
                        new_id
                    )

                    level_sizes[
                        duplicate.symbol
                    ][duplicate.side][
                        duplicate.price
                    ] -= duplicate.shares

                orders[new_id] = Order(
                    order_id=new_id,
                    side=old_order.side,
                    shares=new_shares,
                    price=new_price,
                    symbol=symbol,
                )

                level_sizes[
                    symbol
                ][old_order.side][
                    new_price
                ] += new_shares

                replace_messages += 1

            for current_symbol in TARGET_LOCATES:
                for side in ("B", "S"):
                    negative_levels = [
                        quantity
                        for quantity in level_sizes[
                            current_symbol
                        ][side].values()
                        if quantity < 0
                    ]

                    if negative_levels:
                        invariant_failures[
                            "negative_level_depth"
                        ] += 1

    print()
    print("ITCH RECONSTRUCTION SMOKE TEST")
    print("=" * 80)

    print(
        f"Messages scanned: "
        f"{total_messages:,}"
    )

    print(
        f"Target-symbol messages: "
        f"{target_messages:,}"
    )

    print()
    print("Target message types")
    print("-" * 80)

    for message_type, count in sorted(
        target_message_counts.items()
    ):
        print(
            f"{message_type}: "
            f"{count:,}"
        )

    print()
    print("Orders remaining in book")
    print("-" * 80)

    for symbol in sorted(
        TARGET_LOCATES
    ):
        count = sum(
            1
            for order in orders.values()
            if order.symbol == symbol
        )

        print(
            f"{symbol}: {count:,}"
        )

    print()
    print("Invariant checks")
    print("-" * 80)

    if invariant_failures:
        for name, count in (
            invariant_failures.most_common()
        ):
            print(
                f"{name}: {count:,}"
            )
    else:
        print(
            "No invariant violations detected."
        )

    print()
    print("Message-type totals")
    print("-" * 80)

    for message_type, count in (
        message_counts.most_common()
    ):
        print(
            f"{message_type}: "
            f"{count:,}"
        )

    print()
    print(
        "Timestamp range (ITCH nanoseconds "
        "since midnight):"
    )

    print(
        f"{first_timestamp} -> "
        f"{last_timestamp}"
    )

    print()
    print(
        f"Add messages processed: "
        f"{add_messages:,}"
    )

    print(
        f"Execution messages processed: "
        f"{execution_messages:,}"
    )

    print(
        f"Cancel/delete messages processed: "
        f"{cancel_messages:,}"
    )

    print(
        f"Replace messages processed: "
        f"{replace_messages:,}"
    )


if __name__ == "__main__":
    main()