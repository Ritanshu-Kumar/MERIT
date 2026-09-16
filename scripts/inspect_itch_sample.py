from __future__ import annotations

import gzip
from collections import Counter
from pathlib import Path


PATH = Path(
    "data/itch/07302019.NASDAQ_ITCH50.gz"
)

TARGETS = {
    "AAPL",
    "AMZN",
    "GOOG",
    "INTC",
    "MSFT",
}


def decode_stock(value: bytes) -> str:
    return value.decode(
        "ascii",
        errors="ignore",
    ).strip()


def main() -> None:
    message_counts = Counter()
    target_symbols = {}
    total_messages = 0
    total_bytes = 0

    print("ITCH SAMPLE INSPECTION")
    print("=" * 80)
    print(f"File: {PATH}")

    with gzip.open(PATH, "rb") as file:
        while True:
            header = file.read(2)

            if not header:
                break

            if len(header) != 2:
                raise ValueError(
                    "Truncated message-length header."
                )

            message_length = int.from_bytes(
                header,
                byteorder="big",
                signed=False,
            )

            if message_length <= 0:
                raise ValueError(
                    f"Invalid message length: "
                    f"{message_length}"
                )

            payload = file.read(
                message_length
            )

            if len(payload) != message_length:
                raise ValueError(
                    "Truncated ITCH message."
                )

            total_messages += 1
            total_bytes += (
                2 + message_length
            )

            message_type = chr(
                payload[0]
            )

            message_counts[
                message_type
            ] += 1

            if message_type == "R":
                if len(payload) < 21:
                    continue

                stock = decode_stock(
                    payload[11:19]
                )

                if stock in TARGETS:
                    target_symbols[
                        stock
                    ] = {
                        "stock_locate": int.from_bytes(
                            payload[1:3],
                            byteorder="big",
                        ),
                        "tracking_number": int.from_bytes(
                            payload[3:5],
                            byteorder="big",
                        ),
                    }

    print()
    print("SCAN COMPLETE")
    print("-" * 80)
    print(
        f"Messages:      {total_messages:,}"
    )
    print(
        f"Compressed stream bytes read: "
        f"{total_bytes:,}"
    )

    print()
    print("Target symbols")
    print("-" * 80)

    for symbol in sorted(TARGETS):
        if symbol in target_symbols:
            info = target_symbols[symbol]

            print(
                f"{symbol}: FOUND "
                f"(stock_locate="
                f"{info['stock_locate']})"
            )
        else:
            print(
                f"{symbol}: NOT FOUND"
            )

    print()
    print("Message types")
    print("-" * 80)

    for message_type, count in (
        message_counts.most_common()
    ):
        print(
            f"{message_type!r}: "
            f"{count:,}"
        )


if __name__ == "__main__":
    main()