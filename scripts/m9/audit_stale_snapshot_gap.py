from collections import Counter, defaultdict
from pathlib import Path

from build_state_transitions import (
    discover_stock_locates,
    iter_itch_messages,
    parse_stock_locate,
    parse_timestamp_ns,
)

INPUT = Path("data/itch/07302019.NASDAQ_ITCH50.gz")
MAX_MESSAGES = 2_000_000
TARGET_SYMBOLS = {"AAPL", "AMZN", "GOOG", "INTC", "MSFT"}

BOOK_UPDATE_TYPES = {"A", "F", "D", "X", "E", "C", "U"}

locate_to_symbol = discover_stock_locates(INPUT, TARGET_SYMBOLS)

all_type_counts = defaultdict(Counter)
book_update_counts = Counter()
timestamp_ranges = {}
last_updates = defaultdict(list)

for index, msg in enumerate(
    iter_itch_messages(INPUT, max_messages=MAX_MESSAGES),
    start=1,
):
    if len(msg) < 11:
        continue

    locate = parse_stock_locate(msg)
    symbol = locate_to_symbol.get(locate)

    if symbol is None:
        continue

    message_type = chr(msg[0])
    timestamp_ns = parse_timestamp_ns(msg)

    all_type_counts[symbol][message_type] += 1

    if symbol not in timestamp_ranges:
        timestamp_ranges[symbol] = [timestamp_ns, timestamp_ns]
    else:
        timestamp_ranges[symbol][0] = min(
            timestamp_ranges[symbol][0], timestamp_ns
        )
        timestamp_ranges[symbol][1] = max(
            timestamp_ranges[symbol][1], timestamp_ns
        )

    if message_type in BOOK_UPDATE_TYPES:
        book_update_counts[symbol] += 1
        last_updates[symbol].append(
            (index, timestamp_ns, message_type)
        )
        last_updates[symbol] = last_updates[symbol][-5:]

print("Discovered symbol locates:", locate_to_symbol)

for symbol in sorted(TARGET_SYMBOLS):
    print(f"\n--- {symbol} ---")
    print("All message counts by type:", dict(all_type_counts[symbol]))
    print("Recognized book-update count:", book_update_counts[symbol])
    print("Timestamp range (ns):", timestamp_ranges.get(symbol))
    print("Last five book updates:", last_updates[symbol])