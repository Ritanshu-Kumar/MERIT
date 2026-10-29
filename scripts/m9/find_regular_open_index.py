from pathlib import Path

from build_state_transitions import (
    iter_itch_messages,
    parse_timestamp_ns,
)

INPUT = Path("data/itch/07302019.NASDAQ_ITCH50.gz")
REGULAR_OPEN_NS = 34_200_000_000_000  # 09:30:00

for index, msg in enumerate(iter_itch_messages(INPUT), start=1):
    if len(msg) < 11:
        continue

    timestamp_ns = parse_timestamp_ns(msg)

    if timestamp_ns >= REGULAR_OPEN_NS:
        seconds = timestamp_ns / 1_000_000_000

        print("First message at/after regular open")
        print("Message index:", index)
        print("Timestamp (ns):", timestamp_ns)
        print(
            "Time since midnight:",
            f"{int(seconds // 3600):02d}:"
            f"{int((seconds % 3600) // 60):02d}:"
            f"{seconds % 60:06.3f}",
        )
        print("Suggested bounded smoke-test limit:", index + 500_000)
        break
else:
    print("No message reached the regular-session open.")