from datetime import date
from itertools import islice
from pathlib import Path

from merit.book.l2_book import L2OrderBook
from merit.data.lobster import read_messages
from merit.data.lobster_orderbook import read_orderbooks

BASE = Path("data/sample")

MESSAGE_FILE = BASE / "AAPL_2012-06-21_34200000_57600000_message_10.csv"
BOOK_FILE = BASE / "AAPL_2012-06-21_34200000_57600000_orderbook_10.csv"


def main() -> None:
    messages = iter(
        read_messages(
            MESSAGE_FILE,
            "AAPL",
            date(2012, 6, 21),
        )
    )

    references = iter(
        read_orderbooks(
            BOOK_FILE,
            levels=10,
        )
    )

    first_reference = next(references)

    book = L2OrderBook("AAPL")
    book.load_snapshot(*first_reference)

    next(messages)

    checked = 0

    for event, reference in islice(zip(messages, references), 1000):
        book.apply(event)

        actual_bids, actual_asks = book.depth(10)
        expected_bids, expected_asks = reference

        event_number = checked + 2

        if actual_bids != expected_bids:
            raise AssertionError(
                f"Bid mismatch at event {event_number}\n"
                f"Expected: {expected_bids}\n"
                f"Actual:   {actual_bids}"
            )

        if actual_asks != expected_asks:
            raise AssertionError(
                f"Ask mismatch at event {event_number}\n"
                f"Expected: {expected_asks}\n"
                f"Actual:   {actual_asks}"
            )

        checked += 1

    print(f"Validated {checked} book transitions successfully.")


if __name__ == "__main__":
    main()