import csv
import time
from datetime import date
from pathlib import Path

from merit.execution.queue_model import QueueModel
from merit.research.lobster_m8 import (
    build_lobster_research_dataset,
)


BASE = Path("data/sample")

SYMBOLS = [
    "AAPL",
    "AMZN",
    "GOOG",
    "INTC",
    "MSFT",
]

TRADING_DATE = date(2012, 6, 21)

QUEUE_AHEAD_FRACTION = 0.0
ORDER_QUANTITY = 100
LEVELS = 10

OUTPUT_FILE = BASE / "M8_replication_research_dataset.csv"


def get_message_file(symbol: str) -> Path:
    return BASE / (
        f"{symbol}_2012-06-21_34200000_57600000"
        "_message_10.csv"
    )


def get_orderbook_file(symbol: str) -> Path:
    return BASE / (
        f"{symbol}_2012-06-21_34200000_57600000"
        "_orderbook_10.csv"
    )


def get_symbol_output(symbol: str) -> Path:
    return BASE / f"M8_replication_{symbol}.csv"


def write_dataset(path: Path, observations) -> None:
    if not observations:
        print(f"  No observations for {path.stem}")
        return

    fieldnames = list(
        observations[0].__dataclass_fields__.keys()
    )

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

        for observation in observations:
            writer.writerow(
                {
                    field: getattr(
                        observation,
                        field,
                    )
                    for field in fieldnames
                }
            )


def combine_datasets(
    symbol_files: list[Path],
    output_file: Path,
) -> None:
    combined_rows = []

    for path in symbol_files:
        if not path.exists():
            continue

        with path.open(
            "r",
            newline="",
            encoding="utf-8",
        ) as file:
            reader = csv.DictReader(file)
            combined_rows.extend(reader)

    if not combined_rows:
        raise ValueError(
            "No symbol datasets were available."
        )

    fieldnames = list(combined_rows[0].keys())

    with output_file.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(combined_rows)


def main() -> None:
    completed_files = []

    for symbol in SYMBOLS:
        message_file = get_message_file(symbol)
        orderbook_file = get_orderbook_file(symbol)
        symbol_output = get_symbol_output(symbol)

        if not message_file.exists():
            raise FileNotFoundError(message_file)

        if not orderbook_file.exists():
            raise FileNotFoundError(orderbook_file)

        print(f"\nProcessing {symbol}...")

        start = time.perf_counter()

        observations = build_lobster_research_dataset(
            message_path=message_file,
            orderbook_path=orderbook_file,
            symbol=symbol,
            trading_date=TRADING_DATE,
            queue_model=QueueModel(
                ahead_fraction=QUEUE_AHEAD_FRACTION,
            ),
            order_quantity=ORDER_QUANTITY,
            levels=LEVELS,
            max_observations=None,
        )

        elapsed = time.perf_counter() - start

        write_dataset(
            symbol_output,
            observations,
        )

        print(
            f"  Observations: {len(observations):,}"
        )
        print(
            f"  Time: {elapsed:.2f} seconds"
        )
        print(
            f"  Saved: {symbol_output}"
        )

        completed_files.append(symbol_output)

    combine_datasets(
        completed_files,
        OUTPUT_FILE,
    )

    print(
        f"\nCombined output: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()