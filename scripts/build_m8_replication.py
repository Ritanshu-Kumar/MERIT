from __future__ import annotations

import argparse
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

LEVELS = 10
QUEUE_AHEAD_FRACTION = 0.0
ORDER_QUANTITY = 100


def parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Date must use YYYY-MM-DD."
        ) from exc


def get_message_file(
    symbol: str,
    trading_date: date,
) -> Path:
    return BASE / (
        f"{symbol}_{trading_date.isoformat()}"
        "_34200000_57600000"
        "_message_10.csv"
    )


def get_orderbook_file(
    symbol: str,
    trading_date: date,
) -> Path:
    return BASE / (
        f"{symbol}_{trading_date.isoformat()}"
        "_34200000_57600000"
        "_orderbook_10.csv"
    )


def get_symbol_output(
    symbol: str,
    trading_date: date,
) -> Path:
    return BASE / (
        f"M8_replication_"
        f"{symbol}_{trading_date.isoformat()}.csv"
    )


def get_combined_output(
    trading_date: date,
) -> Path:
    return BASE / (
        f"M8_replication_research_dataset_"
        f"{trading_date.isoformat()}.csv"
    )


def write_dataset(
    path: Path,
    observations,
) -> None:
    if not observations:
        print(
            f"  No observations for {path.stem}"
        )
        return

    fieldnames = list(
        observations[0]
        .__dataclass_fields__
        .keys()
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

    fieldnames = list(
        combined_rows[0].keys()
    )

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
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--date",
        required=True,
        type=parse_date,
        help="Trading date in YYYY-MM-DD format.",
    )

    args = parser.parse_args()

    trading_date = args.date

    print()
    print("M8 RESEARCH DATASET BUILDER")
    print("=" * 80)
    print(
        f"Trading date: {trading_date}"
    )

    completed_files = []

    for symbol in SYMBOLS:
        message_file = get_message_file(
            symbol,
            trading_date,
        )

        orderbook_file = get_orderbook_file(
            symbol,
            trading_date,
        )

        symbol_output = get_symbol_output(
            symbol,
            trading_date,
        )

        if not message_file.exists():
            raise FileNotFoundError(
                f"Missing message file: "
                f"{message_file}"
            )

        if not orderbook_file.exists():
            raise FileNotFoundError(
                f"Missing orderbook file: "
                f"{orderbook_file}"
            )

        print(
            f"\nProcessing {symbol}..."
        )

        start = time.perf_counter()

        observations = (
            build_lobster_research_dataset(
                message_path=message_file,
                orderbook_path=orderbook_file,
                symbol=symbol,
                trading_date=trading_date,
                queue_model=QueueModel(
                    ahead_fraction=(
                        QUEUE_AHEAD_FRACTION
                    ),
                ),
                order_quantity=ORDER_QUANTITY,
                levels=LEVELS,
                max_observations=None,
            )
        )

        elapsed = (
            time.perf_counter()
            - start
        )

        write_dataset(
            symbol_output,
            observations,
        )

        print(
            f"  Observations: "
            f"{len(observations):,}"
        )

        print(
            f"  Time: {elapsed:.2f} seconds"
        )

        print(
            f"  Saved: {symbol_output}"
        )

        completed_files.append(
            symbol_output
        )

    combined_output = (
        get_combined_output(
            trading_date
        )
    )

    combine_datasets(
        completed_files,
        combined_output,
    )

    print(
        f"\nCombined output: "
        f"{combined_output}"
    )


if __name__ == "__main__":
    main()