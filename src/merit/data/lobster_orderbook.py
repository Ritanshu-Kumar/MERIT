import csv
from collections.abc import Iterator
from decimal import Decimal
from pathlib import Path


PRICE_SCALE = Decimal("10000")


def read_orderbooks(
    path: str | Path,
    levels: int = 10,
) -> Iterator[
    tuple[
        tuple[tuple[Decimal, int], ...],
        tuple[tuple[Decimal, int], ...],
    ]
]:
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(path)

    expected_columns = levels * 4

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        reader = csv.reader(file)

        for row in reader:
            if not row:
                continue

            if len(row) != expected_columns:
                raise ValueError(
                    f"Expected {expected_columns} columns, got {len(row)}"
                )

            values = [int(value) for value in row]

            asks = tuple(
                (
                    Decimal(values[index]) / PRICE_SCALE,
                    values[index + 1],
                )
                for index in range(0, expected_columns, 4)
            )

            bids = tuple(
                (
                    Decimal(values[index]) / PRICE_SCALE,
                    values[index + 1],
                )
                for index in range(2, expected_columns, 4)
            )

            yield bids, asks