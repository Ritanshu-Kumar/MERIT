from decimal import Decimal

import pytest

from merit.data.lobster_orderbook import read_orderbooks


def write_book(path, row: str) -> None:
    path.write_text(row + "\n", encoding="utf-8")


def test_reads_level_10_book(tmp_path) -> None:
    path = tmp_path / "book.csv"

    values = []
    for level in range(10):
        values.extend(
            [
                str(5859400 + level * 100),
                str(200 + level),
                str(5853300 - level * 100),
                str(18 + level),
            ]
        )

    write_book(path, ",".join(values))

    bids, asks = next(read_orderbooks(path))

    assert asks[0] == (Decimal("585.94"), 200)
    assert bids[0] == (Decimal("585.33"), 18)
    assert len(asks) == 10
    assert len(bids) == 10


def test_rejects_wrong_column_count(tmp_path) -> None:
    path = tmp_path / "book.csv"
    write_book(path, "1,2,3")

    with pytest.raises(ValueError, match="Expected 40 columns"):
        next(read_orderbooks(path))