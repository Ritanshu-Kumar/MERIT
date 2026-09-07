from datetime import date

import pytest

from merit.data.lobster import read_messages


def test_non_order_events_are_skipped(tmp_path) -> None:
    path = tmp_path / "messages.csv"

    path.write_text(
        "34200.0,5,0,0,0,0\n"
        "34200.1,6,0,0,0,0\n"
        "34200.2,7,0,0,0,0\n",
        encoding="utf-8",
    )

    events = list(
        read_messages(
            path,
            "AAPL",
            date(2012, 6, 21),
        )
    )

    assert events == []