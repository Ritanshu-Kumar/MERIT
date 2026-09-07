from datetime import date, datetime, timezone
from decimal import Decimal

from merit.data.lobster import read_messages
from merit.research.lobster_replay import (
    build_feature_snapshot_from_book,
    replay_lobster_snapshots,
)


TRADING_DATE = date(2012, 6, 21)


def test_snapshot_features() -> None:
    bids = (
        (Decimal("100.00"), 200),
        (Decimal("99.90"), 100),
    )

    asks = (
        (Decimal("100.10"), 100),
        (Decimal("100.20"), 100),
    )

    features = build_feature_snapshot_from_book(
        bids=bids,
        asks=asks,
    )

    assert features.mid_price == Decimal("100.05")
    assert features.spread == Decimal("0.10")
    assert features.relative_spread == (
        Decimal("0.10") / Decimal("100.05")
    )
    assert features.bid_size == 200
    assert features.ask_size == 100
    assert features.imbalance == Decimal("1") / Decimal("3")


def test_one_sided_snapshot() -> None:
    features = build_feature_snapshot_from_book(
        bids=((Decimal("100.00"), 100),),
        asks=(),
    )

    assert features.mid_price is None
    assert features.spread is None
    assert features.relative_spread is None
    assert features.bid_size == 100
    assert features.ask_size == 0
    assert features.imbalance is None
    assert features.microprice is None


def test_empty_snapshot() -> None:
    features = build_feature_snapshot_from_book(
        bids=(),
        asks=(),
    )

    assert features.mid_price is None
    assert features.spread is None
    assert features.relative_spread is None
    assert features.bid_size == 0
    assert features.ask_size == 0
    assert features.imbalance is None
    assert features.microprice is None


def test_replay_uses_message_timestamps_and_snapshot_state(
    tmp_path,
) -> None:
    message_path = tmp_path / "messages.csv"
    book_path = tmp_path / "orderbook.csv"

    message_path.write_text(
        "\n".join(
            [
                "34200.000000,1,1,100,1000000,1",
                "34200.100000,1,2,100,1001000,-1",
            ]
        ),
        encoding="utf-8",
    )

    book_path.write_text(
        "\n".join(
            [
                "1001000,100,1000000,100",
                "1001000,100,1000000,100",
            ]
        ),
        encoding="utf-8",
    )

    events = read_messages(
        message_path,
        "AAPL",
        TRADING_DATE,
    )

    states = list(
        replay_lobster_snapshots(
            events=events,
            orderbook_path=book_path,
            levels=1,
        )
    )

    assert len(states) == 2

    assert states[0].timestamp == datetime(
        2012,
        6,
        21,
        9,
        30,
        tzinfo=timezone.utc,
    )

    assert states[1].timestamp == datetime(
        2012,
        6,
        21,
        9,
        30,
        0,
        100000,
        tzinfo=timezone.utc,
    )

    assert states[0].mid_price == Decimal("100.05")
    assert states[1].mid_price == Decimal("100.05")

    assert states[1].features.bid_size == 100
    assert states[1].features.ask_size == 100