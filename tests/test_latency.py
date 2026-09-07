from datetime import datetime, timedelta

import pytest

from merit.execution.latency import LatencyModel


def test_activation_time() -> None:
    model = LatencyModel(
        submit_latency=timedelta(milliseconds=500)
    )

    created_at = datetime(2012, 6, 21, 9, 30)

    assert model.activation_time(created_at) == datetime(
        2012,
        6,
        21,
        9,
        30,
        0,
        500_000,
    )


def test_zero_latency() -> None:
    model = LatencyModel(
        submit_latency=timedelta(0)
    )

    created_at = datetime(2012, 6, 21, 9, 30)

    assert model.activation_time(created_at) == created_at


def test_negative_latency_rejected() -> None:
    with pytest.raises(ValueError):
        LatencyModel(
            submit_latency=timedelta(milliseconds=-1)
        )