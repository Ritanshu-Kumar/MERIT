import pytest

from merit.execution.participation import ParticipationModel


def test_full_participation() -> None:
    model = ParticipationModel(rate=1.0)

    assert model.available_quantity(500) == 500


def test_ten_percent_participation() -> None:
    model = ParticipationModel(rate=0.10)

    assert model.available_quantity(500) == 50


def test_fractional_quantity_is_floored() -> None:
    model = ParticipationModel(rate=0.10)

    assert model.available_quantity(55) == 5


def test_zero_trade() -> None:
    model = ParticipationModel(rate=0.10)

    assert model.available_quantity(0) == 0


def test_invalid_rate() -> None:
    with pytest.raises(ValueError):
        ParticipationModel(rate=0)

    with pytest.raises(ValueError):
        ParticipationModel(rate=-0.1)

    with pytest.raises(ValueError):
        ParticipationModel(rate=1.1)


def test_negative_trade_rejected() -> None:
    model = ParticipationModel(rate=0.10)

    with pytest.raises(ValueError):
        model.available_quantity(-1)