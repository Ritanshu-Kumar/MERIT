from __future__ import annotations

import pandas as pd
import pytest

from evaluate_m8_ml_economics import (
    add_candidate_pnl,
    validate_markout_timestamps,
)


def valid_timestamp_frame() -> pd.DataFrame:
    fill = 1_000_000_000
    row = {
        "fill_timestamp_ns": fill,
        "future_mid_10ms": 100.1,
        "future_mid_10ms_timestamp_ns": fill + 10_000_000,
        "future_mid_100ms": 100.2,
        "future_mid_100ms_timestamp_ns": fill + 100_000_000,
        "future_mid_500ms": 100.3,
        "future_mid_500ms_timestamp_ns": fill + 500_000_000,
        "future_mid_1s": 100.4,
        "future_mid_1s_timestamp_ns": fill + 1_000_000_000,
    }
    return pd.DataFrame([row])


def test_timestamp_validator_accepts_at_horizon_observations() -> None:
    validate_markout_timestamps(valid_timestamp_frame())


def test_timestamp_validator_rejects_observation_before_horizon() -> None:
    df = valid_timestamp_frame()
    df.loc[0, "future_mid_100ms_timestamp_ns"] -= 1

    with pytest.raises(ValueError, match="precede target horizon"):
        validate_markout_timestamps(df)


def test_timestamp_validator_rejects_price_timestamp_mismatch() -> None:
    df = valid_timestamp_frame()
    df.loc[0, "future_mid_500ms_timestamp_ns"] = float("nan")

    with pytest.raises(ValueError, match="midpoint/timestamp mismatches"):
        validate_markout_timestamps(df)


def test_primary_pnl_uses_full_fills_and_reports_partial_sensitivity_separately() -> None:
    df = pd.DataFrame([
        {
            "timestamp_ns": 1_000_000_000,
            "quote_side": "BUY",
            "quote_price": 100.0,
            "quote_qty": 10.0,
            "filled_qty": 10.0,
            "resolution_mid": 101.0,
            "outcome": "FILL",
            "future_mid_100ms": 100.5,
        },
        {
            "timestamp_ns": 2_000_000_000,
            "quote_side": "BUY",
            "quote_price": 100.0,
            "quote_qty": 10.0,
            "filled_qty": 4.0,
            "resolution_mid": 99.0,
            "outcome": "ADVERSE",
            "future_mid_100ms": float("nan"),
        },
        {
            "timestamp_ns": 3_000_000_000,
            "quote_side": "SELL",
            "quote_price": 100.0,
            "quote_qty": 10.0,
            "filled_qty": 0.0,
            "resolution_mid": 100.0,
            "outcome": "TIMEOUT",
            "future_mid_100ms": float("nan"),
        },
    ])

    result = add_candidate_pnl(
        df, "100ms", fee_bps=0.0, qty_haircut=1.0
    )

    assert result["net_mtm_pnl"].tolist() == pytest.approx(
        [5.0, 0.0, 0.0]
    )
    assert result["partial_sensitivity_net_pnl"].tolist() == pytest.approx(
        [0.0, -4.0, 0.0]
    )
    assert result["primary_full_fill_candidate"].tolist() == [
        True, False, False
    ]
    assert result["partial_filled_candidate"].tolist() == [
        False, True, False
    ]
