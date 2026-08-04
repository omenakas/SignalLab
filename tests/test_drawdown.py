import pandas as pd
import pytest

from analytics.drawdown import (
    calculate_drawdown_series,
)


def test_drawdown_requires_expected_columns():
    history = pd.DataFrame(
        {
            "date": ["2025-01-01"],
        }
    )

    with pytest.raises(
        ValueError,
        match="strategy_value",
    ):
        calculate_drawdown_series(history)


def test_drawdown_series_tracks_peak_to_trough_decline():
    history = pd.DataFrame(
        {
            "date": pd.date_range(
                "2025-01-01",
                periods=5,
                freq="D",
            ),
            "strategy_value": [
                100.0,
                120.0,
                90.0,
                108.0,
                130.0,
            ],
        }
    )

    result = calculate_drawdown_series(history)

    assert result["drawdown_pct"].tolist() == pytest.approx(
        [
            0.0,
            0.0,
            -25.0,
            -10.0,
            0.0,
        ]
    )


def test_drawdown_sorts_dates_and_ignores_invalid_rows():
    history = pd.DataFrame(
        {
            "date": [
                "2025-01-03",
                "invalid",
                "2025-01-01",
                "2025-01-02",
            ],
            "strategy_value": [
                90.0,
                999.0,
                100.0,
                120.0,
            ],
        }
    )

    result = calculate_drawdown_series(history)

    assert result["date"].is_monotonic_increasing
    assert result["drawdown_pct"].iloc[-1] == pytest.approx(
        -25.0
    )