import math

import pandas as pd
import pytest

from analytics.rolling import calculate_rolling_sharpe


def test_rolling_sharpe_requires_expected_columns():
    history = pd.DataFrame(
        {
            "date": ["2025-01-01"],
        }
    )

    with pytest.raises(
        ValueError,
        match="strategy_value",
    ):
        calculate_rolling_sharpe(history)


def test_rolling_sharpe_rejects_invalid_window():
    history = pd.DataFrame(
        {
            "date": pd.date_range(
                "2025-01-01",
                periods=5,
                freq="D",
            ),
            "strategy_value": [
                100.0,
                101.0,
                102.0,
                103.0,
                104.0,
            ],
        }
    )

    with pytest.raises(
        ValueError,
        match="at least 2",
    ):
        calculate_rolling_sharpe(
            history,
            window=1,
        )


def test_rolling_sharpe_requires_enough_history():
    history = pd.DataFrame(
        {
            "date": pd.date_range(
                "2025-01-01",
                periods=5,
                freq="D",
            ),
            "strategy_value": [
                100.0,
                101.0,
                102.0,
                103.0,
                104.0,
            ],
        }
    )

    with pytest.raises(
        ValueError,
        match="Not enough history",
    ):
        calculate_rolling_sharpe(
            history,
            window=5,
        )


def test_rolling_sharpe_returns_finite_values():
    history = pd.DataFrame(
        {
            "date": pd.date_range(
                "2025-01-01",
                periods=10,
                freq="D",
            ),
            "strategy_value": [
                100.0,
                102.0,
                101.0,
                104.0,
                103.0,
                106.0,
                105.0,
                108.0,
                107.0,
                110.0,
            ],
        }
    )

    result = calculate_rolling_sharpe(
        history,
        window=3,
    )

    assert not result.empty
    assert result["date"].is_monotonic_increasing
    assert all(
        math.isfinite(value)
        for value in result["rolling_sharpe"]
    )