import pandas as pd
import pytest

from analytics.monthly_returns import (
    calculate_monthly_returns,
)


def test_requires_columns():

    history = pd.DataFrame(
        {
            "date": ["2025-01-01"],
        }
    )

    with pytest.raises(
        ValueError,
        match="strategy_value",
    ):
        calculate_monthly_returns(
            history
        )


def test_returns_expected_columns():

    history = pd.DataFrame(
        {
            "date": pd.date_range(
                "2025-01-01",
                periods=90,
                freq="D",
            ),
            "strategy_value": range(
                100,
                190,
            ),
        }
    )

    result = (
        calculate_monthly_returns(
            history
        )
    )

    assert list(result.columns) == [
        "year",
        "month",
        "monthly_return",
    ]


def test_monthly_returns_not_empty():

    history = pd.DataFrame(
        {
            "date": pd.date_range(
                "2025-01-01",
                periods=365,
                freq="D",
            ),
            "strategy_value": range(
                100,
                465,
            ),
        }
    )

    result = (
        calculate_monthly_returns(
            history
        )
    )

    assert not result.empty