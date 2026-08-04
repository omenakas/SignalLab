import math

import pandas as pd
import pytest

from analytics.performance import (
    PerformanceMetrics,
    calculate_performance_metrics,
)


def test_metrics_require_date_and_strategy_value():
    history = pd.DataFrame(
        {
            "date": pd.date_range(
                "2025-01-01",
                periods=3,
                freq="D",
            ),
        }
    )

    with pytest.raises(
        ValueError,
        match="strategy_value",
    ):
        calculate_performance_metrics(history)


def test_metrics_return_zero_for_insufficient_history():
    history = pd.DataFrame(
        {
            "date": ["2025-01-01"],
            "strategy_value": [500.0],
        }
    )

    metrics = calculate_performance_metrics(history)

    assert metrics == PerformanceMetrics(
        sharpe_ratio=0.0,
        cagr=0.0,
    )


def test_flat_portfolio_has_zero_sharpe_and_cagr():
    history = pd.DataFrame(
        {
            "date": pd.date_range(
                "2025-01-01",
                periods=366,
                freq="D",
            ),
            "strategy_value": [500.0] * 366,
        }
    )

    metrics = calculate_performance_metrics(history)

    assert metrics.sharpe_ratio == 0.0
    assert metrics.cagr == pytest.approx(
        0.0,
        abs=1e-12,
    )


def test_portfolio_doubling_in_one_year_has_100_percent_cagr():
    history = pd.DataFrame(
        {
            "date": [
                "2025-01-01",
                "2026-01-01",
            ],
            "strategy_value": [
                500.0,
                1000.0,
            ],
        }
    )

    metrics = calculate_performance_metrics(history)

    expected_cagr = (
        2 ** (365.25 / 365)
        - 1
    ) * 100

    assert metrics.cagr == pytest.approx(
        expected_cagr,
        rel=1e-10,
    )


def test_metrics_ignore_invalid_rows_and_sort_dates():
    history = pd.DataFrame(
        {
            "date": [
                "2026-01-01",
                "invalid",
                "2025-01-01",
            ],
            "strategy_value": [
                600.0,
                550.0,
                500.0,
            ],
        }
    )

    metrics = calculate_performance_metrics(history)

    assert math.isfinite(metrics.sharpe_ratio)
    assert metrics.cagr > 0

def test_performance_metrics_as_dict():
    metrics = PerformanceMetrics(
        sharpe_ratio=1.25,
        cagr=18.5,
    )

    assert metrics.as_dict() == {
        "Sharpe ratio": 1.25,
        "CAGR (%)": 18.5,
    }


def test_performance_metrics_as_dict_returns_new_dictionary():
    metrics = PerformanceMetrics(
        sharpe_ratio=1.25,
        cagr=18.5,
    )

    first = metrics.as_dict()
    second = metrics.as_dict()

    assert first == second
    assert first is not second