import math

import pandas as pd
import pytest

from analytics.trade_metrics import (
    TradeMetrics,
    calculate_trade_metrics,
)


def test_trade_metrics_require_profit_column():
    trades = pd.DataFrame(
        {
            "action": [
                "BUY",
                "SELL",
            ],
        }
    )

    with pytest.raises(
        ValueError,
        match="profit",
    ):
        calculate_trade_metrics(trades)


def test_empty_trade_log_has_zero_profit_factor():
    trades = pd.DataFrame(
        {
            "action": pd.Series(dtype=str),
            "profit": pd.Series(dtype=float),
        }
    )

    metrics = calculate_trade_metrics(trades)

    assert metrics == TradeMetrics(
        profit_factor=0.0,
        expectancy=0.0,
    )


def test_profit_factor_uses_gross_profit_and_loss():
    trades = pd.DataFrame(
        {
            "action": [
                "BUY",
                "SELL",
                "SELL",
                "SELL",
                "SELL",
            ],
            "profit": [
                0.0,
                100.0,
                -40.0,
                50.0,
                -10.0,
            ],
        }
    )

    metrics = calculate_trade_metrics(trades)

    assert metrics.profit_factor == pytest.approx(3.0)
    assert metrics.expectancy == pytest.approx(25.0)


def test_all_losing_trades_have_zero_profit_factor():
    trades = pd.DataFrame(
        {
            "action": [
                "SELL",
                "SELL",
            ],
            "profit": [
                -25.0,
                -75.0,
            ],
        }
    )

    metrics = calculate_trade_metrics(trades)

    assert metrics.profit_factor == 0.0
    assert metrics.expectancy == pytest.approx(-50.0)

def test_profitable_trades_without_losses_have_infinite_profit_factor():
    trades = pd.DataFrame(
        {
            "action": [
                "SELL",
                "SELL",
            ],
            "profit": [
                25.0,
                75.0,
            ],
        }
    )

    metrics = calculate_trade_metrics(trades)

    assert math.isinf(metrics.profit_factor)
    assert metrics.expectancy == pytest.approx(50.0)


def test_invalid_profit_values_are_ignored():
    trades = pd.DataFrame(
        {
            "action": [
                "SELL",
                "SELL",
                "SELL",
                "SELL",
            ],
            "profit": [
                "invalid",
                50.0,
                -25.0,
                None,
            ],
        }
    )

    metrics = calculate_trade_metrics(trades)

    assert metrics.profit_factor == pytest.approx(2.0)
    assert metrics.expectancy == pytest.approx(12.5)


def test_trade_metrics_as_dict():
    metrics = TradeMetrics(
        profit_factor=1.75,
        expectancy=12.5,
    )

    assert metrics.as_dict() == {
        "Profit factor": 1.75,
        "Expectancy (€)": 12.5,
    }

def test_expectancy_is_average_completed_trade_profit():
    trades = pd.DataFrame(
        {
            "action": [
                "BUY",
                "SELL",
                "BUY",
                "SELL",
                "BUY",
            ],
            "profit": [
                None,
                30.0,
                None,
                -10.0,
                None,
            ],
        }
    )

    metrics = calculate_trade_metrics(trades)

    assert metrics.expectancy == pytest.approx(10.0)


def test_expectancy_ignores_open_entries():
    trades = pd.DataFrame(
        {
            "action": [
                "BUY",
                "SELL",
                "BUY",
            ],
            "profit": [
                -999.0,
                20.0,
                -999.0,
            ],
        }
    )

    metrics = calculate_trade_metrics(trades)

    assert metrics.expectancy == pytest.approx(20.0)