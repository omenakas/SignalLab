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
            "profit": pd.Series(
                dtype=float
            ),
        }
    )

    metrics = calculate_trade_metrics(
        trades
    )

    assert metrics == TradeMetrics(
        profit_factor=0.0,
    )


def test_profit_factor_uses_gross_profit_and_loss():
    trades = pd.DataFrame(
        {
            "profit": [
                0.0,
                100.0,
                -40.0,
                50.0,
                -10.0,
            ],
        }
    )

    metrics = calculate_trade_metrics(
        trades
    )

    # Gross profit = 150
    # Gross loss = 50
    assert metrics.profit_factor == pytest.approx(
        3.0
    )


def test_all_losing_trades_have_zero_profit_factor():
    trades = pd.DataFrame(
        {
            "profit": [
                -25.0,
                -75.0,
            ],
        }
    )

    metrics = calculate_trade_metrics(
        trades
    )

    assert metrics.profit_factor == 0.0


def test_profitable_trades_without_losses_have_infinite_profit_factor():
    trades = pd.DataFrame(
        {
            "profit": [
                25.0,
                75.0,
            ],
        }
    )

    metrics = calculate_trade_metrics(
        trades
    )

    assert math.isinf(
        metrics.profit_factor
    )


def test_invalid_profit_values_are_ignored():
    trades = pd.DataFrame(
        {
            "profit": [
                "invalid",
                50.0,
                -25.0,
                None,
            ],
        }
    )

    metrics = calculate_trade_metrics(
        trades
    )

    assert metrics.profit_factor == pytest.approx(
        2.0
    )


def test_trade_metrics_as_dict():
    metrics = TradeMetrics(
        profit_factor=1.75,
    )

    assert metrics.as_dict() == {
        "Profit factor": 1.75,
    }