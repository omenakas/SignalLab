import pandas as pd
import pytest

from walk_forward import (
    GenericWalkForwardResult,
    RollingWalkForwardResult,
)


def make_window(
    test_return: float,
    excess_return: float,
    drawdown: float,
    win_rate: float,
) -> GenericWalkForwardResult:
    return GenericWalkForwardResult(
        strategy_name="Test Strategy",
        best_parameters={},
        optimization_target="strategy_return",
        split_index=10,
        train_rows=10,
        test_rows=5,
        train_return_pct=1.0,
        test_return_pct=test_return,
        test_buy_hold_return_pct=0.0,
        test_excess_return_pct=excess_return,
        test_final_value=500.0,
        test_buy_hold_final_value=500.0,
        test_max_drawdown_pct=drawdown,
        test_trades=2,
        test_win_rate_pct=win_rate,
        optimization_results=pd.DataFrame(),
        test_equity_curve=pd.DataFrame(),
        test_trade_log=pd.DataFrame(),
    )

def test_rolling_result_summarizes_windows():
    result = RollingWalkForwardResult(
        windows=[
            make_window(
                test_return=10.0,
                excess_return=5.0,
                drawdown=8.0,
                win_rate=60.0,
            ),
            make_window(
                test_return=-2.0,
                excess_return=-4.0,
                drawdown=12.0,
                win_rate=40.0,
            ),
            make_window(
                test_return=7.0,
                excess_return=3.0,
                drawdown=10.0,
                win_rate=50.0,
            ),
        ]
    )

    assert result.number_of_windows == 3
    assert result.average_test_return == pytest.approx(
        5.0
    )
    assert result.average_excess_return == pytest.approx(
        4.0 / 3.0
    )
    assert result.average_drawdown == pytest.approx(
        10.0
    )
    assert result.average_win_rate == pytest.approx(
        50.0
    )

def test_rolling_result_identifies_best_and_worst_windows():
    best = make_window(
        test_return=12.0,
        excess_return=5.0,
        drawdown=8.0,
        win_rate=60.0,
    )

    worst = make_window(
        test_return=-6.0,
        excess_return=-8.0,
        drawdown=18.0,
        win_rate=20.0,
    )

    middle = make_window(
        test_return=3.0,
        excess_return=1.0,
        drawdown=10.0,
        win_rate=50.0,
    )

    result = RollingWalkForwardResult(
        windows=[
            middle,
            worst,
            best,
        ]
    )

    assert result.best_window is best
    assert result.worst_window is worst