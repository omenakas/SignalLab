import pandas as pd
import pytest

from optimizer import (
    _sort_optimizer_results,
    optimize_strategy,
)
from strategies.registry import get_strategy

def test_sort_prefers_highest_return():
    results = pd.DataFrame(
        {
            "strategy_return": [
                5.0,
                12.0,
                8.0,
            ],
            "max_drawdown": [
                10.0,
                20.0,
                5.0,
            ],
        }
    )

    sorted_results = _sort_optimizer_results(
        results
    )

    assert (
        sorted_results.iloc[0]["strategy_return"]
        == pytest.approx(12.0)
    )

def test_sort_prefers_lower_drawdown_when_returns_tie():
    results = pd.DataFrame(
        {
            "strategy_return": [
                10.0,
                10.0,
                8.0,
            ],
            "max_drawdown": [
                30.0,
                10.0,
                5.0,
            ],
        }
    )

    sorted_results = _sort_optimizer_results(
        results
    )

    assert (
        sorted_results.iloc[0]["max_drawdown"]
        == pytest.approx(10.0)
    )

@pytest.fixture
def history() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range(
                "2025-01-01",
                periods=10,
                freq="D",
            ),
            "open": [
                100.0,
                101.0,
                102.0,
                103.0,
                104.0,
                105.0,
                106.0,
                107.0,
                108.0,
                109.0,
            ],
            "high": [
                101.0,
                102.0,
                103.0,
                104.0,
                105.0,
                106.0,
                107.0,
                108.0,
                109.0,
                110.0,
            ],
            "low": [
                99.0,
                100.0,
                101.0,
                102.0,
                103.0,
                104.0,
                105.0,
                106.0,
                107.0,
                108.0,
            ],
            "close": [
                100.0,
                101.0,
                102.0,
                103.0,
                104.0,
                105.0,
                106.0,
                107.0,
                108.0,
                109.0,
            ],
            "volume": [
                1_000.0
            ] * 10,
        }
    )

def test_optimizer_rejects_empty_history():
    strategy = get_strategy(
        "Moving Average"
    )

    with pytest.raises(
        ValueError,
        match="historical dataframe is empty",
    ):
        optimize_strategy(
            df=pd.DataFrame(),
            strategy=strategy,
            parameter_grid={
                "fast_window": [2],
                "slow_window": [5],
            },
        )

def test_optimizer_rejects_empty_parameter_grid(
    history,
):
    strategy = get_strategy(
        "Moving Average"
    )

    with pytest.raises(
        ValueError,
        match="parameter grid is empty",
    ):
        optimize_strategy(
            df=history,
            strategy=strategy,
            parameter_grid={},
        )

def test_optimizer_rejects_non_positive_capital(
    history,
):
    strategy = get_strategy(
        "Moving Average"
    )

    with pytest.raises(
        ValueError,
        match="Initial capital must be positive",
    ):
        optimize_strategy(
            df=history,
            strategy=strategy,
            parameter_grid={
                "fast_window": [2],
                "slow_window": [5],
            },
            initial_capital=0.0,
        )

@pytest.mark.parametrize(
    "fee_rate",
    [
        -0.01,
        1.0,
        1.5,
    ],
)
def test_optimizer_rejects_invalid_fee_rate(
    history,
    fee_rate,
):
    strategy = get_strategy(
        "Moving Average"
    )

    with pytest.raises(
        ValueError,
        match="Fee rate must be between 0 and 1",
    ):
        optimize_strategy(
            df=history,
            strategy=strategy,
            parameter_grid={
                "fast_window": [2],
                "slow_window": [5],
            },
            fee_rate=fee_rate,
        )

def test_optimizer_rejects_unknown_parameters(
    history,
):
    strategy = get_strategy(
        "Moving Average"
    )

    with pytest.raises(
        ValueError,
        match="Unknown parameters",
    ):
        optimize_strategy(
            df=history,
            strategy=strategy,
            parameter_grid={
                "imaginary_window": [10],
            },
        )

def test_optimizer_rejects_empty_parameter_values(
    history,
):
    strategy = get_strategy(
        "Moving Average"
    )

    with pytest.raises(
        ValueError,
        match="cannot be empty",
    ):
        optimize_strategy(
            df=history,
            strategy=strategy,
            parameter_grid={
                "fast_window": [],
                "slow_window": [5],
            },
        )

def test_sort_can_maximize_sharpe_ratio():
    results = pd.DataFrame(
        {
            "strategy_return": [
                20.0,
                10.0,
                15.0,
            ],
            "sharpe_ratio": [
                0.5,
                1.5,
                1.0,
            ],
            "max_drawdown": [
                15.0,
                10.0,
                12.0,
            ],
        }
    )

    sorted_results = _sort_optimizer_results(
        results=results,
        optimization_target="sharpe_ratio",
    )

    assert sorted_results.iloc[0][
        "sharpe_ratio"
    ] == pytest.approx(1.5)

def test_sort_can_minimize_maximum_drawdown():
    results = pd.DataFrame(
        {
            "strategy_return": [
                20.0,
                10.0,
                15.0,
            ],
            "max_drawdown": [
                30.0,
                5.0,
                15.0,
            ],
        }
    )

    sorted_results = _sort_optimizer_results(
        results=results,
        optimization_target="max_drawdown",
    )

    assert sorted_results.iloc[0][
        "max_drawdown"
    ] == pytest.approx(5.0)

def test_sort_uses_lower_drawdown_as_tie_breaker():
    results = pd.DataFrame(
        {
            "strategy_return": [
                12.0,
                10.0,
            ],
            "sharpe_ratio": [
                1.25,
                1.25,
            ],
            "max_drawdown": [
                20.0,
                8.0,
            ],
        }
    )

    sorted_results = _sort_optimizer_results(
        results=results,
        optimization_target="sharpe_ratio",
    )

    assert sorted_results.iloc[0][
        "max_drawdown"
    ] == pytest.approx(8.0)

def test_sort_rejects_unsupported_target():
    results = pd.DataFrame(
        {
            "strategy_return": [
                10.0,
            ],
            "max_drawdown": [
                5.0,
            ],
        }
    )

    with pytest.raises(
        ValueError,
        match="Unsupported optimization target",
    ):
        _sort_optimizer_results(
            results=results,
            optimization_target="magic_score",
        )

def test_optimizer_rejects_unsupported_target(
    history,
):
    strategy = get_strategy(
        "Moving Average"
    )

    with pytest.raises(
        ValueError,
        match="Unsupported optimization target",
    ):
        optimize_strategy(
            df=history,
            strategy=strategy,
            parameter_grid={
                "fast_window": [2],
                "slow_window": [5],
            },
            optimization_target="magic_score",
        )

