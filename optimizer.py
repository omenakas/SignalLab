from itertools import product
from typing import Any

import pandas as pd

from engine.simulator import run_position_backtest
from strategies.registry import StrategyDefinition, get_strategy

from analytics.performance import (
    calculate_performance_metrics,
)

from analytics.trade_metrics import (
    calculate_trade_metrics,
)

OPTIMIZATION_OBJECTIVES = {
    "strategy_return": False,
    "sharpe_ratio": False,
    "sortino_ratio": False,
    "cagr": False,
    "calmar_ratio": False,
    "profit_factor": False,
    "expectancy": False,
    "max_drawdown": True,
    "volatility": True,
}

def _sort_optimizer_results(
    results: pd.DataFrame,
    optimization_target: str = "strategy_return",
) -> pd.DataFrame:
    """
    Rank optimizer results using the selected objective.

    Higher values are preferred for return and quality
    metrics. Lower values are preferred for risk metrics
    such as drawdown and volatility.

    Maximum drawdown is used as a secondary tie-breaker
    unless it is already the primary objective.
    """

    if optimization_target not in OPTIMIZATION_OBJECTIVES:
        raise ValueError(
            "Unsupported optimization target: "
            f"{optimization_target}"
        )

    if optimization_target not in results.columns:
        raise ValueError(
            "Optimization results are missing the target column: "
            f"{optimization_target}"
        )

    primary_ascending = (
        OPTIMIZATION_OBJECTIVES[
            optimization_target
        ]
    )

    if optimization_target == "max_drawdown":
        sort_columns = [
            "max_drawdown",
            "strategy_return",
        ]

        ascending = [
            True,
            False,
        ]

    else:
        sort_columns = [
            optimization_target,
            "max_drawdown",
        ]

        ascending = [
            primary_ascending,
            True,
        ]

    return (
        results
        .sort_values(
            by=sort_columns,
            ascending=ascending,
        )
        .reset_index(drop=True)
    )

def _build_optimizer_result(
    parameters: dict[str, Any],
    simulation: Any,
    performance_metrics: Any,
    trade_metrics: Any,
) -> dict[str, Any]:
    """
    Build one optimizer result row from parameters,
    backtest results, and calculated analytics.
    """

    return {
        **parameters,
        "final_value": simulation.final_value,
        "strategy_return": simulation.strategy_return,
        "buy_hold_return": simulation.buy_hold_return,
        "excess_return": simulation.excess_return,
        "max_drawdown": simulation.max_drawdown,
        "completed_trades": simulation.completed_trades,
        "win_rate": simulation.win_rate,
        "sharpe_ratio": performance_metrics.sharpe_ratio,
        "sortino_ratio": performance_metrics.sortino_ratio,
        "cagr": performance_metrics.cagr,
        "calmar_ratio": performance_metrics.calmar_ratio,
        "volatility": performance_metrics.volatility,
        "profit_factor": trade_metrics.profit_factor,
        "expectancy": trade_metrics.expectancy,
    }

def optimize_strategy(
    df: pd.DataFrame,
    strategy: StrategyDefinition,
    parameter_grid: dict[str, list[Any]],
    initial_capital: float = 500.0,
    fee_rate: float = 0.001,
    min_trades: int = 0,
    optimization_target: str = "strategy_return",
) -> pd.DataFrame:
    """
    Evaluate every combination in a strategy parameter grid.

    The strategy generator produces positions, and the generic
    simulator evaluates those positions.

    Example parameter grid:

        {
            "fast_window": [5, 10, 15],
            "slow_window": [50, 100, 150],
        }
    """

    if df is None or df.empty:
        raise ValueError("The historical dataframe is empty.")

    if not parameter_grid:
        raise ValueError("The parameter grid is empty.")

    if initial_capital <= 0:
        raise ValueError("Initial capital must be positive.")

    if not 0 <= fee_rate < 1:
        raise ValueError("Fee rate must be between 0 and 1.")

    valid_parameter_names = {
        parameter.name
        for parameter in strategy.parameters
    }

    supplied_parameter_names = set(parameter_grid)

    unknown_parameters = (
        supplied_parameter_names - valid_parameter_names
    )

    if unknown_parameters:
        raise ValueError(
            "Unknown parameters for "
            f"{strategy.name}: "
            f"{sorted(unknown_parameters)}"
        )

    empty_parameters = [
        name
        for name, values in parameter_grid.items()
        if not values
    ]

    if empty_parameters:
        raise ValueError(
            "Parameter grid values cannot be empty: "
            f"{empty_parameters}"
        )

    parameter_names = list(parameter_grid)
    parameter_value_lists = [
        parameter_grid[name]
        for name in parameter_names
    ]

    if optimization_target not in OPTIMIZATION_OBJECTIVES:
        raise ValueError(
            "Unsupported optimization target: "
            f"{optimization_target}"
        )

    results: list[dict[str, Any]] = []

    for combination in product(*parameter_value_lists):
        parameters = dict(
            zip(
                parameter_names,
                combination,
                strict=True,
            )
        )

        try:
            positions = strategy.generator(
                df=df,
                **parameters,
            )

            simulation = run_position_backtest(
                df=positions,
                initial_capital=initial_capital,
                fee_rate=fee_rate,
            )


        except ValueError:
            # Invalid combinations such as fast MA >= slow MA
            # are skipped rather than stopping the entire search.
            continue

        if simulation.completed_trades < min_trades:
            continue

        performance_metrics = calculate_performance_metrics(
            history=simulation.history,
        )

        trade_metrics = calculate_trade_metrics(
            trades=simulation.trades,
        )

        results.append(
            _build_optimizer_result(
                parameters=parameters,
                simulation=simulation,
                performance_metrics=performance_metrics,
                trade_metrics=trade_metrics,
            )
        )

    if not results:
        return pd.DataFrame()

    result_df = pd.DataFrame(results)

    return _sort_optimizer_results(
        results=result_df,
        optimization_target=optimization_target,
    )


def optimize_ma_strategy(
    df: pd.DataFrame,
    fast_values: list[int],
    slow_values: list[int],
    initial_capital: float = 500.0,
    fee_rate: float = 0.001,
    min_trades: int = 0,
) -> pd.DataFrame:
    """
    Compatibility wrapper for the existing MA Strategy Lab.

    Internally, this now uses the generic optimizer.
    """

    strategy = get_strategy("Moving Average")

    results = optimize_strategy(
        df=df,
        strategy=strategy,
        parameter_grid={
            "fast_window": fast_values,
            "slow_window": slow_values,
        },
        initial_capital=initial_capital,
        fee_rate=fee_rate,
        min_trades=min_trades,
    )

    if results.empty:
        return results

    # Preserve the column names currently expected by app.py
    # and walk_forward.py.
    return results.rename(
        columns={
            "fast_window": "fast_ma",
            "slow_window": "slow_ma",
        }
    )