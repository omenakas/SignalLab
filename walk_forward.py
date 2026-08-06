from dataclasses import dataclass
from typing import Any

import pandas as pd

from engine.simulator import run_position_backtest
from optimizer import optimize_strategy
from strategies.registry import StrategyDefinition, get_strategy


@dataclass
class GenericWalkForwardResult:
    strategy_name: str
    best_parameters: dict[str, Any]
    optimization_target: str

    split_index: int
    train_rows: int
    test_rows: int

    train_return_pct: float
    test_return_pct: float
    test_buy_hold_return_pct: float
    test_excess_return_pct: float

    test_final_value: float
    test_buy_hold_final_value: float
    test_max_drawdown_pct: float
    test_trades: int
    test_win_rate_pct: float

    optimization_results: pd.DataFrame
    test_equity_curve: pd.DataFrame
    test_trade_log: pd.DataFrame


@dataclass
class WalkForwardResult:
    """
    Compatibility result used by the existing MA Walk-Forward UI.
    """

    fast_ma: int
    slow_ma: int

    split_index: int
    train_rows: int
    test_rows: int

    train_return_pct: float
    test_return_pct: float
    test_buy_hold_return_pct: float
    test_excess_return_pct: float

    test_final_value: float
    test_buy_hold_final_value: float
    test_max_drawdown_pct: float
    test_trades: int
    test_win_rate_pct: float

    optimization_results: pd.DataFrame
    test_equity_curve: pd.DataFrame
    test_trade_log: pd.DataFrame


def _validate_parameter_grid(
    strategy: StrategyDefinition,
    parameter_grid: dict[str, list[Any]],
) -> None:
    """
    Ensure the supplied grid contains only parameters registered
    for the selected strategy.
    """

    if not parameter_grid:
        raise ValueError("The parameter grid is empty.")

    valid_names = {
        parameter.name
        for parameter in strategy.parameters
    }

    supplied_names = set(parameter_grid)

    unknown_names = supplied_names - valid_names

    if unknown_names:
        raise ValueError(
            f"Unknown parameters for {strategy.name}: "
            f"{sorted(unknown_names)}"
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


def _test_unseen_period(
    full_df: pd.DataFrame,
    split_index: int,
    strategy: StrategyDefinition,
    parameters: dict[str, Any],
    initial_capital: float,
    fee_rate: float,
):
    """
    Generate positions using the complete historical dataframe so
    indicators have access to pre-test context, but simulate only rows
    belonging to the unseen test period.
    """

    positions = strategy.generator(
        df=full_df,
        **parameters,
    )

    if positions is None or positions.empty:
        raise ValueError(
            f"{strategy.name} generated no valid positions."
        )

    if "date" not in full_df.columns:
        raise ValueError(
            "Historical data must contain a 'date' column."
        )

    if "date" not in positions.columns:
        raise ValueError(
            "Strategy output must contain a 'date' column."
        )

    test_start_date = full_df.iloc[split_index]["date"]

    test_positions = positions.loc[
        positions["date"] >= test_start_date
    ].copy()

    if test_positions.empty:
        raise ValueError(
            "The testing period contains no valid strategy rows."
        )

    return run_position_backtest(
        df=test_positions,
        initial_capital=initial_capital,
        fee_rate=fee_rate,
    )


def generic_walk_forward_test(
    df: pd.DataFrame,
    strategy: StrategyDefinition,
    parameter_grid: dict[str, list[Any]],
    train_fraction: float = 0.70,
    initial_capital: float = 500.0,
    fee_rate: float = 0.001,
    min_trades: int = 1,
    optimization_target: str = "strategy_return",
) -> GenericWalkForwardResult:
    """
    Optimize a registered strategy on the earlier training period,
    freeze its best parameters, and evaluate them once on the later
    unseen period.
    """

    if df is None or df.empty:
        raise ValueError("The dataframe is empty.")

    if not 0.5 <= train_fraction <= 0.9:
        raise ValueError(
            "train_fraction must be between 0.5 and 0.9."
        )

    if initial_capital <= 0:
        raise ValueError(
            "Initial capital must be positive."
        )

    if not 0 <= fee_rate < 1:
        raise ValueError(
            "Fee rate must be between 0 and 1."
        )

    if min_trades < 0:
        raise ValueError(
            "Minimum trades cannot be negative."
        )

    _validate_parameter_grid(
        strategy=strategy,
        parameter_grid=parameter_grid,
    )

    data = df.copy()

    if "date" not in data.columns:
        raise ValueError(
            "Historical data must contain a 'date' column."
        )

    data["date"] = pd.to_datetime(
        data["date"],
        errors="coerce",
    )

    data = (
        data
        .dropna(subset=["date"])
        .sort_values("date")
        .reset_index(drop=True)
    )

    split_index = int(len(data) * train_fraction)

    train_df = data.iloc[:split_index].copy()
    test_df = data.iloc[split_index:].copy()

    if len(train_df) < 30:
        raise ValueError(
            "The training period is too short."
        )

    if len(test_df) < 20:
        raise ValueError(
            "The testing period is too short."
        )

    optimization_results = optimize_strategy(
        df=train_df,
        strategy=strategy,
        parameter_grid=parameter_grid,
        initial_capital=initial_capital,
        fee_rate=fee_rate,
        min_trades=min_trades,
        optimization_target=optimization_target,
    )

    if optimization_results.empty:
        raise ValueError(
            "No valid strategies were found in the training period. "
            "Try lowering the minimum trade count or changing the "
            "parameter ranges."
        )

    best_row = optimization_results.iloc[0]

    selected_parameter_definitions = [
        parameter
        for parameter in strategy.parameters
        if parameter.name in parameter_grid
    ]

    best_parameters: dict[str, Any] = {}

    for parameter in selected_parameter_definitions:
        value = best_row[parameter.name]

        # Convert NumPy scalar values into ordinary Python values.
        if hasattr(value, "item"):
            value = value.item()

        # Pandas may convert integer parameters to floats when reading
        # a mixed-type result row, so restore the registered type.
        if parameter.parameter_type == "int":
            value = int(value)

        elif parameter.parameter_type == "float":
            value = float(value)

        best_parameters[parameter.name] = value

    train_return_pct = float(
        best_row["strategy_return"]
    )

    test_simulation = _test_unseen_period(
        full_df=data,
        split_index=split_index,
        strategy=strategy,
        parameters=best_parameters,
        initial_capital=initial_capital,
        fee_rate=fee_rate,
    )

    return GenericWalkForwardResult(
        strategy_name=strategy.name,
        best_parameters=best_parameters,
        optimization_target=optimization_target,
        split_index=split_index,
        train_rows=len(train_df),
        test_rows=len(test_df),
        train_return_pct=train_return_pct,
        test_return_pct=test_simulation.strategy_return,
        test_buy_hold_return_pct=(
            test_simulation.buy_hold_return
        ),
        test_excess_return_pct=(
            test_simulation.excess_return
        ),
        test_final_value=test_simulation.final_value,
        test_buy_hold_final_value=(
            test_simulation.buy_hold_final_value
        ),
        test_max_drawdown_pct=(
            test_simulation.max_drawdown
        ),
        test_trades=test_simulation.completed_trades,
        test_win_rate_pct=test_simulation.win_rate,
        optimization_results=optimization_results,
        test_equity_curve=test_simulation.history,
        test_trade_log=test_simulation.trades,
    )


def walk_forward_test(
    df: pd.DataFrame,
    fast_values,
    slow_values,
    train_fraction: float = 0.70,
    initial_capital: float = 500.0,
    fee_rate: float = 0.001,
    min_trades: int = 1,
    optimization_target: str = "strategy_return",
) -> WalkForwardResult:
    """
    Compatibility wrapper for the current MA-specific Streamlit tab.

    Internally, this uses the generic walk-forward implementation.
    """

    strategy = get_strategy("Moving Average")

    generic_result = generic_walk_forward_test(
        df=df,
        strategy=strategy,
        parameter_grid={
            "fast_window": list(fast_values),
            "slow_window": list(slow_values),
        },
        train_fraction=train_fraction,
        initial_capital=initial_capital,
        fee_rate=fee_rate,
        min_trades=min_trades,
        optimization_target=optimization_target,
    )

    fast_ma = int(
        generic_result.best_parameters["fast_window"]
    )

    slow_ma = int(
        generic_result.best_parameters["slow_window"]
    )

    optimization_results = (
        generic_result.optimization_results.rename(
            columns={
                "fast_window": "fast_ma",
                "slow_window": "slow_ma",
            }
        )
    )

    return WalkForwardResult(
        fast_ma=fast_ma,
        slow_ma=slow_ma,
        split_index=generic_result.split_index,
        train_rows=generic_result.train_rows,
        test_rows=generic_result.test_rows,
        train_return_pct=(
            generic_result.train_return_pct
        ),
        test_return_pct=(
            generic_result.test_return_pct
        ),
        test_buy_hold_return_pct=(
            generic_result.test_buy_hold_return_pct
        ),
        test_excess_return_pct=(
            generic_result.test_excess_return_pct
        ),
        test_final_value=(
            generic_result.test_final_value
        ),
        test_buy_hold_final_value=(
            generic_result.test_buy_hold_final_value
        ),
        test_max_drawdown_pct=(
            generic_result.test_max_drawdown_pct
        ),
        test_trades=generic_result.test_trades,
        test_win_rate_pct=(
            generic_result.test_win_rate_pct
        ),
        optimization_results=optimization_results,
        test_equity_curve=(
            generic_result.test_equity_curve
        ),
        test_trade_log=generic_result.test_trade_log,
    )