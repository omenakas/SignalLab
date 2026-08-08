from dataclasses import dataclass
from typing import Any

import pandas as pd
import statistics

from engine.simulator import run_position_backtest
from optimizer import optimize_strategy
from strategies.registry import StrategyDefinition


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

    window_number: int | None = None

    train_start_date: pd.Timestamp | None = None
    train_end_date: pd.Timestamp | None = None
    test_start_date: pd.Timestamp | None = None
    test_end_date: pd.Timestamp | None = None

@dataclass
class RollingWalkForwardResult:
    """
    Result of a rolling walk-forward analysis.
    """

    windows: list[GenericWalkForwardResult]

    @property
    def number_of_windows(self) -> int:
        return len(self.windows)

    @property
    def average_test_return(self) -> float:
        return float(
            sum(
                window.test_return_pct
                for window in self.windows
            )
            / len(self.windows)
        )

    @property
    def average_excess_return(self) -> float:
        return float(
            sum(
                window.test_excess_return_pct
                for window in self.windows
            )
            / len(self.windows)
        )

    @property
    def average_drawdown(self) -> float:
        return float(
            sum(
                window.test_max_drawdown_pct
                for window in self.windows
            )
            / len(self.windows)
        )

    @property
    def average_win_rate(self) -> float:
        return float(
            sum(
                window.test_win_rate_pct
                for window in self.windows
            )
            / len(self.windows)
        )
    
    @property
    def best_window(
        self,
    ) -> GenericWalkForwardResult:
        return max(
            self.windows,
            key=lambda window: (
                window.test_return_pct
            ),
        )

    @property
    def worst_window(
        self,
    ) -> GenericWalkForwardResult:
        return min(
            self.windows,
            key=lambda window: (
                window.test_return_pct
            ),
        )
    
    @property
    def summary_table(
        self,
    ) -> pd.DataFrame:
        rows = []

        for window in self.windows:
            rows.append(
                {
                    "Window": window.window_number,
                    "Train start": (
                        window.train_start_date
                    ),
                    "Train end": (
                        window.train_end_date
                    ),
                    "Test start": (
                        window.test_start_date
                    ),
                    "Test end": (
                        window.test_end_date
                    ),
                    "Best parameters": (
                        window.best_parameters
                    ),
                    "Training return (%)": (
                        window.train_return_pct
                    ),
                    "Testing return (%)": (
                        window.test_return_pct
                    ),
                    "Excess return (pp)": (
                        window.test_excess_return_pct
                    ),
                    "Max drawdown (%)": (
                        window.test_max_drawdown_pct
                    ),
                    "Trades": window.test_trades,
                    "Win rate (%)": (
                        window.test_win_rate_pct
                    ),
                }
            )

        return pd.DataFrame(rows)

    @property
    def parameter_history(
        self,
    ) -> dict[str, list[int | float]]:
        """
        Return the selected parameter values across
        all valid rolling walk-forward windows.
        """

        history: dict[
            str,
            list[int | float],
        ] = {}

        for window in self.windows:
            for (
                parameter_name,
                value,
            ) in window.best_parameters.items():

                history.setdefault(
                    parameter_name,
                    []
                ).append(value)

        return history
    
    @property
    def parameter_statistics(
        self,
    ) -> dict[str, dict[str, float]]:
        """
        Return descriptive statistics for each
        optimized parameter.
        """

        statistics_summary: dict[
            str,
            dict[str, float],
        ] = {}

        for (
            parameter_name,
            values,
        ) in self.parameter_history.items():

            statistics_summary[
                parameter_name
            ] = {
                "minimum": min(values),
                "maximum": max(values),
                "mean": statistics.mean(values),
                "range": (
                    max(values)
                    - min(values)
                ),
                "standard_deviation": (
                    statistics.pstdev(values)
                ),
            }

        return statistics_summary
    
    @property
    def parameter_summary_table(
        self,
    ) -> pd.DataFrame:
        """
        Return parameter stability statistics as a table.
        """

        rows = []

        for (
            parameter_name,
            statistics_summary,
        ) in self.parameter_statistics.items():

            rows.append(
                {
                    "Parameter": parameter_name,
                    "Mean": statistics_summary[
                        "mean"
                    ],
                    "Minimum": statistics_summary[
                        "minimum"
                    ],
                    "Maximum": statistics_summary[
                        "maximum"
                    ],
                    "Range": statistics_summary[
                        "range"
                    ],
                    "Std Dev": statistics_summary[
                        "standard_deviation"
                    ],
                }
            )

        return pd.DataFrame(rows)

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

def rolling_walk_forward_test(
    df: pd.DataFrame,
    strategy: StrategyDefinition,
    parameter_grid: dict[str, list[Any]],
    training_rows: int,
    testing_rows: int,
    step_rows: int,
    initial_capital: float = 500.0,
    fee_rate: float = 0.001,
    min_trades: int = 1,
    optimization_target: str = "strategy_return",
) -> RollingWalkForwardResult:
    """
    Execute repeated walk-forward analyses using rolling
    training and testing windows.
    """

    if df is None or df.empty:
        raise ValueError(
            "The dataframe is empty."
        )

    if training_rows <= 0:
        raise ValueError(
            "Training rows must be positive."
        )

    if testing_rows <= 0:
        raise ValueError(
            "Testing rows must be positive."
        )

    if step_rows <= 0:
        raise ValueError(
            "Step rows must be positive."
        )

    if len(df) < training_rows + testing_rows:
        raise ValueError(
            "Not enough history for one rolling window."
        )

    windows: list[
        GenericWalkForwardResult
    ] = []

    start = 0

    while True:
        train_start = start
        train_end = train_start + training_rows

        test_start = train_end
        test_end = test_start + testing_rows

        if test_end > len(df):
            break

        window_df = df.iloc[
            train_start:test_end
        ].copy()

        try:
            window_result = generic_walk_forward_test(
                df=window_df,
                strategy=strategy,
                parameter_grid=parameter_grid,
                train_fraction=(
                    training_rows
                    / (
                        training_rows
                        + testing_rows
                    )
                ),
                initial_capital=initial_capital,
                fee_rate=fee_rate,
                min_trades=min_trades,
                optimization_target=optimization_target,
            )

        except ValueError:
            start += step_rows
            continue


        window_result.window_number = (
            len(windows) + 1
        )

        window_result.train_start_date = (
            window_df.iloc[0]["date"]
        )

        window_result.train_end_date = (
            window_df.iloc[
                training_rows - 1
            ]["date"]
        )

        window_result.test_start_date = (
            window_df.iloc[
                training_rows
            ]["date"]
        )

        window_result.test_end_date = (
            window_df.iloc[-1]["date"]
        )

        windows.append(
            window_result
        )
        
        start += step_rows

    if not windows:
        raise ValueError(
            "No valid rolling walk-forward windows "
            "were produced."
        )

    return RollingWalkForwardResult(
        windows=windows,
    )