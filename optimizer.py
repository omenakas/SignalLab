from dataclasses import dataclass

import pandas as pd

from engine.simulator import run_position_backtest
from strategies.ma_crossover import generate_ma_positions


@dataclass
class OptimizationBacktestResult:
    fast_ma: int
    slow_ma: int
    final_value: float
    strategy_return: float
    buy_hold_return: float
    excess_return: float
    max_drawdown: float
    completed_trades: int
    win_rate: float


def run_ma_crossover_backtest(
    df: pd.DataFrame,
    fast_window: int,
    slow_window: int,
    initial_capital: float = 500.0,
    fee_rate: float = 0.001,
) -> OptimizationBacktestResult:
    """
    Generate MA crossover positions and evaluate them using the
    generic trading simulator.
    """

    positions = generate_ma_positions(
        df=df,
        fast_window=fast_window,
        slow_window=slow_window,
    )

    simulation = run_position_backtest(
        df=positions,
        initial_capital=initial_capital,
        fee_rate=fee_rate,
    )

    return OptimizationBacktestResult(
        fast_ma=fast_window,
        slow_ma=slow_window,
        final_value=simulation.final_value,
        strategy_return=simulation.strategy_return,
        buy_hold_return=simulation.buy_hold_return,
        excess_return=simulation.excess_return,
        max_drawdown=simulation.max_drawdown,
        completed_trades=simulation.completed_trades,
        win_rate=simulation.win_rate,
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
    Test several moving-average combinations and rank them by return.
    """

    results = []

    for fast_window in fast_values:
        for slow_window in slow_values:
            if fast_window >= slow_window:
                continue

            try:
                result = run_ma_crossover_backtest(
                    df=df,
                    fast_window=fast_window,
                    slow_window=slow_window,
                    initial_capital=initial_capital,
                    fee_rate=fee_rate,
                )

            except ValueError:
                continue

            if result.completed_trades < min_trades:
                continue

            results.append(
                {
                    "fast_ma": result.fast_ma,
                    "slow_ma": result.slow_ma,
                    "final_value": result.final_value,
                    "strategy_return": result.strategy_return,
                    "buy_hold_return": result.buy_hold_return,
                    "excess_return": result.excess_return,
                    "max_drawdown": result.max_drawdown,
                    "completed_trades": result.completed_trades,
                    "win_rate": result.win_rate,
                }
            )

    if not results:
        return pd.DataFrame()

    result_df = pd.DataFrame(results)

    return result_df.sort_values(
        by=[
            "strategy_return",
            "max_drawdown",
        ],
        ascending=[
            False,
            False,
        ],
    ).reset_index(drop=True)