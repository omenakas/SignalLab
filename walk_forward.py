from dataclasses import dataclass

import pandas as pd

from engine.simulator import run_position_backtest
from optimizer import optimize_ma_strategy
from strategies.ma_crossover import generate_ma_positions


@dataclass
class WalkForwardResult:
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


def _find_price_column(df: pd.DataFrame) -> str:
    """
    Find a likely price column without assuming one exact spelling.
    """

    candidates = [
        "price",
        "Price",
        "close",
        "Close",
        "usd",
        "USD",
    ]

    for column in candidates:
        if column in df.columns:
            return column

    numeric_columns = df.select_dtypes(include="number").columns.tolist()

    if len(numeric_columns) == 1:
        return numeric_columns[0]

    raise ValueError(
        "Could not determine the price column. "
        f"Available columns: {list(df.columns)}"
    )


def _get_value(row: pd.Series, possible_names: list[str]):
    """
    Read a value from an optimizer result even if its column naming
    differs slightly.
    """

    for name in possible_names:
        if name in row.index:
            return row[name]

    raise KeyError(
        f"None of these columns were found: {possible_names}. "
        f"Available columns: {list(row.index)}"
    )


def _test_unseen_period(
    full_df: pd.DataFrame,
    split_index: int,
    fast_ma: int,
    slow_ma: int,
    initial_capital: float,
    fee_rate: float,
) -> dict:
    """
    Generate MA positions using the full history for indicator context,
    then evaluate only the unseen test period.
    """

    positions = generate_ma_positions(
        df=full_df,
        fast_window=fast_ma,
        slow_window=slow_ma,
    )

    if positions.empty:
        raise ValueError("No valid MA positions were generated.")

    original_test_start_date = full_df.iloc[split_index]["date"]

    test_positions = positions.loc[
        positions["date"] >= original_test_start_date
    ].copy()

    if test_positions.empty:
        raise ValueError(
            "The testing period contains no valid strategy rows."
        )

    simulation = run_position_backtest(
        df=test_positions,
        initial_capital=initial_capital,
        fee_rate=fee_rate,
    )

    return {
        "strategy_return_pct": simulation.strategy_return,
        "buy_hold_return_pct": simulation.buy_hold_return,
        "excess_return_pct": simulation.excess_return,
        "final_value": simulation.final_value,
        "buy_hold_final_value": simulation.buy_hold_final_value,
        "max_drawdown_pct": simulation.max_drawdown,
        "trades": simulation.completed_trades,
        "win_rate_pct": simulation.win_rate,
        "equity_curve": simulation.history,
        "trade_log": simulation.trades,
    }

def walk_forward_test(
    df: pd.DataFrame,
    fast_values,
    slow_values,
    train_fraction: float = 0.70,
    initial_capital: float = 500.0,
    fee_rate: float = 0.001,
    min_trades: int = 1,
) -> WalkForwardResult:
    """
    Optimize on the earlier training period and evaluate the winning
    parameters once on the later unseen period.
    """

    if df is None or df.empty:
        raise ValueError("The dataframe is empty.")

    if not 0.5 <= train_fraction <= 0.9:
        raise ValueError(
            "train_fraction must be between 0.5 and 0.9."
        )

    data = df.copy().sort_index()

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

    optimization_results = optimize_ma_strategy(
        train_df,
        fast_values=fast_values,
        slow_values=slow_values,
        initial_capital=initial_capital,
        fee_rate=fee_rate,
        min_trades=min_trades,
    )

    if optimization_results is None or optimization_results.empty:
        raise ValueError(
            "No valid strategies were found in the training period. "
            "Try lowering the minimum trade count or changing the "
            "moving-average ranges."
        )

    best_row = optimization_results.iloc[0]

    fast_ma = int(
        _get_value(
            best_row,
            ["fast_ma", "Fast MA", "fast", "Fast"],
        )
    )

    slow_ma = int(
        _get_value(
            best_row,
            ["slow_ma", "Slow MA", "slow", "Slow"],
        )
    )

    train_return_pct = float(
        _get_value(
            best_row,
            [
                "strategy_return",
                "strategy_return_pct",
                "return_pct",
                "Return (%)",
                "Strategy Return (%)",
            ],
        )
    )

    test_result = _test_unseen_period(
        full_df=data,
        split_index=split_index,
        fast_ma=fast_ma,
        slow_ma=slow_ma,
        initial_capital=initial_capital,
        fee_rate=fee_rate,
    )

    return WalkForwardResult(
        fast_ma=fast_ma,
        slow_ma=slow_ma,
        split_index=split_index,
        train_rows=len(train_df),
        test_rows=len(test_df),
        train_return_pct=train_return_pct,
        test_return_pct=test_result["strategy_return_pct"],
        test_buy_hold_return_pct=test_result[
            "buy_hold_return_pct"
        ],
        test_excess_return_pct=test_result[
            "excess_return_pct"
        ],
        test_final_value=test_result["final_value"],
        test_buy_hold_final_value=test_result[
            "buy_hold_final_value"
        ],
        test_max_drawdown_pct=test_result[
            "max_drawdown_pct"
        ],
        test_trades=test_result["trades"],
        test_win_rate_pct=test_result["win_rate_pct"],
        optimization_results=optimization_results,
        test_equity_curve=test_result["equity_curve"],
        test_trade_log=test_result["trade_log"],
    )