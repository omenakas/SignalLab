from dataclasses import dataclass
from strategies.ma_crossover import generate_ma_positions

import pandas as pd


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


def calculate_max_drawdown(values: pd.Series) -> float:
    running_max = values.cummax()
    drawdown = values / running_max - 1

    return float(drawdown.min() * 100)


def run_ma_crossover_backtest(
    df: pd.DataFrame,
    fast_window: int,
    slow_window: int,
    initial_capital: float = 500.0,
    fee_rate: float = 0.001,
) -> OptimizationBacktestResult:
    if fast_window >= slow_window:
        raise ValueError(
            "Fast moving average must be shorter than slow moving average."
        )

    if initial_capital <= 0:
        raise ValueError("Initial capital must be positive.")

    if not 0 <= fee_rate < 1:
        raise ValueError("Fee rate must be between 0 and 1.")

    data = generate_ma_positions(
    df=df,
    fast_window=fast_window,
    slow_window=slow_window,
    )

    cash = float(initial_capital)
    bitcoin = 0.0
    entry_cost = None

    portfolio_values = []
    completed_trade_profits = []

    for row in data.itertuples():
        price = float(row.price)
        position = row.position

        if position == 1 and bitcoin == 0:
            fee = cash * fee_rate
            invested_amount = cash - fee

            bitcoin = invested_amount / price
            entry_cost = cash
            cash = 0.0

        elif position == 0 and bitcoin > 0:
            gross_sale_value = bitcoin * price
            fee = gross_sale_value * fee_rate

            cash = gross_sale_value - fee

            if entry_cost is not None:
                completed_trade_profits.append(
                    cash - entry_cost
                )

            bitcoin = 0.0
            entry_cost = None

        portfolio_value = cash + bitcoin * price
        portfolio_values.append(portfolio_value)

    data["strategy_value"] = portfolio_values

    last_price = float(data.iloc[-1]["price"])

    # Value the open position as though it were sold at the end.
    if bitcoin > 0:
        final_value = (
            bitcoin
            * last_price
            * (1 - fee_rate)
        )
    else:
        final_value = cash

    first_price = float(data.iloc[0]["price"])

    buy_fee = initial_capital * fee_rate
    buy_hold_bitcoin = (
        initial_capital - buy_fee
    ) / first_price

    buy_hold_final_value = (
        buy_hold_bitcoin
        * last_price
        * (1 - fee_rate)
    )

    strategy_return = (
        final_value / initial_capital - 1
    ) * 100

    buy_hold_return = (
        buy_hold_final_value / initial_capital - 1
    ) * 100

    max_drawdown = calculate_max_drawdown(
        data["strategy_value"]
    )

    completed_trades = len(completed_trade_profits)

    winning_trades = sum(
        profit > 0
        for profit in completed_trade_profits
    )

    if completed_trades > 0:
        win_rate = (
            winning_trades
            / completed_trades
            * 100
        )
    else:
        win_rate = 0.0

    return OptimizationBacktestResult(
        fast_ma=fast_window,
        slow_ma=slow_window,
        final_value=final_value,
        strategy_return=strategy_return,
        buy_hold_return=buy_hold_return,
        excess_return=(
            strategy_return - buy_hold_return
        ),
        max_drawdown=max_drawdown,
        completed_trades=completed_trades,
        win_rate=win_rate,
    )


def optimize_ma_strategy(
    df: pd.DataFrame,
    fast_values: list[int],
    slow_values: list[int],
    initial_capital: float = 500.0,
    fee_rate: float = 0.001,
    min_trades: int = 0,
) -> pd.DataFrame:
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

