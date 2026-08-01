from dataclasses import dataclass

import pandas as pd

from strategy import add_signals


@dataclass
class BacktestResult:
    history: pd.DataFrame
    trades: pd.DataFrame
    initial_capital: float
    final_value: float
    strategy_return: float
    buy_hold_final_value: float
    buy_hold_return: float
    max_drawdown: float
    completed_trades: int
    winning_trades: int
    win_rate: float


def calculate_max_drawdown(values: pd.Series) -> float:
    running_max = values.cummax()
    drawdowns = values / running_max - 1

    return float(drawdowns.min() * 100)


def run_backtest(
    df: pd.DataFrame,
    initial_capital: float = 500.0,
    fee_rate: float = 0.001,
) -> BacktestResult:
    if initial_capital <= 0:
        raise ValueError("Initial capital must be positive.")

    if not 0 <= fee_rate < 1:
        raise ValueError("Fee rate must be between 0 and 1.")

    data = add_signals(df)
    data = data.dropna(
        subset=["price", "MA20", "MA50", "RSI", "signal"]
    ).copy()

    if data.empty:
        raise ValueError("Not enough data for the backtest.")

    cash = float(initial_capital)
    bitcoin = 0.0

    entry_cost = None
    trade_records = []
    portfolio_values = []
    position_states = []

    for row in data.itertuples():
        price = float(row.price)
        signal = row.signal
        date = row.date

        if signal == "Bullish" and bitcoin == 0:
            fee = cash * fee_rate
            amount_invested = cash - fee
            bitcoin = amount_invested / price
            entry_cost = cash
            cash = 0.0

            trade_records.append(
                {
                    "date": date,
                    "action": "BUY",
                    "price": price,
                    "fee": fee,
                    "portfolio_value": amount_invested,
                    "profit": None,
                }
            )

        elif signal == "Bearish" and bitcoin > 0:
            gross_sale_value = bitcoin * price
            fee = gross_sale_value * fee_rate
            cash = gross_sale_value - fee

            profit = (
                cash - entry_cost
                if entry_cost is not None
                else None
            )

            trade_records.append(
                {
                    "date": date,
                    "action": "SELL",
                    "price": price,
                    "fee": fee,
                    "portfolio_value": cash,
                    "profit": profit,
                }
            )

            bitcoin = 0.0
            entry_cost = None

        portfolio_value = cash + bitcoin * price

        portfolio_values.append(portfolio_value)
        position_states.append(bitcoin > 0)

    data["strategy_value"] = portfolio_values
    data["holding_bitcoin"] = position_states

    first_price = float(data.iloc[0]["price"])

    buy_hold_fee = initial_capital * fee_rate
    buy_hold_bitcoin = (
        initial_capital - buy_hold_fee
    ) / first_price

    data["buy_hold_value"] = (
        buy_hold_bitcoin * data["price"]
    )

    # Apply a theoretical final selling fee to both strategies,
    # allowing a fair comparison of cash values.
    last_price = float(data.iloc[-1]["price"])

    if bitcoin > 0:
        final_value = bitcoin * last_price * (1 - fee_rate)
    else:
        final_value = cash

    buy_hold_final_value = (
        buy_hold_bitcoin * last_price * (1 - fee_rate)
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

    trades = pd.DataFrame(trade_records)

    if trades.empty:
        completed_profits = pd.Series(dtype=float)
    else:
        completed_profits = trades.loc[
            trades["action"] == "SELL",
            "profit",
        ].dropna()

    completed_trades = len(completed_profits)
    winning_trades = int((completed_profits > 0).sum())

    if completed_trades > 0:
        win_rate = winning_trades / completed_trades * 100
    else:
        win_rate = 0.0

    return BacktestResult(
        history=data,
        trades=trades,
        initial_capital=initial_capital,
        final_value=final_value,
        strategy_return=strategy_return,
        buy_hold_final_value=buy_hold_final_value,
        buy_hold_return=buy_hold_return,
        max_drawdown=max_drawdown,
        completed_trades=completed_trades,
        winning_trades=winning_trades,
        win_rate=win_rate,
    )