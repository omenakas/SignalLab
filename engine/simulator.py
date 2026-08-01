from dataclasses import dataclass

import pandas as pd


@dataclass
class SimulationResult:
    history: pd.DataFrame
    trades: pd.DataFrame

    initial_capital: float
    final_value: float
    strategy_return: float

    buy_hold_final_value: float
    buy_hold_return: float
    excess_return: float

    max_drawdown: float
    completed_trades: int
    winning_trades: int
    win_rate: float


def calculate_max_drawdown(values: pd.Series) -> float:
    """
    Return the largest peak-to-trough decline as a percentage.
    """

    running_max = values.cummax()
    drawdowns = values / running_max - 1

    return float(drawdowns.min() * 100)


def run_position_backtest(
    df: pd.DataFrame,
    initial_capital: float = 500.0,
    fee_rate: float = 0.001,
) -> SimulationResult:
    """
    Simulate an all-in/all-out strategy.

    Required columns:
    - date
    - price
    - position

    Position meanings:
    - 0: hold cash
    - 1: hold the asset

    The strategy is assumed to have already handled signal timing.
    For example, if signals must be executed one day later, the
    strategy module should shift the position before calling this
    simulator.
    """

    if initial_capital <= 0:
        raise ValueError("Initial capital must be positive.")

    if not 0 <= fee_rate < 1:
        raise ValueError("Fee rate must be between 0 and 1.")

    required_columns = {"date", "price", "position"}

    if not required_columns.issubset(df.columns):
        missing = required_columns - set(df.columns)

        raise ValueError(
            f"Backtest data is missing columns: {sorted(missing)}"
        )

    data = df.copy()

    data["price"] = pd.to_numeric(
        data["price"],
        errors="coerce",
    )

    data["position"] = pd.to_numeric(
        data["position"],
        errors="coerce",
    )

    data = data.dropna(
        subset=["date", "price", "position"]
    ).copy()

    if data.empty:
        raise ValueError("No valid data remains for the backtest.")

    invalid_positions = ~data["position"].isin([0, 1])

    if invalid_positions.any():
        raise ValueError(
            "Position values must contain only 0 or 1."
        )

    data["position"] = data["position"].astype(int)

    cash = float(initial_capital)
    units = 0.0
    entry_cost: float | None = None

    trade_records: list[dict] = []
    portfolio_values: list[float] = []
    holding_states: list[bool] = []

    for row in data.itertuples():
        date = row.date
        price = float(row.price)
        desired_position = int(row.position)

        # Enter the market.
        if desired_position == 1 and units == 0:
            fee = cash * fee_rate
            invested_amount = cash - fee

            units = invested_amount / price
            entry_cost = cash
            cash = 0.0

            trade_records.append(
                {
                    "date": date,
                    "action": "BUY",
                    "price": price,
                    "fee": fee,
                    "portfolio_value": invested_amount,
                    "profit": None,
                }
            )

        # Exit the market.
        elif desired_position == 0 and units > 0:
            gross_sale_value = units * price
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

            units = 0.0
            entry_cost = None

        portfolio_value = cash + units * price

        portfolio_values.append(portfolio_value)
        holding_states.append(units > 0)

    data["strategy_value"] = portfolio_values
    data["holding_asset"] = holding_states

    first_price = float(data.iloc[0]["price"])
    last_price = float(data.iloc[-1]["price"])

    # Buy-and-hold benchmark:
    # one purchase at the beginning and one theoretical sale at the end.
    buy_hold_purchase_fee = initial_capital * fee_rate

    buy_hold_units = (
        initial_capital - buy_hold_purchase_fee
    ) / first_price

    data["buy_hold_value"] = (
        buy_hold_units * data["price"]
    )

    buy_hold_final_value = (
        buy_hold_units
        * last_price
        * (1 - fee_rate)
    )

    # If the strategy finishes invested, apply a theoretical final
    # selling fee so its final cash value is comparable with buy-and-hold.
    if units > 0:
        final_value = (
            units
            * last_price
            * (1 - fee_rate)
        )
    else:
        final_value = cash

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
        win_rate = (
            winning_trades
            / completed_trades
            * 100
        )
    else:
        win_rate = 0.0

    return SimulationResult(
        history=data,
        trades=trades,
        initial_capital=initial_capital,
        final_value=final_value,
        strategy_return=strategy_return,
        buy_hold_final_value=buy_hold_final_value,
        buy_hold_return=buy_hold_return,
        excess_return=strategy_return - buy_hold_return,
        max_drawdown=max_drawdown,
        completed_trades=completed_trades,
        winning_trades=winning_trades,
        win_rate=win_rate,
    )