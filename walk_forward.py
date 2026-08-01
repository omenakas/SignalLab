from dataclasses import dataclass

import numpy as np
import pandas as pd

from optimizer import optimize_ma_strategy


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


def _backtest_unseen_period(
    full_df: pd.DataFrame,
    split_index: int,
    fast_ma: int,
    slow_ma: int,
    initial_capital: float,
    fee_rate: float,
) -> dict:
    """
    Calculate indicators using earlier data for context, but begin the
    simulated portfolio only at the start of the unseen test period.

    Signals calculated on one row are executed on the following row.
    """

    data = full_df.copy()
    price_column = _find_price_column(data)

    data[price_column] = pd.to_numeric(
        data[price_column],
        errors="coerce",
    )

    data = data.dropna(subset=[price_column]).copy()

    if split_index >= len(data):
        raise ValueError("The split index is outside the dataframe.")

    data["fast_ma"] = data[price_column].rolling(fast_ma).mean()
    data["slow_ma"] = data[price_column].rolling(slow_ma).mean()

    # 1 means invested, 0 means cash.
    data["raw_position"] = np.where(
        data["fast_ma"] > data["slow_ma"],
        1,
        0,
    )

    # Avoid look-ahead bias:
    # today's signal becomes tomorrow's position.
    data["position"] = data["raw_position"].shift(1).fillna(0).astype(int)

    test_data = data.iloc[split_index:].copy()

    if test_data.empty:
        raise ValueError("The testing period contains no rows.")

    cash = float(initial_capital)
    units = 0.0
    previous_position = 0

    equity_values: list[float] = []
    trade_log: list[dict] = []

    open_trade_cost: float | None = None
    completed_trade_returns: list[float] = []

    for index, row in test_data.iterrows():
        price = float(row[price_column])
        desired_position = int(row["position"])

        # Enter the market.
        if desired_position == 1 and previous_position == 0:
            purchase_fee = cash * fee_rate
            investable_cash = cash - purchase_fee

            units = investable_cash / price
            cash = 0.0
            open_trade_cost = initial_capital if not equity_values else equity_values[-1]

            trade_log.append(
                {
                    "date": index,
                    "action": "BUY",
                    "price": price,
                    "fee": purchase_fee,
                }
            )

        # Exit the market.
        elif desired_position == 0 and previous_position == 1:
            gross_sale_value = units * price
            sale_fee = gross_sale_value * fee_rate
            cash = gross_sale_value - sale_fee
            units = 0.0

            if open_trade_cost is not None and open_trade_cost > 0:
                trade_return = (
                    (cash / open_trade_cost) - 1
                ) * 100

                completed_trade_returns.append(trade_return)

            trade_log.append(
                {
                    "date": index,
                    "action": "SELL",
                    "price": price,
                    "fee": sale_fee,
                }
            )

            open_trade_cost = None

        portfolio_value = cash + units * price
        equity_values.append(portfolio_value)
        previous_position = desired_position

    # Value any remaining position at the final market price.
    final_price = float(test_data[price_column].iloc[-1])
    final_value = cash + units * final_price

    first_test_price = float(test_data[price_column].iloc[0])

    buy_hold_purchase_fee = initial_capital * fee_rate
    buy_hold_units = (
        initial_capital - buy_hold_purchase_fee
    ) / first_test_price

    buy_hold_gross_value = buy_hold_units * final_price
    buy_hold_sale_fee = buy_hold_gross_value * fee_rate
    buy_hold_final_value = buy_hold_gross_value - buy_hold_sale_fee

    strategy_return_pct = (
        (final_value / initial_capital) - 1
    ) * 100

    buy_hold_return_pct = (
        (buy_hold_final_value / initial_capital) - 1
    ) * 100

    equity_curve = test_data[
        [price_column, "fast_ma", "slow_ma", "position"]
    ].copy()

    equity_curve["strategy_value"] = equity_values

    equity_curve["buy_hold_value"] = (
        buy_hold_units * equity_curve[price_column]
    )

    equity_curve["running_peak"] = (
        equity_curve["strategy_value"].cummax()
    )

    equity_curve["drawdown_pct"] = (
        (
            equity_curve["strategy_value"]
            / equity_curve["running_peak"]
        )
        - 1
    ) * 100

    max_drawdown_pct = float(
        equity_curve["drawdown_pct"].min()
    )

    completed_trades = len(completed_trade_returns)

    winning_trades = sum(
        trade_return > 0
        for trade_return in completed_trade_returns
    )

    win_rate_pct = (
        winning_trades / completed_trades * 100
        if completed_trades > 0
        else 0.0
    )

    return {
        "strategy_return_pct": strategy_return_pct,
        "buy_hold_return_pct": buy_hold_return_pct,
        "excess_return_pct": (
            strategy_return_pct - buy_hold_return_pct
        ),
        "final_value": final_value,
        "buy_hold_final_value": buy_hold_final_value,
        "max_drawdown_pct": max_drawdown_pct,
        "trades": completed_trades,
        "win_rate_pct": win_rate_pct,
        "equity_curve": equity_curve,
        "trade_log": pd.DataFrame(trade_log),
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

    test_result = _backtest_unseen_period(
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