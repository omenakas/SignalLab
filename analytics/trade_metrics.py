from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TradeMetrics:
    """
    Analytics calculated from completed trade outcomes.
    """

    profit_factor: float
    expectancy: float

    def as_dict(self) -> dict[str, float]:
        return {
            "Profit factor": self.profit_factor,
            "Expectancy (€)": self.expectancy,
        }


def calculate_trade_metrics(
    trades: pd.DataFrame,
) -> TradeMetrics:
    """
    Calculate analytics from a simulator trade log.

    Profit factor is:

        gross profit / absolute gross loss

    The trade log must contain a numeric 'profit' column.
    Rows without a realized profit, such as entry transactions,
    do not affect the calculation.
    """
    required_columns = {
        "action",
        "profit",
        }

    missing_columns = required_columns - set(trades.columns)

    if missing_columns:
        raise ValueError(
            "Trades dataframe is missing columns: "
            f"{sorted(missing_columns)}"
        )
    
    if trades is None:
        raise ValueError(
            "Trades dataframe cannot be None."
        )

    if "profit" not in trades.columns:
        raise ValueError(
            "Trades dataframe is missing the 'profit' column."
        )

    completed_trades = trades.loc[
        trades["action"]
        .astype(str)
        .str.upper()
        .eq("SELL")
    ].copy()

    profits = pd.to_numeric(
        completed_trades["profit"],
        errors="coerce",
    ).dropna()

    gross_profit = float(
        profits.loc[profits > 0].sum()
    )

    gross_loss = abs(
        float(
            profits.loc[profits < 0].sum()
        )
    )

    if profits.empty:
        expectancy = 0.0
    else:
        expectancy = float(
            profits.mean()
        )

    if gross_loss == 0:
        if gross_profit > 0:
            profit_factor = float("inf")
        else:
            profit_factor = 0.0

    else:
        profit_factor = (
            gross_profit / gross_loss
        )

    if np.isnan(profit_factor):
        profit_factor = 0.0

    return TradeMetrics(
        profit_factor=float(profit_factor),
        expectancy=float(expectancy),
    )