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

    def as_dict(self) -> dict[str, float]:
        """
        Return display-ready metric names and values.
        """

        return {
            "Profit factor": self.profit_factor,
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

    if trades is None:
        raise ValueError(
            "Trades dataframe cannot be None."
        )

    if "profit" not in trades.columns:
        raise ValueError(
            "Trades dataframe is missing the 'profit' column."
        )

    profits = pd.to_numeric(
        trades["profit"],
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
    )