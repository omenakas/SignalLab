from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

@dataclass(frozen=True)
class PerformanceMetrics:
    """
    Risk-adjusted performance metrics calculated from a completed
    backtest.
    """

    sharpe_ratio: float

def calculate_performance_metrics(
    history: pd.DataFrame,
    risk_free_rate: float = 0.0,
) -> PerformanceMetrics:
    """
    Calculate risk-adjusted performance metrics from a completed
    backtest history.

    Parameters
    ----------
    history
        Simulator history dataframe.

    risk_free_rate
        Annual risk-free rate expressed as a decimal.
        Default is 0.0.
    """

    required_columns = {
        "strategy_value",
    }

    missing = required_columns - set(history.columns)

    if missing:
        raise ValueError(
            "History is missing columns: "
            f"{sorted(missing)}"
        )

    values = (
        history["strategy_value"]
        .astype(float)
        .copy()
    )

    returns = values.pct_change().dropna()

    if returns.empty:
        return PerformanceMetrics(
            sharpe_ratio=0.0,
        )

    volatility = returns.std()

    if volatility == 0:
        sharpe = 0.0

    else:
        sharpe = (
            (
                returns.mean()
                - risk_free_rate / 365
            )
            / volatility
        ) * np.sqrt(365)

    return PerformanceMetrics(
        sharpe_ratio=float(sharpe),
    )