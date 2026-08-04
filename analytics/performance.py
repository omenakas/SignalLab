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
    cagr: float

    def as_dict(self) -> dict[str, float]:
        """
        Return display-ready metric names and values.
        """

        return {
            "Sharpe ratio": self.sharpe_ratio,
            "CAGR (%)": self.cagr,
        }
    
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
        Simulator history dataframe containing:
        - date
        - strategy_value

    risk_free_rate
        Annual risk-free rate expressed as a decimal.
    """

    required_columns = {
        "date",
        "strategy_value",
    }

    missing_columns = (
        required_columns - set(history.columns)
    )

    if missing_columns:
        raise ValueError(
            "History is missing columns: "
            f"{sorted(missing_columns)}"
        )

    data = history[
        ["date", "strategy_value"]
    ].copy()

    data["date"] = pd.to_datetime(
        data["date"],
        errors="coerce",
    )

    data["strategy_value"] = pd.to_numeric(
        data["strategy_value"],
        errors="coerce",
    )

    data = (
        data
        .dropna(subset=["date", "strategy_value"])
        .sort_values("date")
        .drop_duplicates(
            subset=["date"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    if len(data) < 2:
        return PerformanceMetrics(
            sharpe_ratio=0.0,
            cagr=0.0,
        )

    values = data["strategy_value"]

    start_date = data["date"].iloc[0]
    end_date = data["date"].iloc[-1]

    elapsed_days = (
        end_date - start_date
    ).total_seconds() / 86_400

    years = elapsed_days / 365.25

    initial_value = float(values.iloc[0])
    final_value = float(values.iloc[-1])

    if (
        years <= 0
        or initial_value <= 0
        or final_value <= 0
    ):
        cagr = 0.0

    else:
        cagr = (
            (
                final_value / initial_value
            )
            ** (1 / years)
            - 1
        ) * 100

    returns = (
        values
        .pct_change(fill_method=None)
        .dropna()
    )

    if returns.empty:
        sharpe = 0.0

    else:
        volatility = float(
            returns.std(ddof=1)
        )

        if not np.isfinite(volatility) or volatility == 0:
            sharpe = 0.0

        else:
            daily_risk_free_rate = (
                (1 + risk_free_rate)
                ** (1 / 365)
                - 1
            )

            sharpe = (
                (
                    float(returns.mean())
                    - daily_risk_free_rate
                )
                / volatility
                * np.sqrt(365)
            )

            if not np.isfinite(sharpe):
                sharpe = 0.0

    return PerformanceMetrics(
        sharpe_ratio=float(sharpe),
        cagr=float(cagr),
    )
