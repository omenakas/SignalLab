from __future__ import annotations

import numpy as np
import pandas as pd


def calculate_rolling_sharpe(
    history: pd.DataFrame,
    window: int = 30,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 365,
) -> pd.DataFrame:
    """
    Calculate annualized rolling Sharpe ratio.

    Required columns:
    - date
    - strategy_value

    Returns:
    - date
    - rolling_sharpe
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

    if window < 2:
        raise ValueError(
            "Rolling window must be at least 2."
        )

    if periods_per_year <= 0:
        raise ValueError(
            "Periods per year must be positive."
        )

    if risk_free_rate <= -1:
        raise ValueError(
            "Risk-free rate must be greater than -1."
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

    if len(data) <= window:
        raise ValueError(
            "Not enough history for the selected rolling window."
        )

    data["return"] = (
        data["strategy_value"]
        .pct_change(fill_method=None)
    )

    periodic_risk_free_rate = (
        (1 + risk_free_rate)
        ** (1 / periods_per_year)
        - 1
    )

    excess_returns = (
        data["return"]
        - periodic_risk_free_rate
    )

    rolling_mean = excess_returns.rolling(
        window=window,
        min_periods=window,
    ).mean()

    rolling_volatility = data["return"].rolling(
        window=window,
        min_periods=window,
    ).std(ddof=1)

    rolling_sharpe = (
        rolling_mean
        / rolling_volatility
        * np.sqrt(periods_per_year)
    )

    rolling_sharpe = rolling_sharpe.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    data["rolling_sharpe"] = rolling_sharpe

    return (
        data[
            [
                "date",
                "rolling_sharpe",
            ]
        ]
        .dropna(subset=["rolling_sharpe"])
        .reset_index(drop=True)
    )