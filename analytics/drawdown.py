from __future__ import annotations

import pandas as pd


def calculate_drawdown_series(
    history: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate the portfolio drawdown through time.

    Required columns:
    - date
    - strategy_value

    Returns:
    - date
    - drawdown_pct
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

    if data.empty:
        raise ValueError(
            "No valid history remains for drawdown analysis."
        )

    running_max = data["strategy_value"].cummax()

    data["drawdown_pct"] = (
        data["strategy_value"]
        / running_max
        - 1
    ) * 100

    return data[
        ["date", "drawdown_pct"]
    ]