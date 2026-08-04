from __future__ import annotations

import pandas as pd


def calculate_monthly_returns(
    history: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate monthly portfolio returns.

    Required columns:
    - date
    - strategy_value

    Returns
    -------
    DataFrame

    year
    month
    monthly_return
    """

    required_columns = {
        "date",
        "strategy_value",
    }

    missing = (
        required_columns - set(history.columns)
    )

    if missing:
        raise ValueError(
            "History is missing columns: "
            f"{sorted(missing)}"
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
        .dropna()
        .sort_values("date")
        .drop_duplicates(
            subset=["date"],
            keep="last",
        )
    )

    monthly = (
        data
        .set_index("date")
        .resample("ME")
        .last()
    )

    monthly["monthly_return"] = (
        monthly["strategy_value"]
        .pct_change(fill_method=None)
        * 100
    )

    monthly = (
        monthly
        .dropna(subset=["monthly_return"])
        .reset_index()
    )

    monthly["year"] = (
        monthly["date"].dt.year
    )

    monthly["month"] = (
        monthly["date"]
        .dt.strftime("%b")
    )

    return monthly[
        [
            "year",
            "month",
            "monthly_return",
        ]
    ]