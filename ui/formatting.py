from __future__ import annotations

import pandas as pd

def format_currency(value: float) -> str:
    """
    Format a monetary value using two decimals and
    place the minus sign before the currency symbol.
    """

    if pd.isna(value):
        return "—"

    if value < 0:
        return f"-€{abs(value):,.2f}"

    return f"€{value:,.2f}"


def format_percentage(value: float) -> str:
    """
    Format a percentage using two decimals.
    """

    if pd.isna(value):
        return "—"

    return f"{value:.2f}%"

BACKTEST_FORMATS = {
    "price": "{:,.2f}",
    "fee": format_currency,
    "portfolio_value": format_currency,
    "profit": format_currency,
}

TRADE_HISTORY_FORMATS = {
    "Entry price": "{:,.2f}",
    "Exit price": "{:,.2f}",
    "Profit": format_currency,
    "Return (%)": format_percentage,
    "Portfolio value": format_currency,
}

def format_dataframe(
    dataframe: pd.DataFrame,
    formats: dict[str, str],
) -> pd.io.formats.style.Styler:
    """
    Apply presentation formatting to a DataFrame without
    modifying the underlying values.
    """

    applicable_formats = {
        column: fmt
        for column, fmt in formats.items()
        if column in dataframe.columns
    }

    return dataframe.style.format(
        applicable_formats,
        na_rep="—",
    )

