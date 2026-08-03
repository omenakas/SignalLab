from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go


def plot_price_with_trades(
    price_history: pd.DataFrame,
    trade_log: pd.DataFrame | None,
    title: str = "Price and trades",
) -> go.Figure:
    """
    Plot historical prices with BUY and SELL markers.

    Required price-history columns:
    - date
    - price

    Expected trade-log columns:
    - date
    - action
    - price
    """

    required_price_columns = {"date", "price"}

    missing_price_columns = (
        required_price_columns - set(price_history.columns)
    )

    if missing_price_columns:
        raise ValueError(
            "Price history is missing columns: "
            f"{sorted(missing_price_columns)}"
        )

    chart_data = price_history[["date", "price"]].copy()

    chart_data["date"] = pd.to_datetime(
        chart_data["date"],
        errors="coerce",
    )

    chart_data["price"] = pd.to_numeric(
        chart_data["price"],
        errors="coerce",
    )

    chart_data = (
        chart_data
        .dropna(subset=["date", "price"])
        .sort_values("date")
    )

    if chart_data.empty:
        raise ValueError(
            "No valid price data remains for the chart."
        )

    figure = go.Figure()

    figure.add_trace(
        go.Scatter(
            x=chart_data["date"],
            y=chart_data["price"],
            mode="lines",
            name="Price",
            hovertemplate=(
                "%{x|%Y-%m-%d}<br>"
                "Price: €%{y:,.2f}"
                "<extra></extra>"
            ),
        )
    )

    if trade_log is not None and not trade_log.empty:
        required_trade_columns = {
            "date",
            "action",
            "price",
        }

        missing_trade_columns = (
            required_trade_columns - set(trade_log.columns)
        )

        if missing_trade_columns:
            raise ValueError(
                "Trade log is missing columns: "
                f"{sorted(missing_trade_columns)}"
            )

        trades = trade_log[
            ["date", "action", "price"]
        ].copy()

        trades["date"] = pd.to_datetime(
            trades["date"],
            errors="coerce",
        )

        trades["price"] = pd.to_numeric(
            trades["price"],
            errors="coerce",
        )

        trades["action"] = (
            trades["action"]
            .astype(str)
            .str.upper()
        )

        trades = trades.dropna(
            subset=["date", "price"]
        )

        buys = trades.loc[
            trades["action"] == "BUY"
        ]

        sells = trades.loc[
            trades["action"] == "SELL"
        ]

        if not buys.empty:
            figure.add_trace(
                go.Scatter(
                    x=buys["date"],
                    y=buys["price"],
                    mode="markers",
                    name="Buy",
                    marker={
                        "symbol": "triangle-up",
                        "size": 12,
                    },
                    hovertemplate=(
                        "BUY<br>"
                        "%{x|%Y-%m-%d}<br>"
                        "Price: €%{y:,.2f}"
                        "<extra></extra>"
                    ),
                )
            )

        if not sells.empty:
            figure.add_trace(
                go.Scatter(
                    x=sells["date"],
                    y=sells["price"],
                    mode="markers",
                    name="Sell",
                    marker={
                        "symbol": "triangle-down",
                        "size": 12,
                    },
                    hovertemplate=(
                        "SELL<br>"
                        "%{x|%Y-%m-%d}<br>"
                        "Price: €%{y:,.2f}"
                        "<extra></extra>"
                    ),
                )
            )

    figure.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Price (€)",
        hovermode="x unified",
        legend_title_text="Series",
        margin={
            "l": 40,
            "r": 20,
            "t": 60,
            "b": 40,
        },
    )

    return figure