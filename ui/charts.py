from __future__ import annotations
from strategies.registry import ChartPanel

import pandas as pd
import plotly.graph_objects as go


def plot_price_with_trades(
    price_history: pd.DataFrame,
    trade_log: pd.DataFrame | None,
    title: str = "Price and trades",
    overlays: dict[str, str] | None = None,
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

    if overlays:
        for display_name, column_name in overlays.items():
            if column_name not in price_history.columns:
                continue

            overlay_data = price_history[
                ["date", column_name]
            ].copy()

            overlay_data["date"] = pd.to_datetime(
                overlay_data["date"],
                errors="coerce",
            )

            overlay_data[column_name] = pd.to_numeric(
                overlay_data[column_name],
                errors="coerce",
            )

            overlay_data = overlay_data.dropna(
                subset=["date", column_name]
            )

            if overlay_data.empty:
                continue

            figure.add_trace(
                go.Scatter(
                    x=overlay_data["date"],
                    y=overlay_data[column_name],
                    mode="lines",
                    name=display_name,
                    hovertemplate=(
                        f"{display_name}<br>"
                        "%{x|%Y-%m-%d}<br>"
                        "Value: €%{y:,.2f}"
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

def plot_indicator_panel(
    dataframe: pd.DataFrame,
    panel: ChartPanel,
) -> go.Figure:
    """
    Plot a generic indicator panel from ChartPanel metadata.
    """

    figure = go.Figure()

    data = dataframe.copy()

    if "date" not in data.columns:
        raise ValueError(
            "Dataframe must contain a 'date' column."
        )

    data["date"] = pd.to_datetime(
        data["date"],
        errors="coerce",
    )

    for series in panel.series:

        if series.column not in data.columns:
            continue

        plot_data = data[
            ["date", series.column]
        ].copy()

        plot_data[series.column] = pd.to_numeric(
            plot_data[series.column],
            errors="coerce",
        )

        plot_data = plot_data.dropna()

        if plot_data.empty:
            continue

        if series.chart_type == "line":

            figure.add_trace(
                go.Scatter(
                    x=plot_data["date"],
                    y=plot_data[series.column],
                    mode="lines",
                    name=series.label,
                )
            )

        elif series.chart_type == "bar":

            figure.add_trace(
                go.Bar(
                    x=plot_data["date"],
                    y=plot_data[series.column],
                    name=series.label,
                )
            )

    #
    # Reference lines
    #

    if panel.reference_lines:

        x_min = data["date"].min()
        x_max = data["date"].max()

        for value in panel.reference_lines:

            figure.add_trace(
                go.Scatter(
                    x=[x_min, x_max],
                    y=[value, value],
                    mode="lines",
                    name=str(value),
                    line=dict(
                        dash="dash",
                    ),
                    hoverinfo="skip",
                )
            )

    figure.update_layout(
        title=panel.title,
        xaxis_title="Date",
        yaxis_title=panel.y_axis_title,
        hovermode="x unified",
        barmode="relative",
        margin=dict(
            l=40,
            r=20,
            t=60,
            b=40,
        ),
    )

    return figure