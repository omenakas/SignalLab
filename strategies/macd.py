import pandas as pd
from ta.trend import MACD


def generate_macd_positions(
    df: pd.DataFrame,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> pd.DataFrame:
    """
    Generate long-only MACD positions.

    Position meanings:
        0 = hold cash
        1 = hold the asset

    Enter after the MACD line crosses above the signal line.
    Exit after the MACD line crosses below the signal line.

    A crossover calculated from today's closing price is executed
    on the following row to avoid look-ahead bias.
    """

    required_columns = {"date", "price"}

    if not required_columns.issubset(df.columns):
        raise ValueError(
            "The DataFrame must contain 'date' and 'price' columns."
        )

    if fast_period <= 0:
        raise ValueError("Fast MACD period must be positive.")

    if slow_period <= 0:
        raise ValueError("Slow MACD period must be positive.")

    if signal_period <= 0:
        raise ValueError("MACD signal period must be positive.")

    if fast_period >= slow_period:
        raise ValueError(
            "Fast MACD period must be shorter than slow MACD period."
        )

    data = df[["date", "price"]].copy()

    data["price"] = pd.to_numeric(
        data["price"],
        errors="coerce",
    )

    data = data.dropna(
        subset=["date", "price"]
    ).copy()

    macd_indicator = MACD(
        close=data["price"],
        window_fast=fast_period,
        window_slow=slow_period,
        window_sign=signal_period,
    )

    data["macd"] = macd_indicator.macd()
    data["signal_line"] = macd_indicator.macd_signal()
    data["macd_histogram"] = macd_indicator.macd_diff()

    data = data.dropna(
        subset=[
            "macd",
            "signal_line",
            "macd_histogram",
        ]
    ).copy()

    if len(data) < 2:
        raise ValueError(
            "Not enough historical data for these MACD periods."
        )

    previous_macd = data["macd"].shift(1)
    previous_signal = data["signal_line"].shift(1)

    data["bullish_crossover"] = (
        (previous_macd <= previous_signal)
        & (data["macd"] > data["signal_line"])
    )

    data["bearish_crossover"] = (
        (previous_macd >= previous_signal)
        & (data["macd"] < data["signal_line"])
    )

    raw_positions: list[int] = []
    current_position = 0

    for row in data.itertuples():
        if row.bullish_crossover:
            current_position = 1

        elif row.bearish_crossover:
            current_position = 0

        raw_positions.append(current_position)

    data["raw_position"] = raw_positions

    # Avoid look-ahead bias:
    # today's completed crossover is executed on the next row.
    data["position"] = (
        data["raw_position"]
        .shift(1)
        .fillna(0)
        .astype(int)
    )

    return data.reset_index(drop=True)