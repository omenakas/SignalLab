import pandas as pd


def generate_rsi_positions(
    df: pd.DataFrame,
    rsi_period: int = 14,
    oversold: float = 30.0,
    overbought: float = 70.0,
) -> pd.DataFrame:
    """
    Generate long-only trading positions using RSI.

    Position:
        0 = Cash
        1 = Invested
    """

    required_columns = {"date", "price"}

    if not required_columns.issubset(df.columns):
        raise ValueError(
            "DataFrame must contain 'date' and 'price'."
        )

    data = df.copy()

    price = data["price"]

    delta = price.diff()

    gain = delta.clip(lower=0)

    loss = -delta.clip(upper=0)

    average_gain = (
        gain
        .rolling(rsi_period)
        .mean()
    )

    average_loss = (
        loss
        .rolling(rsi_period)
        .mean()
    )

    rs = average_gain / average_loss

    data["RSI"] = 100 - (
        100 / (1 + rs)
    )

    data = data.dropna(
        subset=["RSI"]
    ).copy()

    positions = []

    current_position = 0

    for rsi in data["RSI"]:

        if rsi < oversold:
            current_position = 1

        elif rsi > overbought:
            current_position = 0

        positions.append(current_position)

    data["raw_position"] = positions

    # Execute tomorrow.
    data["position"] = (
        data["raw_position"]
        .shift(1)
        .fillna(0)
        .astype(int)
    )

    return data.reset_index(drop=True)