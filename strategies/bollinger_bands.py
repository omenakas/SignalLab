import pandas as pd
from ta.volatility import BollingerBands


def generate_bollinger_positions(
    df: pd.DataFrame,
    window: int = 20,
    window_dev: float = 2.0,
) -> pd.DataFrame:
    """
    Generate long-only Bollinger Bands positions.

    Position meanings:
        0 = hold cash
        1 = hold the asset

    Enter when price closes below the lower band.
    Exit when price closes above the upper band.

    Signals are executed on the following row to avoid look-ahead bias.
    """

    required_columns = {"date", "price"}

    if not required_columns.issubset(df.columns):
        raise ValueError(
            "The DataFrame must contain 'date' and 'price' columns."
        )

    if window <= 1:
        raise ValueError(
            "Bollinger window must be greater than 1."
        )

    if window_dev <= 0:
        raise ValueError(
            "Bollinger deviation must be positive."
        )

    data = df[["date", "price"]].copy()

    data["price"] = pd.to_numeric(
        data["price"],
        errors="coerce",
    )

    data = data.dropna(
        subset=["date", "price"]
    ).copy()

    indicator = BollingerBands(
        close=data["price"],
        window=window,
        window_dev=window_dev,
    )

    data["bb_middle"] = indicator.bollinger_mavg()
    data["bb_upper"] = indicator.bollinger_hband()
    data["bb_lower"] = indicator.bollinger_lband()

    data = data.dropna(
        subset=[
            "bb_middle",
            "bb_upper",
            "bb_lower",
        ]
    ).copy()

    if len(data) < 2:
        raise ValueError(
            "Not enough historical data for these Bollinger settings."
        )

    raw_positions: list[int] = []
    current_position = 0

    for row in data.itertuples():
        if row.price < row.bb_lower:
            current_position = 1

        elif row.price > row.bb_upper:
            current_position = 0

        raw_positions.append(current_position)

    data["raw_position"] = raw_positions

    data["position"] = (
        data["raw_position"]
        .shift(1)
        .fillna(0)
        .astype(int)
    )

    return data.reset_index(drop=True)