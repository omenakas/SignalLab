import pandas as pd


def generate_ma_positions(
    df: pd.DataFrame,
    fast_window: int,
    slow_window: int,
) -> pd.DataFrame:
    """
    Calculate moving-average crossover positions.

    The returned DataFrame contains:

    - date
    - price
    - fast_ma
    - slow_ma
    - raw_position
    - position

    Position meanings:
    0 = hold cash
    1 = hold Bitcoin

    The signal calculated from today's closing price is shifted forward
    by one row, so it is executed using the following day's price.
    """

    if fast_window <= 0 or slow_window <= 0:
        raise ValueError(
            "Moving-average windows must be positive."
        )

    if fast_window >= slow_window:
        raise ValueError(
            "Fast moving average must be shorter than slow moving average."
        )

    required_columns = {"date", "price"}

    if not required_columns.issubset(df.columns):
        raise ValueError(
            "The DataFrame must contain 'date' and 'price' columns."
        )

    data = df[["date", "price"]].copy()

    data["price"] = pd.to_numeric(
        data["price"],
        errors="coerce",
    )

    data = data.dropna(subset=["price"]).copy()

    data["fast_ma"] = (
        data["price"]
        .rolling(window=fast_window)
        .mean()
    )

    data["slow_ma"] = (
        data["price"]
        .rolling(window=slow_window)
        .mean()
    )

    data = data.dropna(
        subset=["fast_ma", "slow_ma"]
    ).copy()

    if len(data) < 2:
        raise ValueError(
            "Not enough historical data for these moving-average windows."
        )

    data["raw_position"] = (
        data["fast_ma"] > data["slow_ma"]
    ).astype(int)

    # Avoid look-ahead bias:
    # today's completed signal becomes tomorrow's position.
    data["position"] = (
        data["raw_position"]
        .shift(1)
        .fillna(0)
        .astype(int)
    )

    return data.reset_index(drop=True)