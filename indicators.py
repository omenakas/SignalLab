import pandas as pd
from ta.momentum import RSIIndicator


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicators to a historical-price DataFrame.

    Expected input columns:
    - date
    - price
    """

    result = df.copy()

    result["MA20"] = result["price"].rolling(window=20).mean()
    result["MA50"] = result["price"].rolling(window=50).mean()

    rsi = RSIIndicator(
        close=result["price"],
        window=14,
    )

    result["RSI"] = rsi.rsi()

    return result