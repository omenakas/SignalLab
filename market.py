from typing import Optional

import pandas as pd
import requests


BASE_URL = "https://api.coingecko.com/api/v3"


def get_prices() -> Optional[dict]:
    url = f"{BASE_URL}/simple/price"

    params = {
        "ids": "bitcoin,ethereum,solana",
        "vs_currencies": "eur",
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        return response.json()

    except requests.RequestException as error:
        print(f"Price request failed: {error}")
        return None


def get_history(coin_id: str, days: int = 365) -> Optional[pd.DataFrame]:
    url = f"{BASE_URL}/coins/{coin_id}/market_chart"

    params = {
        "vs_currency": "eur",
        "days": days,
    }

    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()

    except requests.RequestException as error:
        print(f"History request failed: {error}")
        return None

    prices = data.get("prices")

    if not prices:
        return None

    df = pd.DataFrame(
        prices,
        columns=["timestamp", "price"],
    )

    df["date"] = pd.to_datetime(
        df["timestamp"],
        unit="ms",
        utc=True,
    )

    daily = (
        df.set_index("date")[["price"]]
        .resample("1D")
        .last()
        .dropna()
        .reset_index()
    )

    return daily


if __name__ == "__main__":
    print(get_prices())
    print(get_history("bitcoin").tail())