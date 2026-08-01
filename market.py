from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import requests


BASE_URL = "https://api.coingecko.com/api/v3"

DATA_DIR = Path(__file__).parent / "data"
CACHE_MAX_AGE = timedelta(hours=6)

def _history_cache_path(
    coin_id: str,
    currency: str,
    days: int,
) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    safe_coin = coin_id.lower().replace("/", "_")
    safe_currency = currency.lower().replace("/", "_")

    return DATA_DIR / (
        f"{safe_coin}_{safe_currency}_{days}d.csv"
    )


def _load_cached_history(
    cache_path: Path,
) -> Optional[pd.DataFrame]:
    if not cache_path.exists():
        return None

    try:
        data = pd.read_csv(
            cache_path,
            parse_dates=["date"],
        )

    except (OSError, ValueError, pd.errors.ParserError) as error:
        print(f"Could not read cache {cache_path}: {error}")
        return None

    required_columns = {"date", "price"}

    if not required_columns.issubset(data.columns):
        print(f"Invalid cache columns in {cache_path}")
        return None

    return data


def _cache_is_fresh(cache_path: Path) -> bool:
    if not cache_path.exists():
        return False

    modified_at = datetime.fromtimestamp(
        cache_path.stat().st_mtime,
        tz=timezone.utc,
    )

    return (
        datetime.now(timezone.utc) - modified_at
        <= CACHE_MAX_AGE
    )


def _save_cached_history(
    data: pd.DataFrame,
    cache_path: Path,
) -> None:
    try:
        data.to_csv(cache_path, index=False)

    except OSError as error:
        # The app can still operate even if caching fails.
        print(f"Could not save cache {cache_path}: {error}")


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


def get_history(
    coin_id: str,
    days: int = 365,
    currency: str = "eur",
    force_refresh: bool = False,
) -> Optional[pd.DataFrame]:
    days = min(days, 365)

    cache_path = _history_cache_path(
        coin_id=coin_id,
        currency=currency,
        days=days,
    )

    cached_data = _load_cached_history(cache_path)

    if (
        not force_refresh
        and cached_data is not None
        and _cache_is_fresh(cache_path)
    ):
        print(f"Using fresh cache: {cache_path.name}")
        return cached_data

    url = f"{BASE_URL}/coins/{coin_id}/market_chart"

    params = {
        "vs_currency": currency,
        "days": days,
    }

    try:
        response = requests.get(
            url,
            params=params,
            timeout=15,
            headers={
                "accept": "application/json",
                "User-Agent": "crypto-assistant/0.5",
            },
        )

        response.raise_for_status()
        payload = response.json()

        prices = payload.get("prices")

        if not prices:
            raise ValueError(
                "CoinGecko response contained no price data."
            )

        data = pd.DataFrame(
            prices,
            columns=["timestamp", "price"],
        )

        data["date"] = pd.to_datetime(
            data["timestamp"],
            unit="ms",
            utc=True,
        )

        daily = (
            data.set_index("date")[["price"]]
            .resample("1D")
            .last()
            .dropna()
            .reset_index()
        )

        _save_cached_history(
            data=daily,
            cache_path=cache_path,
        )

        print(f"Updated cache: {cache_path.name}")
        return daily

    except (
        requests.RequestException,
        ValueError,
        KeyError,
    ) as error:
        print(f"CoinGecko history request failed: {error}")

        if cached_data is not None:
            print(
                "Using stale cached history because "
                "fresh data could not be downloaded."
            )
            return cached_data

        return None


if __name__ == "__main__":
    print(get_prices())
    print(get_history("bitcoin").tail())