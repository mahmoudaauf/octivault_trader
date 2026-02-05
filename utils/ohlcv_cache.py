# utils/ohlcv_cache.py

import os
import pandas as pd
import logging
from datetime import datetime
import requests

logger = logging.getLogger("OHLCVCache")
logging.basicConfig(level=logging.INFO)

DATA_PATH = "data/ohlcv/"
BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"


def ensure_data_dir(path=DATA_PATH):
    os.makedirs(path, exist_ok=True)


def save_ohlcv_to_csv(symbol: str, df: pd.DataFrame, path: str = DATA_PATH):
    ensure_data_dir(path)
    file_path = os.path.join(path, f"{symbol}.csv")
    df.to_csv(file_path, index=False)
    logger.info(f"✅ Saved OHLCV for {symbol} to {file_path}")


def load_ohlcv_from_cache(symbol: str, path: str = DATA_PATH) -> pd.DataFrame | None:
    file_path = os.path.join(path, f"{symbol}.csv")
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        logger.info(f"📥 Loaded cached OHLCV for {symbol}")
        return df
    logger.warning(f"⚠️ No cached OHLCV found for {symbol}")
    return None


def fetch_and_cache_ohlcv(symbol: str, interval: str = "1m", limit: int = 500) -> pd.DataFrame:
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit
    }

    try:
        logger.info(f"📡 Fetching OHLCV from Binance: {symbol}, interval={interval}, limit={limit}")
        response = requests.get(BINANCE_KLINES_URL, params=params)
        response.raise_for_status()
        raw_data = response.json()

        df = pd.DataFrame(raw_data, columns=[
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "num_trades",
            "taker_buy_base", "taker_buy_quote", "ignore"
        ])
        df = df[["timestamp", "open", "high", "low", "close", "volume"]]
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df = df.astype({
            "open": "float", "high": "float", "low": "float",
            "close": "float", "volume": "float"
        })

        save_ohlcv_to_csv(symbol, df)
        return df

    except requests.RequestException as e:
        logger.error(f"❌ Failed to fetch OHLCV for {symbol}: {e}")
        return pd.DataFrame()
