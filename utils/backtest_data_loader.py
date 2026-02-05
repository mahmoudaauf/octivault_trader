import logging
import pandas as pd
import os
from pathlib import Path
from typing import Dict, Optional, List

logger = logging.getLogger("BacktestDataLoader")

class BacktestDataLoader:
    """
    Loads historical market data (CSV) and prepares it for the Backtesting Engine.
    Expected CSV columns: [timestamp, open, high, low, close, volume]
    """
    def __init__(self, data_dir: str = "data/historical"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.registry: Dict[str, pd.DataFrame] = {}

    def load_csv(self, symbol: str, timeframe: str = "5m") -> Optional[pd.DataFrame]:
        """
        Loads a CSV file for a specific symbol and timeframe.
        File naming convention: {symbol}_{timeframe}.csv (e.g., BTCUSDT_5m.csv)
        """
        filename = f"{symbol}_{timeframe}.csv"
        filepath = self.data_dir / filename

        if not filepath.exists():
            logger.error(f"Historical data file not found: {filepath}")
            return None

        try:
            df = pd.read_csv(filepath)
            # Standardize columns
            df.columns = [c.lower() for c in df.columns]
            
            # Ensure required columns
            required = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required):
                logger.error(f"Missing columns in {filename}. Expected: {required}")
                return None

            # Sort by timestamp
            df = df.sort_values('timestamp').reset_index(drop=True)
            self.registry[f"{symbol}_{timeframe}"] = df
            logger.info(f"Loaded {len(df)} rows for {symbol} ({timeframe})")
            return df
        except Exception as e:
            logger.exception(f"Error loading {filename}: {e}")
            return None

    def get_data(self, symbol: str, timeframe: str = "5m") -> Optional[pd.DataFrame]:
        return self.registry.get(f"{symbol}_{timeframe}")

    def list_available_symbols(self) -> List[str]:
        return [f.stem.split('_')[0] for f in self.data_dir.glob("*.csv")]
