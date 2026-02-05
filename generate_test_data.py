import pandas as pd
import numpy as np
import os
from pathlib import Path

def generate_synthetic_data(symbol: str, timeframe: str = "5m", n_rows: int = 1500):
    data_dir = Path("data/historical")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate prices: Random walk with drift
    np.random.seed(42)
    start_price = 50000.0 if "BTC" in symbol else 3000.0
    drift = 0.0001
    volatility = 0.005
    
    returns = np.random.normal(drift, volatility, n_rows)
    price_series = start_price * (1 + returns).cumprod()
    
    data = {
        "timestamp": range(1672531200000, 1672531200000 + n_rows * 300000, 300000), # 5m intervals
        "open": price_series * (1 - 0.001),
        "high": price_series * (1 + 0.002),
        "low": price_series * (1 - 0.002),
        "close": price_series,
        "volume": np.random.randint(10, 100, n_rows) * 100.0
    }
    
    df = pd.DataFrame(data)
    filepath = data_dir / f"{symbol}_{timeframe}.csv"
    df.to_csv(filepath, index=False)
    print(f"Generated synthetic data: {filepath}")

if __name__ == "__main__":
    generate_synthetic_data("BTCUSDT")
    generate_synthetic_data("ETHUSDT")
