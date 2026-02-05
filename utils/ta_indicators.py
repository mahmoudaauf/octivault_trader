# utils/ta_indicators.py

import numpy as np
import pandas as pd

def calculate_ema(series, period):
    return pd.Series(series).ewm(span=period, adjust=False).to_numpy()

def calculate_rsi(series, period=14):
    delta = np.diff(series)
    up = delta.clip(min=0)
    down = -1 * delta.clip(max=0)
    avg_gain = pd.Series(up).rolling(window=period).mean()
    avg_loss = pd.Series(down).rolling(window=period).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return np.append([50] * (period - 1), rsi.fillna(50).to_numpy())

def calculate_volume_surge(volumes, threshold_ratio=1.5):
    recent_vol = volumes[-1]
    avg_vol = np.mean(volumes[-21:-1])
    return recent_vol > avg_vol * threshold_ratio
