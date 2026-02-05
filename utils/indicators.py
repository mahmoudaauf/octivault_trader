import numpy as np
import pandas as pd

__all__ = [
    "compute_ema",
    "compute_bollinger_bands",
    "compute_atr",
    "compute_rsi",
    "compute_macd",
    "compute_stochastic",
    "compute_obv",
    "compute_vwap",
    "calculate_indicators",
    "scale_features",
]

from typing import Iterable, Tuple, Optional, Union

def _as_series(x, name: str = "value") -> pd.Series:
    if isinstance(x, pd.Series):
        return x
    return pd.Series(list(x), name=name)

def _as_dataframe(df_or_hlc) -> pd.DataFrame:
    if isinstance(df_or_hlc, pd.DataFrame):
        return df_or_hlc
    # Expecting a tuple/list of (highs, lows, closes)
    try:
        highs, lows, closes = df_or_hlc
        return pd.DataFrame({
            "high": _as_series(highs, "high").astype(float),
            "low": _as_series(lows, "low").astype(float),
            "close": _as_series(closes, "close").astype(float),
        })
    except Exception:
        raise TypeError("compute_atr expects a DataFrame with columns ['high','low','close'] "
                        "or an iterable triple (highs, lows, closes).")

def compute_ema(series: Union[Iterable[float], pd.Series], span: int) -> pd.Series:
    """Exponential Moving Average (accepts list/array/Series)."""
    s = _as_series(series, "close").astype(float)
    return s.ewm(span=span, adjust=False).mean()

def compute_bollinger_bands(series: Union[Iterable[float], pd.Series], period: int = 20, std_dev: float = 2
                            ) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate Bollinger Bands (accepts list/array/Series). Returns (upper, middle, lower)."""
    s = _as_series(series, "close").astype(float)
    sma = s.rolling(window=period, min_periods=period).mean()
    std = s.rolling(window=period, min_periods=period).std(ddof=0)
    upper_band = sma + std_dev * std
    lower_band = sma - std_dev * std
    return upper_band, sma, lower_band

def compute_atr(df_or_hlc, period: int = 14) -> pd.Series:
    """
    Average True Range (accepts DataFrame with ['high','low','close'] OR a triple of iterables).
    Returns a pandas Series aligned to the inputs.
    """
    df = _as_dataframe(df_or_hlc)
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    # Wilder's ATR uses RMA; here we keep simple rolling mean for speed/compat unless min_periods provided.
    return true_range.rolling(window=period, min_periods=period).mean()

def compute_rsi(series: Union[Iterable[float], pd.Series], period: int = 14) -> pd.Series:
    """Relative Strength Index (accepts list/array/Series)."""
    s = _as_series(series, "close").astype(float)
    delta = s.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, pd.NA)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_macd(series: Union[Iterable[float], pd.Series], fast: int = 12, slow: int = 26, signal: int = 9
                 ) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Moving Average Convergence Divergence (accepts list/array/Series). Returns (macd, signal, histogram)."""
    s = _as_series(series, "close").astype(float)
    ema_fast = compute_ema(s, span=fast)
    ema_slow = compute_ema(s, span=slow)
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - signal_line
    return macd, signal_line, hist

def compute_stochastic(df_or_hlc, k_period: int = 14, d_period: int = 3
                       ) -> Tuple[pd.Series, pd.Series]:
    """Stochastic Oscillator (accepts DataFrame or triple of iterables)."""
    df = _as_dataframe(df_or_hlc)
    low_min = df["low"].rolling(window=k_period, min_periods=k_period).min()
    high_max = df["high"].rolling(window=k_period, min_periods=k_period).max()
    denom = (high_max - low_min).replace(0, pd.NA)
    k_percent = 100 * ((df["close"] - low_min) / denom)
    d_percent = k_percent.rolling(window=d_period, min_periods=d_period).mean()
    return k_percent, d_percent

def compute_obv(df: pd.DataFrame) -> pd.Series:
    """On-Balance Volume."""
    direction = np.sign(df['close'].diff()).fillna(0)
    obv = (df['volume'] * direction).cumsum()
    return obv

def compute_vwap(df: pd.DataFrame) -> pd.Series:
    """Volume Weighted Average Price."""
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    vwap = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
    return vwap

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Apply common technical indicators to the DataFrame."""
    df = df.copy()
    df["ema20"] = compute_ema(df["close"], span=20)
    df["ema50"] = compute_ema(df["close"], span=50)
    df["atr14"] = compute_atr(df, 14)

    upper, middle, lower = compute_bollinger_bands(df["close"], period=20, std_dev=2)
    df["bb_upper"] = upper
    df["bb_middle"] = middle
    df["bb_lower"] = lower

    df["rsi14"] = compute_rsi(df["close"], 14)

    macd_line, signal_line, hist = compute_macd(df["close"])
    df["macd"] = macd_line
    df["macd_signal"] = signal_line
    df["macd_hist"] = hist

    k, d = compute_stochastic(df)
    df["stoch_k"] = k
    df["stoch_d"] = d

    df["obv"] = compute_obv(df)
    df["vwap"] = compute_vwap(df)
    return df

def scale_features(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """Normalize feature columns between 0 and 1. Skips missing columns gracefully."""
    for col in feature_cols:
        if col not in df.columns:
            continue
        min_val = df[col].min()
        max_val = df[col].max()
        if pd.notna(max_val) and pd.notna(min_val) and max_val > min_val:
            df[col] = (df[col] - min_val) / (max_val - min_val)
        else:
            df[col] = 0.0
    return df
