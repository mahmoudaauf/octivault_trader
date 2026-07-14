"""
Native L0: Math Utilities — Performance Metrics & Statistical Functions

Provides:
- Sharpe ratio (risk-adjusted return)
- Sortino ratio (downside risk only)
- Calmar ratio (return / max drawdown)
- Max drawdown calculation
- Cumulative returns

All functions operate on returns arrays (daily/cycle returns as %),
designed for portfolio performance analysis.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def sharpe_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252,
) -> float:
    """
    Calculate Sharpe ratio (return / volatility).

    Args:
        returns: Array of period returns (as decimals, e.g., 0.01 for 1%)
        risk_free_rate: Annual risk-free rate (default 2%)
        periods_per_year: Number of periods per year (252 for daily, 365 for hourly trades)

    Returns:
        Sharpe ratio (higher is better, >1.0 is good, >2.0 is excellent)

    Example:
        >>> daily_returns = np.array([0.01, -0.005, 0.02, 0.003])
        >>> sr = sharpe_ratio(daily_returns)
    """
    if len(returns) < 2:
        return 0.0

    excess_returns = returns - (risk_free_rate / periods_per_year)
    return float(
        np.mean(excess_returns) / (np.std(excess_returns) + 1e-8) * np.sqrt(periods_per_year)
    )


def sortino_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252,
) -> float:
    """
    Calculate Sortino ratio (return / downside volatility).

    Only penalizes downside volatility, not upside.

    Args:
        returns: Array of period returns (as decimals)
        risk_free_rate: Annual risk-free rate
        periods_per_year: Periods per year

    Returns:
        Sortino ratio (higher is better)

    Example:
        >>> daily_returns = np.array([0.01, -0.005, 0.02, 0.003])
        >>> sr = sortino_ratio(daily_returns)
    """
    if len(returns) < 2:
        return 0.0

    excess_returns = returns - (risk_free_rate / periods_per_year)
    downside = np.where(excess_returns < 0, excess_returns, 0)
    downside_std = float(np.std(downside) + 1e-8)

    return float(np.mean(excess_returns) / downside_std * np.sqrt(periods_per_year))


def calmar_ratio(
    returns: np.ndarray,
    periods_per_year: int = 252,
) -> float:
    """
    Calculate Calmar ratio (annual return / max drawdown).

    Captures return per unit of maximum loss experienced.

    Args:
        returns: Array of period returns (as decimals)
        periods_per_year: Periods per year

    Returns:
        Calmar ratio (higher is better, >0.5 is good)

    Example:
        >>> daily_returns = np.array([0.01, -0.005, 0.02, 0.003])
        >>> cr = calmar_ratio(daily_returns)
    """
    if len(returns) < 2:
        return 0.0

    annual_return = float(np.mean(returns) * periods_per_year)
    max_dd = max_drawdown(returns)

    if max_dd >= 0:  # No drawdown or NaN
        return 0.0

    return float(annual_return / abs(max_dd))


def max_drawdown(returns: np.ndarray) -> float:
    """
    Calculate maximum drawdown (worst peak-to-trough decline).

    Args:
        returns: Array of period returns (as decimals)

    Returns:
        Max drawdown as decimal (e.g., -0.15 for 15% drawdown)

    Example:
        >>> daily_returns = np.array([0.01, -0.005, 0.02, -0.05, 0.01])
        >>> dd = max_drawdown(daily_returns)
    """
    if len(returns) < 2:
        return 0.0

    cumulative = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max

    return float(np.min(drawdown))


def cumulative_returns(returns: np.ndarray) -> float:
    """
    Calculate cumulative return from a series of period returns.

    Args:
        returns: Array of period returns (as decimals)

    Returns:
        Total cumulative return (e.g., 0.25 for +25%)

    Example:
        >>> daily_returns = np.array([0.01, -0.005, 0.02, 0.003])
        >>> cum_ret = cumulative_returns(daily_returns)
    """
    if len(returns) == 0:
        return 0.0

    return float(np.prod(1 + returns) - 1)


def win_rate(returns: np.ndarray) -> float:
    """
    Calculate win rate (% of positive periods).

    Args:
        returns: Array of period returns (as decimals)

    Returns:
        Win rate as decimal (e.g., 0.55 for 55%)

    Example:
        >>> daily_returns = np.array([0.01, -0.005, 0.02, 0.003])
        >>> wr = win_rate(daily_returns)
    """
    if len(returns) == 0:
        return 0.0

    return float(np.sum(returns > 0) / len(returns))


def profit_factor(returns: np.ndarray) -> float:
    """
    Calculate profit factor (gross wins / gross losses).

    Args:
        returns: Array of period returns (as decimals)

    Returns:
        Profit factor (>1.5 is good, >2.0 is excellent)

    Example:
        >>> daily_returns = np.array([0.01, -0.005, 0.02, -0.01])
        >>> pf = profit_factor(daily_returns)
    """
    wins = np.sum(np.where(returns > 0, returns, 0))
    losses = np.sum(np.abs(np.where(returns < 0, returns, 0)))

    if losses == 0:
        return 0.0 if wins == 0 else float("inf")

    return float(wins / losses)


def volatility(returns: np.ndarray, periods_per_year: int = 252) -> float:
    """
    Calculate annualized volatility.

    Args:
        returns: Array of period returns (as decimals)
        periods_per_year: Periods per year

    Returns:
        Annualized volatility (e.g., 0.15 for 15%)

    Example:
        >>> daily_returns = np.array([0.01, -0.005, 0.02, 0.003])
        >>> vol = volatility(daily_returns)
    """
    if len(returns) < 2:
        return 0.0

    return float(np.std(returns) * np.sqrt(periods_per_year))


def compute_atr_from_candles(candles: list, lookback: int = 14) -> float:
    """Compute ATR (average true range, absolute price units) from candlestick data.

    Handles both formats:
      - Dict:  {"high": h, "low": l, "close": c}  (WebSocket live candles)
      - List:  [ts, open, high, low, close, ...]    (legacy kline format)

    ATR = SMA(TR) where TR = max(H-L, abs(H-PC), abs(L-PC))

    Shared by NativeTPSLEngine (TP/SL sizing) and NativeCapitalAllocator
    (position-sizing volatility input) — moved here rather than duplicated
    since ATR computation is real, non-trivial logic worth keeping in sync.
    """
    try:
        if len(candles) < 2:
            return 0.0

        true_ranges = []
        prev_close = None

        for candle in candles[-lookback:]:
            if isinstance(candle, dict):
                high = float(candle.get("high") or candle.get("h") or 0.0)
                low = float(candle.get("low") or candle.get("l") or 0.0)
                close = float(candle.get("close") or candle.get("c") or 0.0)
            elif isinstance(candle, (list, tuple)) and len(candle) >= 5:
                high = float(candle[2] or 0.0)
                low = float(candle[3] or 0.0)
                close = float(candle[4] or 0.0)
            else:
                continue

            if high <= 0 or low <= 0 or close <= 0:
                continue

            if prev_close is None:
                prev_close = close
                continue

            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            true_ranges.append(tr)
            prev_close = close

        if not true_ranges:
            return 0.0

        return float(sum(true_ranges) / len(true_ranges))

    except Exception as e:
        logger.error("compute_atr_from_candles failed: %s", e)
        return 0.0


def compute_atr(shared_state: Any, symbol: str, lookback: int = 14) -> float:
    """Compute ATR(lookback) (absolute price units) for ``symbol`` from shared_state.

    Strategy:
      1. Try live candles from WebSocket: market_data[(symbol, tf)] (tuple key, dict format)
      2. Try cached ATR scalar: market_data[symbol]["atr"] (legacy format)
      3. Try klines attribute: klines[symbol]["1m"] (legacy format)
      4. Fallback to 0.8% of last price
    """
    try:
        market_data = getattr(shared_state, "market_data", {}) or {}

        for tf in ("1m", "5m", "15m"):
            candles = market_data.get((symbol, tf))
            if isinstance(candles, list) and len(candles) >= max(lookback, 3):
                atr = compute_atr_from_candles(candles, min(lookback, len(candles)))
                if atr > 0:
                    return atr

        sym_md = market_data.get(symbol)
        if isinstance(sym_md, dict):
            cached_atr = float(sym_md.get("atr") or 0.0)
            if cached_atr > 0:
                return cached_atr

        klines = getattr(shared_state, "klines", {}) or {}
        if symbol in klines:
            candles = klines.get(symbol, {}).get("1m", [])
            if isinstance(candles, list) and len(candles) >= 3:
                atr = compute_atr_from_candles(candles, min(lookback, len(candles)))
                if atr > 0:
                    return atr

        prices = getattr(shared_state, "prices", {}) or {}
        if symbol in prices:
            last_price = float(prices[symbol] or 0.0)
            if last_price > 0:
                return last_price * 0.008

        logger.warning("compute_atr: %s no data for ATR, returning 0", symbol)
        return 0.0

    except Exception as e:
        logger.error("compute_atr failed for %s: %s", symbol, e)
        return 0.0


def compute_atr_pct(shared_state: Any, symbol: str, lookback: int = 14) -> float:
    """ATR normalized to a fraction of the current price (e.g. 0.008 = 0.8%).

    Returns 0.0 if no ATR or no current price is available — callers should
    apply their own floor (e.g. MIN_ATR_PCT) rather than treat 0.0 as "flat."
    """
    atr = compute_atr(shared_state, symbol, lookback)
    if atr <= 0:
        return 0.0
    prices = getattr(shared_state, "prices", {}) or {}
    price = float(prices.get(symbol, 0.0) or 0.0)
    if price <= 0:
        return 0.0
    return atr / price


__all__ = [
    "sharpe_ratio",
    "sortino_ratio",
    "calmar_ratio",
    "max_drawdown",
    "cumulative_returns",
    "win_rate",
    "profit_factor",
    "volatility",
    "compute_atr_from_candles",
    "compute_atr",
    "compute_atr_pct",
]
