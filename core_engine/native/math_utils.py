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

import numpy as np


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


__all__ = [
    "sharpe_ratio",
    "sortino_ratio",
    "calmar_ratio",
    "max_drawdown",
    "cumulative_returns",
    "win_rate",
    "profit_factor",
    "volatility",
]
