"""
Native L3: Signal Engine (Phase 8.2.4)

Pure-numpy technical signal generation. Replaces ~1500 LOC legacy
``signal_engine/`` with a focused single-file plugin registry.

Design choices
--------------
* Pure functions for indicators (RSI, MACD, MA crossover) — no pandas.
* Strategy plugins are simple callables: ``(closes: np.ndarray) → Signal | None``.
* Aggregation: weighted average of per-strategy scores; conviction in
  [0, 1].
* Hysteresis / cooldown: a symbol that fired recently is suppressed for
  ``cooldown_sec`` seconds to prevent flip-flopping.
* No I/O. Caller supplies klines (typically from L2 NativeMarketData).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

Direction = str  # "BUY" | "SELL" | "HOLD"

# ─────────────────────────────────────────────────────────────────────
# Public types
# ─────────────────────────────────────────────────────────────────────


@dataclass
class Signal:
    """Single-strategy signal."""

    symbol: str
    direction: Direction  # "BUY" | "SELL" | "HOLD"
    score: float  # conviction in [0, 1]
    strategy: str
    meta: dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)


@dataclass
class AggregatedSignal:
    """Per-symbol aggregate of all strategy outputs."""

    symbol: str
    direction: Direction
    score: float
    contributions: list[Signal] = field(default_factory=list)
    ts: float = field(default_factory=time.time)


StrategyFn = Callable[[np.ndarray], Optional[Signal]]


# ─────────────────────────────────────────────────────────────────────
# Indicators (pure numpy)
# ─────────────────────────────────────────────────────────────────────
def rsi(closes: np.ndarray, period: int = 14) -> Optional[float]:
    """
    Wilder's RSI of the most recent bar. Returns None if insufficient data.
    """
    if closes.size < period + 1:
        return None
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    # Wilder's smoothing (recursive). Use mean of first ``period`` as seed.
    avg_gain = float(gains[:period].mean())
    avg_loss = float(losses[:period].mean())
    for i in range(period, deltas.size):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return float(100.0 - (100.0 / (1.0 + rs)))


def _ema(values: np.ndarray, period: int) -> np.ndarray:
    """Exponential moving average. Length matches input."""
    if values.size == 0:
        return values
    alpha = 2.0 / (period + 1.0)
    out = np.empty_like(values, dtype=np.float64)
    out[0] = values[0]
    for i in range(1, values.size):
        out[i] = alpha * values[i] + (1.0 - alpha) * out[i - 1]
    return out


def macd(
    closes: np.ndarray,
    fast: int = 12,
    slow: int = 26,
    signal_period: int = 9,
) -> Optional[tuple[float, float, float]]:
    """
    Returns ``(macd_line, signal_line, histogram)`` for the most recent bar,
    or None if insufficient data.
    """
    if closes.size < slow + signal_period:
        return None
    fast_ema = _ema(closes, fast)
    slow_ema = _ema(closes, slow)
    macd_line = fast_ema - slow_ema
    sig_line = _ema(macd_line, signal_period)
    hist = macd_line - sig_line
    return float(macd_line[-1]), float(sig_line[-1]), float(hist[-1])


def ma_crossover(
    closes: np.ndarray, fast: int = 10, slow: int = 30
) -> Optional[tuple[float, float, int]]:
    """
    Returns ``(fast_ma, slow_ma, cross)`` where ``cross`` ∈ {-1, 0, +1}:
    +1 fast crossed above slow on most recent bar, -1 below, 0 no cross.
    None if insufficient data.
    """
    if closes.size < slow + 1:
        return None
    fast_ma_now = float(closes[-fast:].mean())
    slow_ma_now = float(closes[-slow:].mean())
    fast_ma_prev = float(closes[-fast - 1 : -1].mean())
    slow_ma_prev = float(closes[-slow - 1 : -1].mean())

    cross = 0
    if fast_ma_prev <= slow_ma_prev and fast_ma_now > slow_ma_now:
        cross = 1
    elif fast_ma_prev >= slow_ma_prev and fast_ma_now < slow_ma_now:
        cross = -1
    return fast_ma_now, slow_ma_now, cross


# ─────────────────────────────────────────────────────────────────────
# Built-in strategies
# ─────────────────────────────────────────────────────────────────────
def strategy_rsi(closes: np.ndarray, *, symbol: str = "") -> Optional[Signal]:
    val = rsi(closes, period=14)
    if val is None:
        return None
    if val <= 30:
        # 30 → 0.5, 0 → 1.0
        score = min(1.0, (30.0 - val) / 30.0 + 0.5)
        return Signal(symbol, "BUY", score, "rsi", {"rsi": val})
    if val >= 70:
        score = min(1.0, (val - 70.0) / 30.0 + 0.5)
        return Signal(symbol, "SELL", score, "rsi", {"rsi": val})
    return Signal(symbol, "HOLD", 0.0, "rsi", {"rsi": val})


def strategy_macd(closes: np.ndarray, *, symbol: str = "") -> Optional[Signal]:
    res = macd(closes)
    if res is None:
        return None
    macd_line, sig_line, hist = res
    # Score scaled by histogram magnitude relative to recent price.
    px = float(closes[-1]) if closes.size else 1.0
    norm = abs(hist) / max(px * 0.001, 1e-9)  # 0.1% of price = score 1.0
    score = float(min(1.0, norm))
    if hist > 0 and macd_line > sig_line:
        return Signal(symbol, "BUY", score, "macd", {"hist": hist})
    if hist < 0 and macd_line < sig_line:
        return Signal(symbol, "SELL", score, "macd", {"hist": hist})
    return Signal(symbol, "HOLD", 0.0, "macd", {"hist": hist})


def strategy_ma_crossover(closes: np.ndarray, *, symbol: str = "") -> Optional[Signal]:
    res = ma_crossover(closes)
    if res is None:
        return None
    fast_ma, slow_ma, cross = res
    if cross == 1:
        return Signal(symbol, "BUY", 0.7, "ma_cross", {"fast": fast_ma, "slow": slow_ma})
    if cross == -1:
        return Signal(symbol, "SELL", 0.7, "ma_cross", {"fast": fast_ma, "slow": slow_ma})
    return Signal(symbol, "HOLD", 0.0, "ma_cross", {"fast": fast_ma, "slow": slow_ma})


BUILTIN_STRATEGIES: dict[str, Callable[..., Optional[Signal]]] = {
    "rsi": strategy_rsi,
    "macd": strategy_macd,
    "ma_cross": strategy_ma_crossover,
}


# ─────────────────────────────────────────────────────────────────────
# Engine
# ─────────────────────────────────────────────────────────────────────
class NativeSignalEngine:
    """
    Strategy plugin registry + aggregator with cooldown.

    Usage::

        eng = NativeSignalEngine()  # all builtins enabled
        agg = eng.evaluate("BTCUSDT", klines)
        # ⇒ AggregatedSignal | None (None when no strategy produced output)
    """

    def __init__(
        self,
        *,
        weights: Optional[dict[str, float]] = None,
        enabled: Optional[list[str]] = None,
        cooldown_sec: float = 0.0,
    ) -> None:
        self._strategies: dict[str, Callable[..., Optional[Signal]]] = dict(BUILTIN_STRATEGIES)
        self._weights: dict[str, float] = dict(weights or {})
        self._enabled: set[str] = set(enabled if enabled is not None else self._strategies.keys())
        self._cooldown_sec = max(0.0, float(cooldown_sec))
        self._last_fired: dict[str, float] = {}  # symbol → ts

    # ──────────────────────────────────────────────────────────────────
    # Registry
    # ──────────────────────────────────────────────────────────────────
    def register_strategy(
        self,
        name: str,
        fn: Callable[..., Optional[Signal]],
        *,
        weight: float = 1.0,
        enabled: bool = True,
    ) -> None:
        self._strategies[name] = fn
        self._weights[name] = float(weight)
        if enabled:
            self._enabled.add(name)
        else:
            self._enabled.discard(name)

    def enable(self, name: str) -> None:
        if name not in self._strategies:
            raise KeyError(f"unknown strategy: {name}")
        self._enabled.add(name)

    def disable(self, name: str) -> None:
        self._enabled.discard(name)

    @property
    def enabled_strategies(self) -> list[str]:
        return sorted(self._enabled)

    # ──────────────────────────────────────────────────────────────────
    # Evaluation
    # ──────────────────────────────────────────────────────────────────
    def evaluate(self, symbol: str, klines: list[list[Any]]) -> Optional[AggregatedSignal]:
        """
        Evaluate all enabled strategies on the given klines.

        ``klines`` is the Binance kline format: list of rows where
        index 4 is the close price. Returns ``None`` when in cooldown
        or when no strategy produced a directional signal.
        """
        # Cooldown gate
        if self._cooldown_sec > 0:
            last = self._last_fired.get(symbol)
            if last is not None and (time.time() - last) < self._cooldown_sec:
                return None

        closes = self._extract_closes(klines)
        if closes.size == 0:
            return None

        contributions: list[Signal] = []
        for name in self._enabled:
            fn = self._strategies.get(name)
            if fn is None:
                continue
            try:
                sig = fn(closes, symbol=symbol)
            except Exception as e:  # pragma: no cover — defensive
                logger.exception("strategy %s failed for %s: %s", name, symbol, e)
                continue
            if sig is None:
                continue
            contributions.append(sig)

        if not contributions:
            return None

        agg = self._aggregate(symbol, contributions)
        if agg.direction != "HOLD":
            self._last_fired[symbol] = time.time()
        return agg

    def evaluate_many(
        self, klines_by_symbol: dict[str, list[list[Any]]]
    ) -> dict[str, AggregatedSignal]:
        """Convenience: evaluate each symbol; HOLD/None outputs are dropped."""
        out: dict[str, AggregatedSignal] = {}
        for sym, kl in klines_by_symbol.items():
            agg = self.evaluate(sym, kl)
            if agg is not None and agg.direction != "HOLD":
                out[sym] = agg
        return out

    # ──────────────────────────────────────────────────────────────────
    # Internals
    # ──────────────────────────────────────────────────────────────────
    @staticmethod
    def _extract_closes(klines: list[list[Any]]) -> np.ndarray:
        if not klines:
            return np.array([], dtype=np.float64)
        try:
            return np.asarray([float(row[4]) for row in klines], dtype=np.float64)
        except (IndexError, TypeError, ValueError):
            return np.array([], dtype=np.float64)

    def _aggregate(self, symbol: str, sigs: list[Signal]) -> AggregatedSignal:
        # Weighted vote: BUY = +score, SELL = -score, HOLD ignored.
        total_w = 0.0
        net = 0.0
        for s in sigs:
            w = float(self._weights.get(s.strategy, 1.0))
            total_w += w
            if s.direction == "BUY":
                net += w * s.score
            elif s.direction == "SELL":
                net -= w * s.score
        if total_w == 0:
            return AggregatedSignal(symbol, "HOLD", 0.0, sigs)

        normalized = net / total_w  # in [-1, +1]
        if normalized > 1e-9:
            return AggregatedSignal(symbol, "BUY", float(min(1.0, normalized)), sigs)
        if normalized < -1e-9:
            return AggregatedSignal(symbol, "SELL", float(min(1.0, -normalized)), sigs)
        return AggregatedSignal(symbol, "HOLD", 0.0, sigs)

    # ──────────────────────────────────────────────────────────────────
    # Facade adapter methods (for compatibility with SituationEngine)
    # ──────────────────────────────────────────────────────────────────
    def get_all_signals(self) -> list[dict[str, Any]]:
        """
        Evaluate all symbols and return signals.
        **This requires market_data to be available in app_ctx.**

        For use by SituationEngine via the implementations bridge.
        Returns a list of signal dicts with: symbol, signal_type, edge_score.
        """
        # This is stateless — we have no klines cached. Real use is through
        # evaluate_with_market_data() which the facade should call.
        return []

    def get_signals(self, symbol: Optional[str] = None) -> list[dict[str, Any]]:
        """Alias for get_all_signals() for compatibility."""
        return self.get_all_signals()

    def evaluate_with_market_data(
        self, market_data: Any, symbols: Optional[list[str]] = None
    ) -> list[dict[str, Any]]:
        """
        Evaluate signals for all symbols using market_data._klines cache.

        Args:
            market_data: NativeMarketData instance with _klines cache
            symbols: Symbols to evaluate (default: auto-detect from cache)

        Returns:
            List of signal dicts: {symbol, signal_type, edge_score}
        """
        signals = []

        if not market_data:
            logger.debug("❌ evaluate_with_market_data: no market_data provided")
            return signals

        # Get the internal klines cache (OrderedDict with key=(symbol, interval, limit))
        klines_cache = getattr(market_data, "_klines", {})
        if not klines_cache:
            logger.debug("❌ evaluate_with_market_data: _klines cache is empty or missing")
            return signals

        # Collect unique symbols from cache keys
        cache_symbols: set[str] = set()
        for key in klines_cache:
            if isinstance(key, tuple) and len(key) >= 1:
                cache_symbols.add(key[0])

        eval_symbols = symbols or list(cache_symbols)
        if not eval_symbols:
            return signals

        for symbol in eval_symbols:
            # Find klines for this symbol (prefer the most complete cache entry)
            klines = None
            best_key = None
            best_size = 0

            for key, (_ts, data) in klines_cache.items():
                if isinstance(key, tuple) and key[0] == symbol:
                    data_size = len(data) if isinstance(data, (list, tuple)) else 0
                    if data_size > best_size:
                        klines = data
                        best_key = key
                        best_size = data_size

            if not klines or len(klines) < 14:  # Need at least 14 bars for RSI
                continue

            # Evaluate this symbol
            agg = self.evaluate(symbol, klines)
            if agg is None or agg.direction == "HOLD":
                continue

            # Convert to dict for facade
            signals.append(
                {
                    "symbol": agg.symbol,
                    "signal_type": agg.direction,  # "BUY" or "SELL"
                    "edge_score": float(agg.score),
                    "timestamp": agg.ts,
                }
            )

        logger.info(
            f"✅ evaluate_with_market_data: found {len(signals)} signals from {len(eval_symbols)} symbols"
        )
        return signals
