"""
Native market regime detector.

Per-symbol regime classification. Each symbol gets its own trend score
so a flat BTC doesn't block a trending SOL.

Thresholds:
  gap >= +0.30%  → UPTREND
  gap <= -0.30%  → DOWNTREND
  0.10% < |gap| < 0.30% → CHOPPY (momentum bursts possible)
  |gap| <= 0.10% → RANGING (no directional edge)
"""

from __future__ import annotations

import math
from typing import Any


class NativeMarketRegimeDetector:
    def __init__(
        self,
        *,
        market_data: Any | None = None,
        shared_state: Any | None = None,
        signal_engine: Any | None = None,
    ) -> None:
        self._market_data = market_data
        self._shared_state = shared_state
        self._signal_engine = signal_engine

    async def get_regime(self) -> dict[str, str]:
        """Return overall regime using the best-trending symbol, not just BTC."""
        klines_cache = getattr(self._market_data, "_klines", {}) if self._market_data else {}

        volatility_regime = "NORMAL"
        symbol_regimes: dict[str, str] = {}
        all_returns: list[float] = []

        # REST klines cache
        for key, (_ts, data) in klines_cache.items():
            if not data or len(data) < 10:
                continue
            symbol = key[0] if isinstance(key, tuple) else "UNKNOWN"
            closes = self._extract_closes(data)
            if len(closes) < 10:
                continue
            for i in range(1, len(closes)):
                if closes[i - 1] > 0:
                    all_returns.append((closes[i] - closes[i - 1]) / closes[i - 1])
            symbol_regimes[symbol] = self._classify_trend(closes)

        # WebSocket klines — shared_state.market_data[(symbol, tf)]
        if self._shared_state is not None:
            ws_md = getattr(self._shared_state, "market_data", {}) or {}
            for (sym, _tf), data in ws_md.items() if isinstance(ws_md, dict) else []:
                if sym in symbol_regimes or not isinstance(data, list) or len(data) < 10:
                    continue
                closes = []
                for c in data[-25:]:
                    try:
                        if isinstance(c, dict):
                            closes.append(float(c.get("close") or c.get("c") or 0))
                        elif isinstance(c, (list, tuple)) and len(c) >= 5:
                            closes.append(float(c[4]))
                    except (ValueError, TypeError):
                        continue
                closes = [v for v in closes if v > 0]
                if len(closes) < 10:
                    continue
                for i in range(1, len(closes)):
                    if closes[i - 1] > 0:
                        all_returns.append((closes[i] - closes[i - 1]) / closes[i - 1])
                symbol_regimes[sym] = self._classify_trend(closes)

        # Volatility from aggregated returns
        if all_returns:
            vol = self._stddev(all_returns[-40:])
            if vol >= 0.03:
                volatility_regime = "HIGH"
            elif vol <= 0.003:
                volatility_regime = "LOW"

        # Overall regime = best across all symbols
        # Priority: UPTREND > CHOPPY > DOWNTREND > RANGING
        trend_regime = self._aggregate_regime(symbol_regimes)

        # Store per-symbol for debug
        if self._shared_state and hasattr(self._shared_state, "metrics"):
            self._shared_state.metrics["symbol_regimes"] = symbol_regimes

        nav_regime = self._nav_regime()
        overall_health = "OK"
        if bool(getattr(self._shared_state, "trading_halted", False)):
            overall_health = "CRISIS"
        elif (
            bool(getattr(self._shared_state, "exchange_throttled", False))
            or volatility_regime == "HIGH"
        ):
            overall_health = "WARN"

        return {
            "volatility_regime": volatility_regime,
            "trend_regime": trend_regime,
            "nav_regime": nav_regime,
            "overall_health": overall_health,
        }

    def get_symbol_regime(self, symbol: str) -> str:
        """Per-symbol regime for gate_3 to use instead of global regime."""
        # Primary: REST klines cache (keyed by (symbol, interval, limit))
        klines_cache = getattr(self._market_data, "_klines", {}) if self._market_data else {}
        for key, (_ts, data) in klines_cache.items():
            if isinstance(key, tuple) and key[0] == symbol and data and len(data) >= 10:
                closes = self._extract_closes(data)
                if len(closes) >= 10:
                    return self._classify_trend(closes)
        # Secondary: WebSocket klines stored in shared_state.market_data[(symbol, tf)]
        # This is the live path when polling is disabled and WebSocket is active.
        if self._shared_state is not None:
            ws_md = getattr(self._shared_state, "market_data", {}) or {}
            for tf in ("1m", "3m", "5m"):
                data = ws_md.get((symbol, tf)) or ws_md.get(symbol, [])
                if isinstance(data, list) and len(data) >= 10:
                    closes = []
                    for c in data[-25:]:
                        try:
                            if isinstance(c, dict):
                                closes.append(float(c.get("close") or c.get("c") or 0))
                            elif isinstance(c, (list, tuple)) and len(c) >= 5:
                                closes.append(float(c[4]))
                        except (ValueError, TypeError):
                            continue
                    closes = [v for v in closes if v > 0]
                    if len(closes) >= 10:
                        return self._classify_trend(closes)
        # Last resort: cached metrics
        stored = {}
        if self._shared_state and hasattr(self._shared_state, "metrics"):
            stored = self._shared_state.metrics.get("symbol_regimes", {})
        return stored.get(symbol, "RANGING")

    # ── Internal ─────────────────────────────────────────────────────

    @staticmethod
    def _extract_closes(data: list) -> list[float]:
        closes = []
        for row in data or []:
            try:
                closes.append(float(row[4]))
            except (IndexError, TypeError, ValueError):
                continue
        return closes

    @staticmethod
    def _classify_trend(closes: list[float]) -> str:
        """
        MA5 vs MA20 gap classification.
          gap >= +0.30%  → UPTREND
          gap <= -0.30%  → DOWNTREND
          |gap| 0.10-0.30% → CHOPPY
          |gap| <= 0.10% → RANGING
        """
        n = len(closes)
        fast = sum(closes[-5:]) / min(5, n)
        slow = sum(closes[-20:]) / min(20, n)
        if slow <= 0:
            return "RANGING"
        gap = (fast / slow) - 1.0
        if gap >= 0.003:  # +0.30%
            return "UPTREND"
        if gap <= -0.003:  # -0.30%
            return "DOWNTREND"
        if abs(gap) >= 0.001:  # 0.10-0.30%
            return "CHOPPY"
        return "RANGING"

    @staticmethod
    def _aggregate_regime(symbol_regimes: dict[str, str]) -> str:
        """Pick the most favourable regime across all symbols."""
        if not symbol_regimes:
            return "RANGING"
        counts = {"UPTREND": 0, "CHOPPY": 0, "DOWNTREND": 0, "RANGING": 0}
        for r in symbol_regimes.values():
            counts[r] = counts.get(r, 0) + 1
        # If any symbol is trending up → trade it
        if counts["UPTREND"] >= 1:
            return "UPTREND"
        # If majority are choppy → CHOPPY (tradeable with floor bump)
        if counts["CHOPPY"] >= max(1, len(symbol_regimes) // 3):
            return "CHOPPY"
        if counts["DOWNTREND"] >= 1:
            return "DOWNTREND"
        return "RANGING"

    def _nav_regime(self) -> str:
        nav = float(getattr(self._shared_state, "nav_usdt", 0.0) or 0.0)
        peak = float(getattr(self._shared_state, "metrics", {}).get("peak_nav", 0.0) or 0.0)
        anchor = float(getattr(self._shared_state, "session_anchor_nav", 0.0) or 0.0)
        baseline = peak or anchor or nav
        if baseline <= 0:
            return "GROWTH"
        return "GROWTH" if nav >= baseline * 0.98 else "DECAY"

    @staticmethod
    def _stddev(values: list[float]) -> float:
        if not values:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        return math.sqrt(max(variance, 0.0))


__all__ = ["NativeMarketRegimeDetector"]
