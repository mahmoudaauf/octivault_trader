"""
Native market regime detector.
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
        closes = self._collect_reference_closes()
        volatility_regime = "NORMAL"
        trend_regime = "RANGING"

        if len(closes) >= 10:
            returns = []
            for i in range(1, len(closes)):
                prev = closes[i - 1]
                cur = closes[i]
                if prev > 0:
                    returns.append((cur - prev) / prev)
            vol = self._stddev(returns[-20:]) if returns else 0.0
            if vol >= 0.03:
                volatility_regime = "HIGH"
            elif vol <= 0.003:
                volatility_regime = "LOW"

            fast = sum(closes[-5:]) / min(5, len(closes))
            slow = sum(closes[-20:]) / min(20, len(closes))
            if slow > 0:
                gap = (fast / slow) - 1.0
                if gap >= 0.01:
                    trend_regime = "UPTREND"
                elif gap <= -0.01:
                    trend_regime = "DOWNTREND"

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

    def _collect_reference_closes(self) -> list[float]:
        klines_cache = getattr(self._market_data, "_klines", {}) if self._market_data else {}
        best: list[float] = []
        for _key, (_ts, data) in klines_cache.items():
            closes = []
            for row in data or []:
                try:
                    closes.append(float(row[4]))
                except (IndexError, TypeError, ValueError):
                    continue
            if len(closes) > len(best):
                best = closes
        return best

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
