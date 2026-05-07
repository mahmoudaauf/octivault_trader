"""
Native regime gate.

Lightweight market-condition filter inspired by the legacy regime and
arbitration layers. This gate is intentionally simple: it consumes signal-side
metadata when present and decides whether opening a new position is sensible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RegimeDecision:
    allowed: bool
    confidence_floor_bump: float = 0.0
    reason: str = ""


class NativeRegimeGate:
    def evaluate(self, signal: dict[str, Any]) -> RegimeDecision:
        regime = str(signal.get("regime", signal.get("market_regime", "")) or "").lower()
        liquidity_score = float(signal.get("liquidity_score", 1.0) or 0.0)
        volatility_score = float(signal.get("volatility_score", 0.5) or 0.0)
        spread_pct = float(signal.get("spread_pct", 0.0) or 0.0)

        if regime in {"crisis", "halted", "low_liquidity"}:
            return RegimeDecision(False, reason=f"regime_blocked:{regime}")
        if spread_pct > 0.01:
            return RegimeDecision(False, reason="spread_too_wide")
        if liquidity_score < 0.15:
            return RegimeDecision(False, reason="liquidity_too_low")
        if regime == "volatile" or volatility_score >= 0.85:
            return RegimeDecision(True, confidence_floor_bump=0.10, reason="volatile_guard")
        if regime == "range":
            return RegimeDecision(True, confidence_floor_bump=0.05, reason="range_guard")
        return RegimeDecision(True, confidence_floor_bump=0.0, reason="regime_ok")
