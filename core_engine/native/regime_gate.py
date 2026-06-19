"""
Native regime gate.

Lightweight market-condition filter inspired by the legacy regime and
arbitration layers. This gate is intentionally simple: it consumes signal-side
metadata when present and decides whether opening a new position is sensible.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

_log = logging.getLogger(__name__)

# Order-book / flow tuning (the supply/demand layer that leads indicators).
# Spread guard: block entries when the real bid/ask spread is too wide (bad fills).
_SPREAD_MAX_PCT = float(os.getenv("ORDERBOOK_SPREAD_MAX_PCT", "0.005") or 0.005)
# Imbalance veto: bid_qty/(bid_qty+ask_qty) below this = supply-heavy (ask wall);
# vetoing a BUY into a wall. OBSERVE-ONLY unless the enable flag is set, so it
# can't disturb the live edge sample until we've watched it behave.
_IMBALANCE_BUY_MIN = float(os.getenv("ORDERBOOK_IMBALANCE_BUY_MIN", "0.35") or 0.35)
_IMBALANCE_VETO_ENABLED = str(
    os.getenv("ORDERBOOK_IMBALANCE_VETO_ENABLED", "false")
).strip().lower() in {"1", "true", "yes", "on"}
# Only trust book-derived guards when the snapshot is fresh (stale book = ignore).
_BOOK_MAX_AGE_S = float(os.getenv("ORDERBOOK_MAX_AGE_S", "10.0") or 10.0)


@dataclass(frozen=True)
class RegimeDecision:
    allowed: bool
    confidence_floor_bump: float = 0.0
    reason: str = ""


class NativeRegimeGate:
    def __init__(self, shared_state: Any = None, **kwargs: Any) -> None:
        self._shared_state = shared_state

    def evaluate(self, signal: dict[str, Any]) -> RegimeDecision:
        regime = str(signal.get("regime", signal.get("market_regime", "")) or "").lower()
        liquidity_score = float(signal.get("liquidity_score", 1.0) or 0.0)
        volatility_score = float(signal.get("volatility_score", 0.5) or 0.0)
        spread_pct = float(signal.get("spread_pct", 0.0) or 0.0)

        # --- Real-time order-book layer (overrides the always-0.0 signal spread) ---
        symbol = str(signal.get("symbol", "") or "")
        side = str(signal.get("signal_type", signal.get("direction", "")) or "").upper()
        ss = self._shared_state
        if symbol and ss is not None and hasattr(ss, "get_book"):
            try:
                if float(ss.get_book_age(symbol)) <= _BOOK_MAX_AGE_S:
                    real_spread = float(ss.get_spread_pct(symbol))
                    if real_spread > 0:
                        spread_pct = real_spread
                    imbalance = float(ss.get_book_imbalance(symbol))
                    # Imbalance veto (observe-only by default): buying into an ask wall.
                    if side == "BUY" and imbalance < _IMBALANCE_BUY_MIN:
                        if _IMBALANCE_VETO_ENABLED:
                            return RegimeDecision(
                                False, reason=f"book_imbalance_veto:{imbalance:.2f}"
                            )
                        _log.info(
                            "[orderbook] %s WOULD veto BUY: imbalance=%.2f < %.2f "
                            "(ask-heavy) [observe-only]",
                            symbol, imbalance, _IMBALANCE_BUY_MIN,
                        )
            except Exception as e:  # never let the book layer break the gate
                _log.debug("[orderbook] book read failed for %s: %s", symbol, e)

        if regime in {"crisis", "halted", "low_liquidity"}:
            return RegimeDecision(False, reason=f"regime_blocked:{regime}")
        if spread_pct > _SPREAD_MAX_PCT:
            return RegimeDecision(False, reason=f"spread_too_wide:{spread_pct:.4f}")
        if liquidity_score < 0.15:
            return RegimeDecision(False, reason="liquidity_too_low")

        # --- Momentum regime gates ---
        # RANGING: price is flat (MA gap ≤ 0.30%) — momentum signals are noise.
        # Block ALL new BUY entries. Allow SELL to exit existing positions.
        if regime == "range":
            if side == "BUY":
                return RegimeDecision(False, reason="regime_ranging:no_momentum_for_buy")
            # SELL is fine — let existing positions exit
            return RegimeDecision(True, confidence_floor_bump=0.0, reason="regime_ranging:sell_allowed")

        # CHOPPY: MA gap 0.30–1.00% but no clean trend — require higher conviction.
        # Also check per-symbol RANGING/CHOPPY from the shared_state metrics.
        symbol_regimes = {}
        if self._shared_state is not None:
            try:
                symbol_regimes = (
                    getattr(self._shared_state, "metrics", {}) or {}
                ).get("symbol_regimes", {}) or {}
            except Exception:
                symbol_regimes = {}
        sym_regime = str(symbol_regimes.get(symbol, "") or "").upper()
        if sym_regime in {"RANGING", "CHOPPY"}:
            if side == "BUY":
                return RegimeDecision(False, reason=f"symbol_regime_{sym_regime.lower()}:no_momentum_for_buy")
            return RegimeDecision(True, confidence_floor_bump=0.0, reason=f"symbol_regime_{sym_regime.lower()}:sell_allowed")

        if regime == "volatile" or volatility_score >= 0.85:
            return RegimeDecision(True, confidence_floor_bump=0.10, reason="volatile_guard")
        return RegimeDecision(True, confidence_floor_bump=0.0, reason="regime_ok")
