"""
Opportunity Ranking Engine (Capital-First)

Purpose:
- Convert symbol-first signal handling into a capital-first opportunity list.
- Score BUY opportunities using multi-factor inputs (signal strength, regime
  alignment, liquidity, volatility context, agent confidence).
- Prune to the top-N ideas that fit the account size (small NAV focus).

Design notes:
- Lightweight and synchronous — safe to call inside MetaController decision
  loop without extra awaits.
- Uses SharedState data where possible (unified score, latest_prices).
- Safe defaults so missing fields never block decisions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple
import math


@dataclass
class RankedOpportunity:
    symbol: str
    action: str
    signal: Dict[str, Any]
    score: float


class OpportunityRanker:
    """Small-NAV friendly opportunity scorer."""

    def __init__(self, shared_state, logger=None):
        self.shared_state = shared_state
        self.logger = logger

        # Weighting scheme balances agent conviction with market quality.
        self.weights = {
            "signal_strength": 0.40,
            "regime_alignment": 0.15,
            "liquidity": 0.15,
            "volatility": 0.10,
            "agent_confidence": 0.10,
            "market_quality": 0.10,  # shared_state unified score
        }

    # ---- Public API -----------------------------------------------------
    def recommended_max_positions(self, nav_usd: float) -> int:
        """
        Capital-first limit tuned for very small accounts.
        $0-150 → 1 slot, $151-350 → 2 slots, $351-750 → 3 slots, else defer to mode caps.
        """
        if nav_usd <= 150:
            return 1
        if nav_usd <= 350:
            return 2
        if nav_usd <= 750:
            return 3
        return 0  # 0 = no additional cap (fall back to regime/max_pos)

    def rank_and_prune(
        self,
        decisions: Sequence[Tuple[str, str, Dict[str, Any]]],
        max_buys: int,
    ) -> List[Tuple[str, str, Dict[str, Any]]]:
        """
        Rank BUY decisions, keep top `max_buys`, preserve SELLs.
        """
        sells: List[Tuple[str, str, Dict[str, Any]]] = []
        buy_candidates: List[RankedOpportunity] = []

        for sym, action, sig in decisions:
            if action == "BUY":
                score = self.score_signal(sym, sig)
                buy_candidates.append(RankedOpportunity(sym, action, sig, score))
            else:
                sells.append((sym, action, sig))

        # Sort BUYs by score (desc), tie-breaker: higher confidence then symbol
        buy_candidates.sort(key=lambda r: (r.score, r.signal.get("confidence", 0.0), r.symbol), reverse=True)

        kept_buys = buy_candidates[:max_buys] if max_buys > 0 else buy_candidates

        dropped = len(buy_candidates) - len(kept_buys)
        if dropped > 0 and self.logger:
            self.logger.info(
                "[OppRanker] Pruned %d BUYs -> kept %d (limit=%d)",
                dropped,
                len(kept_buys),
                max_buys,
            )

        # Preserve SELL ordering, append ranked BUYs after risk exits
        ranked_decisions: List[Tuple[str, str, Dict[str, Any]]] = list(sells)
        ranked_decisions.extend([(r.symbol, r.action, r.signal) for r in kept_buys])
        return ranked_decisions

    # ---- Scoring --------------------------------------------------------
    def score_signal(self, symbol: str, signal: Dict[str, Any]) -> float:
        """
        Compute multi-factor opportunity score. Outputs ~0-1.5 range.
        Safe defaults avoid NaNs if data is missing.
        """
        sym = str(symbol or "").upper()
        confidence = float(signal.get("confidence", 0.0) or 0.0)
        agent_conf = float(signal.get("agent_confidence", confidence) or confidence)

        # Signal strength considers composite_edge if present
        edge = float(signal.get("composite_edge", signal.get("edge", 0.0)) or 0.0)
        signal_strength = max(confidence, abs(edge))

        regime_alignment = float(signal.get("regime_alignment", 0.5) or 0.5)
        liquidity_score = float(signal.get("liquidity_score", self._estimate_liquidity(sym)) or 0.0)
        volatility_score = float(signal.get("volatility_score", self._estimate_volatility(sym)) or 0.0)
        market_quality = self._safe_unified_score(sym)

        w = self.weights
        score = (
            signal_strength * w["signal_strength"]
            + regime_alignment * w["regime_alignment"]
            + liquidity_score * w["liquidity"]
            + volatility_score * w["volatility"]
            + agent_conf * w["agent_confidence"]
            + market_quality * w["market_quality"]
        )
        return float(score)

    # ---- Helpers --------------------------------------------------------
    def _safe_unified_score(self, symbol: str) -> float:
        try:
            if hasattr(self.shared_state, "get_unified_score"):
                return float(self.shared_state.get_unified_score(symbol) or 0.0)
        except Exception:
            pass
        return 0.0

    def _estimate_liquidity(self, symbol: str) -> float:
        """
        Estimate liquidity using latest_prices snapshot (volume + spread).
        Mirrors the scoring logic used in SharedState.
        """
        try:
            lp = getattr(self.shared_state, "latest_prices", {}) or {}
            px = lp.get(symbol, {})
            quote_volume = float(px.get("quote_volume", 0.0) or 0.0)
            spread = float(px.get("spread", 0.01) or 0.01)
            return min(quote_volume / 100000, 1.0) * max(0.0, 1.0 - min(spread, 0.05))
        except Exception:
            return 0.0

    def _estimate_volatility(self, symbol: str) -> float:
        """
        Use regime label to nudge volatility alignment. Neutral = 0.5.
        """
        try:
            regimes = getattr(self.shared_state, "volatility_regimes", {}) or {}
            tf_regimes = regimes.get(symbol) or regimes.get("GLOBAL") or {}
            # pick any timeframe entry
            if tf_regimes:
                regime = next(iter(tf_regimes.values())).get("regime", "normal")
                regime = str(regime or "normal").lower()
                if regime == "bull":
                    return 0.8  # higher tolerance for volatility
                if regime == "bear":
                    return 0.6
            return 0.5
        except Exception:
            return 0.5

