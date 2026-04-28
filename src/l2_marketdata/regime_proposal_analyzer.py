"""
Phase 3: Regime-Based Proposal Weighting
core/regime_proposal_analyzer.py

Analyzes and weights discovery proposals based on current market regime.
Improves signal quality by prioritizing regime-aligned opportunities.
"""

from typing import Dict, List, Optional, Tuple
from enum import Enum
import asyncio


class ProposalType(Enum):
    """Classification of proposal types."""
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    VOLATILITY_PLAY = "volatility_play"
    NEW_DISCOVERY = "new_discovery"


class RegimeProposalAnalyzer:
    """
    Analyzes and weights discovery proposals based on market regime.
    
    Scoring Strategy:
    1. Get current market regime (TREND, RANGE, VOLATILE, BREAKOUT, LOW_LIQUIDITY)
    2. Classify proposal type (momentum, mean-reversion, volatility, discovery)
    3. Apply regime-specific multiplier (0.6 - 1.5)
    4. Apply volatility bonus
    5. Apply liquidity penalty
    6. Calculate final weighted score (0-1)
    7. Sort by weighted score descending
    """
    
    def __init__(self, shared_state, config, logger):
        """Initialize analyzer with system dependencies."""
        self.ss = shared_state
        self.cfg = config
        self.logger = logger
        
        # Regime scoring matrix
        # [regime][proposal_type] = multiplier
        self.regime_multipliers = {
            "trend": {
                "momentum": 1.30,
                "mean_reversion": 0.80,
                "volatility_play": 1.10,
                "new_discovery": 1.15,
            },
            "range": {
                "momentum": 0.80,
                "mean_reversion": 1.40,
                "volatility_play": 0.70,
                "new_discovery": 1.15,
            },
            "volatile": {
                "momentum": 1.40,
                "mean_reversion": 0.90,
                "volatility_play": 1.50,
                "new_discovery": 1.20,
            },
            "breakout": {
                "momentum": 1.50,
                "mean_reversion": 0.70,
                "volatility_play": 1.20,
                "new_discovery": 1.25,
            },
            "low_liquidity": {
                "momentum": 0.60,
                "mean_reversion": 0.70,
                "volatility_play": 0.50,
                "new_discovery": 0.60,
            },
        }
        
        # Metrics tracking
        self.metrics = {
            "total_analyzed": 0,
            "total_weighted": 0,
            "regime_bonuses_applied": {},
            "proposal_types_counted": {},
            "score_distribution": {
                "high": 0,  # > 0.75
                "medium": 0,  # 0.5-0.75
                "low": 0,  # < 0.5
            },
            "last_regime": None,
            "last_analysis_time": None,
        }
    
    async def analyze_proposal(self, proposal: Dict) -> Dict:
        """
        Analyze and weight a single proposal.
        
        Args:
            proposal: {symbol, confidence, source, ...metadata}
        
        Returns:
            Enhanced proposal with:
            {
                ...original proposal,
                base_confidence: float,
                proposal_type: str,
                regime_bonus: float,
                volatility_bonus: float,
                liquidity_penalty: float,
                weighted_score: float,
                regime: str,
            }
        """
        try:
            # Get base confidence
            base_conf = float(proposal.get("confidence", 0.5))
            
            # Get current regime
            regime = await self._get_current_regime()
            
            # Classify proposal type
            prop_type = await self._classify_proposal_type(proposal)
            
            # Get regime multiplier
            regime_mult = self._get_regime_multiplier(regime, prop_type)
            regime_bonus = regime_mult - 1.0  # Convert to bonus (-0.4 to +0.5)
            
            # Get volatility bonus (if available)
            vol_bonus = await self._get_volatility_bonus(proposal.get("symbol"), regime)
            
            # Get liquidity penalty (if available)
            liq_penalty = await self._get_liquidity_penalty(proposal.get("symbol"))
            
            # Calculate final weighted score
            weighted_score = self._calculate_weighted_score(
                base_conf, regime_bonus, vol_bonus, liq_penalty
            )
            
            # Clamp score to 0-1
            weighted_score = max(0.0, min(1.0, weighted_score))
            
            # Add to metrics
            self.metrics["total_analyzed"] += 1
            self.metrics["total_weighted"] += 1
            self.metrics["last_regime"] = regime
            
            # Track regime bonus
            if regime not in self.metrics["regime_bonuses_applied"]:
                self.metrics["regime_bonuses_applied"][regime] = 0
            self.metrics["regime_bonuses_applied"][regime] += 1
            
            # Track proposal type
            if prop_type not in self.metrics["proposal_types_counted"]:
                self.metrics["proposal_types_counted"][prop_type] = 0
            self.metrics["proposal_types_counted"][prop_type] += 1
            
            # Track score distribution
            if weighted_score > 0.75:
                self.metrics["score_distribution"]["high"] += 1
            elif weighted_score > 0.5:
                self.metrics["score_distribution"]["medium"] += 1
            else:
                self.metrics["score_distribution"]["low"] += 1
            
            # Return enhanced proposal
            return {
                **proposal,
                "base_confidence": base_conf,
                "proposal_type": prop_type,
                "regime_bonus": round(regime_bonus, 3),
                "volatility_bonus": round(vol_bonus, 3),
                "liquidity_penalty": round(liq_penalty, 3),
                "weighted_score": round(weighted_score, 3),
                "regime": regime,
            }
        
        except Exception as e:
            self.logger.error(f"[ANALYZER] Error analyzing proposal: {e}")
            # Safe fallback: return with original confidence
            return {
                **proposal,
                "base_confidence": float(proposal.get("confidence", 0.5)),
                "proposal_type": "unknown",
                "regime_bonus": 0.0,
                "volatility_bonus": 0.0,
                "liquidity_penalty": 0.0,
                "weighted_score": float(proposal.get("confidence", 0.5)),
                "regime": "unknown",
            }
    
    async def batch_analyze_proposals(self, proposals: List[Dict]) -> List[Dict]:
        """
        Efficiently analyze multiple proposals.
        Returns proposals sorted by weighted_score descending.
        
        Args:
            proposals: List of proposal dicts
        
        Returns:
            List of enhanced proposals sorted by weighted_score (descending)
        """
        try:
            # Analyze all proposals in parallel (if possible)
            analyzed = []
            for proposal in proposals:
                result = await self.analyze_proposal(proposal)
                analyzed.append(result)
            
            # Sort by weighted_score descending
            analyzed.sort(
                key=lambda p: p.get("weighted_score", 0.5),
                reverse=True
            )
            
            self.logger.debug(
                f"[ANALYZER] Analyzed {len(analyzed)} proposals, "
                f"top score: {analyzed[0].get('weighted_score', 0) if analyzed else 0}"
            )
            
            return analyzed
        
        except Exception as e:
            self.logger.error(f"[ANALYZER] Error in batch analysis: {e}")
            # Return original proposals (no weighting applied)
            return proposals
    
    async def _get_current_regime(self) -> str:
        """
        Get current market regime from MarketRegimeDetector.
        
        Returns:
            One of: "trend", "range", "volatile", "breakout", "low_liquidity"
        """
        try:
            # Try to get regime from SharedState
            if hasattr(self.ss, "get_market_regime"):
                regime_result = await self._maybe_await(
                    self.ss.get_market_regime()
                )
                if isinstance(regime_result, dict):
                    regime = str(regime_result.get("regime", "trend")).lower()
                    if regime in self.regime_multipliers:
                        return regime
            
            # Fallback to "trend" (neutral)
            return "trend"
        
        except Exception as e:
            self.logger.debug(f"[ANALYZER] Error getting regime: {e}")
            return "trend"  # Safe default
    
    async def _classify_proposal_type(self, proposal: Dict) -> str:
        """
        Classify proposal as momentum, mean-reversion, volatility, or discovery.
        
        Strategy:
        - If from symbol_screener with high atr_pct → momentum
        - If from wallet_scanner with old symbol → mean_reversion
        - If volatility_regime == HIGH → volatility_play
        - Otherwise → new_discovery (conservative)
        """
        try:
            source = str(proposal.get("source", "")).lower()
            conf = float(proposal.get("confidence", 0.5))
            
            # Momentum: screener with high confidence
            if source == "symbol_screener" and conf > 0.65:
                return ProposalType.MOMENTUM.value
            
            # Volatility play: from screener in volatile regime
            if source == "symbol_screener":
                regime = await self._get_current_regime()
                if regime in ("volatile", "breakout"):
                    return ProposalType.VOLATILITY_PLAY.value
            
            # Mean-reversion: wallet scanner (older symbols)
            if source in ("wallet_scanner", "symbol_discoverer"):
                return ProposalType.MEAN_REVERSION.value
            
            # Default: discovery
            return ProposalType.NEW_DISCOVERY.value
        
        except Exception as e:
            self.logger.debug(f"[ANALYZER] Error classifying proposal: {e}")
            return ProposalType.NEW_DISCOVERY.value
    
    def _get_regime_multiplier(self, regime: str, proposal_type: str) -> float:
        """
        Get multiplier for regime+type combination.
        Ranges from 0.6 (very bad match) to 1.5 (excellent match).
        
        Args:
            regime: "trend", "range", "volatile", "breakout", "low_liquidity"
            proposal_type: "momentum", "mean_reversion", "volatility_play", "new_discovery"
        
        Returns:
            Multiplier (0.6 - 1.5)
        """
        regime = regime.lower()
        prop_type = proposal_type.lower()
        
        # Get from matrix, default to 1.0
        multiplier = self.regime_multipliers.get(regime, {}).get(prop_type, 1.0)
        
        return multiplier
    
    async def _get_volatility_bonus(self, symbol: Optional[str], regime: str) -> float:
        """
        Get volatility bonus for this symbol in current regime.
        
        Returns:
            Bonus (-0.2 to +0.2) applied to score
        """
        try:
            if not symbol:
                return 0.0
            
            # In high volatility regime, volatility is good
            if regime in ("volatile", "breakout"):
                return 0.1
            
            # In range/trend, high volatility is risky
            if regime == "range":
                return -0.05
            
            return 0.0
        
        except Exception as e:
            self.logger.debug(f"[ANALYZER] Error getting volatility bonus: {e}")
            return 0.0
    
    async def _get_liquidity_penalty(self, symbol: Optional[str]) -> float:
        """
        Get liquidity penalty for this symbol.
        
        Returns:
            Penalty (-0.1 to 0.0) applied to score
        """
        try:
            if not symbol:
                return 0.0
            
            # Try to check liquidity from SharedState
            if hasattr(self.ss, "get_symbol_liquidity"):
                liq = await self._maybe_await(
                    self.ss.get_symbol_liquidity(symbol)
                )
                if isinstance(liq, dict):
                    liq_status = str(liq.get("status", "ok")).lower()
                    if liq_status == "low":
                        return -0.1
                    elif liq_status == "very_low":
                        return -0.2
            
            return 0.0
        
        except Exception as e:
            self.logger.debug(f"[ANALYZER] Error getting liquidity penalty: {e}")
            return 0.0
    
    def _calculate_weighted_score(
        self,
        base_conf: float,
        regime_bonus: float,
        vol_bonus: float,
        liq_penalty: float,
    ) -> float:
        """
        Calculate final weighted score.
        
        Formula:
        weighted_score = base_conf * (1 + regime_bonus + vol_bonus) - liq_penalty
        
        Result clamped to 0-1 range.
        """
        aggressiveness = float(self.cfg("WEIGHTING_AGGRESSIVENESS", 0.8))
        
        # Apply regime and volatility bonuses
        score = base_conf * (1 + (regime_bonus * aggressiveness) + (vol_bonus * aggressiveness))
        
        # Apply liquidity penalty
        score = score + (liq_penalty * aggressiveness)
        
        return score
    
    async def _maybe_await(self, result):
        """Handle both sync and async results."""
        if hasattr(result, "__await__"):
            return await result
        return result
    
    def get_metrics(self) -> Dict:
        """Get analyzer metrics."""
        return {
            **self.metrics,
            "last_analysis_time": self.metrics.get("last_analysis_time"),
        }
    
    def reset_metrics(self):
        """Reset metrics counters."""
        self.metrics = {
            "total_analyzed": 0,
            "total_weighted": 0,
            "regime_bonuses_applied": {},
            "proposal_types_counted": {},
            "score_distribution": {"high": 0, "medium": 0, "low": 0},
            "last_regime": None,
            "last_analysis_time": None,
        }
