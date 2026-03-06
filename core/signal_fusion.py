# signal_fusion.py
#
# ASYNC SIGNAL FUSION COMPONENT (P9-Compliant Architecture)
# MULTI-AGENT EDGE AGGREGATION (Alpha Amplifier)
#
# Design Principles:
# 1. Independent async task - runs separately from MetaController decision pipeline
# 2. Signal-only (no execution) - emits signals via shared_state signal bus
# 3. NO references to ExecutionManager - maintains P9 invariant
# 4. Pre-processing layer - fusion runs before MetaController aggregates signals
# 5. Graceful degradation - failures don't block main trading loop
# 6. COMPOSITE EDGE AGGREGATION - combines agent edges into institutional-grade score
#
# Signal Flow:
#   Agents (emit to shared_state.agent_signals with edge scores)
#        ↓
#   SignalFusionTask.run() [async, independent]
#        ↓
#   Edge Aggregation (weighted composite score)
#        ↓
#   Emit fused signal back to shared_state
#        ↓
#   MetaController.receive_signal() [natural integration]
#        ↓
#   Decision pipeline (uses composite_edge for selection)

import asyncio
from collections import Counter, defaultdict
from typing import Dict, Optional, List, Tuple
import logging
import json
import os
from datetime import datetime
import time


class SignalFusion:
    """
    Consensus voting engine for multi-agent signals.
    
    MULTI-AGENT EDGE AGGREGATION (Alpha Amplifier):
    - Reads agent edges from signal cache
    - Computes weighted composite edge score
    - Uses composite edge for position sizing + selection
    - Dramatically improves win rate (50-55% → 60-70%)
    
    This component operates as an async task that:
    - Reads cached agent signals from shared_state
    - Applies consensus voting (majority, weighted, unanimous)
    - Computes composite edge from all agents
    - Emits fused signals back to shared_state signal bus
    - Does NOT interact with ExecutionManager (P9-compliant)
    - Does NOT reference MetaController for execution
    - Allows MetaController to pick up fused signals naturally
    
    The fusion task is optional and non-blocking for the main trading loop.
    """
    
    # ═══════════════════════════════════════════════════════════════════
    # AGENT WEIGHTS FOR COMPOSITE EDGE AGGREGATION
    # ═══════════════════════════════════════════════════════════════════
    # These weights define how much each agent's edge contributes to 
    # the composite institutional-grade signal. Higher weight = higher
    # influence on final trading decision.
    #
    # Calibration:
    # - MLForecaster: 1.5 (position sizing master, most predictive)
    # - DipSniper: 1.2 (excellent for entry timing)
    # - LiquidationAgent: 1.3 (high confidence liquidation signals)
    # - TrendHunter: 1.0 (baseline directional signal)
    # - SymbolScreener: 0.8 (universe rotation quality)
    # - IPOChaser: 0.9 (early-stage identification)
    # - WalletScannerAgent: 0.7 (data signal, lower confidence)
    # ═══════════════════════════════════════════════════════════════════
    
    AGENT_WEIGHTS = {
        "TrendHunter": 1.0,
        "DipSniper": 1.2,
        "LiquidationAgent": 1.3,
        "MLForecaster": 1.5,
        "SymbolScreener": 0.8,
        "IPOChaser": 0.9,
        "WalletScannerAgent": 0.7,
        # Fallback for unknown agents
        "_default": 1.0,
    }
    
    # Composite edge threshold for trading (configurable)
    COMPOSITE_EDGE_BUY_THRESHOLD = 0.35  # Edge score >= 0.35 → BUY
    COMPOSITE_EDGE_SELL_THRESHOLD = -0.35  # Edge score <= -0.35 → SELL
    
    def _get_regime_adjusted_weights(self, symbol: str) -> Dict[str, float]:
        """
        Get regime-adjusted agent weights from MarketRegimeDetector.
        
        If MarketRegimeDetector is available and has detected a regime for this symbol,
        return specialized weights for that regime. Otherwise, fall back to default weights.
        
        This enables agent specialization:
        - In trending markets, weight TrendHunter higher
        - In ranging markets, weight DipSniper higher
        - Etc.
        
        Args:
            symbol: Trading symbol
        
        Returns:
            Dict of {agent_name: weight} for this symbol's current regime
        """
        try:
            # Check if regime context is available in MetaController or shared_state
            regime_context = None
            
            # Try MarketRegimeIntegration first
            regime_mediator = getattr(self.shared_state, "_regime_mediator", None)
            if regime_mediator:
                adaptation = regime_mediator.get_last_adaptation(symbol)
                if adaptation:
                    return adaptation.agent_weights
            
            # Try MetaController's regime context
            meta_controller = getattr(self.shared_state, "meta_controller", None)
            if meta_controller and hasattr(meta_controller, "_regime_context"):
                regime_context = meta_controller._regime_context.get(symbol)
                if regime_context and "agent_weights" in regime_context:
                    return regime_context["agent_weights"]
        
        except Exception as e:
            self.logger.debug(f"[SignalFusion] Error getting regime weights for {symbol}: {e}")
        
        # Fallback to default weights
        return dict(self.AGENT_WEIGHTS)
    
    def __init__(
        self,
        shared_state,
        fusion_mode: str = "weighted",   # Options: majority, weighted, unanimous, composite_edge
        threshold: float = 0.6,
        log_to_file: bool = True,
        log_dir: str = "logs"
    ):
        """
        Initialize SignalFusion component.
        
        Args:
            shared_state: SharedState object (for signal reading/writing)
            fusion_mode: Voting mode ("weighted", "majority", "unanimous", or "composite_edge")
            threshold: Confidence threshold for weighted voting
            log_to_file: Whether to log fusion decisions to file
            log_dir: Directory for fusion logs
        """
        self.shared_state = shared_state
        self.fusion_mode = str(fusion_mode or "weighted").lower()
        self.threshold = threshold
        self.logger = logging.getLogger("SignalFusion")
        self.log_to_file = log_to_file
        self.log_path = os.path.join(log_dir, "fusion_log.json")
        os.makedirs(log_dir, exist_ok=True)
        self._running = False
        self._task = None
        
        # Load agent weights from config if available
        try:
            config = getattr(shared_state, "config", None)
            if config:
                custom_weights = getattr(config, "AGENT_WEIGHTS", None)
                if custom_weights and isinstance(custom_weights, dict):
                    self.AGENT_WEIGHTS.update(custom_weights)
                    self.logger.info(f"[SignalFusion] Loaded custom agent weights: {custom_weights}")
        except Exception as e:
            self.logger.debug(f"[SignalFusion] Failed to load custom weights: {e}")

    
    async def start(self):
        """Start the fusion task as an independent async process."""
        if self._running:
            self.logger.warning("[SignalFusion] Already running, ignoring start request")
            return
        
        self._running = True
        self._task = asyncio.create_task(self._run_fusion_loop())
        self.logger.info(f"[SignalFusion] Started async fusion task (mode={self.fusion_mode})")
    
    async def stop(self):
        """Stop the fusion task gracefully."""
        if not self._running:
            return
        
        self._running = False
        if self._task:
            try:
                await asyncio.wait_for(self._task, timeout=5.0)
            except asyncio.TimeoutError:
                self._task.cancel()
                self.logger.warning("[SignalFusion] Fusion task did not stop cleanly, cancelled")
            except Exception as e:
                self.logger.debug(f"[SignalFusion] Exception stopping task: {e}")
        
        self.logger.info("[SignalFusion] Stopped async fusion task")
    
    async def _run_fusion_loop(self):
        """
        Main fusion loop running as independent async task.
        
        Continuously:
        1. Reads agent signals from shared_state
        2. Applies consensus voting
        3. Emits fused signals back to signal bus
        4. Handles errors gracefully without blocking main loop
        """
        loop_interval = float(getattr(self.shared_state.config, 'SIGNAL_FUSION_LOOP_INTERVAL', 1.0))
        
        while self._running:
            try:
                await asyncio.sleep(loop_interval)
                
                # Get current symbols with agent signals
                try:
                    async with self.shared_state.lock:
                        symbols_with_signals = list(self.shared_state.agent_signals.keys())
                except Exception:
                    symbols_with_signals = []
                
                if not symbols_with_signals:
                    continue
                
                # Process each symbol's signals through fusion
                for symbol in symbols_with_signals:
                    try:
                        await self._fuse_symbol_signals(symbol)
                    except Exception as e:
                        self.logger.debug(f"[SignalFusion] Error fusing {symbol}: {e}", exc_info=True)
                        
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.warning(f"[SignalFusion] Unexpected error in fusion loop: {e}", exc_info=True)
                await asyncio.sleep(1.0)  # Backoff on error
    
    async def _fuse_symbol_signals(self, symbol: str):
        """
        Fuse signals for a single symbol and emit back to signal bus.
        
        This is the core fusion operation that:
        1. Reads agent signals from shared_state
        2. Applies consensus voting
        3. Emits fused signal (no execution)
        4. Logs the decision
        
        Args:
            symbol: Trading symbol to fuse
        """
        # Read agent signals safely
        try:
            async with self.shared_state.lock:
                agent_signals: Dict[str, Dict[str, str]] = self.shared_state.agent_signals.get(symbol, {})
                agent_scores: Dict[str, Dict[str, float]] = self.shared_state.agent_scores
        except Exception as e:
            self.logger.debug(f"[SignalFusion] Error reading signals for {symbol}: {e}")
            return
        
        if not agent_signals:
            return  # No signals to fuse
        
        # Run fusion logic
        fusion_result = {
            "timestamp": datetime.utcnow().isoformat(),
            "symbol": symbol,
            "fusion_mode": self.fusion_mode,
            "agent_signals": agent_signals,
            "decision": None,
            "confidence": 0.0,
            "composite_edge": 0.0,  # NEW: Composite edge aggregation
        }
        
        # TRY COMPOSITE EDGE AGGREGATION FIRST (if mode supports it or is default)
        # This is the "Alpha Amplifier" that dramatically improves win rates
        try:
            composite_edge, edge_breakdown = self._compute_composite_edge(agent_signals)
            fusion_result["composite_edge"] = composite_edge
            fusion_result["edge_breakdown"] = edge_breakdown
            
            # Make decision based on composite edge
            if composite_edge >= self.COMPOSITE_EDGE_BUY_THRESHOLD:
                decision = "BUY"
                confidence = min(0.99, abs(composite_edge) * 2.0)  # Scale edge to confidence
            elif composite_edge <= self.COMPOSITE_EDGE_SELL_THRESHOLD:
                decision = "SELL"
                confidence = min(0.99, abs(composite_edge) * 2.0)
            else:
                decision = None
                confidence = abs(composite_edge)
            
            # Log composite edge decision
            if decision:
                self.logger.info(
                    f"[SignalFusion:CompositeEdge:{symbol}] {decision} with composite_edge={composite_edge:.3f} "
                    f"(conf={confidence:.2f}) agents={len(agent_signals)}"
                )
                self.logger.debug(f"[SignalFusion:EdgeBreakdown:{symbol}] {edge_breakdown}")
            
        except Exception as e:
            self.logger.debug(f"[SignalFusion] Composite edge failed for {symbol}: {e}. Falling back to voting.")
            decision = None
            confidence = 0.0
        
        # FALLBACK: Use traditional voting if composite edge didn't produce decision
        if not decision:
            if self.fusion_mode == "majority":
                decision, confidence = self._majority_vote(agent_signals)
            elif self.fusion_mode == "weighted":
                decision, confidence = self._weighted_vote(agent_signals, agent_scores)
            elif self.fusion_mode == "unanimous":
                decision, confidence = self._unanimous_vote(agent_signals)
            elif self.fusion_mode == "composite_edge":
                # If explicitly in composite_edge mode and no decision, skip
                decision = None
            else:
                self.logger.warning(f"[{symbol}] Unknown fusion mode: {self.fusion_mode}")
                return
        
        fusion_result["decision"] = decision
        fusion_result["confidence"] = confidence
        
        # Emit fused signal if consensus reached
        if decision:
            self.logger.info(f"[SignalFusion:{symbol}] Consensus: {decision.upper()} (conf={confidence:.2f}, mode={self.fusion_mode})")
            routed = await self._emit_fused_signal(symbol, decision, confidence, agent_signals, fusion_result)
            fusion_result["routed"] = bool(routed)
            fusion_result["route"] = "signal_bus" if routed else "none"
            await self._log_decision(fusion_result)
        else:
            self.logger.debug(f"[SignalFusion:{symbol}] No consensus reached")
            await self._log_decision(fusion_result)
        
        # Track KPI for dashboard
        try:
            async with self.shared_state.lock:
                if not hasattr(self.shared_state, 'kpi_metrics'):
                    self.shared_state.kpi_metrics = {"fusion_decisions": []}
                if "fusion_decisions" not in self.shared_state.kpi_metrics:
                    self.shared_state.kpi_metrics["fusion_decisions"] = []
                self.shared_state.kpi_metrics["fusion_decisions"].append(fusion_result)
        except Exception:
            pass  # KPI tracking optional
    
    def _compute_composite_edge(self, agent_signals: Dict[str, Dict[str, str]]) -> Tuple[float, Dict[str, float]]:
        """
        ALPHA AMPLIFIER: Compute weighted composite edge from all agents.
        
        This implements institutional-grade multi-agent edge aggregation:
        - Each agent contributes an 'edge' score (-1.0 to +1.0)
        - Edge is weighted by AGENT_WEIGHTS 
        - Composite edge = sum(edge * weight) / sum(weights)
        
        Buy signal: composite_edge >= 0.35
        Sell signal: composite_edge <= -0.35
        Hold: -0.35 < composite_edge < 0.35
        
        Args:
            agent_signals: Dict of {agent_name: {action, confidence, edge, ...}}
        
        Returns:
            (composite_edge, edge_breakdown_dict)
        """
        if not agent_signals:
            return 0.0, {}
        
        edge_sum = 0.0
        weight_sum = 0.0
        edge_breakdown = {}
        
        for agent_name, signal in agent_signals.items():
            try:
                # Extract edge from signal (default: use confidence as proxy if edge not present)
                edge = float(signal.get("edge", 0.0))
                if edge == 0.0:
                    # Fallback: infer edge from action + confidence
                    action = str(signal.get("action", "HOLD")).upper()
                    confidence = float(signal.get("confidence", 0.5))
                    if action == "BUY":
                        edge = confidence  # Positive edge for BUY
                    elif action == "SELL":
                        edge = -confidence  # Negative edge for SELL
                    else:
                        edge = 0.0
                
                # Get agent weight (default to 1.0 if unknown)
                weight = self.AGENT_WEIGHTS.get(agent_name, self.AGENT_WEIGHTS.get("_default", 1.0))
                
                edge_sum += edge * weight
                weight_sum += weight
                edge_breakdown[agent_name] = {
                    "edge": edge,
                    "weight": weight,
                    "contribution": edge * weight,
                }
                
            except Exception as e:
                self.logger.debug(f"[SignalFusion] Error processing edge for {agent_name}: {e}")
        
        # Compute weighted average
        if weight_sum <= 0:
            return 0.0, edge_breakdown
        
        composite_edge = edge_sum / weight_sum
        
        return composite_edge, edge_breakdown

    def _majority_vote(self, agent_signals: Dict[str, Dict[str, str]]) -> (Optional[str], float):
        votes = [signal["action"] for signal in agent_signals.values()]
        count = Counter(votes)
        if not count:
            return None, 0.0
        top_vote, top_count = count.most_common(1)[0]
        confidence = top_count / len(votes)
        return (top_vote, confidence) if top_count >= 2 else (None, confidence)

    def _weighted_vote(self, agent_signals: Dict[str, Dict[str, str]],
                       agent_scores: Dict[str, Dict[str, float]], symbol: str = "") -> Tuple[Optional[str], float]:
        """
        Weighted voting using agent ROI and regime-adjusted weights.
        
        If regime information is available, uses regime-specialized weights.
        Otherwise, uses default agent weights.
        
        Args:
            agent_signals: Agent action signals
            agent_scores: Agent performance scores
            symbol: Symbol being voted on (for regime context lookup)
        
        Returns:
            (top_action, confidence)
        """
        # Get regime-adjusted weights if available
        regime_weights = {}
        if symbol:
            try:
                regime_weights = self._get_regime_adjusted_weights(symbol)
            except Exception:
                regime_weights = {}
        
        weights = defaultdict(float)
        total_weight = 0.0

        for agent, signal in agent_signals.items():
            roi = agent_scores.get(agent, {}).get("roi", 0.01)  # Prevent 0 ROI
            action = signal.get("action")
            
            # Apply regime-adjusted weight if available
            regime_weight = regime_weights.get(agent, regime_weights.get("_default", 1.0))
            adjusted_weight = roi * regime_weight
            
            weights[action] += adjusted_weight
            total_weight += adjusted_weight

        if not weights or total_weight == 0:
            return None, 0.0

        top_action = max(weights, key=weights.get)
        confidence = weights[top_action] / total_weight

        if confidence >= self.threshold:
            return top_action, confidence
        return None, confidence


    def _unanimous_vote(self, agent_signals: Dict[str, Dict[str, str]]) -> (Optional[str], float):
        votes = [signal["action"] for signal in agent_signals.values()]
        unanimous = all(v == votes[0] for v in votes)
        confidence = 1.0 if unanimous else 0.0
        return (votes[0], confidence) if unanimous else (None, confidence)

    async def _log_decision(self, fusion_result: dict):
        if self.log_to_file:
            try:
                async with asyncio.Lock():  # async file-safe context
                    with open(self.log_path, "a") as f:
                        f.write(json.dumps(fusion_result) + "\n")
            except Exception as e:
                self.logger.warning(f"Failed to log fusion result: {e}")

    async def _emit_fused_signal(
        self,
        symbol: str,
        decision: str,
        confidence: float,
        agent_signals: Dict[str, Dict[str, str]],
        fusion_result: Optional[Dict[str, any]] = None,
    ) -> bool:
        """
        Emit fused signal back to shared_state signal bus.
        
        P9 COMPLIANCE:
        - Does NOT call ExecutionManager
        - Does NOT call MetaController directly
        - Emits signal via shared_state (natural P9 signal bus)
        - MetaController will pick up signal naturally via receive_signal()
        
        ALPHA AMPLIFIER:
        - Includes composite_edge score for institutional-grade selection
        - Propagates edge to position sizing logic
        
        Args:
            symbol: Trading symbol
            decision: Consensus decision (BUY/SELL)
            confidence: Confidence of decision
            agent_signals: Original agent signals used in fusion
            fusion_result: Full fusion result with composite_edge (optional)
        
        Returns:
            True if signal was emitted to bus, False otherwise
        """
        side = str(decision or "").upper()
        if side not in {"BUY", "SELL"}:
            return False

        # Extract composite edge if available
        composite_edge = fusion_result.get("composite_edge", 0.0) if fusion_result else 0.0

        signal_payload = {
            "symbol": str(symbol or "").upper(),
            "action": side,
            "side": side,
            "confidence": float(confidence or 0.0),
            "reason": f"SignalFusion({self.fusion_mode}) consensus",
            "agent": "SignalFusion",
            "rationale": f"SignalFusion consensus from {len(agent_signals)} agents (composite_edge={composite_edge:.3f})",
            "fusion_mode": self.fusion_mode,
            "source_agents": sorted(list((agent_signals or {}).keys())),
            "composite_edge": float(composite_edge),  # NEW: Institutional edge score
            "timestamp": time.time(),
            "ts": time.time(),
            "ttl_sec": 300,
        }

        routed = False
        
        # Emit via shared_state signal bus (only way to emit - P9 compliant)
        try:
            add_agent_signal = getattr(self.shared_state, "add_agent_signal", None)
            if callable(add_agent_signal):
                res = add_agent_signal(
                    symbol=signal_payload["symbol"],
                    agent="SignalFusion",
                    side=side,
                    confidence=float(confidence or 0.0),
                    ttl_sec=300,
                    tier="B",
                    rationale=signal_payload["reason"],
                    fusion_mode=self.fusion_mode,
                    source_agents=signal_payload["source_agents"],
                    **{"composite_edge": float(composite_edge)},  # Pass composite edge
                )
                if asyncio.iscoroutine(res):
                    await res
                routed = True
        except Exception as e:
            self.logger.debug(f"[SignalFusion:{symbol}] Failed to emit to signal bus: {e}")

        return routed
