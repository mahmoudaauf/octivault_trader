# signal_fusion.py
#
# ASYNC SIGNAL FUSION COMPONENT (P9-Compliant Architecture)
#
# Design Principles:
# 1. Independent async task - runs separately from MetaController decision pipeline
# 2. Signal-only (no execution) - emits signals via shared_state signal bus
# 3. NO references to ExecutionManager - maintains P9 invariant
# 4. Pre-processing layer - fusion runs before MetaController aggregates signals
# 5. Graceful degradation - failures don't block main trading loop
#
# Signal Flow:
#   Agents (emit to shared_state.agent_signals)
#        ↓
#   SignalFusionTask.run() [async, independent]
#        ↓
#   Emit fused signal back to shared_state
#        ↓
#   MetaController.receive_signal() [natural integration]
#        ↓
#   Decision pipeline (unaware of fusion origin)

import asyncio
from collections import Counter, defaultdict
from typing import Dict, Optional
import logging
import json
import os
from datetime import datetime
import time


class SignalFusion:
    """
    Consensus voting engine for multi-agent signals.
    
    This component operates as an async task that:
    - Reads cached agent signals from shared_state
    - Applies consensus voting (majority, weighted, unanimous)
    - Emits fused signals back to shared_state signal bus
    - Does NOT interact with ExecutionManager (P9-compliant)
    - Does NOT reference MetaController for execution
    - Allows MetaController to pick up fused signals naturally
    
    The fusion task is optional and non-blocking for the main trading loop.
    """
    
    def __init__(
        self,
        shared_state,
        fusion_mode: str = "weighted",   # Options: majority, weighted, unanimous
        threshold: float = 0.6,
        log_to_file: bool = True,
        log_dir: str = "logs"
    ):
        """
        Initialize SignalFusion component.
        
        Args:
            shared_state: SharedState object (for signal reading/writing)
            fusion_mode: Voting mode ("weighted", "majority", or "unanimous")
            threshold: Confidence threshold for weighted voting
            log_to_file: Whether to log fusion decisions to file
            log_dir: Directory for fusion logs
        """
        self.shared_state = shared_state
        self.fusion_mode = fusion_mode
        self.threshold = threshold
        self.logger = logging.getLogger("SignalFusion")
        self.log_to_file = log_to_file
        self.log_path = os.path.join(log_dir, "fusion_log.json")
        os.makedirs(log_dir, exist_ok=True)
        self._running = False
        self._task = None
    
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
            "confidence": 0.0
        }
        
        if self.fusion_mode == "majority":
            decision, confidence = self._majority_vote(agent_signals)
        elif self.fusion_mode == "weighted":
            decision, confidence = self._weighted_vote(agent_signals, agent_scores)
        elif self.fusion_mode == "unanimous":
            decision, confidence = self._unanimous_vote(agent_signals)
        else:
            self.logger.warning(f"[{symbol}] Unknown fusion mode: {self.fusion_mode}")
            return
        
        fusion_result["decision"] = decision
        fusion_result["confidence"] = confidence
        
        # Emit fused signal if consensus reached
        if decision:
            self.logger.info(f"[SignalFusion:{symbol}] Consensus: {decision.upper()} (conf={confidence:.2f}, mode={self.fusion_mode})")
            routed = await self._emit_fused_signal(symbol, decision, confidence, agent_signals)
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

    def _majority_vote(self, agent_signals: Dict[str, Dict[str, str]]) -> (Optional[str], float):
        votes = [signal["action"] for signal in agent_signals.values()]
        count = Counter(votes)
        if not count:
            return None, 0.0
        top_vote, top_count = count.most_common(1)[0]
        confidence = top_count / len(votes)
        return (top_vote, confidence) if top_count >= 2 else (None, confidence)

    def _weighted_vote(self, agent_signals: Dict[str, Dict[str, str]],
                       agent_scores: Dict[str, Dict[str, float]]) -> (Optional[str], float):
        weights = defaultdict(float)
        total_weight = 0.0

        for agent, signal in agent_signals.items():
            roi = agent_scores.get(agent, {}).get("roi", 0.01)  # Prevent 0 ROI
            action = signal.get("action")
            weights[action] += roi
            total_weight += roi

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
    ) -> bool:
        """
        Emit fused signal back to shared_state signal bus.
        
        P9 COMPLIANCE:
        - Does NOT call ExecutionManager
        - Does NOT call MetaController directly
        - Emits signal via shared_state (natural P9 signal bus)
        - MetaController will pick up signal naturally via receive_signal()
        
        Args:
            symbol: Trading symbol
            decision: Consensus decision (BUY/SELL)
            confidence: Confidence of decision
            agent_signals: Original agent signals used in fusion
        
        Returns:
            True if signal was emitted to bus, False otherwise
        """
        side = str(decision or "").upper()
        if side not in {"BUY", "SELL"}:
            return False

        signal_payload = {
            "symbol": str(symbol or "").upper(),
            "action": side,
            "side": side,
            "confidence": float(confidence or 0.0),
            "reason": f"SignalFusion({self.fusion_mode}) consensus",
            "agent": "SignalFusion",
            "rationale": f"SignalFusion consensus from {len(agent_signals)} agents",
            "fusion_mode": self.fusion_mode,
            "source_agents": sorted(list((agent_signals or {}).keys())),
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
                )
                if asyncio.iscoroutine(res):
                    await res
                routed = True
        except Exception as e:
            self.logger.debug(f"[SignalFusion:{symbol}] Failed to emit to signal bus: {e}")

        return routed
