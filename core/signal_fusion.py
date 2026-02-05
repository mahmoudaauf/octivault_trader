# signal_fusion.py

import asyncio
from collections import Counter, defaultdict
from typing import Dict, Optional
import logging
import json
import os
from datetime import datetime


class SignalFusion:
    def __init__(
        self,
        shared_state,
        execution_manager,
        fusion_mode: str = "weighted",   # Options: majority, weighted, unanimous
        threshold: float = 0.6,
        log_to_file: bool = True,
        log_dir: str = "logs"
    ):
        self.shared_state = shared_state
        self.execution_manager = execution_manager
        self.fusion_mode = fusion_mode
        self.threshold = threshold
        self.logger = logging.getLogger("SignalFusion")
        self.log_to_file = log_to_file
        self.log_path = os.path.join(log_dir, "fusion_log.json")
        os.makedirs(log_dir, exist_ok=True)

    async def fuse_and_execute(self, symbol: str):
        async with self.shared_state.lock:
            agent_signals: Dict[str, Dict[str, str]] = self.shared_state.agent_signals.get(symbol, {})
            agent_scores: Dict[str, Dict[str, float]] = self.shared_state.agent_scores

        if not agent_signals:
            self.logger.info(f"[{symbol}] No agent signals available for fusion.")
            return

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

        if decision:
            self.logger.info(f"[{symbol}] Fusion decision: {decision.upper()} with confidence {confidence:.2f}")
            await self.execution_manager.execute_trade(symbol, decision)
            await self._log_decision(fusion_result)
        else:
            self.logger.info(f"[{symbol}] No consensus reached, trade not executed.")
            await self._log_decision(fusion_result)

        # Track KPI for dashboard
        async with self.shared_state.lock:
            self.shared_state.kpi_metrics["fusion_decisions"].append(fusion_result)

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
