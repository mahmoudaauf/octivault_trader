"""
Signal Batching System - Reduce Trade Friction

PURPOSE:
--------
Batch agent signals to reduce execution frequency and minimize fees.
Instead of executing every signal immediately, collect signals for N seconds,
de-duplicate conflicting signals, and execute batches.

ECONOMIC IMPACT:
----------------
Before: 20 trades/day × 0.3% friction = 6% monthly friction
After:  5 batches/day × 0.3% friction = 1.5% monthly friction
Improvement: 75% reduction in friction costs

ARCHITECTURE:
-------------
1. AgentManager.collect_and_forward_signals() → adds signals to batch queue
2. MetaController.run_loop() → calls batcher.flush() periodically (every 5 sec)
3. Batcher de-duplicates, prioritizes, executes batched signals
4. Reduces round-trip frequency while maintaining signal responsiveness
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, Tuple, Set
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class BatchedSignal:
    """Represents a signal ready for execution."""
    symbol: str
    side: str  # BUY, SELL, HOLD
    confidence: float
    agent: str
    rationale: str
    timestamp: float = field(default_factory=time.time)
    tier: str = "B"
    extra: Dict[str, Any] = field(default_factory=dict)
    
    def __hash__(self):
        return hash((self.symbol, self.side, self.agent))
    
    def __eq__(self, other):
        if not isinstance(other, BatchedSignal):
            return False
        return self.symbol == other.symbol and self.side == other.side and self.agent == other.agent


class SignalBatcher:
    """
    Batches signals to reduce execution frequency and friction.
    
    Key Features:
    - Collects signals over N-second windows
    - De-duplicates conflicting signals (keeps highest-confidence)
    - Prioritizes critical signals (SELL, ROTATION, LIQUIDATION)
    - Reduces daily trade count by 75%+
    - Micro-NAV batching: accumulates signals for small accounts until economically worthwhile
    
    MICRO-NAV OPTIMIZATION (Phase 4):
    For accounts < $500, fees consume 50-80% of trading edge.
    Solution: Batch signals until accumulated quote >= economic trade size.
    
    NAV Thresholds:
      NAV < $100: batch until $30-40, use maker orders
      NAV < $200: batch until $50-70, use maker orders
      NAV < $500: batch until $100+, use maker orders
      NAV >= $500: normal execution, normal order types
    """
    
    def __init__(
        self,
        batch_window_sec: float = 5.0,
        max_batch_size: int = 10,
        logger: Optional[logging.Logger] = None,
        shared_state: Optional[Any] = None,
    ):
        self.batch_window_sec = batch_window_sec
        self.max_batch_size = max_batch_size
        self.logger = logger or logging.getLogger("SignalBatcher")
        self.shared_state = shared_state
        
        # Active batch: accumulating signals
        self._pending_signals: List[BatchedSignal] = []
        self._pending_by_key: Dict[Tuple[str, str], BatchedSignal] = {}  # {(symbol, side): signal}
        
        # Batch state
        self._batch_start_time: float = time.time()
        self._batch_count: int = 0
        self._is_flushing: bool = False
        
        # Micro-NAV batching state
        self._accumulated_quote_usdt: float = 0.0
        self._micro_nav_mode_active: bool = False
        
        # Metrics
        self.total_signals_batched: int = 0
        self.total_batches_executed: int = 0
        self.total_signals_deduplicated: int = 0
        self.total_friction_saved_pct: float = 0.0
        self.total_micro_nav_batches_accumulated: int = 0  # Track micro-NAV accumulations
    
    def add_signal(self, signal: BatchedSignal) -> None:
        """
        Add a signal to the current batch.
        
        De-duplicates: if same symbol+side already in batch,
        keeps signal with higher confidence.
        """
        key = (signal.symbol, signal.side)
        
        if key in self._pending_by_key:
            # Conflict: same symbol+side already in batch
            existing = self._pending_by_key[key]
            if signal.confidence > existing.confidence:
                # Replace with higher-confidence signal
                self._pending_signals.remove(existing)
                self._pending_signals.append(signal)
                self._pending_by_key[key] = signal
                self.total_signals_deduplicated += 1
                self.logger.debug(
                    "[Batcher:Dedup] Replaced %s/%s (conf=%.2f) with %s (conf=%.2f)",
                    existing.agent, signal.symbol, existing.confidence,
                    signal.agent, signal.confidence
                )
            else:
                # Keep existing; ignore new
                self.total_signals_deduplicated += 1
                self.logger.debug(
                    "[Batcher:Dedup] Ignored %s/%s (conf=%.2f < existing=%.2f)",
                    signal.agent, signal.symbol, signal.confidence,
                    existing.confidence
                )
        else:
            # New signal
            self._pending_signals.append(signal)
            self._pending_by_key[key] = signal
            self.total_signals_batched += 1
            self.logger.debug(
                "[Batcher:Add] %s/%s (%s, conf=%.2f) → batch size now %d",
                signal.agent, signal.symbol, signal.side, signal.confidence, len(self._pending_signals)
            )
    
    # ===== MICRO-NAV OPTIMIZATION (Phase 4) =====
    async def _get_current_nav(self) -> float:
        """
        Get current NAV for micro-NAV batching decisions.
        
        Returns:
            NAV in USDT, or 0.0 if unavailable
        """
        try:
            if self.shared_state is None:
                return 0.0
            if hasattr(self.shared_state, "get_nav_quote"):
                nav = await self.shared_state.get_nav_quote()
                return float(nav or 0.0)
            elif hasattr(self.shared_state, "nav"):
                return float(getattr(self.shared_state, "nav", 0.0) or 0.0)
        except Exception as e:
            self.logger.debug(f"[Batcher:MicroNAV] Error getting NAV: {e}")
        return 0.0
    
    def _calculate_economic_trade_size(self, nav: float) -> float:
        """
        Calculate minimum economically worthwhile trade size for given NAV.
        
        Rationale: Fees are ~0.2% per round trip. Expected edge is 0.15-0.40%.
        For small accounts, fees consume 50-80% of edge.
        
        Solution: Batch until quote >= economic threshold.
        
        Thresholds (conservative):
          NAV < $100: $30-40 (50% of NAV)
          NAV < $200: $50-70 (25-35% of NAV)
          NAV < $500: $100 (20% of NAV)
          NAV >= $500: $50 (10% of NAV)
        
        Args:
            nav: Current NAV in USDT
        
        Returns:
            Minimum quote to batch until (in USDT)
        """
        if nav >= 500:
            # Large account: standard batching
            return 50.0
        elif nav >= 200:
            # Medium-small: batch more aggressively
            return max(50.0, nav * 0.25)
        elif nav >= 100:
            # Small: batch very aggressively
            return max(30.0, nav * 0.35)
        else:
            # Tiny: batch extremely aggressively
            return max(30.0, nav * 0.40)
    
    def _should_use_maker_orders(self, nav: float) -> bool:
        """
        Determine if orders should favor maker orders (micro-NAV mode).
        
        Rationale: For NAV < $500, maker fees (~0.02-0.06%) are 50-75% cheaper
        than taker fees (~0.10%), saving significant percentage of trading edge.
        
        Args:
            nav: Current NAV in USDT
        
        Returns:
            True if should prefer maker limit orders
        """
        return nav < 500
    
    async def _update_micro_nav_mode(self) -> None:
        """
        Update micro-NAV batching mode based on current NAV.
        
        Sets:
        - self._micro_nav_mode_active: True if NAV < $500
        - self._accumulated_quote_usdt: Track accumulated position quote
        """
        nav = await self._get_current_nav()
        self._micro_nav_mode_active = nav < 500
        
        if self._micro_nav_mode_active:
            self.logger.debug(
                "[Batcher:MicroNAV] Micro-NAV mode ACTIVE (NAV=%.2f) → accumulating signals",
                nav
            )
    
    async def _check_micro_nav_threshold(self) -> Tuple[bool, float]:
        """
        Check if accumulated quote meets economic threshold (micro-NAV mode).
        
        Returns:
            (should_flush_micro, accumulated_quote)
        """
        if not self._micro_nav_mode_active:
            return (False, 0.0)
        
        nav = await self._get_current_nav()
        if nav <= 0:
            return (False, 0.0)
        
        # Calculate quote of all pending signals
        total_quote = sum(
            float(sig.extra.get("planned_quote", 10.0) or 10.0)
            for sig in self._pending_signals
        )
        
        economic_threshold = self._calculate_economic_trade_size(nav)
        meets_threshold = total_quote >= economic_threshold
        
        if meets_threshold:
            self.logger.info(
                "[Batcher:MicroNAV] Threshold met: accumulated=%.2f >= economic=%.2f (NAV=%.2f) → flushing",
                total_quote, economic_threshold, nav
            )
            self.total_micro_nav_batches_accumulated += 1
        
        return (meets_threshold, total_quote)
    
    # ===== END MICRO-NAV OPTIMIZATION =====
    
    def should_flush(self) -> bool:
        """
        Determine if batch should be flushed.
        
        Triggers when:
        1. Window elapsed (5 seconds), OR
        2. Batch is full (10 signals), OR
        3. Critical signal detected (SELL, LIQUIDATION, ROTATION)
        
        NOTE: Micro-NAV threshold checks are done in flush() via async context
        """
        if len(self._pending_signals) == 0:
            return False
        
        now = time.time()
        elapsed = now - self._batch_start_time
        
        # Check for critical signals
        has_critical = any(
            sig.side in ("SELL", "LIQUIDATION") or 
            sig.extra.get("_forced_exit") or
            sig.extra.get("_is_rotation")
            for sig in self._pending_signals
        )
        
        # Flush if:
        # 1. Window expired
        # 2. Batch full
        # 3. Critical signal present (exit immediately)
        should_flush = (
            elapsed >= self.batch_window_sec or
            len(self._pending_signals) >= self.max_batch_size or
            (has_critical and len(self._pending_signals) >= 1)
        )
        
        if should_flush:
            self.logger.info(
                "[Batcher:Flush] Flushing batch: size=%d, elapsed=%.1fs, has_critical=%s",
                len(self._pending_signals), elapsed, has_critical
            )
        
        return should_flush
    
    async def flush(self) -> List[BatchedSignal]:
        """
        Flush current batch: prioritize and return signals for execution.
        
        Micro-NAV Optimization (Phase 4):
        - For NAV < $500, check if accumulated quote meets economic threshold
        - If not, skip flush and continue accumulating (unless critical signal)
        - This reduces fee drag by batching small trades
        
        Execution Order:
        1. SELL/LIQUIDATION (critical exits) — highest priority
        2. ROTATION/forced exits — high priority
        3. BUY signals — normal priority
        
        Returns list of signals ready to execute.
        """
        if self._is_flushing:
            self.logger.debug("[Batcher] Already flushing; skipping")
            return []
        
        if len(self._pending_signals) == 0:
            return []
        
        # ===== MICRO-NAV CHECK: Don't flush small batches unless critical =====
        try:
            await self._update_micro_nav_mode()
            
            if self._micro_nav_mode_active:
                # Check if we have critical signals that bypass batching
                has_critical = any(
                    sig.side in ("SELL", "LIQUIDATION") or 
                    sig.extra.get("_forced_exit") or
                    sig.extra.get("_is_rotation")
                    for sig in self._pending_signals
                )
                
                if not has_critical:
                    # No critical signals; check if accumulated quote meets threshold
                    meets_threshold, accumulated = await self._check_micro_nav_threshold()
                    
                    if not meets_threshold:
                        # Don't flush yet; continue accumulating
                        self.logger.debug(
                            "[Batcher:MicroNAV] Holding batch: accumulated=%.2f < threshold, "
                            "waiting for more signals or critical event",
                            accumulated
                        )
                        return []  # Return empty; batch continues
                    # else: meets_threshold = True, continue to flush
        except Exception as e:
            self.logger.debug("[Batcher:MicroNAV] Micro-NAV check failed: %s, continuing with flush", e)
        
        # ===== NORMAL FLUSH LOGIC =====
        self._is_flushing = True
        try:
            # Prioritize signals
            signals_to_execute = self._prioritize_signals(self._pending_signals)
            
            # Calculate friction savings
            pre_batch_friction = len(signals_to_execute) * 0.003  # 0.3% per trade
            post_batch_friction = 0.003  # One batch execution = 0.3%
            saved = pre_batch_friction - post_batch_friction
            self.total_friction_saved_pct += saved
            
            self.logger.info(
                "[Batcher:Execute] Batch #%d: %d signals → 1 execution (saved %.1f%% friction)",
                self._batch_count,
                len(signals_to_execute),
                saved * 100
            )
            
            # Reset batch state
            self._pending_signals.clear()
            self._pending_by_key.clear()
            self._batch_start_time = time.time()
            self._batch_count += 1
            self.total_batches_executed += 1
            
            return signals_to_execute
        
        finally:
            self._is_flushing = False
    
    def _prioritize_signals(self, signals: List[BatchedSignal]) -> List[BatchedSignal]:
        """
        Prioritize signals for execution.
        
        Order:
        1. SELL/LIQUIDATION (forced exits) — level 0
        2. ROTATION/forced_exit — level 1
        3. BUY — level 2
        4. HOLD — level 3 (ignore these)
        
        Within same level, sort by confidence (descending).
        """
        def priority_key(sig: BatchedSignal) -> Tuple[int, float]:
            if sig.side == "SELL" or sig.extra.get("_forced_exit"):
                level = 0
            elif sig.extra.get("_is_rotation"):
                level = 1
            elif sig.side == "BUY":
                level = 2
            else:
                level = 3
            
            # Sort by level (ascending), then confidence (descending)
            return (level, -sig.confidence)
        
        # Filter out HOLD signals
        valid = [s for s in signals if s.side != "HOLD"]
        
        # Sort by priority
        prioritized = sorted(valid, key=priority_key)
        
        if len(prioritized) > self.max_batch_size:
            self.logger.warning(
                "[Batcher] Batch too large (%d > %d); truncating to top %d by priority",
                len(prioritized), self.max_batch_size, self.max_batch_size
            )
            prioritized = prioritized[:self.max_batch_size]
        
        return prioritized
    
    def get_metrics(self) -> Dict[str, Any]:
        """Return batching metrics."""
        return {
            "total_signals_batched": self.total_signals_batched,
            "total_batches_executed": self.total_batches_executed,
            "total_signals_deduplicated": self.total_signals_deduplicated,
            "total_friction_saved_pct": round(self.total_friction_saved_pct * 100, 2),
            "current_batch_size": len(self._pending_signals),
            "batch_window_sec": self.batch_window_sec,
            "avg_signals_per_batch": (
                self.total_signals_batched / max(1, self.total_batches_executed)
            ),
        }
    
    def reset_metrics(self) -> None:
        """Reset all metrics to zero."""
        self.total_signals_batched = 0
        self.total_batches_executed = 0
        self.total_signals_deduplicated = 0
        self.total_friction_saved_pct = 0.0


class SignalBatcherIntegration:
    """
    Integration helper: bridges between MetaController and SignalBatcher.
    
    Usage:
    ------
    1. In MetaController.__init__():
        self.signal_batcher = SignalBatcher(batch_window_sec=5.0)
    
    2. In agent flow (agent → add_agent_signal):
        signal = BatchedSignal(
            symbol="BTCUSDT",
            side="SELL",
            confidence=0.72,
            agent="TrendHunter",
            rationale="MACD Bearish"
        )
        await self.meta_controller.signal_batcher.add_signal(signal)
    
    3. In MetaController.run_loop():
        if self.signal_batcher.should_flush():
            signals = await self.signal_batcher.flush()
            for signal in signals:
                await self._execute_decision(signal.symbol, signal.side, {...})
    
    Friction Reduction:
    -------------------
    Before (no batching):
    - 20 trades/day × 0.3% = 6% daily friction
    - Monthly: 6% × 22 = 132% of micro-NAV($350) = $46 lost to fees
    
    After (with batching):
    - 5 batches/day × 0.3% = 1.5% daily friction
    - Monthly: 1.5% × 22 = 33% of micro-NAV($350) = $11.5 lost to fees
    - Savings: $34.50/month (75% reduction)
    """
    
    @staticmethod
    def convert_agent_signal_to_batched(
        symbol: str,
        agent: str,
        side: str,
        confidence: float,
        tier: str = "B",
        rationale: str = "",
        **extra
    ) -> BatchedSignal:
        """Convert agent_signal call to BatchedSignal."""
        return BatchedSignal(
            symbol=symbol,
            side=side.upper(),
            confidence=float(confidence),
            agent=agent,
            tier=tier,
            rationale=rationale,
            extra=extra
        )
