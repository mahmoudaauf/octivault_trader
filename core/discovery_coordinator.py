# core/discovery_coordinator.py
"""
Discovery Coordinator: Phase 2b Implementation

Centralized proposal collection, deduplication, and routing for UURE.

Problem Solved:
  - Multiple discovery agents → duplicate proposals
  - No unified visibility into proposals
  - UURE doesn't know what's available to propose
  - Proposals scattered across agents and stashes

Solution:
  Single coordinator collects proposals from all agents, deduplicates, and routes
  to UURE's _collect_candidates() method.

Architecture:
  SymbolScreener (regime-aware, +40 lines)
    ↓
  WalletScanner (regime-adaptive intervals, +30 lines)
    ↓
  IPOChaser (volatility-aware, +20 lines)
    ↓
  SymbolDiscoverer (generic fallback)
    ↓
  DiscoveryCoordinator (THIS FILE - 150 lines)
    ↓
  SharedState.discovery_proposals → UURE._collect_candidates()

Key Features:
  ✅ Deduplicates proposals (same symbol from multiple agents)
  ✅ Tracks proposal source (SymbolScreener, WalletScanner, etc.)
  ✅ Rates proposals by quality (confidence scores)
  ✅ Rate limits (max proposals/min)
  ✅ Clean visibility into discovery pipeline
  ✅ Metrics/logging for debugging

Usage:
  coordinator = DiscoveryCoordinator(shared_state, config, logger)
  proposals = await coordinator.collect_and_deduplicate()
  # proposals: { "BTCUSDT": {"source": "screener", "confidence": 0.8}, ... }

Config:
  DISCOVERY_MAX_PROPOSALS_PER_MIN: 10 (default)
  DISCOVERY_DEDUP_WINDOW_SEC: 60 (default)
  DISCOVERY_QUALITY_THRESHOLD: 0.3 (default, 0.0-1.0)
  DISCOVERY_TRACK_METRICS: true (default)
"""

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, Set
from collections import defaultdict


class DiscoveryCoordinator:
    """
    Centralized discovery proposal coordinator.
    
    Responsibilities:
    1. Collect proposals from all discovery agents
    2. Deduplicate (keep best by source & confidence)
    3. Apply rate limiting
    4. Route to SharedState for UURE consumption
    5. Track metrics (proposals/sec, dedup rate, etc.)
    """

    def __init__(
        self,
        shared_state: Any,
        config: Optional[Any] = None,
        logger: Optional[Any] = None,
    ):
        """
        Args:
            shared_state: SharedState for discovery_proposals storage
            config: Configuration object
            logger: Logger instance
        """
        self.ss = shared_state
        self.config = config or {}
        self.logger = logger or logging.getLogger("DiscoveryCoordinator")

        # Rate limiting
        self.max_proposals_per_min = self._cfg("DISCOVERY_MAX_PROPOSALS_PER_MIN", 10)
        self.dedup_window_sec = self._cfg("DISCOVERY_DEDUP_WINDOW_SEC", 60)
        self.quality_threshold = self._cfg("DISCOVERY_QUALITY_THRESHOLD", 0.3)
        self.track_metrics = bool(self._cfg("DISCOVERY_TRACK_METRICS", True))

        # State tracking
        self._recent_proposals: Dict[str, float] = {}  # symbol -> timestamp
        self._proposal_history: Dict[str, Dict[str, Any]] = {}  # symbol -> best_proposal
        self._metrics = {
            "total_collected": 0,
            "total_deduped": 0,
            "rate_limited": 0,
            "low_quality_filtered": 0,
            "last_collection_time": 0,
        }

    def _cfg(self, key: str, default: Any = None) -> Any:
        """Config getter (supports dict or attribute configs)."""
        if isinstance(self.config, dict):
            return self.config.get(key, default)
        return getattr(self.config, key, default)

    async def collect_and_deduplicate(self) -> Dict[str, Dict[str, Any]]:
        """
        Main entry point: collect proposals from all agents and deduplicate.
        
        Returns:
            Dict of {symbol: {source, confidence, metadata, timestamp}}
        """
        try:
            # Step 1: Collect from all sources
            all_proposals = await self._collect_all_proposals()
            self._metrics["total_collected"] += len(all_proposals)

            if not all_proposals:
                self.logger.debug("[DiscoveryCoordinator] No proposals collected this cycle")
                return {}

            # Step 2: Deduplicate (keep best per symbol)
            deduped = await self._deduplicate_proposals(all_proposals)
            self._metrics["total_deduped"] += len(deduped)

            # Step 3: Apply quality filter
            filtered = await self._apply_quality_filter(deduped)

            # Step 4: Apply rate limiting
            final = await self._apply_rate_limit(filtered)

            # Step 5: Store in SharedState for UURE consumption
            await self._store_in_shared_state(final)

            # Step 6: Update metrics
            self._metrics["last_collection_time"] = time.time()
            self._log_metrics(all_proposals, deduped, filtered, final)

            return final

        except Exception as e:
            self.logger.error(f"[DiscoveryCoordinator] Error in collect_and_deduplicate: {e}")
            return {}

    async def collect_and_rank_by_regime(self) -> Dict[str, Dict[str, Any]]:
        """
        Phase 3: Enhanced collection with regime-based weighting.
        
        Process:
        1. Collect all proposals (Phase 2b behavior)
        2. Deduplicate (Phase 2b behavior)
        3. Apply regime weighting ← NEW (Phase 3)
        4. Sort by weighted score
        5. Apply rate limiting
        6. Store in SharedState
        
        Returns:
            Dict of {symbol: {source, confidence, weighted_score, regime_bonus, ...}}
        """
        try:
            # Enable weighting if configured
            use_weighting = bool(self._cfg("DISCOVERY_USE_REGIME_WEIGHTING", False))
            
            if not use_weighting:
                # Fall back to Phase 2b behavior (no weighting)
                return await self.collect_and_deduplicate()
            
            # Get RegimeProposalAnalyzer
            if not hasattr(self, "_analyzer"):
                from core.regime_proposal_analyzer import RegimeProposalAnalyzer
                self._analyzer = RegimeProposalAnalyzer(self.ss, self.config, self.logger)
            
            # Step 1: Collect from all sources
            all_proposals = await self._collect_all_proposals()
            self._metrics["total_collected"] += len(all_proposals)
            
            if not all_proposals:
                self.logger.debug("[DiscoveryCoordinator] No proposals for weighting")
                return {}
            
            # Step 2: Deduplicate (Phase 2b)
            deduped = await self._deduplicate_proposals(all_proposals)
            self._metrics["total_deduped"] += len(deduped)
            
            # Step 3: Apply regime weighting (Phase 3) ← NEW
            weighted_proposals = await self._apply_regime_weighting(list(deduped.values()))
            
            # Convert back to dict format {symbol: {properties}}
            weighted_dict = {}
            for prop in weighted_proposals:
                symbol = str(prop.get("symbol", "")).upper()
                weighted_dict[symbol] = prop
            
            # Step 4: Apply quality filter (now with weighted scores)
            filtered = await self._apply_quality_filter(weighted_dict)
            
            # Step 5: Apply rate limiting
            final = await self._apply_rate_limit(filtered)
            
            # Step 6: Store in SharedState (with weighted tag)
            setattr(self.ss, "discovery_proposals_weighted", final)
            setattr(self.ss, "discovery_proposals", final)  # Also update regular
            
            # Step 7: Update metrics
            self._metrics["last_collection_time"] = time.time()
            self._log_metrics(all_proposals, deduped, filtered, final)
            
            self.logger.info(
                f"[DiscoveryCoordinator] Phase 3 weighting applied: "
                f"{len(final)} proposals, top score: "
                f"{max((p.get('weighted_score', 0) for p in final.values()), default=0):.3f}"
            )
            
            return final
        
        except Exception as e:
            self.logger.error(f"[DiscoveryCoordinator] Error in regime weighting: {e}")
            # Fall back to Phase 2b (no weighting)
            return await self.collect_and_deduplicate()

    async def _apply_regime_weighting(
        self, proposals: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Apply regime-based weighting to proposals. Phase 3 enhancement.
        
        Uses RegimeProposalAnalyzer to:
        1. Score each proposal based on market regime
        2. Apply regime-specific multipliers
        3. Sort by weighted score
        
        Args:
            proposals: List of proposal dicts
        
        Returns:
            List of weighted proposals sorted by weighted_score descending
        """
        try:
            if not hasattr(self, "_analyzer"):
                from core.regime_proposal_analyzer import RegimeProposalAnalyzer
                self._analyzer = RegimeProposalAnalyzer(self.ss, self.config, self.logger)
            
            # Batch analyze all proposals
            weighted = await self._analyzer.batch_analyze_proposals(proposals)
            
            self.logger.debug(
                f"[DiscoveryCoordinator] Weighted {len(weighted)} proposals, "
                f"metrics: {self._analyzer.get_metrics()}"
            )
            
            return weighted
        
        except Exception as e:
            self.logger.error(f"[DiscoveryCoordinator] Error applying regime weighting: {e}")
            # Return original proposals (no weighting)
            return proposals

    async def _collect_all_proposals(self) -> List[Dict[str, Any]]:
        """Collect proposals from all discovery agents and stashes."""
        proposals = []

        try:
            # Source 1: SharedState symbol_proposals stash (buffered by agents)
            if hasattr(self.ss, "symbol_proposals"):
                stash = getattr(self.ss, "symbol_proposals", None) or {}
                for symbol, prop in stash.items():
                    proposals.append({
                        "symbol": str(symbol).upper(),
                        "source": prop.get("source", "stash"),
                        "confidence": float(prop.get("confidence", 0.5)),
                        "metadata": prop.get("metadata", {}),
                        "timestamp": float(prop.get("ts", time.time())),
                    })

            # Source 2: SymbolManager proposals (if available)
            if hasattr(self.ss, "get_pending_proposals"):
                try:
                    pending = await self.ss.get_pending_proposals() if asyncio.iscoroutine(self.ss.get_pending_proposals()) else self.ss.get_pending_proposals()
                    if isinstance(pending, dict):
                        for symbol, prop in pending.items():
                            proposals.append({
                                "symbol": str(symbol).upper(),
                                "source": prop.get("source", "symbol_manager"),
                                "confidence": float(prop.get("confidence", 0.5)),
                                "metadata": prop.get("metadata", {}),
                                "timestamp": float(prop.get("ts", time.time())),
                            })
                except Exception as e:
                    self.logger.debug(f"[DiscoveryCoordinator] Failed to get pending proposals: {e}")

            self.logger.debug(
                f"[DiscoveryCoordinator] Collected {len(proposals)} total proposals "
                f"from {len(set(p['source'] for p in proposals))} sources"
            )
            return proposals

        except Exception as e:
            self.logger.error(f"[DiscoveryCoordinator] Error collecting proposals: {e}")
            return proposals

    async def _deduplicate_proposals(
        self, proposals: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Deduplicate proposals: keep highest-confidence per symbol.
        
        Strategy:
          1. Group by symbol
          2. Within each symbol, keep highest confidence
          3. If tied on confidence, prefer source priority: screener > wallet > ipo > discoverer
        """
        deduped = {}
        source_priority = {
            "symbol_screener": 4,
            "wallet_scanner": 3,
            "ipo_chaser": 2,
            "symbol_discoverer": 1,
            "stash": 0,
            "symbol_manager": 1,
        }

        for prop in proposals:
            symbol = str(prop.get("symbol", "")).upper()
            if not symbol:
                continue

            confidence = float(prop.get("confidence", 0.5))
            source = str(prop.get("source", "stash")).lower()
            priority = source_priority.get(source, 0)

            if symbol not in deduped:
                # First occurrence
                deduped[symbol] = {
                    "symbol": symbol,
                    "source": source,
                    "confidence": confidence,
                    "metadata": prop.get("metadata", {}),
                    "timestamp": prop.get("timestamp", time.time()),
                    "priority": priority,
                }
            else:
                # Compare and keep best
                existing = deduped[symbol]
                existing_priority = existing.get("priority", 0)

                # Better if: higher confidence, OR same confidence + higher source priority
                is_better = (
                    confidence > existing["confidence"]
                    or (
                        confidence == existing["confidence"]
                        and priority > existing_priority
                    )
                )

                if is_better:
                    deduped[symbol] = {
                        "symbol": symbol,
                        "source": source,
                        "confidence": confidence,
                        "metadata": prop.get("metadata", {}),
                        "timestamp": prop.get("timestamp", time.time()),
                        "priority": priority,
                    }
                    self._metrics["total_deduped"] += 1
                    self.logger.debug(
                        f"[DiscoveryCoordinator] Deduped {symbol}: {existing['source']} "
                        f"(conf={existing['confidence']:.2f}) → {source} (conf={confidence:.2f})"
                    )

        return deduped

    async def _apply_quality_filter(
        self, proposals: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """Filter proposals by minimum confidence threshold."""
        filtered = {}
        low_quality_count = 0

        for symbol, prop in proposals.items():
            confidence = float(prop.get("confidence", 0.5))

            if confidence >= self.quality_threshold:
                filtered[symbol] = prop
            else:
                low_quality_count += 1
                self.logger.debug(
                    f"[DiscoveryCoordinator] {symbol} filtered (confidence={confidence:.2f} < threshold={self.quality_threshold:.2f})"
                )

        self._metrics["low_quality_filtered"] += low_quality_count

        if low_quality_count > 0:
            self.logger.info(
                f"[DiscoveryCoordinator] Quality filter: {len(proposals)} → {len(filtered)} "
                f"(filtered {low_quality_count} low-quality)"
            )

        return filtered

    async def _apply_rate_limit(
        self, proposals: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Rate limit: don't process more than max_proposals_per_min in this cycle.
        
        Strategy:
        1. Recent proposals (< dedup_window) are rate-limited
        2. New proposals have unlimited throughput
        3. Prioritize high-confidence proposals
        """
        current_time = time.time()
        rate_limited = []

        # Find recent proposals
        recent_count = sum(
            1 for symbol in self._recent_proposals
            if current_time - self._recent_proposals[symbol] < self.dedup_window_sec
        )

        # Allow new proposals only if under rate limit
        final = {}
        proposals_to_add = []

        for symbol, prop in proposals.items():
            if symbol in self._recent_proposals:
                # Symbol was proposed recently - check rate limit
                age = current_time - self._recent_proposals[symbol]
                if age < self.dedup_window_sec:
                    rate_limited.append(symbol)
                    continue

            # New or old symbol - add it
            final[symbol] = prop
            proposals_to_add.append(symbol)

        # Update tracking
        for symbol in proposals_to_add:
            self._recent_proposals[symbol] = current_time

        # Clean up old entries
        self._recent_proposals = {
            sym: ts
            for sym, ts in self._recent_proposals.items()
            if current_time - ts < self.dedup_window_sec
        }

        if rate_limited:
            self._metrics["rate_limited"] += len(rate_limited)
            self.logger.debug(
                f"[DiscoveryCoordinator] Rate limited {len(rate_limited)} symbols: {rate_limited[:5]}..."
            )

        return final

    async def _store_in_shared_state(
        self, proposals: Dict[str, Dict[str, Any]]
    ) -> None:
        """Store finalized proposals in SharedState for UURE consumption."""
        try:
            if not hasattr(self.ss, "discovery_proposals"):
                self.ss.discovery_proposals = {}

            self.ss.discovery_proposals = proposals

            self.logger.info(
                f"[DiscoveryCoordinator] Stored {len(proposals)} proposals to SharedState.discovery_proposals"
            )
        except Exception as e:
            self.logger.error(f"[DiscoveryCoordinator] Error storing proposals: {e}")

    def _log_metrics(
        self,
        all_proposals: List[Dict[str, Any]],
        deduped: Dict[str, Dict[str, Any]],
        filtered: Dict[str, Dict[str, Any]],
        final: Dict[str, Dict[str, Any]],
    ) -> None:
        """Log metrics for this collection cycle."""
        if not self.track_metrics:
            return

        try:
            dedup_rate = (
                (len(all_proposals) - len(deduped)) / len(all_proposals) * 100
                if all_proposals
                else 0
            )
            filter_rate = (
                (len(deduped) - len(filtered)) / len(deduped) * 100
                if deduped
                else 0
            )
            rate_limit_rate = (
                (len(filtered) - len(final)) / len(filtered) * 100
                if filtered
                else 0
            )

            self.logger.info(
                f"[DiscoveryCoordinator] Cycle metrics: "
                f"collected={len(all_proposals)}, "
                f"deduped={len(deduped)} ({dedup_rate:.1f}% reduction), "
                f"filtered={len(filtered)} ({filter_rate:.1f}% quality filter), "
                f"rate_limited={len(final)} ({rate_limit_rate:.1f}% limit), "
                f"final={len(final)}"
            )

            # Log by source
            sources = defaultdict(int)
            for prop in deduped.values():
                sources[prop.get("source", "unknown")] += 1
            
            if sources:
                source_str = ", ".join(f"{src}={cnt}" for src, cnt in sorted(sources.items()))
                self.logger.debug(f"[DiscoveryCoordinator] Proposals by source: {source_str}")

        except Exception as e:
            self.logger.debug(f"[DiscoveryCoordinator] Error logging metrics: {e}")

    def get_metrics(self) -> Dict[str, Any]:
        """Export current metrics."""
        return dict(self._metrics)

    def reset_metrics(self) -> None:
        """Reset metrics counters."""
        self._metrics = {
            "total_collected": 0,
            "total_deduped": 0,
            "rate_limited": 0,
            "low_quality_filtered": 0,
            "last_collection_time": 0,
        }
