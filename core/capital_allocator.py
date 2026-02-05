from __future__ import annotations

"""
CapitalAllocator (P9-compliant)

Purpose
-------
Allocate per-agent capital budgets based on recent performance & risk posture without ever
placing orders. Emits an AllocationPlan event and applies weights/budgets to StrategyManager.

Conformance (Spec 2025-08-20)
-----------------------------
- Emits AllocationPlan (C/Q/E contract) via SharedState.emit_event("AllocationPlan", payload)
- Emits HealthStatus events (startup, running, warning, error)
- Respects phase gates: waits for AcceptedSymbolsReady & MarketDataReady
- Reads configuration from Config under CAPITAL_ALLOCATOR.*
- Integrates RiskManager to avoid allocations that would push risk beyond caps
- Integrates StrategyManager to apply budgets/weights per agent
- No direct order placement (ExecutionManager is the single order path)

Config Keys (with defaults)
---------------------------
CAPITAL_ALLOCATOR:
  ENABLED: true
  INTERVAL_MIN: 15            # periodic planning cadence
  TIERS:                      # tiered budget split for agent classes
    core: 0.50                # e.g., stable agents
    growth: 0.35              # e.g., mid-risk agents
    experimental: 0.15        # e.g., new/volatile agents
  IPO_POOL_RATIO: 0.10        # optional carve-out for IPO/Discovery agents
  MIN_AGENT_BUDGET: 10.0      # minimum USDT to allocate per enabled agent (if possible)
  MAX_GLOBAL_ALLOC_RATIO: 0.65# cap relative to free USDT even if headroom suggests more
  REQUIRE_PERF_SOURCE: false  # if true, skip planning unless perf source is available

Performance Inputs
------------------
- Prefers PerformanceEvaluator / PerformanceMonitor metrics if available in SharedState.kpi_metrics
- Falls back to SharedState.agent_scores / roi_log if provided by the repo

Notes
-----
- Designed to be async; schedule with your AppContext.
- Does not assume specific storage internals; uses safe getters and graceful fallbacks.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Tuple


@dataclass
class AllocationTierConfig:
    core: float = 0.50
    growth: float = 0.35
    experimental: float = 0.15

class CapitalAllocator:
    component_name = "CapitalAllocator"

    def __init__(
        self,
        config: Any,
        shared_state: Any,
        risk_manager: Any,
        sstools: Any = None,
        logger: Optional[logging.Logger] = None,
        profit_target_engine: Any = None,
        strategy_manager: Any = None,
        agent_manager: Any = None,  # P9: Add AgentManager for agent discovery
        liquidation_agent: Any = None,  # OPPORTUNISTIC: Add LiquidationAgent
        **kwargs
    ) -> None:
        self.config = config
        self.ss = shared_state
        self.strategy_manager = strategy_manager
        self.agent_manager = agent_manager  # P9: Store AgentManager
        self.risk = risk_manager
        self.sstools = sstools
        self.profit_target = profit_target_engine
        self.liq_agent = liquidation_agent  # OPPORTUNISTIC: Store reference

        self.logger = logger or logging.getLogger(self.component_name)
        if not self.logger.handlers:
            self.logger.setLevel(logging.INFO)

        ca_cfg = getattr(self.config, "CAPITAL_ALLOCATOR", None) or {}
             
        self.enabled: bool = bool(ca_cfg.get("ENABLED", True))
        self.interval_min: float = float(ca_cfg.get("INTERVAL_MIN", 15))
        
        # Critical Logic Gap: Define targets for NAV-based and Cash-based allocation
        self.target_exposure_pct = float(ca_cfg.get("TARGET_EXPOSURE_PCT", 0.20))
        self.target_quote_pool_ratio = float(ca_cfg.get("TARGET_QUOTE_POOL_RATIO", 0.25))
        
        tiers = ca_cfg.get("TIERS", {}) or {}
        self.tiers = AllocationTierConfig(
            core=float(tiers.get("core", 0.50)),
            growth=float(tiers.get("growth", 0.35)),
            experimental=float(tiers.get("experimental", 0.15)),
        )
        self.ipo_pool_ratio: float = float(ca_cfg.get("IPO_POOL_RATIO", 0.10))
        self.min_agent_budget: float = float(ca_cfg.get("MIN_AGENT_BUDGET", 10.0))
        self.max_global_alloc_ratio: float = float(ca_cfg.get("MAX_GLOBAL_ALLOC_RATIO", 0.65))
        self.require_perf_source: bool = bool(ca_cfg.get("REQUIRE_PERF_SOURCE", False))
        
        # EXECUTABLE CAPITAL UNIT: Safety factor for sustainable allocations
        # Allocations must be >= (minNotional × safety_factor) to account for:
        # - Fees (0.1% taker)
        # - Rounding (stepSize)
        # - Retries (failed orders)
        # - Symbol switching (multi-symbol rotation)
        self.executable_capital_safety_factor: float = float(ca_cfg.get("EXECUTABLE_CAPITAL_SAFETY_FACTOR", 2.0))

        # Capital Hysteresis Tracking (Delegated to SharedState)
        self.hysteresis_mul: float = 1.5               # Require 1.5x capital to un-fail
        self.failure_cooldown: float = 300.0           # 5 minutes backoff
        
        # DUAL-MODE ALLOCATION: EXECUTE vs ACCUMULATE
        # EXECUTE: Full cost, real order, has fees
        # ACCUMULATE: Partial capital, no order, zero fees
        self.min_accumulation_unit: float = float(ca_cfg.get("MIN_ACCUMULATION_UNIT", 1.0))
        self.max_accumulation_per_symbol: float = float(ca_cfg.get("MAX_ACCUMULATION_PER_SYMBOL", 50.0))
        
        # Pending Liquidation Registry (Throttling)
        self.pending_liquidations: Dict[str, float] = {} # {symbol: timestamp}

        # P9 lifecycle
        self._stop_event = asyncio.Event()
        self._task: Optional[asyncio.Task] = None
    async def start(self):
        """
        P9 contract: start() spawns the periodic planning loop once (idempotent).
        """
        if getattr(self, "_task", None) and not self._task.done():
            return
        self._stop_event.clear()
        # Early health so Watchdog sees us even before the first plan
        await self.emit_health("OK", "Allocator ready")
        self._task = asyncio.create_task(self.run_forever(), name="ops.capital_allocator")
        self.logger.info("🚀 CapitalAllocator start() launched background loop.")

    async def stop(self):
        """
        P9 contract: stop() requests the loop to end and waits for it.
        """
        self._stop_event.set()
        t = getattr(self, "_task", None)
        self._task = None
        if t:
            try:
                t.cancel()
                try:
                    await asyncio.wait_for(t, timeout=5.0)
                except asyncio.CancelledError:
                    pass
            except Exception:
                self.logger.debug("CapitalAllocator stop wait failed", exc_info=True)
        await self.emit_health("OK", "Allocator stopped by request")
        self.logger.info("🛑 CapitalAllocator stopped.")


    # -------------------- Utility & Phase Gates --------------------
    def _now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    async def emit_health(self, status: str, message: str) -> None:
        """Async health emitter aligned with SharedState event bus (levels: OK|DEGRADED|ERROR)."""
        payload = {
            "component": self.component_name,
            "level": status,  # OK | DEGRADED | ERROR
            "details": {"message": message},
            "ts": self._now_iso(),
        }
        try:
            if hasattr(self.ss, "emit_event") and asyncio.iscoroutinefunction(self.ss.emit_event):
                await self.ss.emit_event("HealthStatus", payload)
            elif self.sstools and hasattr(self.sstools, "emit_event"):
                self.sstools.emit_event("HealthStatus", payload)
        except Exception:
            self.logger.debug("HealthStatus", exc_info=True)

    def _cfg(self, key: str, default: Any = None) -> Any:
        """Helper to safely read config values with fallback."""
        try:
            # Try nested key access (e.g., "CAPITAL.TRADE_UNIT_RATIO")
            if "." in key:
                parts = key.split(".")
                val = self.config
                for part in parts:
                    val = getattr(val, part, None)
                    if val is None:
                        return default
                return val
            # Direct attribute access
            return getattr(self.config, key, default)
        except Exception:
            return default

    def _gate_ready(self) -> bool:
        """Phase gates: STRICT invariants required for planning.
        
        MANDATORY (both required):
        - MarketDataReady: Prices must be available
        - AcceptedSymbolsReady: Symbol universe must be populated
        
        OPTIONAL (nice to have):
        - BalancesReady: Can proceed with stale balances in emergencies
        """
        def _ev(evt) -> bool:
            """Check if event is set."""
            try:
                return evt and hasattr(evt, "is_set") and evt.is_set()
            except Exception:
                return False

        # MANDATORY: Market data AND symbols
        mdr_ev = getattr(self.ss, "market_data_ready_event", None)
        asr_ev = getattr(self.ss, "accepted_symbols_ready_event", None)
        
        market_ready = _ev(mdr_ev)
        symbols_ready = _ev(asr_ev)
        
        # Hard invariant: BOTH must be true
        if not (market_ready and symbols_ready):
            missing = []
            if not market_ready:
                missing.append("market_data")
            if not symbols_ready:
                missing.append("symbols")
            # G001/G002: Market Data & Accepted Symbols gates - ELEVATED to INFO
            self.logger.info(f"[EXEC_BLOCK] gate=PHASE_GATES reason={'_'.join(missing).upper()} component=CapitalAllocator action=SKIP_ALLOCATION")
            return False
        
        # OPTIONAL: Balances (log warning but don't block)
        bal_ev = getattr(self.ss, "balances_ready_event", None)
        if not _ev(bal_ev):
            # G004: BalancesReady gate - SOFT (warning only, don't block)
            self.logger.info(f"[EXEC_BLOCK] gate=BALANCES_READY reason=STALE_DATA component=CapitalAllocator action=PROCEED_WITH_CAUTION")
        
        return True

    # -------------------- Performance Snapshot --------------------
    def _snapshot_performance(self) -> Tuple[Dict[str, Any], bool]:
        """Collect per-agent KPIs from SharedState or perf components.

        Returns (perf_map, perf_ok). perf_map format:
        {
          agent_name: {
            "tier": "core|growth|experimental|ipo",
            "roi": float,
            "win_rate": float,
            "drawdown": float,
            "tph": float,  # trades per hour or throughput
          }, ...
        }
        """
        perf_ok = True
        perf_map: Dict[str, Any] = {}

        try:
            # Preferred: SharedState.kpi_metrics (if your repo fills this via PerformanceMonitor/Evaluator)
            kpi = getattr(self.ss, "kpi_metrics", None)
            if isinstance(kpi, dict) and kpi:
                for agent, stats in kpi.get("per_agent", {}).items():
                    perf_map[agent] = {
                        "tier": stats.get("tier", "core"),
                        "roi": float(stats.get("roi", 0.0)),
                        "win_rate": float(stats.get("win_rate", 0.0)),
                        "drawdown": float(stats.get("max_drawdown", 0.0)),
                        "tph": float(stats.get("trades_per_hour", 0.0)),
                    }
            else:
                # Fallback: legacy structures some repos keep in SharedState
                agent_scores = getattr(self.ss, "agent_scores", None) or {}
                for agent, meta in agent_scores.items():
                    perf_map[agent] = {
                        "tier": meta.get("tier", "core"),
                        "roi": float(meta.get("roi", 0.0)),
                        "win_rate": float(meta.get("win_rate", 0.0)),
                        "drawdown": float(meta.get("drawdown", 0.0)),
                        "tph": float(meta.get("tph", 0.0)),
                    }

            # BOOTSTRAP FIX: Ensure all registered agents are in the map
            # This handles both new agents AND cold-start with zero performance data
            # P9: Use AgentManager as primary source
            source = self.agent_manager or self.strategy_manager
            agents = {}
            if source:
                if hasattr(source, "get_agents"):
                    agents = source.get_agents() or {}
                elif hasattr(source, "agents"):
                    agents = source.agents or {}
            
            # BOOTSTRAP FIX: Ensure agents is not None before iterating
            if agents:
                for name in agents:
                    if name not in perf_map:
                        # New agent or cold-start: Default to 'growth' tier with equal weight
                        perf_map[name] = {
                            "tier": "growth",
                            "roi": 0.0,
                            "win_rate": 0.5,  # Neutral assumption for bootstrap
                            "drawdown": 0.0,
                            "tph": 0.0,
                        }
                        perf_ok = True  # We have agents now
            
            # BOOTSTRAP FIX: If we still have no agents, log warning but don't crash
            if not perf_map:
                self.logger.warning("[Allocator] No agents registered; cannot allocate capital")
                perf_ok = False
                
        except Exception:
            self.logger.warning("Failed to snapshot performance; falling back to empty map", exc_info=True)
            perf_ok = False

        # BOOTSTRAP FIX: Remove hard gate - allow allocation even without performance data
        # The system must be able to allocate capital on first run to enable trading
        if not perf_ok and not perf_map:
            self.logger.warning("[Allocator] BOOTSTRAP MODE: No performance data available")
            return {}, False

        return perf_map, True

    # -------------------- Resource Headroom --------------------
    async def _free_usdt(self) -> float:
        """Read authoritative free USDT from SharedState (Point 1)."""
        try:
            if hasattr(self.ss, "get_spendable_balance"):
                # Use a small 5% reserve ratio for safety
                return await self.ss.get_spendable_balance("USDT", reserve_ratio=0.05)
            # Fallback to direct balance read
            bal = self.ss.get_balance_snapshot()
            usdt = bal.get("USDT", {})
            return float(usdt.get("free", 0.0))
        except Exception:
            self.logger.warning("[Allocator] Real funds read failed, using 0.0", exc_info=True)
        return 0.0

    async def _safe_await(self, obj):
        """Helper to await if object is awaitable, otherwise return as is."""
        if asyncio.iscoroutine(obj):
            return await obj
        return obj

    async def _nav_quote(self) -> float:
        """
        Get current NAV (Net Asset Value) in quote terms.
        Point 1 Safety: If NAV is unknown, we cannot safely calculate headroom.
        """
        try:
            # 1) Try standard SharedState method
            if hasattr(self.ss, "get_nav_quote"):
                raw_nav = await self._safe_await(self.ss.get_nav_quote())
                nav = float(raw_nav)
                if nav > 0: return nav
            
            # 2) Fallback: Check SharedState total_value mirror (updated by PnLCalculator)
            tv = float(getattr(self.ss, "total_value", 0.0))
            if tv > 0: return tv
            
            # 3) Fallback: Portfolio snapshot
            if hasattr(self.ss, "get_portfolio_snapshot"):
                snap = await self._safe_await(self.ss.get_portfolio_snapshot())
                if isinstance(snap, dict):
                    val = float(snap.get("nav", 0.0))
                    if val > 0: return val

            # 🔥 BOOTSTRAP FALLBACK
            # If NAV is still 0 (cold start), rely on free USDT to bootstrap the first trade.
            free = await self._free_usdt()
            if free > 0:
                self.logger.info(f"[Allocator] Bootstrap NAV from free USDT: {free:.2f}")
                return free
        except Exception as e:
            self.logger.debug(f"NAV read failed: {e}", exc_info=False)
        return 0.0

    def _exposure_target(self) -> float:
        """Global risk exposure limit (e.g. 0.20 = 20%)."""
        try:
            # 1) Config-driven (preferred)
            if self.config:
                ca_cfg = getattr(self.config, "CAPITAL_ALLOCATOR", None) or {}
                # Modern key
                v = ca_cfg.get("TARGET_EXPOSURE_PCT")
                if v is not None: return float(v)
                
                # Legacy keys
                v = ca_cfg.get("MAX_EXPOSURE_RATIO") or ca_cfg.get("max_exposure_ratio")
                if v is not None: return float(v)

            # 2) Fallback to instance attribute or SharedState
            return float(getattr(self.ss, "exposure_target", self.target_exposure_pct))
        except Exception:
            return self.target_exposure_pct

    def _current_exposure_quote(self) -> float:
        """Approximate current exposure in quote terms if available, else 0."""
        try:
            metrics = getattr(self.ss, "metrics", None) or {}
            return float(metrics.get("current_exposure_quote", 0.0))
        except Exception:
            return 0.0

    async def _exposure_headroom_quote(self) -> float:
        """ISSUE #1 FIX: Made async to properly await _nav_quote().
        Returns how much quote we can allocate without exceeding target exposure.
        """
        try:
            nav = await self._nav_quote()
            tgt = self._exposure_target()
            cur = self._current_exposure_quote()
            headroom = (nav * tgt) - cur
            self.logger.info(f"[Allocator] Headroom: NAV={nav:.2f} Tgt={tgt:.2f} Cur={cur:.2f} -> Headroom={headroom:.2f}")
            return float(headroom)
        except Exception:
            self.logger.warning("_exposure_headroom_quote failed", exc_info=True)
            # ISSUE #2 FIX: Return reasonable default instead of 0 in bootstrap
            try:
                if hasattr(self.ss, "is_bootstrap_mode") and self.ss.is_bootstrap_mode():
                    free = await self._free_usdt()
                    return max(0.0, free * 0.8)  # Allow 80% in bootstrap
            except Exception:
                pass
            return 0.0

    # -------------------- Reserve/Target Helpers --------------------
    def _target_free_usdt(self) -> float:
        """
        Read desired free USDT buffer from config.
        Supports multiple shapes:
          - getattr(self.config, "CAPITAL", {}).get("TARGET_FREE_USDT")
          - getattr(self.config, "capital", {}).get("target_free_usdt")
          - getattr(self.config, "CAPITAL_ALLOCATOR", {}).get("MIN_FREE_RESERVE_USDT")  # legacy fallback
        """
        try:
            # Common shapes
            cap_upper = getattr(self.config, "CAPITAL", None) or {}
            cap_lower = getattr(self.config, "capital", None) or {}
            v = cap_upper.get("TARGET_FREE_USDT", cap_lower.get("target_free_usdt", None))
            if v is None:
                # legacy/fallback knobs
                ca_cfg = getattr(self.config, "CAPITAL_ALLOCATOR", None) or {}
                v = ca_cfg.get("MIN_FREE_RESERVE_USDT", 0.0)
            return float(v or 0.0)
        except Exception:
            return 0.0

    def _min_free_reserve_usdt(self) -> float:
        """
        Minimum free USDT to always keep un-allocated (execution safety buffer).
        Reads from EXECUTION.MIN_FREE_RESERVE_USDT or execution.min_free_reserve_usdt.
        """
        try:
            exe_upper = getattr(self.config, "EXECUTION", None) or {}
            exe_lower = getattr(self.config, "execution", None) or {}
            v = exe_upper.get("MIN_FREE_RESERVE_USDT", exe_lower.get("min_free_reserve_usdt", 0.0))
            return float(v or 0.0)
        except Exception:
            return 0.0

    # -------------------- Planning Logic --------------------
    def _score_to_weight(self, roi: float, win_rate: float, drawdown: float, tph: float) -> float:
        # Simple composite score; can be replaced by a learned function
        # Normalize ranges conservatively
        roi_s = max(0.0, min(1.0, roi))
        win_s = max(0.0, min(1.0, win_rate))
        dd_s = max(0.0, min(1.0, 1.0 - max(0.0, drawdown)))  # lower drawdown -> higher score
        tph_s = max(0.0, min(1.0, tph))
        return 0.40 * roi_s + 0.35 * win_s + 0.15 * dd_s + 0.10 * tph_s

    def _tier_split(self, total: float, tier: str) -> float:
        t = tier.lower().strip()
        if t == "core":
            return total * self.tiers.core
        if t == "growth":
            return total * self.tiers.growth
        if t == "experimental":
            return total * self.tiers.experimental
        if t == "ipo":  # treated as part of discovery carve-out; handled separately
            return 0.0
        return total * self.tiers.growth  # default to growth if unknown

    def _split_by_weights(self, amount: float, weights: Dict[str, float]) -> Dict[str, float]:
        s = sum(max(0.0, w) for w in weights.values())
        if s <= 0.0:
            n = max(1, len(weights))
            return {k: amount / n for k in weights}
        return {k: amount * max(0.0, w) / s for k, w in weights.items()}

    def _identify_agent_tiers(self, perf_map: Dict[str, Any]) -> Dict[str, str]:
        tiers: Dict[str, str] = {}
        for agent, stats in perf_map.items():
            t = str(stats.get("tier", "core")).lower()
            if t not in {"core", "growth", "experimental", "ipo"}:
                t = "growth"
            tiers[agent] = t
        return tiers

    def _select_ipo_agents(self, perf_map: Dict[str, Any]) -> Dict[str, Any]:
        return {a: s for a, s in perf_map.items() if str(s.get("tier", "")).lower() == "ipo"}

    # -------------------- Symbol Metadata Normalization --------------------

    def _is_bootstrap_complete(self, free_usdt: float) -> bool:
        """
        Determine if we should exit aggressive bootstrap mode.
        Exit ONLY if we have at least one confirmed trade to ensure NAV is initialized.
        """
        try:
            metrics = getattr(self.ss, "metrics", {})
            trades = int(metrics.get("total_trades_executed", 0))

            if trades >= 1:
                return True

        except Exception:
            pass
        return False

    async def _validate_budgets(self, agent_budgets: Dict[str, float]) -> Dict[str, float]:
        """Validate that agent budgets meet minimum allocation requirements."""
        validated = {}
        for agent, budget in agent_budgets.items():
            if budget >= self.min_agent_budget:
                validated[agent] = budget
            else:
                self.logger.debug(f"[Allocator] ✗ {agent}: Budget {budget:.2f} < min_agent_budget {self.min_agent_budget:.2f}")
        return validated

    def _agent_alloc_ranges(self) -> Dict[str, Tuple[float, float]]:
        ca_cfg = getattr(self.config, "CAPITAL_ALLOCATOR", None) or {}
        ranges = ca_cfg.get("AGENT_ALLOC_RANGES", {}) or ca_cfg.get("AGENT_ALLOC_BOUNDS", {}) or {}
        normalized: Dict[str, Tuple[float, float]] = {}
        if isinstance(ranges, dict):
            for agent, bounds in ranges.items():
                try:
                    if isinstance(bounds, (list, tuple)) and len(bounds) >= 2:
                        mn = float(bounds[0]); mx = float(bounds[1])
                    elif isinstance(bounds, dict):
                        mn = float(bounds.get("min", 0.0)); mx = float(bounds.get("max", 1.0))
                    else:
                        continue
                    mn = max(0.0, min(1.0, mn))
                    mx = max(mn, min(1.0, mx))
                    normalized[str(agent)] = (mn, mx)
                except Exception:
                    continue
        return normalized

    def _apply_agent_alloc_ranges(
        self,
        agent_budgets: Dict[str, float],
        pool_quote: float,
    ) -> Dict[str, float]:
        ranges = self._agent_alloc_ranges()
        if not ranges or pool_quote <= 0:
            return agent_budgets

        result = dict(agent_budgets)
        for agent in ranges.keys():
            if agent not in result:
                result[agent] = 0.0

        override_sum = 0.0
        for agent, (mn, mx) in ranges.items():
            target = (mn + mx) / 2.0
            desired = target * pool_quote
            desired = max(mn * pool_quote, min(mx * pool_quote, desired))
            result[agent] = round(desired, 6)
            override_sum += result[agent]

        remaining_agents = [a for a in result.keys() if a not in ranges]
        remaining_pool = max(pool_quote - override_sum, 0.0)
        if remaining_agents:
            current_sum = sum(result.get(a, 0.0) for a in remaining_agents)
            if current_sum > 0:
                scale = remaining_pool / current_sum
                for a in remaining_agents:
                    result[a] = round(result.get(a, 0.0) * scale, 6)
            else:
                equal = remaining_pool / len(remaining_agents)
                for a in remaining_agents:
                    result[a] = round(equal, 6)
        elif override_sum > pool_quote and override_sum > 0:
            scale = pool_quote / override_sum
            for a in ranges.keys():
                result[a] = round(result[a] * scale, 6)

        self.logger.info(
            "[Allocator] Applied agent allocation ranges: %s (pool=%.2f)",
            ", ".join(f"{a}={ranges[a][0]*100:.0f}-{ranges[a][1]*100:.0f}%" for a in ranges),
            pool_quote,
        )
        return result


    async def _build_plan(self, perf_map: Dict[str, Any]) -> Dict[str, Any]:
        """
        P9 Structural Compliant Allocation
        
        Logic:
        1. Calculate Usable Pool based on Headroom (Risk Limit).
        2. Bootstrap Reserve Guard: Enforce $15 minimum if flat.
        3. Split Pool among Agents based on performance weights.
        4. Emit budgets per agent; symbols are selected by MetaController.
        """
        # 1. Get Capacities
        spendable_free = await self._free_usdt()
        headroom = await self._exposure_headroom_quote()
        
        is_bootstrap = True
        try:
             if self._is_bootstrap_complete(spendable_free):
                 is_bootstrap = False
             elif hasattr(self.ss, "is_bootstrap_mode") and self.ss.is_bootstrap_mode():
                 is_bootstrap = True
        except Exception: pass

        # P9 MANDATORY: Bootstrap Reserve Guard (I1 + I2 Invariant)
        # If in bootstrap (flat portfolio), enforce minimum reserve before allocation
        # PRIORITY 2 FIX (EMERGENCY OVERRIDE): Use account-aware reserve scaling
        # Default to 2.0 USDT for small accounts (< 10 USDT spendable)
        default_bootstrap_reserve = 2.0  # EMERGENCY: Reduced from 15.0 for account compatibility
        try:
            # Get min_notional from risk manager for dynamic calculation
            min_notional = float(getattr(self.risk, "min_notional_usdt", 4.0))  # Reduced default
            default_bootstrap_reserve = min_notional / 2.0  # Half of min notional
            # SAFETY: Cap reserve at 30% of spendable to prevent deadlock
            spendable = await self._free_usdt()
            if spendable > 0:
                max_allowed_reserve = spendable * 0.30
                default_bootstrap_reserve = min(default_bootstrap_reserve, max_allowed_reserve)
            self.logger.debug(f"[Allocator:Bootstrap] Calculated reserve: min_notional={min_notional:.2f} -> reserve={default_bootstrap_reserve:.2f}")
        except Exception:
            pass  # Use default
        
        bootstrap_reserve_usdt = float(self._cfg("BOOTSTRAP_RESERVE_USDT", default_bootstrap_reserve))
        
        # CRITICAL FIX: If spendable is dangerously low, clear stale quote reservations
        # This prevents deadlock when reservations from failed orders lock capital
        if is_bootstrap and spendable_free < bootstrap_reserve_usdt * 0.5:
            self.logger.warning(
                f"[Allocator:Bootstrap] ⚠️  Spendable critically low: {spendable_free:.2f} < {bootstrap_reserve_usdt * 0.5:.2f}. "
                f"Clearing stale reservations to recover capital..."
            )
            try:
                # Clear all expired quote reservations
                await self.ss.prune_reservations()
                # Recalculate after cleanup
                spendable_free = await self._free_usdt()
                self.logger.info(
                    f"[Allocator:Bootstrap] After pruning: spendable_free={spendable_free:.2f}"
                )
            except Exception as e:
                self.logger.warning(f"[Allocator:Bootstrap] Prune failed: {e}")
        
        if is_bootstrap and spendable_free < bootstrap_reserve_usdt:
            # DEADLOCK PREVENTION: Not enough capital for bootstrap minimum
            self.logger.warning(
                f"[Allocator:Bootstrap] ⛔ Reserve insufficient: spendable={spendable_free:.2f} < reserve={bootstrap_reserve_usdt:.2f}. "
                f"Blocking all allocations to prevent fragmentation."
            )
            # Structured capital block log
            self.logger.info(f"[CAPITAL_BLOCK] reason=BOOTSTRAP_RESERVE_INSUFFICIENT spendable={spendable_free:.2f} required={bootstrap_reserve_usdt:.2f} action=WAIT_FOR_INFLOW")
            
            # Emit event for observability
            try:
                await self.ss.emit_event("AllocationBlocked", {
                    "reason": "BOOTSTRAP_RESERVE_INSUFFICIENT",
                    "spendable_free": float(spendable_free),
                    "bootstrap_reserve": float(bootstrap_reserve_usdt),
                    "shortfall": float(bootstrap_reserve_usdt - spendable_free),
                    "ts": time.time(),
                    "component": "CapitalAllocator"
                })
            except Exception:
                pass
            
            return {"agent_budgets": {}, "per_agent_usdt": {}, "reason": "bootstrap_reserve_insufficient", "pool_quote": 0.0}

        if is_bootstrap:
            # Bootstrap: Reserve floor, then allocate 90% of remainder
            post_reserve_free = spendable_free - bootstrap_reserve_usdt
            usable_pool = max(0.0, post_reserve_free * 0.90)
            self.logger.info(
                f"[Allocator:Bootstrap] Reserve applied: spendable={spendable_free:.2f} - reserve={bootstrap_reserve_usdt:.2f} "
                f"= post_reserve={post_reserve_free:.2f} → usable_pool={usable_pool:.2f}"
            )
        else:
            # Normal: Cap is Headroom (Risk Limit)
            # Logic Gap Fix: Ensure we also respect the TARGET_QUOTE_POOL_RATIO (e.g. 25% of available)
            target_quote_pool = spendable_free * getattr(self, "target_quote_pool_ratio", 0.25)
            
            # usable_pool is the headroom, but we allow it to be at least target_quote_pool 
            # if we have the cash and are not severely over-exposed.
            usable_pool = max(headroom, 0.0)
            
            if usable_pool <= 0 and target_quote_pool > 0:
                # If headroom is zero but we have a target pool and available cash, 
                # we use the target pool to ensure trading (unless headroom is negative/overexposed)
                if headroom >= -1.0: # Allow slight overexposure/rounding jitter
                    usable_pool = target_quote_pool
                    self.logger.info(f"[Allocator] 🛡️ Forcing usable_pool to target_quote_pool ({usable_pool:.2f}) to enable trading.")

        if usable_pool <= 0:
             # G007: UsablePoolZero gate - ELEVATED to INFO
             self.logger.info(f"[EXEC_BLOCK] gate=USABLE_POOL_ZERO reason=INSUFFICIENT_LIQUIDITY_HEADROOM component=CapitalAllocator action=DENY_ALLOCATION")
             return {"agent_budgets": {}, "per_agent_usdt": {}, "reason": "usable_pool_zero", "pool_quote": 0.0}

        # 2. Agent Weighting (with Rejection-Aware Filtering - I2 Invariant)
        agent_tiers = self._identify_agent_tiers(perf_map)
        candidates = []
        now = time.time()
        
        # P9 MANDATORY: Rejection-Aware Agent Filtering (I2 Invariant)
        # Prevent repeated allocations to agents in deadlock loops
        deadlock_threshold = int(self._cfg("DEADLOCK_REJECTION_THRESHOLD", 10))

        ignore_csv = str(self._cfg("DEADLOCK_REJECTION_IGNORE_REASONS", "COLD_BOOTSTRAP_BLOCK,PORTFOLIO_FULL") or "")
        ignore_reasons = {r.strip().upper() for r in ignore_csv.split(",") if r.strip()}
        rej_ttl_sec = 300.0
        now_ts_local = time.time()
        total_rejection_count = 0
        try:
            if hasattr(self.ss, "rejection_counters"):
                for (sym, side, reason), count in self.ss.rejection_counters.items():
                    if str(reason).upper() in ignore_reasons:
                        continue
                    ts = self.ss.rejection_timestamps.get((sym, side, reason), now_ts_local)
                    if now_ts_local - ts > rej_ttl_sec:
                        continue
                    total_rejection_count += count
            elif hasattr(self.ss, "get_total_rejections"):
                total_rejection_count = self.ss.get_total_rejections()
        except Exception:
            pass
        
        # If system has too many rejections, signal caution but don't block allocation entirely
        # The idea: let rejections decay by not allocating new capital this cycle
        if total_rejection_count >= deadlock_threshold:
            self.logger.warning(
                f"[Allocator:Deadlock] ⚠️ Total rejections={total_rejection_count} >= threshold={deadlock_threshold}. "
                f"DEADLOCK RISK DETECTED. Skipping allocation to allow rejection decay."
            )
            # Structured deadlock risk log
            self.logger.info(f"[CAPITAL_BLOCK] reason=DEADLOCK_RISK_DETECTED rejections={total_rejection_count} threshold={deadlock_threshold} action=SKIP_ALLOCATION")
            
            try:
                await self.ss.emit_event("AllocationSkipped", {
                    "reason": "DEADLOCK_RISK",
                    "total_rejections": total_rejection_count,
                    "threshold": deadlock_threshold,
                    "ts": time.time(),
                    "component": "CapitalAllocator"
                })
            except Exception:
                pass
            
            return {"agent_budgets": {}, "per_agent_usdt": {}, "reason": "deadlock_risk_detected", "pool_quote": 0.0}
        
        for agent, stats in perf_map.items():
             if agent_tiers.get(agent) == "ipo": continue
             
             # Hysteresis check
             last_fail = getattr(self.ss, "get_agent_capital_failure", lambda x: 0)(agent)
             if last_fail > 0 and (now - last_fail < self.failure_cooldown):
                 self.logger.debug(f"[Allocator] {agent} in failure cooldown. Skipping.")
                 continue

             w = self._score_to_weight(
                stats.get("roi", 0.0), stats.get("win_rate", 0.0), 
                stats.get("drawdown", 0.0), stats.get("tph", 0.0)
            )
             tier_mul = getattr(self.tiers, agent_tiers.get(agent, "growth"), 0.35)
             candidates.append((agent, w * tier_mul))

        if not candidates:
            return {"agent_budgets": {}, "per_agent_usdt": {}, "reason": "no_candidates"}

        # 3. Allocation Split
        total_w = sum(c[1] for c in candidates)
        agent_budgets = {}
        if total_w > 0:
            for agent, w in candidates:
                # Distribute usable_pool among candidates proportionally
                agent_budgets[agent] = round((w / total_w) * usable_pool, 6)
        
        # 4. Filter by Risk & Min Budget
        agent_budgets = await self._validate_budgets(agent_budgets)
        # 4.1 Apply per-agent allocation ranges (e.g., MLForecaster 40-60%)
        agent_budgets = self._apply_agent_alloc_ranges(agent_budgets, usable_pool)
        
        # 5. Handle Opportunity Gaps (Abstracted)
        # If usable_pool is significantly below Nav-based target, emit general liquidity needed
        nav = await self._nav_quote()
        target_nav_pool = nav * self._exposure_target()
        if is_bootstrap is False and usable_pool < (target_nav_pool * 0.5):
            await self.ss.emit_event("LIQUIDITY_NEEDED", {
                "gap_usdt": target_nav_pool - usable_pool,
                "reason": "risk_capacity_underfill"
            })

        total_final = sum(agent_budgets.values())
        
        # 6. Accumulation Plan (Issue #4 FIX: Event-driven)
        if spendable_free > total_final:
            accum_pool = min(spendable_free - total_final, usable_pool * 0.1) 
            if accum_pool > self.min_accumulation_unit:
                await self.ss.emit_event("ACCUMULATION_PLAN", {
                    "pool_quote": accum_pool,
                    "max_per_symbol": self.max_accumulation_per_symbol,
                    "ts": self._now_iso()
                })

        return {
            "agent_budgets": agent_budgets,
            "per_agent_usdt": agent_budgets, # Legacy compatibility
            "effective_ts": self._now_iso(),
            "reason": "agent_focused_plan",
            "pool_quote": round(total_final, 2),
            "free_usdt": round(float(spendable_free), 6),
            "is_bootstrap": is_bootstrap,
            "accumulate_count": 0, # Logic moved to event consumer
        }

    async def _old_build_plan(self, perf_map: Dict[str, Any]) -> Dict[str, Any]:
        """❌ DEPRECATED: Use _build_plan instead. Structural violation in P6."""
        raise RuntimeError("Deprecated allocator path: _old_build_plan is removed in Phase 6")
    # -------------------- Metrics --------------------
    def _publish_metrics(self, plan: Dict[str, Any]) -> None:
        try:
            if not isinstance(plan, dict):
                return
            metrics = getattr(self.ss, "metrics", None)
            if isinstance(metrics, dict):
                metrics["alloc_last_pool_quote"] = float(plan.get("pool_quote", 0.0) or 0.0)
                metrics["alloc_last_agents"] = int(len(plan.get("per_agent_usdt", {})))
                metrics["alloc_last_free_usdt"] = float(plan.get("free_usdt", 0.0) or 0.0)
                metrics["alloc_last_keep_free"] = float(plan.get("keep_free", 0.0) or 0.0)
                metrics["alloc_last_headroom_quote"] = float(plan.get("headroom_quote", 0.0) or 0.0)
                metrics["alloc_last_ts"] = self._now_iso()
        except Exception:
            self.logger.debug("Allocation metrics publish failed", exc_info=True)

    def _validate_with_risk(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Ask RiskManager whether the implied increases are permissible; if not, reduce budgets."""
        if not plan or not plan.get("per_agent_usdt"):
            return plan
        pac = plan.get("per_agent_usdt", {}).copy()
        try:
            for agent, quote in list(pac.items()):
                ok = True
                if hasattr(self.risk, "can_increase_exposure") and callable(self.risk.can_increase_exposure):
                    # Prefer the newer signature with symbol=None & trade_quote
                    try:
                        ok = bool(self.risk.can_increase_exposure(symbol=None, trade_quote=float(quote)))
                    except TypeError:
                        # Older signature without symbol
                        ok = bool(self.risk.can_increase_exposure(float(quote)))
                if not ok:
                    pac[agent] = 0.0
            plan["per_agent_usdt"] = pac
        except Exception:
            self.logger.warning("Risk validation failed; using unadjusted plan", exc_info=True)
        return plan

    # -------------------- Side-effects --------------------
    async def emit_allocation_plan(self, plan: Dict[str, Any]) -> None:
        try:
            if hasattr(self.ss, "emit_event") and asyncio.iscoroutinefunction(self.ss.emit_event):
                await self.ss.emit_event("AllocationPlan", plan)
            elif self.sstools and hasattr(self.sstools, "emit_event"):
                self.sstools.emit_event("AllocationPlan", plan)
        except Exception:
            self.logger.warning("Failed to emit AllocationPlan event", exc_info=True)

    def apply_plan_to_strategy_manager(self, plan: Dict[str, Any]) -> None:
        if not plan or not plan.get("per_agent_usdt"):
            return
        if not self.strategy_manager:
            return
        pac: Dict[str, float] = plan.get("per_agent_usdt", {})
        try:
            # StrategyManager API varies across repos; handle common shapes
            if hasattr(self.strategy_manager, "set_agent_budget"):
                for agent, quote in pac.items():
                    self.strategy_manager.set_agent_budget(agent, float(quote))
            elif hasattr(self.strategy_manager, "set_weight"):
                # Convert budgets to normalized weights
                total = sum(max(0.0, v) for v in pac.values()) or 1.0
                for agent, quote in pac.items():
                    w = max(0.0, float(quote)) / total
                    self.strategy_manager.set_weight(agent, w)
            # Enable any agent with budget > 0
            if hasattr(self.strategy_manager, "enable"):
                for agent, quote in pac.items():
                    if quote > 0:
                        self.strategy_manager.enable(agent)
        except Exception:
            self.logger.warning("Failed to apply plan to StrategyManager", exc_info=True)

    async def plan_once(self) -> Dict[str, Any]:
        if not self.enabled:
            await self.emit_health("DEGRADED", "CapitalAllocator disabled by config")
            # G008: AllocationDisabledByConfig gate - ELEVATED to INFO
            self.logger.info(f"[EXEC_BLOCK] gate=ALLOCATION_DISABLED reason=CONFIG_DISABLED component=CapitalAllocator action=DENY_ALLOCATION")
            return {"per_agent_usdt": {}, "effective_ts": self._now_iso(), "reason": "disabled"}

        if not self._gate_ready():
            await self.emit_health("DEGRADED", "Phase gates not ready; skipping allocation")
            return {"per_agent_usdt": {}, "effective_ts": self._now_iso(), "reason": "phase_gates"}

        # ---- P9 Governance Cross: Mode Enforcement ----
        # Read the current mode from SharedState (synced by MetaController)
        current_mode = "NORMAL"
        if hasattr(self.ss, "metrics"):
            current_mode = str(self.ss.metrics.get("current_mode", "NORMAL")).upper()
            
        if current_mode in ("PAUSED", "PROTECTIVE"):
            self.logger.info(f"[Allocator] 🛡️ Governance Mode {current_mode}: Zeroing all agent budgets.")
            # Zero out all authoritative reservations
            if hasattr(self.ss, "set_authoritative_reservations"):
                # Get existing agents to zero them out
                agents = {}
                if self.agent_manager:
                    agents = self.agent_manager.get_agents() or {}
                elif self.strategy_manager:
                    agents = self.strategy_manager.agents or {}
                
                zero_reservations = {agent: 0.0 for agent in agents}
                self.ss.set_authoritative_reservations(zero_reservations)
                self.logger.info(f"[Allocator] 🛡️ Cleared {len(zero_reservations)} agent budgets due to {current_mode} mode.")
                
            await self.emit_health("DEGRADED", f"Budgets zeroed due to {current_mode} mode")
            return {"per_agent_usdt": {}, "effective_ts": self._now_iso(), "reason": f"mode_{current_mode.lower()}"}

        # P9: HARD GATE on market data readiness - MUST WAIT, not just check
        md_event = getattr(self.ss, "market_data_ready_event", None)
        if md_event and hasattr(md_event, "wait"):
            try:
                # Wait up to 60s for market data to be ready (prevents spinning)
                await asyncio.wait_for(md_event.wait(), timeout=60.0)
                self.logger.debug("[Allocator] Market data ready event satisfied")
            except asyncio.TimeoutError:
                msg = "[Allocator] Market data ready timeout; data pipeline might be stalled."
                await self.emit_health("ERROR", msg)
                self.logger.error(msg)
                return {"per_agent_usdt": {}, "effective_ts": self._now_iso(), "reason": "market_data_timeout"}
        
        # Double-check: Validate we actually have prices before planning
        if hasattr(self.ss, "has_valid_prices"):
            has_prices = self.ss.has_valid_prices()
            if asyncio.iscoroutine(has_prices):
                has_prices = await has_prices
            if not has_prices:
                msg = "[Allocator] No valid prices available; data pipeline issue."
                await self.emit_health("ERROR", msg)
                self.logger.error(msg)
                return {"per_agent_usdt": {}, "effective_ts": self._now_iso(), "reason": "no_valid_prices"}

        # Check balance staleness
        try:
            bal_ts = self.ss.metrics.get("balances_updated_at", 0)
            if datetime.now().timestamp() - bal_ts > 300: # 5 minutes
                msg = f"[Allocator] Balances are stale (last update {datetime.now().timestamp() - bal_ts:.1f}s ago)"
                await self.emit_health("ERROR", msg)
                self.logger.error(msg)
                return {"per_agent_usdt": {}, "effective_ts": self._now_iso(), "reason": "stale_balances"}
        except Exception:
            pass


        self.logger.info("[Allocator] 📊 Snapshotting agent performance...")
        perf_map, perf_ok = self._snapshot_performance()
        
        self.logger.warning(f"[Allocator] Agent Discovery: found {len(perf_map)} agents: {list(perf_map.keys())}")
        
        if self.require_perf_source and not perf_ok:
            await self.emit_health("DEGRADED", "No performance source; skipping allocation")
            return {"per_agent_usdt": {}, "effective_ts": self._now_iso(), "reason": "no_perf"}

        # P9: Process meta-healing requests before planning
        self._process_pending_reservation_requests()

        self.logger.info(f"[Allocator] 💰 Building allocation plan for {len(perf_map)} agents...")
        plan = await self._build_plan(perf_map)
        plan = self._validate_with_risk(plan)
        
        # ISSUE 5 FIX: Atomic reservation updates (apply all or none)
        pac = plan.get("per_agent_usdt", {})
        
        # Build new reservation state atomically
        new_reservations = {}
        for agent in perf_map.keys():
            new_reservations[agent] = float(pac.get(agent, 0.0))
        
        # ISSUE 5 FIX: Apply all reservations atomically via batch API
        if hasattr(self.ss, "set_authoritative_reservations"):
            self.ss.set_authoritative_reservations(new_reservations)
            funded_count = sum(1 for v in new_reservations.values() if v > 0)
            self.logger.info(f"[Allocator] Atomic batch update: {funded_count} agents funded")
        else:
            # Fallback: per-agent updates (non-atomic, legacy)
            for agent, quote in new_reservations.items():
                if hasattr(self.ss, "set_authoritative_reservation"):
                    self.ss.set_authoritative_reservation(agent, quote)
                    if quote > 0:
                        self.logger.debug(f"[Allocator] Reserved {quote:.2f} for {agent}")

        await self.emit_allocation_plan(plan)
        self.apply_plan_to_strategy_manager(plan)
        self._publish_metrics(plan)

        agent_cnt = len(pac)
        msg = f"Authoritative Allocation plan emitted; agents={agent_cnt}, pool={plan.get('pool_quote', 0.0)}"
        await self.emit_health("OK", msg)
        return plan

    async def run_forever(self) -> None:
        """Periodic planner loop. Schedule from AppContext at P9."""
        self.logger.warning("=" * 80)
        self.logger.warning("🚀 CAPITAL ALLOCATOR STARTING - Planning loop initialized")
        self.logger.warning(f"   Interval: {self.interval_min} minutes")
        self.logger.warning(f"   Component: {self.component_name}")
        self.logger.warning("=" * 80)
        
        await self.emit_health("OK", "Allocator loop starting")
        
        while not self._stop_event.is_set():
            try:
                self.logger.info("[Allocator] 🔄 Starting capital allocation cycle...")
                
                # 1. Plan
                await self.plan_once()
                
                # 2. Wait for next interval OR trigger
                sleep_s = max(5.0, 60.0 * float(self.interval_min))
                replan_ev = getattr(self.ss, "replan_request_event", None)
                
                if replan_ev and hasattr(replan_ev, "wait"):
                    # Create tasks for waiting
                    stop_task = asyncio.create_task(self._stop_event.wait())
                    replan_task = asyncio.create_task(replan_ev.wait())
                    
                    done, pending = await asyncio.wait(
                        [stop_task, replan_task],
                        timeout=sleep_s,
                        return_when=asyncio.FIRST_COMPLETED
                    )
                    
                    # Cleanup pending
                    for t in pending:
                        t.cancel()
                        
                    if replan_task in done:
                        replan_ev.clear()
                        self.logger.info("[Allocator] Re-planning triggered by event request")
                else:
                    await asyncio.sleep(sleep_s)
                    
            except asyncio.CancelledError:
                await self.emit_health("Running", "Allocator loop cancelled")
                raise
            except Exception as e:
                await self.emit_health("ERROR", f"Planning error: {e!r}")
                self.logger.exception("CapitalAllocator cycle failed")
    def _process_pending_reservation_requests(self):
        """
        P9: Authoritatively process meta-healing requests from SharedState.
        Ensures that MetaController's spent or returned budget is reflected
        in the authoritative reservations using jurisdictional APIs.
        """
        if not hasattr(self.ss, "get_pending_reservation_requests"):
            return
            
        to_process = self.ss.get_pending_reservation_requests(drain=True)
        if not to_process:
            return
            
        self.logger.info(f"[Allocator] Processing {len(to_process)} pending reservation adjustments via SharedState API")
        
        # P9 Structural Fix: Use authorized API instead of direct dict access
        if hasattr(self.ss, "apply_reservation_batch"):
            self.ss.apply_reservation_batch(to_process)
            self.logger.info(f"[Allocator] Jurisdictional authority applied {len(to_process)} adjustments")
        else:
            self.logger.warning("[Allocator] SharedState missing apply_reservation_batch; structural violation imminent")
