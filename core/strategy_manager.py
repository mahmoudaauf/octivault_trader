from core.component_status_logger import ComponentStatusLogger as CSL
import logging
import asyncio
import random
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
from collections import defaultdict
import time

logger = logging.getLogger("StrategyManager")

class StrategyManager:
    def __init__(self, shared_state, config, database_manager=None, *, agents=None, logger: logging.Logger = None, **kwargs):
        # Accept both legacy and keyword styles; ignore extra kwargs safely
        self.logger = logger or logging.getLogger(__name__).getChild("StrategyManager")
        self.config = config
        self.shared_state = shared_state
        self.database_manager = database_manager
        self.agents = agents
        self.active_strategies = []

        self.interval = float(getattr(config, "STRATEGY_ANALYSIS_INTERVAL", 300.0))
        self.jitter_pct = float(getattr(config, "STRATEGY_INTERVAL_JITTER_PCT", 0.05))
        self.agent_timeout = float(getattr(config, "STRATEGY_AGENT_TIMEOUT_S", 15.0))
        self.max_concurrency = int(getattr(config, "STRATEGY_MAX_CONCURRENCY", 8))
        self.base_ccy = getattr(config, "BASE_CURRENCY", "USDT")

        # Readiness gating knobs
        self.require_market_data_ready: bool = bool(getattr(config, "REQUIRE_MARKET_DATA_READY", True))
        self.require_balances_ready: bool = bool(getattr(config, "REQUIRE_BALANCES_READY", True))
        self.require_ops_plane_ready: bool = bool(getattr(config, "REQUIRE_OPS_PLANE_READY", True))
        self.readiness_log_interval_s: float = float(getattr(config, "READINESS_LOG_INTERVAL_S", 10.0))
        self._last_ready_log_ts: float = 0.0

        self._last_pull_ts = defaultdict(float)
        self._stop_event = asyncio.Event()
        self._sem = asyncio.Semaphore(self.max_concurrency)

        # Strategy controls
        self._enabled: Dict[str, bool] = {}
        self._weights: Dict[str, float] = {}

        # --- Order guard knobs (with multiple aliases for config sources) ---
        self.exec_max_concurrency = int(getattr(config, "EXECUTION_MAX_CONCURRENCY",
                                    getattr(config, "execution_max_concurrency",
                                    getattr(config, "EXECUTION__MAX_CONCURRENCY", 1))))
        self.order_guard_require_free_usdt = float(getattr(config, "STRATEGY_ORDER_GUARD_REQUIRE_FREE_USDT_GTE",
                                                getattr(config, "strategy_manager_order_guard_require_free_usdt_gte",
                                                getattr(config, "strategy_manager.order_guard.require_free_usdt_gte", 0.0))))
        self.order_guard_cooldown_s = float(getattr(config, "STRATEGY_ORDER_GUARD_COOLDOWN_S_PER_SYMBOL",
                                          getattr(config, "strategy_manager_order_guard_cooldown_s_per_symbol",
                                          getattr(config, "strategy_manager.order_guard.cooldown_s_per_symbol", 0.0))))

        # Execution forwarding state
        self.execution_manager = None
        self._last_signal_ts_per_symbol: Dict[str, float] = {}
        self._inflight_exec = 0
        self._exec_gate_sem = asyncio.Semaphore(max(1, self.exec_max_concurrency))

        # Basic validation (only what you actually have)
        getter = getattr(shared_state, "get_latest_price", None)
        if not (callable(getter) and asyncio.iscoroutinefunction(getter)):
            raise TypeError("Invalid shared_state: need async get_latest_price")

        self.logger.info(
            "StrategyManager initialized (interval=%ss±%s%%, base=%s, timeout=%ss, max_conc=%s).",
            self.interval, int(self.jitter_pct * 100), self.base_ccy, self.agent_timeout, self.max_concurrency
        )

        # Announce via CSL (safe in __init__)
        CSL.log_status("StrategyManager", "Initialized", f"Ready (interval={self.interval}s)")

        # If you want SharedState mirrored here, schedule it (don’t await in __init__)
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self.shared_state.update_component_status(
                "StrategyManager", "Initialized", f"Ready (interval={self.interval}s)"
            ))
        except RuntimeError:
            # no loop yet; fine to skip — CSL will have mirrored if bound
            pass

    async def initialize_strategies(self):
        self.logger.info("Initializing trading strategies from AgentManager...")
        self.active_strategies.clear()
        source = {}
        am = self.agents
        if isinstance(am, dict):
            source = am
        else:
            # AgentManager-style container
            if hasattr(am, "agents") and isinstance(getattr(am, "agents"), dict):
                source = getattr(am, "agents")
        for name, agent in (source or {}).items():
            if getattr(agent, "agent_type", None) == "strategy":
                self.active_strategies.append(agent)
                self._enabled.setdefault(name, True)
                self._weights.setdefault(name, 1.0)
                self.logger.info("✅ Registered strategy agent: %s", name)

        try:
            await self.shared_state.update_component_status(
                "StrategyManager", "Operational",
                f"Strategies initialized: {len(self.active_strategies)}"
            )
        except Exception:
            pass

    def set_execution_manager(self, execution_manager):
        """Wire the ExecutionManager for forwarding orders."""
        self.execution_manager = execution_manager
        self.logger.info("ExecutionManager wired into StrategyManager.")

    async def analyze_strategies(self):
        """Pull a portfolio snapshot and return NAV in base_ccy."""
        snap = self._readiness_snapshot()
        if (self.require_market_data_ready and not snap.get("market_data_ready", True)) or \
           (self.require_balances_ready and not snap.get("balances_ready", True)) or \
           (self.require_ops_plane_ready and not snap.get("ops_plane_ready", True)):
            self.logger.info("Skipping analysis; readiness gates not satisfied: %s", snap)
            return 0.0

        self.logger.debug("Analyzing strategy performance...")
        try:
            snap = await self.shared_state.get_portfolio_snapshot()
            nav = float(snap.get("nav", 0.0))
            self.logger.info("Total portfolio value: %.3f %s", nav, self.base_ccy)
            try:
                ready = self._readiness_snapshot()
                if nav <= 0.0 and ready.get("balances_ready", True):
                    await self.shared_state.update_component_status("StrategyManager", "Degraded", "NAV<=0 with balances_ready")
            except Exception:
                pass
            return nav
        except Exception:
            self.logger.exception("get_portfolio_snapshot failed.")
            return 0.0

    def _readiness_snapshot(self) -> Dict[str, bool]:
        """Best-effort readiness view from SharedState; falls back to ready if unavailable."""
        snap: Dict[str, bool] = {
            "accepted_symbols_ready": True,
            "balances_ready": True,
            "market_data_ready": True,
            "ops_plane_ready": True,
        }
        try:
            getter = getattr(self.shared_state, "get_readiness_snapshot", None)
            if callable(getter):
                got = getter()
                if isinstance(got, dict):
                    snap.update({k: bool(v) for k, v in got.items() if k in snap})
        except Exception:
            pass
        return snap

    async def _get_free_usdt(self) -> float:
        """Best-effort free USDT (or base_ccy) balance for gating."""
        try:
            # Preferred: SharedState.free_usdt()
            fn = getattr(self.shared_state, "free_usdt", None)
            if callable(fn):
                val = await fn() if asyncio.iscoroutinefunction(fn) else fn()
                return float(val or 0.0)
        except Exception:
            pass
        # Fallback: spendable balance in base ccy
        try:
            fn2 = getattr(self.shared_state, "get_spendable_balance", None)
            if callable(fn2):
                val = await fn2(self.base_ccy, reserve_ratio=0.0) if asyncio.iscoroutinefunction(fn2) else fn2(self.base_ccy, reserve_ratio=0.0)
                return float(val or 0.0)
        except Exception:
            pass
        # Last resort: 0.0
        return 0.0

    def _cooldown_left(self, symbol: str) -> float:
        """Seconds of cooldown left for a symbol (0 if none)."""
        last = float(self._last_signal_ts_per_symbol.get(symbol, 0.0))
        try:
            if hasattr(self.shared_state, "get_last_exit_ts"):
                last = max(last, float(self.shared_state.get_last_exit_ts(symbol) or 0.0))
            elif hasattr(self.shared_state, "last_exit_ts"):
                last = max(last, float(self.shared_state.last_exit_ts.get(symbol, 0.0) or 0.0))
        except Exception:
            pass
        cd = float(self.order_guard_cooldown_s or 0.0)
        if cd <= 0:
            return 0.0
        now = time.time()
        left = (last + cd) - now
        return left if left > 0 else 0.0

    async def _can_forward(self, symbol: str) -> tuple[bool, str, dict]:
        """Check hard guards before forwarding to ExecutionManager."""
        meta = {"inflight": int(self._inflight_exec), "max_conc": int(self.exec_max_concurrency)}
        # Concurrency gate (non-blocking)
        if self._inflight_exec >= self.exec_max_concurrency:
            return False, "MaxConcurrency", meta
        # Cooldown gate
        left = self._cooldown_left(symbol)
        if left > 0:
            meta.update({"cooldown_left_s": round(left, 2)})
            return False, "Cooldown", meta
        # Free USDT gate
        try:
            free_usdt = await self._get_free_usdt()
        except Exception:
            free_usdt = 0.0
        req = float(self.order_guard_require_free_usdt or 0.0)
        meta.update({"free_usdt": round(free_usdt, 8), "require_free_usdt_gte": req})
        if free_usdt < req:
            return False, "FreeUSDT", meta
        return True, "OK", meta

    async def _safe_call(self, coro, name: str):
        """Run an agent coroutine under concurrency and timeout guards."""
        try:
            async with self._sem:
                return await asyncio.wait_for(coro, timeout=self.agent_timeout)
        except asyncio.TimeoutError:
            self.logger.warning("Agent timed out: %s", name)
        except Exception:
            self.logger.exception("Agent error: %s", name)
        return None

    async def forward_signal(self, *, symbol: str, side: str, confidence: float = 0.0,
                             planned_quote: Optional[float] = None, tag: str = "meta/Agent") -> Optional[Dict[str, Any]]:
        """
        Hard-guarded forwarding to ExecutionManager. Returns EM result or None if gated.
        """
        if not self.execution_manager:
            self.logger.warning("Gated: NoExecutionManager | symbol=%s side=%s", symbol, side)
            return None

        ok, reason, meta = await self._can_forward(symbol)
        if not ok:
            self.logger.info("Gated: %s | symbol=%s side=%s meta=%s", reason, symbol, side, meta)
            try:
                await self.shared_state.update_component_status("StrategyManager", "Degraded",
                    f"Gated:{reason} symbol={symbol} side={side} meta={meta}")
            except Exception:
                pass
            return None

        # P9 Authoritative Budget Guard
        # Block BUY signals if the agent has no allocated capital (unless allocator disabled)
        allocator_enabled = bool(getattr(self.config, "CAPITAL_ALLOCATOR_ENABLED", True))
        if allocator_enabled and side.lower() == "buy" and hasattr(self.shared_state, "get_authoritative_reservation"):
            agent_id = tag.split("/")[-1] if "/" in tag else tag
            # We treat 'meta' or 'Agent' as generic fallback, but real agents have specific IDs.
            # If specific budget prevents it, block.
            budget = self.shared_state.get_authoritative_reservation(agent_id)
            if budget <= 0:
                # Optional: Check if we are in bootstrap mode? No, strict budget applies always now.
                self.logger.debug("Gated: NoBudget | agent=%s symbol=%s budget=%.2f", agent_id, symbol, budget)
                return None

        # Pass-through to EM with non-blocking concurrency accounting
        self._inflight_exec += 1
        try:
            async with self._exec_gate_sem:
                # Prefer quote-amount path when provided
                if planned_quote and planned_quote > 0 and side.lower() == "buy":
                    res = await self.execution_manager.execute_trade(
                        symbol=symbol, side=side, planned_quote=float(planned_quote), tag=tag
                    )
                else:
                    res = await self.execution_manager.execute_trade(
                        symbol=symbol, side=side, tag=tag
                    )
                # Update cooldown stamp on success/attempt
                self._last_signal_ts_per_symbol[symbol] = time.time()
                return res
        finally:
            self._inflight_exec = max(0, self._inflight_exec - 1)

    async def _wait_for_readiness(self) -> None:
        """Poll readiness gates and log DEGRADED health until all required gates are set or stop is requested."""
        while not self._stop_event.is_set():
            snap = self._readiness_snapshot()
            need = []
            if self.require_market_data_ready and not snap.get("market_data_ready", True):
                need.append("MarketDataReady")
            if self.require_balances_ready and not snap.get("balances_ready", True):
                need.append("BalancesReady")
            if self.require_ops_plane_ready and not snap.get("ops_plane_ready", True):
                need.append("OpsPlaneReady")
            # accepted symbols ready is useful but optional here; gate if provided by config
            if getattr(self.config, "REQUIRE_ACCEPTED_SYMBOLS_READY", False) and not snap.get("accepted_symbols_ready", True):
                need.append("AcceptedSymbolsReady")

            if not need:
                # All good — echo snapshot and accepted symbol breadth
                syms_count = 0
                try:
                    get = getattr(self.shared_state, "get_accepted_symbols", None)
                    if callable(get):
                        syms = await get() if asyncio.iscoroutinefunction(get) else get()
                        syms_count = len(syms or [])
                except Exception:
                    pass
                msg = f"Readiness OK: {snap} | accepted_symbols={syms_count}"
                try:
                    await self.shared_state.update_component_status("StrategyManager", "Operational", msg)
                except Exception:
                    pass
                return

            now = time.time()
            if now - self._last_ready_log_ts >= self.readiness_log_interval_s:
                self._last_ready_log_ts = now
                msg = f"Waiting for gates: {', '.join(need)}"
                self.logger.warning("StrategyManager gated: %s", msg)
                try:
                    await self.shared_state.update_component_status(
                        "StrategyManager", "Degraded", msg)
                except Exception:
                    pass
            # Sleep briefly before re-checking
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                pass

    async def start_periodic_analysis(self):
        # Delegate to canonical loop to avoid drift
        await self.run_loop()

    async def run_loop(self):
        self.logger.info("StrategyManager run_loop started.")
        # Ensure readiness gates are satisfied before running loop
        try:
            await self._wait_for_readiness()
        except Exception:
            self.logger.exception("Readiness wait failed during run_loop")

        if not self.active_strategies:
            try:
                await self.initialize_strategies()
            except Exception:
                self.logger.exception("initialize_strategies failed during run_loop")

        interval = max(5.0, float(getattr(self.config, "STRATEGY_ANALYSIS_INTERVAL", 60)))
        jitter_abs = max(1.0, interval * self.jitter_pct)

        while not self._stop_event.is_set():
            try:
                nav = await self.analyze_strategies()
                # NOTE: Strategy agents should invoke self.forward_signal(...) instead of talking to EM directly.
                CSL.log_status("StrategyManager", "Operational", f"NAV≈{nav:.3f} {self.base_ccy}")
                try:
                    await self.shared_state.update_component_status(
                        "StrategyManager", "Operational",
                        f"Tick OK @ {datetime.utcnow().isoformat()} | NAV≈{nav:.3f} {self.base_ccy}"
                    )
                except Exception:
                    pass
            except asyncio.CancelledError:
                self.logger.warning("StrategyManager run_loop cancelled.")
                raise
            except Exception:
                self.logger.exception("StrategyManager loop error")

            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=interval + random.uniform(-jitter_abs, jitter_abs))
            except asyncio.TimeoutError:
                pass

    async def run(self):
        await self.run_loop()

    def stop(self):
        self._stop_event.set()
        try:
            coro = self.shared_state.update_component_status("StrategyManager", "Stopped", "Stop signal received")
            if asyncio.iscoroutine(coro):
                asyncio.create_task(coro)
        except Exception:
            pass

    def enable_strategy(self, name: str):
        self._enabled[name] = True
        self.logger.info("Strategy enabled: %s", name)

    def disable_strategy(self, name: str):
        self._enabled[name] = False
        self.logger.info("Strategy disabled: %s", name)

    def set_weight(self, name: str, weight: float):
        self._weights[name] = float(weight)
        self.logger.info("Strategy weight set: %s -> %.3f", name, weight)

    def get_weights(self) -> Dict[str, float]:
        return dict(self._weights)
