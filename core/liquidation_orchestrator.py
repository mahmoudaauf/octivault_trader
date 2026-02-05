import asyncio
import logging
import time
import os
from typing import Any, Dict, Optional, List, Callable

class LiquidationOrchestrator:
    """
    P9 Liquidation Orchestrator

    Responsibilities:
      • Bridge LiquidationAgent -> ExecutionManager (drain intents/orders and execute)
      • Observe OpsPlane/health + min-notional issues and trigger freeing USDT when needed
      • Periodically run dust/min-notional rebalance via LiquidationAgent
      • Expose a simple health/readiness surface
    """

    def __init__(
        self,
        shared_state,
        liquidation_agent,
        execution_manager,
        cash_router=None,
        *,
        meta_controller=None,
        position_manager=None,
        risk_manager=None,
        loop_interval_s: int = 5,
        min_usdt_target: float = 0.0,
        min_usdt_floor: float = 0.0,
        min_inventory_usdt: float = 25.0,
        free_usdt_probe_interval_s: int = 10,
        liq_batch_target_usdt: float = 25.0,
        rebalance_interval_s: int = 300,
        planner_timeout_s: float = 10.0,
        name: str = "LiquidationOrchestrator",
    ):
        self.log = logging.getLogger(name)
        self.name = name

        self.ss = shared_state
        self.agent = liquidation_agent
        self.exec = execution_manager
        self.cash = cash_router
        self.meta = meta_controller
        self.pos_mgr = position_manager
        self.risk = risk_manager

        # cadences
        self.loop_interval_s = int(loop_interval_s)
        self.free_usdt_probe_interval_s = int(free_usdt_probe_interval_s)
        self.rebalance_interval_s = int(rebalance_interval_s)

        # thresholds
        self.min_usdt_target = float(min_usdt_target)
        self.min_usdt_floor = float(min_usdt_floor)
        self.min_inventory_usdt = float(min_inventory_usdt)
        self.liq_batch_target_usdt = float(liq_batch_target_usdt)

        # guardrails
        self._planner_timeout_s = float(planner_timeout_s)
        self._tasks: Dict[str, Optional[asyncio.Task]] = {}
        self._running = asyncio.Event()

        # internal ticks
        self._t_last_probe = 0.0
        self._t_last_rebalance = 0.0

        # completion callbacks (e.g., to refresh meta/cash state)
        self._on_completed: List[Callable[[Dict[str, Any]], None]] = []

        self._last_action: Dict[str, Any] = {}  # for health surfacing
        self._rebalance_fail_streak = 0
        self._rebalance_skip_until = 0.0
        self._started_once = False

        # single-flight + throttle for freeing USDT
        self._freeing_lock = asyncio.Lock()
        self._last_free_attempt_ts = 0.0
        self._min_gap_sec_between_free = max(5.0, float(os.getenv("LIQ_ORCH_MIN_GAP_SEC", "15")))

        self.log.info(
            "[%s] initialized floors/targets: min_usdt_floor=%.2f, min_usdt_target=%.2f, liq_batch_target=%.2f (cash_router=%s)",
            self.name, self.min_usdt_floor, self.min_usdt_target, self.liq_batch_target_usdt, bool(self.cash)
        )

    # ----------------- lifecycle -----------------

    async def _async_start(self):
        if self._running.is_set():
            return
        self._running.set()
        if not self._tasks.get("main") or self._tasks["main"].done():
            self._tasks["main"] = asyncio.create_task(self._main_loop(), name=f"{self.name}.main")
        # schedule periodic heartbeat
        if not self._tasks.get("hb") or self._tasks["hb"].done():
            self._tasks["hb"] = asyncio.create_task(self._heartbeat_loop(interval_s=30))
        self.log.info("[%s] started (loop=%ss, probe=%ss, rebalance=%ss)",
                      self.name, self.loop_interval_s, self.free_usdt_probe_interval_s, self.rebalance_interval_s)

    async def start(self, interval_s: int = 15):
        """
        Async start-hook for AppContext P8.
        """
        if self._running.is_set() and self._tasks.get("main") and not self._tasks["main"].done():
            return

        # ensure running flag is set before bootstrapping
        if not self._running.is_set():
            self._running.set()

        # schedule the async starter if not already running
        if not self._tasks.get("bootstrap") or self._tasks["bootstrap"].done():
            self._tasks["bootstrap"] = asyncio.create_task(self._async_start())
        
        self.log.info("[%s] start() scheduled (async bootstrap)", self.name)

    async def stop(self):
        self._running.clear()
        for t in list(self._tasks.values()):
            if t and not t.done():
                t.cancel()
        if self._tasks:
            try:
                await asyncio.gather(*[t for t in self._tasks.values() if t], return_exceptions=True)
            except Exception:
                pass
        self._tasks.clear()
        self.log.info("[%s] stopped", self.name)

    async def _heartbeat_loop(self, interval_s: int = 30):
        """
        Periodically emit a HealthStatus-like dict onto shared_state (if available) for observability.
        """
        while self._running.is_set():
            try:
                status = self.health()
                if self.ss and hasattr(self.ss, "emit_status"):
                    res = self.ss.emit_status(self.name, status)
                    if asyncio.iscoroutine(res):
                        await res
            except Exception:
                self.log.debug("[ORCH] heartbeat emit failed", exc_info=True)
            await asyncio.sleep(interval_s)

    def health(self) -> Dict[str, Any]:
        ready = self._running.is_set()
        ops_ready = False
        try:
            if hasattr(self.ss, "is_ops_plane_ready"):
                res = self.ss.is_ops_plane_ready()
                # if coroutine, we can't await in sync method; fall back to last snapshot
                if not asyncio.iscoroutine(res):
                    ops_ready = bool(res)
        except Exception:
            pass
        # If the readiness is async, try best-effort:
        if not ops_ready and hasattr(self.ss, "ops_plane_last_status"):
            try:
                ops_ready = bool(getattr(self.ss, "ops_plane_last_status", {}).get("ready", False))
            except Exception:
                pass
        return {
            "component": self.name,
            "running": bool(ready),
            "ops_plane_ready": bool(ops_ready),
            "last_action": dict(self._last_action),
            "min_usdt_floor": self.min_usdt_floor,
            "min_usdt_target": self.min_usdt_target,
            "liq_batch_target_usdt": self.liq_batch_target_usdt,
            "min_gap_sec_between_free": self._min_gap_sec_between_free,
            "ts": time.time(),
        }

    def configure_thresholds(
        self,
        *,
        min_usdt_floor: Optional[float] = None,
        min_usdt_target: Optional[float] = None,
        liq_batch_target_usdt: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Dynamically update key thresholds. Returns the effective values.
        """
        if min_usdt_floor is not None:
            self.min_usdt_floor = float(min_usdt_floor)
        if min_usdt_target is not None:
            self.min_usdt_target = float(min_usdt_target)
        if liq_batch_target_usdt is not None:
            self.liq_batch_target_usdt = float(liq_batch_target_usdt)
        self._last_action = {"type": "reconfig", "floor": self.min_usdt_floor, "target": self.min_usdt_target,
                             "batch": self.liq_batch_target_usdt, "ts": time.time()}
        self.log.info("[ORCH] thresholds updated: floor=%.4f target=%.4f batch=%.4f",
                      self.min_usdt_floor, self.min_usdt_target, self.liq_batch_target_usdt)
        return {
            "min_usdt_floor": self.min_usdt_floor,
            "min_usdt_target": self.min_usdt_target,
            "liq_batch_target_usdt": self.liq_batch_target_usdt,
        }

    def on_completed(self, cb):
        """
        Register a callback to be invoked when a meaningful liquidity action completes.
        Signature: cb(context: dict); best-effort, exceptions are swallowed.
        """
        if callable(cb):
            self._on_completed.append(cb)

    def _notify_completed(self, context: Optional[Dict[str, Any]] = None):
        for cb in list(self._on_completed):
            try:
                cb(context or {})
            except Exception:
                # never raise from callbacks
                pass
        # Best-effort: ask meta controller to refresh cash-router state if it exposes a helper
        try:
            if self.meta and hasattr(self.meta, "refresh_cash_router"):
                self.meta.refresh_cash_router()
        except Exception:
            pass

    # ----------------- helpers -----------------

    def _intent_payload(
        self,
        *,
        symbol: str,
        side: str,
        planned_qty: Optional[float] = None,
        planned_quote: Optional[float] = None,
        confidence: float = 0.9,
        ttl_sec: float = 90.0,
        tag: str = "liquidation",
        agent: Optional[str] = None
    ) -> dict:
        """
        Build a TradeIntent-like payload as a plain dict to avoid hard dependency on a specific dataclass.
        """
        payload = {
            "symbol": symbol,
            "side": side.upper(),
            "confidence": float(confidence),
            "ttl_sec": float(ttl_sec),
            "tag": tag,
            "agent": agent or self.name,
        }
        if planned_qty is not None:
            payload["planned_qty"] = float(planned_qty)
        if planned_quote is not None:
            payload["planned_quote"] = float(planned_quote)
        return payload

    async def _emit_trade_intent(self, payload: dict):
        """
        Route a TradeIntent payload to the shared_state bus if available.
        """
        try:
            if self.ss and hasattr(self.ss, "emit_event"):
                res = self.ss.emit_event("TradeIntent", payload)
                if asyncio.iscoroutine(res):
                    await res
            else:
                # No bus available; log for observability
                self.log.warning("[ORCH] No shared_state bus to emit TradeIntent: %s", payload)
        except Exception:
            self.log.debug("[ORCH] emit_event(TradeIntent) failed", exc_info=True)


    async def trigger_liquidity(
        self,
        *,
        symbol: Optional[str] = None,
        required_usdt: Optional[float] = None,
        free_usdt: Optional[float] = None,
        gap_usdt: Optional[float] = None,
        reason: str = "INSUFFICIENT_QUOTE"
    ) -> dict:
        """
        Public API used by AppContext / Readiness checks.
        Prefer to cover a known `gap_usdt`; otherwise fall back to required_usdt - free_usdt.
        """
        target = 0.0
        try:
            if isinstance(gap_usdt, (int, float)) and gap_usdt > 0:
                target = float(gap_usdt)
            elif isinstance(required_usdt, (int, float)) and isinstance(free_usdt, (int, float)):
                target = max(0.0, float(required_usdt) - float(free_usdt))
        except Exception:
            target = 0.0
        res = await self._free_usdt_now(target=target, reason=reason, free_before=free_usdt)
        if isinstance(res, dict):
            return res
        # Back-compat: if old float is returned, normalize
        return {"ok": bool(res and res > 0.0), "freed": float(res or 0.0), "status": "OK" if (res and res > 0.0) else "NOOP"}

    async def _free_usdt_now(self, target: float, reason: str, free_before: Optional[float] = None) -> dict:
        """
        Ask CashRouter first to ensure target free USDT; fall back to LiquidationAgent to PROPOSE liquidation intents (no direct execution here).
        Throttled and single-flight guarded so only one free operation can run at a time.
        Returns: {"ok": bool, "freed": float, "path": "cash_router"|"liquidation_agent"|"none",
                  "submitted": int, "approx_quote": float, "status": "OK"|"NOOP"|"THROTTLED"|"FAILED"}
        """
        if target <= 0:
            return {"ok": False, "freed": 0.0, "path": "none", "submitted": 0, "approx_quote": 0.0, "status": "NOOP"}

        try:
            target = max(0.0, float(target))
        except Exception:
            return {"ok": False, "freed": 0.0, "path": "none", "submitted": 0, "approx_quote": 0.0, "status": "FAILED"}

        now = time.monotonic()
        if (now - self._last_free_attempt_ts) < self._min_gap_sec_between_free:
            self.log.debug("[ORCH] skip free_usdt (throttled %.1fs)", self._min_gap_sec_between_free)
            return {"ok": False, "freed": 0.0, "path": "none", "submitted": 0, "approx_quote": 0.0, "status": "THROTTLED"}

        async with self._freeing_lock:
            # Re-check after acquiring the lock
            now = time.monotonic()
            if (now - self._last_free_attempt_ts) < self._min_gap_sec_between_free:
                self.log.debug("[ORCH] skip free_usdt (throttled %.1fs after lock)", self._min_gap_sec_between_free)
                return {"ok": False, "freed": 0.0, "path": "none", "submitted": 0, "approx_quote": 0.0, "status": "THROTTLED"}
            self._last_free_attempt_ts = now

            freed_total = 0.0
            try:
                headroom = float(os.getenv("LIQ_ORCH_USDT_HEADROOM", "0.5") or 0.5)
            except Exception:
                headroom = 0.5

            # 1) Preferred path: CashRouter.ensure_free_usdt (absolute target)
            try:
                if self.cash and hasattr(self.cash, "ensure_free_usdt") and callable(getattr(self.cash, "ensure_free_usdt")):
                    try:
                        fb = float(free_before) if free_before is not None else await self._current_free_usdt()
                    except Exception:
                        fb = 0.0
                    absolute_target = max(fb + float(target), float(self.min_usdt_target or 0.0))
                    absolute_target = float(f"{absolute_target + headroom:.6f}")
                    res = self.cash.ensure_free_usdt(absolute_target, reason=str(reason))
                    res = await res if asyncio.iscoroutine(res) else res
                    ok = bool(isinstance(res, dict) and res.get("ok")) or (res is True)
                    if ok:
                        freed_total = float((res or {}).get("freed", 0.0)) if isinstance(res, dict) else 0.0
                        if freed_total <= 0.0:
                            try:
                                after = await self._current_free_usdt()
                                freed_total = max(0.0, float(after) - float(fb))
                            except Exception:
                                pass
                        self._last_action = {"type": "free_usdt", "via": "cash_router", "reason": reason,
                                             "target_abs": absolute_target, "freed": freed_total, "ts": time.time()}
                        self.log.info("[ORCH] ensure_free_usdt via CashRouter ok: target_abs=%.4f freed≈%.4f (%s)",
                                      absolute_target, freed_total, reason)
                        self._notify_completed({"type": "free_usdt", "via": "cash_router", "reason": reason, "freed": freed_total})
                        # Ask ops/meta to re-check affordability (BUY retry once policy)
                        try:
                            if self.ss and hasattr(self.ss, "emit_event"):
                                ev = self.ss.emit_event("AffordabilityRecheck", {"reason": reason, "ts": time.time()})
                                if asyncio.iscoroutine(ev):
                                    asyncio.create_task(ev)
                        except Exception:
                            pass
                        return {"ok": True, "freed": max(freed_total, 0.0), "path": "cash_router", "submitted": 0, "approx_quote": 0.0, "status": "OK"}
                    else:
                        self.log.debug("[ORCH] CashRouter.ensure_free_usdt returned not-ok: %s", res)
            except Exception as e:
                self.log.debug("[ORCH] CashRouter.ensure_free_usdt failed: %s", e, exc_info=True)

            # 2) Fallback: ask LiquidationAgent to PROPOSE liquidation intents (no direct execution here)
            if not self.agent:
                self.log.info("[ORCH] BOOTSTRAP: No liquidation agent; no action required")
                return {"ok": True, "freed": 0.0, "path": "no_agent", "submitted": 0, "approx_quote": 0.0, "status": "NO_ACTION_REQUIRED"}
            
            # BOOTSTRAP FIX: Check if we have any positions to liquidate
            has_positions = False
            try:
                if self.ss and hasattr(self.ss, "_positions"):
                    for p in self.ss._positions.values():
                        if float(p.get("quantity", 0.0)) > 0:
                            has_positions = True
                            break
            except Exception:
                pass
            
            if not has_positions and target <= 0:
                self.log.info("[ORCH] BOOTSTRAP: No positions to liquidate; no action required")
                return {"ok": True, "freed": 0.0, "path": "no_positions", "submitted": 0, "approx_quote": 0.0, "status": "NO_ACTION_REQUIRED"}
            
            try:
                intents = []
                # Determine if this is a mandatory/forced liquidation (e.g. triggered by gap filling or floor violation)
                # We treat gap filling, floor violations, and blocks as mandatory.
                is_forced = (target > 0) and any(kw in str(reason) for kw in ("gap", "floor", "block", "INSUFFICIENT_QUOTE"))

                if is_forced:
                    # Rule 6: Liquidation must be allowed to break readiness
                    if self.ss and hasattr(self.ss, "ops_plane_ready_event"):
                        self.ss.ops_plane_ready_event.clear()
                        self.log.info("[ORCH] Forced Liquidation triggered. Readiness = FALSE.")

                # Prefer a targeted proposal API if available
                if hasattr(self.agent, "propose_liquidations") and callable(getattr(self.agent, "propose_liquidations")):
                    intents = await self.agent.propose_liquidations(gap_usdt=float(target), reason=str(reason), force=is_forced)
                # Back-compat: if agent only exposes run_once/produce_orders, call and map to intents
                elif hasattr(self.agent, "produce_orders") and callable(getattr(self.agent, "produce_orders")):
                    orders = await asyncio.wait_for(self.agent.produce_orders(), timeout=self._planner_timeout_s)
                    for eo in orders or []:
                        sym = eo.get("symbol")
                        side = (eo.get("side") or "").upper()
                        qty = eo.get("quantity")
                        if side not in ("SELL", "SELL_SHORT") or not sym or not qty:
                            continue
                        intents.append(self._intent_payload(
                            symbol=sym, side="SELL", planned_qty=float(qty), confidence=0.99, ttl_sec=90.0, tag="liquidation", agent=self.name
                        ))
                # Emit intents to SharedState
                submitted, approx_quote = 0, 0.0
                for it in intents:
                    await self._emit_trade_intent(it)
                    submitted += 1
                    approx_quote += float(it.get("planned_quote") or 0.0)
                if submitted > 0:
                    self._last_action = {"type": "emit_intent_batch", "via": "liquidation_agent", "submitted": submitted,
                                         "approx_quote": approx_quote, "reason": reason, "ts": time.time()}
                    self.log.info("[ORCH] Emitted %d liquidation intents (approx_quote≈%.2f) via LiquidationAgent (%s)",
                                  submitted, approx_quote, reason)
                    self._notify_completed({"type": "intent_batch", "via": "liquidation_agent",
                                            "submitted": submitted, "approx_quote": approx_quote})
                    # Best-effort: ask ops/meta to re-check affordability
                    try:
                        if self.ss and hasattr(self.ss, "emit_event"):
                            ev = self.ss.emit_event("AffordabilityRecheck", {"reason": reason, "ts": time.time()})
                            if asyncio.iscoroutine(ev):
                                asyncio.create_task(ev)
                    except Exception:
                        pass
                return {"ok": False, "freed": 0.0, "path": "liquidation_agent", "submitted": submitted, "approx_quote": approx_quote, "status": "NOOP"}
            except Exception as e:
                self.log.exception("[ORCH] propose_liquidations/produce_orders failed: %s", e)
                return {"ok": False, "freed": 0.0, "path": "liquidation_agent", "submitted": submitted, "approx_quote": approx_quote, "status": "FAILED"}

    async def _current_free_usdt(self) -> float:
        try:
            # Preferred: explicit spendable getter
            if hasattr(self.ss, "get_spendable_usdt"):
                v = self.ss.get_spendable_usdt()
                v = await v if asyncio.iscoroutine(v) else v
                if v is not None:
                    return float(v)
        except Exception:
            pass
        try:
            # Fallback: simple free_usdt() callable
            if hasattr(self.ss, "free_usdt") and callable(getattr(self.ss, "free_usdt")):
                v = self.ss.free_usdt()
                v = await v if asyncio.iscoroutine(v) else v
                if v is not None:
                    return float(v)
        except Exception:
            pass
        try:
            # Fallback: balances structure
            bals = getattr(self.ss, "balances", {}) or {}
            usdt = bals.get("USDT", {}) or {}
            if isinstance(usdt, dict):
                return float(usdt.get("free", 0.0) or 0.0)
            return float(usdt or 0.0)
        except Exception:
            return 0.0

    async def _drain_and_execute_orders(self):
        """
        Deprecated: maintained for compatibility. Delegates to _drain_and_emit_intents.
        """
        await self._drain_and_emit_intents()

    async def _drain_and_emit_intents(self):
        """
        Pull SELL orders from LiquidationAgent and convert them to TradeIntent payloads.
        This preserves the invariant: only ExecutionManager places orders after Meta arbitration.
        """
        if not self.agent:
            return
        orders: List[Dict[str, Any]] = []
        try:
            orders = await asyncio.wait_for(self.agent.produce_orders(), timeout=self._planner_timeout_s)
        except asyncio.TimeoutError:
            self.log.warning("[ORCH] produce_orders timeout after %.1fs", self._planner_timeout_s)
            return
        except Exception:
            self.log.exception("[ORCH] produce_orders failed")
            return

        if not orders:
            return

        submitted = skipped = 0
        for eo in orders:
            symbol = eo.get("symbol")
            side = (eo.get("side") or "").upper()
            qty = eo.get("quantity")
            tag = eo.get("tag") or "liquidation"

            if side not in ("SELL", "SELL_SHORT"):
                skipped += 1
                self.log.debug("[ORCH] skip non-liquidation side=%s symbol=%s", side, symbol)
                continue
            try:
                qf = float(qty or 0.0)
            except Exception:
                qf = 0.0
            if not symbol or qf <= 0.0:
                skipped += 1
                self.log.debug("[ORCH] skip invalid order: %s", eo)
                continue

            payload = self._intent_payload(
                symbol=symbol,
                side="SELL",
                planned_qty=qf,
                confidence=0.99,
                ttl_sec=90.0,
                tag=tag,
                agent=self.name,
            )
            await self._emit_trade_intent(payload)
            submitted += 1
            self._last_action = {"type": "emit_intent", "symbol": symbol, "side": "SELL", "qty": qf, "ts": time.time()}
            self.log.info("[ORCH] EMIT_INTENT SELL %s qty=%.8f tag=%s", symbol, qf, tag)
            await asyncio.sleep(0)  # yield

        if submitted or skipped:
            self.log.info("[ORCH] drain summary: submitted=%d skipped=%d", submitted, skipped)
        if submitted > 0:
            self._notify_completed({"type": "intent_batch", "submitted": submitted, "skipped": skipped})

    async def _maybe_rebalance_min_notional(self):
        if not self.agent:
            return
        now = time.monotonic()

        # backoff guard
        if now < self._rebalance_skip_until:
            return

        if (now - self._t_last_rebalance) < self.rebalance_interval_s:
            return
        self._t_last_rebalance = now
        try:
            summary = await self.agent.rebalance_once()
            self._rebalance_fail_streak = 0
            self._last_action = {"type": "rebalance", "summary": summary, "ts": time.time()}
            self.log.info("[ORCH][REB] %s", summary)
            self._notify_completed({"type": "rebalance", "summary": summary})
        except Exception:
            self._rebalance_fail_streak += 1
            # simple exponential backoff: skip future attempts for up to ~rebalance_interval_s * 2^n (capped)
            backoff = min(self.rebalance_interval_s * (2 ** min(self._rebalance_fail_streak, 4)), 3600)
            self._rebalance_skip_until = now + backoff
            self.log.exception("[ORCH] rebalance_once failed (streak=%d, backoff≈%ss)",
                               self._rebalance_fail_streak, int(backoff))

    async def _read_ops_issues(self) -> Dict[str, Any]:
        """
        Best-effort snapshot of ops-plane issues and recent liquidity events from SharedState.
        Returns dict like: {"issues": [...], "liquidity_gap": float|None}
        """
        issues: List[str] = []
        gap: Optional[float] = None
        try:
            # Preferred: a structured readiness snapshot
            if hasattr(self.ss, "last_readiness_snapshot"):
                snap = self.ss.last_readiness_snapshot()
                snap = await snap if asyncio.iscoroutine(snap) else snap
                if isinstance(snap, dict):
                    issues = list(snap.get("issues", []) or [])
        except Exception:
            pass
        # Fallback: component_statuses message scanning
        try:
            snap = getattr(self.ss, "component_statuses", {}) or {}
            ops = snap.get("OpsPlane") or snap.get("AppContext") or {}
            if isinstance(ops, dict):
                msg = str(ops.get("message", "") or "")
                for key in ("MinNotionalTooHighForConfiguredQuote", "NAVNotReady", "INSUFFICIENT_QUOTE"):
                    if key in msg and key not in issues:
                        issues.append(key)
        except Exception:
            pass
        # Optional: consume recent events if SharedState exposes them
        try:
            if hasattr(self.ss, "get_recent_events"):
                evs = self.ss.get_recent_events("LIQUIDITY_NEEDED", within_sec=120)
                evs = await evs if asyncio.iscoroutine(evs) else evs
                if isinstance(evs, list) and evs:
                    last = evs[-1]
                    gap_val = last.get("gap_usdt")
                    if gap_val is not None:
                        gap = float(gap_val)
        except Exception:
            pass
        return {"issues": issues, "liquidity_gap": gap}

    async def _probe_and_react_min_notional(self):
        now = time.monotonic()
        if (now - self._t_last_probe) < self.free_usdt_probe_interval_s:
            return
        self._t_last_probe = now

        free_usdt = await self._current_free_usdt()
        ops = await self._read_ops_issues()
        issues: List[str] = list(ops.get("issues", []) or [])
        liq_gap = ops.get("liquidity_gap")

        need_reason, target = None, 0.0

        # SAFEGUARD: Check total inventory value before liquidating
        # This prevents "ProfitTargetEngine" dynamics from liquidating simply because cash is low,
        # if the bot itself is just small/starting up.
        total_spot_value = 0.0
        try:
            positions = {}
            if self.agent and hasattr(self.agent, "shared_state"):
                positions = self.agent.shared_state.get_positions_snapshot() or {}
            
            if not positions:
                positions = getattr(self.ss, "get_positions_snapshot", lambda: {})() or {}

            for sym, pos in positions.items():
                qty = float((pos or {}).get("quantity") or (pos or {}).get("qty") or 0.0)
                if qty > 0:
                     px = await self.agent._safe_price(sym) if self.agent else 0.0
                     total_spot_value += (qty * px)
        except Exception as e:
            self.log.warning("[ORCH] failed to calc spot value: %s", e)

        # 1) Hard floor first
        if self.min_usdt_floor and free_usdt < self.min_usdt_floor:
            # Policy: If we have very little inventory, don't cannibalize it just to hit a floor unless CRITICAL
            if total_spot_value < self.min_inventory_usdt:
                 self.log.debug("[ORCH] SKIP raise_free_usdt: inventory=%.2f < min=%.2f (floor=%.2f)",
                                total_spot_value, self.min_inventory_usdt, self.min_usdt_floor)
            else:
                need_reason = f"raise_free_usdt_to_floor({self.min_usdt_floor:.2f})"
                target = max(0.0, self.min_usdt_floor - free_usdt)

        # 2) If there is an explicit ops-plane liquidity gap, try to cover that
        elif isinstance(liq_gap, (int, float)) and liq_gap > 0:
            if total_spot_value >= self.min_inventory_usdt:
                need_reason = "cover_liquidity_gap"
                target = float(liq_gap)

        # 3) If min-notional or insufficient-quote issues are reported, free a batch
        elif any(k in issues for k in ("MinNotionalTooHighForConfiguredQuote", "INSUFFICIENT_QUOTE")):
             if total_spot_value >= self.min_inventory_usdt:
                need_reason = "min_notional_block"
                target = max(self.liq_batch_target_usdt, self.min_usdt_target or self.liq_batch_target_usdt)

        # 4) Soft target
        elif self.min_usdt_target and free_usdt < self.min_usdt_target:
             if total_spot_value >= self.min_inventory_usdt:
                need_reason = f"topup_free_usdt_to_target({self.min_usdt_target:.2f})"
                target = max(0.0, self.min_usdt_target - free_usdt)

        if need_reason and target > 0.0:
            res = await self._free_usdt_now(target=target, reason=need_reason, free_before=free_usdt)
            freed = float(res.get("freed", 0.0)) if isinstance(res, dict) else float(res or 0.0)
            if freed > 0:
                self._last_action = {"type": "free_usdt", "reason": need_reason, "amount": freed,
                                     "free_before": free_usdt, "ts": time.time()}
                # Emit a best-effort event back to SharedState, if supported
                try:
                    if hasattr(self.ss, "emit_event"):
                        ev = self.ss.emit_event("LiquidityFreed", {"amount": freed, "reason": need_reason})
                        if asyncio.iscoroutine(ev):
                            asyncio.create_task(ev)
                except Exception:
                    pass

    async def _process_queued_requests(self):
        """
        Poll for per-symbol liquidation requests from SharedState.
        Serviced as the 'Trigger A' bridge.
        """
        if not self.agent or not hasattr(self.ss, "get_next_liquidation_request"):
            return

        # Limit per tick to avoid locking up
        for _ in range(5):
            req = await self.ss.get_next_liquidation_request()
            if not req:
                break
            
            sym = req.get("symbol")
            reason = req.get("reason", "external_request")
            needed = float(req.get("min_quote_target") or 0.0)
            
            self.log.info("[ORCH] Servicing symbol liquidation request: %s (Reason: %s)", sym, reason)
            
            if sym == "__FREE_QUOTE__" or needed > 0:
                # Requested a specific amount of cash
                await self._free_usdt_now(target=needed or self.liq_batch_target_usdt, reason=reason)
            else:
                # Requested specific symbol exit. MetaController/Rules will decide if it's 'forced'.
                # We ask the agent to plan this specific exit.
                plan = await self.agent.build_plan(target_symbol=sym, needed_quote=0.0, opp_meta={"reason": reason}, force=True)
                if plan.get("status") == "APPROVED":
                    await self._drain_and_emit_intents()
                
            if hasattr(self.ss, "clear_liquidation_flag") and sym:
                self.ss.clear_liquidation_flag(sym)

    # ----------------- main loop -----------------

    async def _main_loop(self):
        # Optional warmup (LiquidationAgent wrapper defines warmup() if missing)
        try:
            if hasattr(self.agent, "warmup"):
                maybe = self.agent.warmup()
                if asyncio.iscoroutine(maybe):
                    await maybe
        except Exception:
            self.log.exception("[ORCH] warmup failed (continuing)")

        # Wait for Ops-plane if you want to be conservative
        try:
            if hasattr(self.ss, "ops_plane_ready_event"):
                await asyncio.wait_for(self.ss.ops_plane_ready_event.wait(), timeout=30.0)
        except Exception:
            # continue anyway — the orchestrator can still help free capital early
            pass

        while self._running.is_set():
            try:
                # 1) Drain SELL intents/orders from LiquidationAgent and execute
                await self._drain_and_emit_intents()

                # 2) Process per-symbol liquidation requests (specific demands)
                await self._process_queued_requests()

                # 3) Periodically free USDT if blocked / below floors (global gaps)
                await self._probe_and_react_min_notional()

                # 4) Periodic dust/min-notional rebalance
                await self._maybe_rebalance_min_notional()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.log.exception("[ORCH] main loop error: %s", e)

            await asyncio.sleep(self.loop_interval_s)

        self.log.info("[%s] main loop exited", self.name)
