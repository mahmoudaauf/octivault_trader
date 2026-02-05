# core/performance_evaluator.py
# PerformanceEvaluator — periodic KPI analytics (P9-aligned)
# Fixes:
#  - Await all async event emissions (no 'coroutine was never awaited')
#  - Warm-up gate (don’t breach until symbols+market+first snapshot are ready)
#  - Structured HealthStatus & PerformanceReport events
#  - Non-fatal operation (never brings down the ops plane)

import asyncio
import logging
import time
from typing import Any, Dict, Optional
from typing import Tuple

from core.stubs import maybe_await, maybe_call  # safe sync/async invocation

logger = logging.getLogger("PerformanceEvaluator")


async def _touch_watchdog(ss: Any, component: str, status: str, detail: str = "") -> None:
    """Best-effort: update timestamp and component status so Watchdog sees activity."""
    try:
        if ss is None:
            return
        if hasattr(ss, "update_timestamp"):
            await maybe_call(ss, "update_timestamp", component)
        if hasattr(ss, "update_component_status"):
            await maybe_call(ss, "update_component_status", component, status, detail)
    except Exception:
        logger.debug("_touch_watchdog failed for %s", component, exc_info=True)


# ---------------------------
# Safe emitter for sync spots
# ---------------------------
def _safe_emit_event(ss, name: str, payload: dict) -> None:
    """
    Fire-and-forget event emission for sync contexts.
    Prefer `await maybe_call(ss, "emit_event", ...)` inside async methods.
    """
    try:
        if not ss or not hasattr(ss, "emit_event"):
            return
        loop = asyncio.get_running_loop()
        loop.create_task(ss.emit_event(name, payload))
    except RuntimeError:
        asyncio.run(ss.emit_event(name, payload))


class PerformanceEvaluator:
    """
    Computes and emits periodic performance analytics used by:
      - CapitalAllocator (agent budgets)
      - StrategyManager (weights)
      - SummaryLog / dashboards

    Emits:
      - HealthStatus (Running / WarmingUp / Error / Breach)
      - PerformanceReport (windowed KPIs)
    """

    def __init__(
        self,
        config: Any,
        shared_state: Any,
        database_manager: Optional[Any] = None,
        notification_manager: Optional[Any] = None,
    ):
        self.config = config
        self.ss = shared_state
        self.db = database_manager
        self.nm = notification_manager

        self.interval_s = int(getattr(config, "PERF_EVAL_INTERVAL_S", 30))
        self.window_s = int(getattr(config, "PERF_EVAL_WINDOW_S", 3600))  # 60m default
        self._running = False

        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(
            "PerformanceEvaluator initialized (interval=%ss, window=%ss).",
            self.interval_s,
            self.window_s,
        )
        self._stop_event = asyncio.Event()
        self._task: Optional[asyncio.Task] = None

    # ---------------------------
    # Public lifecycle
    # ---------------------------
    # ---------------------------
    # Public lifecycle
    # ---------------------------
    async def start(self):
        """Phase 9 self-managed start."""
        if self._task and not self._task.done():
            return
        self._task = asyncio.create_task(self.run(), name="core.performance_evaluator")

    async def run(self):
        """Public entry point (compatible with AppContext.initialize_all)."""
        self._stop_event.clear()
        self.logger.info("🎯 Starting PerformanceEvaluator loop...")
        
        # Emit Initialized so Watchdog doesn't flag no-report
        try:
            await self._emit_health("Initialized", "Ready")
            await _touch_watchdog(self.ss, "PerformanceEvaluator", "Initialized", "Ready")
        except Exception:
            self.logger.debug("PerformanceEvaluator initial health update failed", exc_info=True)

        try:
            while not self._stop_event.is_set():
                try:
                    await self._tick_once()
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    self.logger.error("PerfEval tick error: %s", e, exc_info=True)
                    await self._emit_health("Error", f"Tick failed: {e}")
                await asyncio.sleep(self.interval_s)
        finally:
            self.logger.info("PerformanceEvaluator stopped.")

    async def stop(self):
        """
        P9 contract: stop() requests the loop to end and waits for it.
        """
        self._stop_event.set()
        t = self._task
        self._task = None
        if t:
            try:
                t.cancel()
                try:
                    await asyncio.wait_for(t, timeout=5.0)
                except asyncio.CancelledError:
                    pass
            except Exception:
                self.logger.debug("PerformanceEvaluator stop wait failed", exc_info=True)
        try:
            await _touch_watchdog(self.ss, "PerformanceEvaluator", "Stopped", "Stopped by request")
            await self._emit_health("Stopped", "Stopped by request")
        except Exception:
            self.logger.debug("PerformanceEvaluator final health update failed", exc_info=True)

    # ---------------------------
    # Core logic
    # ---------------------------
    async def _tick_once(self):
        # Warm-up gates (P9): symbols + market + first snapshot (NAV/_total_value)
        if not await self._is_ready():
            await self._emit_health(
                "WarmingUp",
                "Awaiting symbols + market data + first portfolio snapshot",
            )
            await _touch_watchdog(self.ss, "PerformanceEvaluator", "WarmingUp", "Awaiting symbols/market/snapshot")
            return

        # Collect inputs (best-effort; tolerate missing bits)
        now_ts = time.time()
        total_value = float(getattr(self.ss, "_total_value", 0.0) or 0.0)
        realized_pnl_total = float(getattr(self.ss, "realized_pnl", 0.0) or 0.0)

        kpi_metrics = {
            "timestamp": now_ts,
            "nav_quote": total_value,
            "realized_pnl_total": realized_pnl_total,
            # Placeholders for future enrichment (Sharpe, winrate, drawdown, etc.)
            "usdt_per_hour": await self._estimate_usdt_per_hour(default=0.0),
            "drawdown_pct": await self._estimate_drawdown_pct(default=0.0),
            "breaches": int(getattr(self.ss, "kpi_breaches", 0) or 0),
        }

        # Breach logic (soft): only if KPI target is configured and system is ready
        # PHASE A FIX: Dynamic profit target based on NAV
        nav = total_value
        # Use config or default ratio 0.2% per hour
        target_ratio = float(getattr(self.config, "TARGET_PROFIT_RATIO_PER_HOUR", 0.002))
        dynamic_target = max(0.5, nav * target_ratio) # Min 0.5 USDT/h (Phase A)
        
        # Priority: Config 'PROFIT_TARGET_BASE_USD_PER_HOUR' > dynamic (Default 20.0)
        target_usdt_h = float(getattr(self.config, "PROFIT_TARGET_BASE_USD_PER_HOUR", 20.0))
        if target_usdt_h <= 0.0:
            target_usdt_h = dynamic_target

        # Calm watchdog during LOW volatility regime
        regime = ""
        try:
            if hasattr(self.ss, "metrics") and isinstance(self.ss.metrics, dict):
                regime = str(self.ss.metrics.get("volatility_regime") or regime)
            if not regime and hasattr(self.ss, "get_volatility_regime"):
                tf = str(getattr(self.config, "VOLATILITY_REGIME_TIMEFRAME", "5m") or "5m")
                reg = await self.ss.get_volatility_regime("GLOBAL", tf, max_age_seconds=600)
                if reg and reg.get("regime"):
                    regime = str(reg.get("regime"))
        except Exception:
            regime = regime or ""

        regime = (regime or "").lower()
        if regime:
            kpi_metrics["volatility_regime"] = regime

        if regime == "low":
            ignore_breach = bool(getattr(self.config, "LOW_REGIME_IGNORE_BREACH", True))
            low_mult = float(getattr(self.config, "LOW_REGIME_TARGET_MULT", 0.25) or 0.25)
            if ignore_breach:
                target_usdt_h = 0.0
            else:
                target_usdt_h = max(0.0, target_usdt_h * max(0.0, low_mult))
            self.logger.info(
                "[PerfEval] LOW regime: adjusted target_usdt_h=%.3f (ignore_breach=%s)",
                target_usdt_h,
                ignore_breach,
            )
            
        status = "Running"
        msg = "OK"

        # P9 DEADLOCK DETECTION: Check rejection counters for execution deadlock (WITH TTL DECAY)
        # CRITICAL FIX: Apply 5-minute TTL to rejections to prevent stale counters from triggering deadlock
        total_rejections = 0
        max_rejection_count = 0
        deadlock_threshold = int(getattr(self.config, "DEADLOCK_REJECTION_THRESHOLD", 10))
        ignore_csv = str(getattr(self.config, "DEADLOCK_REJECTION_IGNORE_REASONS", "COLD_BOOTSTRAP_BLOCK,PORTFOLIO_FULL") or "")
        ignore_reasons = {r.strip().upper() for r in ignore_csv.split(",") if r.strip()}
        rej_ttl_sec = 300.0  # 5 minutes (must match shared_state.py TTL)
        now_ts_local = time.time()
        
        if hasattr(self.ss, "rejection_counters"):
            for (sym, side, reason), count in self.ss.rejection_counters.items():
                # Apply TTL decay: if no rejection in past 5 min, treat as expired
                ts = self.ss.rejection_timestamps.get((sym, side, reason), now_ts_local)
                if now_ts_local - ts > rej_ttl_sec:
                    # Counter expired; skip it
                    continue
                if str(reason).upper() in ignore_reasons:
                    continue
                total_rejections += count
                if count > max_rejection_count:
                    max_rejection_count = count
        
        if max_rejection_count >= deadlock_threshold:
            status = "Error"
            msg = f"DEADLOCK: Symbol stuck with {max_rejection_count} consecutive rejections"
            self.logger.error("[PerfEval:Deadlock] %s", msg)
        elif total_rejections >= deadlock_threshold * 2:
            status = "Degraded"
            msg = f"DEGRADED: Total rejections={total_rejections} exceeds threshold"
            self.logger.warning("[PerfEval:Degraded] %s", msg)

        # UPTIME GRACE (PHASE A): Ignore noise during first 30 mins
        uptime_grace_min = int(getattr(self.config, "UPTIME_GRACE_PERIOD_MIN", 30))
        # Use SharedState _trading_start_time if available, or current
        start_ts = getattr(self.ss, "_start_time_unix", now_ts)
        uptime_sec = now_ts - start_ts
        
        if status not in ("Error", "Degraded"):  # Don't override deadlock status
            if uptime_sec < (uptime_grace_min * 60):
                 msg = f"WarmUp Grace: uptime={uptime_sec/60:.1f}m < {uptime_grace_min}m"
            elif target_usdt_h > 0.0 and kpi_metrics["usdt_per_hour"] < target_usdt_h:
                status = "Breach"
                msg = f"usdt/h={kpi_metrics['usdt_per_hour']:.3f} < target={target_usdt_h:.2f}"
        
        # P9 ECONOMIC READINESS CHECK
        economically_ready = True
        if hasattr(self.ss, "is_economically_ready"):
            try:
                economically_ready = await self.ss.is_economically_ready(min_executable_symbols=1, threshold=deadlock_threshold)
            except Exception:
                pass
        
        if not economically_ready and status == "Running":
            status = "Degraded"
            msg = "ECONOMIC_STARVATION: No executable symbols available"
            self.logger.warning("[PerfEval:EconomicReadiness] %s", msg)

        # Emit PerformanceReport (awaited)
        await self._emit_report(kpi_metrics, status_msg=msg)

        # Emit HealthStatus - respect deadlock/degraded status
        final_status = status if status in ("Error", "Degraded") else ("Running" if status != "Breach" else "Breach")
        await self._emit_health(final_status, msg)
        await _touch_watchdog(self.ss, "PerformanceEvaluator", final_status, msg)

    # ---------------------------
    # Helpers
    # ---------------------------
    async def _is_ready(self) -> bool:
        """Phase gates + first snapshot/NAV available."""
        try:
            # P9: Robust readiness
            # We no longer check self.ss.is_ops_plane_ready() here to avoid a circular wait,
            # as PerformanceEvaluator itself is a requirement for OpsPlane readiness in Live mode.

            # PHASE A FLEXIBILITY:
            # We are ready if we have market data for our held positions
            # OR if we have market data for at least 3 symbols (discovery)
            has_some_market = False
            ev = getattr(self.ss, "market_data_ready_event", None)
            if ev and hasattr(ev, "is_set") and ev.is_set():
                 has_some_market = True
            else:
                 # Check manually if we have at least 1 bar for any symbol
                 md = getattr(self.ss, "market_data", {})
                 if md and len(md) > 0:
                      # If we have bars for symbols we OWN, we are definitely ready
                      owned = getattr(self.ss, "positions", {})
                      if any( (s, "5m") in md for s in owned):
                           has_some_market = True
                      elif len(md) > 3: # Or generic coverage
                           has_some_market = True

            # Get current NAV and Bootstrap status from SharedState
            nav = 0.0
            if hasattr(self.ss, "get_nav_quote"):
                nav = self.ss.get_nav_quote()
                if asyncio.iscoroutine(nav): nav = await nav
            elif hasattr(self.ss, "metrics"):
                nav = float(self.ss.metrics.get("nav", 0.0))

            is_bootstrap = False
            if hasattr(self.ss, "is_bootstrap_mode"):
                is_bootstrap = self.ss.is_bootstrap_mode()
                if asyncio.iscoroutine(is_bootstrap): is_bootstrap = await is_bootstrap
            
            return bool(has_some_market and (nav > 0 or is_bootstrap))
        except Exception as e:
            self.logger.debug("PerfEval readiness check failed: %s", e)
            return False

    async def _estimate_usdt_per_hour(self, default: float = 0.0) -> float:
        """
        Calculate realized PnL per hour from trade history (Capital Velocity).
        """
        try:
            # 1. Try SharedState trade_history (canonical source)
            history = getattr(self.ss, "trade_history", [])
            
            # 2. Fallback to PnLCalculator's lightweight deque if history is empty/missing
            if not history and hasattr(self.ss, "_realized_pnl"):
                history = [{"ts": x[0], "realized_delta": x[1]} for x in self.ss._realized_pnl]
            
            if not history:
                return default
            
            now = time.time()
            one_hour_ago = now - 3600
            
            # Sum realized_delta for trades within the last hour
            hourly_pnl = sum(
                float(t.get("realized_delta", 0.0)) 
                for t in history 
                if t.get("ts", 0) > one_hour_ago
            )
            
            # Store it back to SS for other consumers (e.g. MetaController)
            setattr(self.ss, "kpi_usdt_per_hour", hourly_pnl)
            
            return hourly_pnl
        except Exception:
            self.logger.debug("Failed to estimate usdt_per_hour", exc_info=True)
            return default

    async def _estimate_drawdown_pct(self, default: float = 0.0) -> float:
        """
        If drawdown tracking is implemented, read from SharedState.
        """
        try:
            return float(getattr(self.ss, "kpi_drawdown_pct", default) or default)
        except Exception:
            return default

    async def _emit_health(self, status: str, message: str):
        payload = {
            "component": "PerformanceEvaluator",
            "status": status,
            "message": message,
            "timestamp": time.time(),
        }
        if self.ss and hasattr(self.ss, "emit_event"):
            await maybe_call(self.ss, "emit_event", "HealthStatus", payload)
        else:
            # As a last resort (shouldn’t happen in P9), log only
            self.logger.debug("HealthStatus (no SS): %s", payload)

    async def _emit_report(self, report: Dict[str, Any], status_msg: str = "OK"):
        payload = {
            "source": "PerformanceEvaluator",
            "status": "OK",
            "message": status_msg,
            "report": report,
        }
        if self.ss and hasattr(self.ss, "emit_event"):
            # This was the warning in your logs — now properly awaited
            await maybe_call(self.ss, "emit_event", "PerformanceReport", payload)
        else:
            self.logger.debug("PerformanceReport (no SS): %s", payload)
