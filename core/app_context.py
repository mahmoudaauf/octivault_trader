# === SECTION: Imports ===
# -*- coding: utf-8 -*-
from __future__ import annotations

# Load .env early (bootstrap safety); do not override existing OS env vars
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=False)

"""
AppContext — Phased Orchestrator (P3→P9) for Octivault Trader (P9)
- Single, clean definition
- Strict canonical imports (no fallbacks) & defensive construction
- Structured logging
- Readiness snapshot with Startup Sanity Gate (filters coverage + free-quote floor)
- Adaptive min-notional floor helper
- Phased init & graceful shutdown
- Startup timeouts with background continuation + completion logs
"""

import asyncio
import logging
import json
import traceback
from datetime import datetime, timezone
import contextlib
import inspect
import os
import importlib

LOGGER = logging.getLogger("AppContext")

from typing import Optional, Dict, Any, List, Awaitable, Iterable, Tuple, Union


# === SECTION: Strict Import Helpers ===
def _import_strict(path: str):
    """
    Import a required module by canonical path only.
    If import fails, raise ImportError immediately (no fallbacks).
    """
    try:
        return importlib.import_module(path)
    except Exception as e:
        LOGGER.error("STRICT_IMPORTS: required module missing: %s (%s)", path, e)
        raise

def _import_optional(path: str):
    """
    Import an optional module by canonical path only.
    If import fails, return None (no fallbacks).
    """
    try:
        return importlib.import_module(path)
    except Exception as e:
        LOGGER.warning("STRICT_IMPORTS: optional module skipping: %s (reason: %s)", path, e)
        return None


# === SECTION: Core Modules (Strict Canonical Paths) ===
# Required core
_exchange_mod       = _import_strict("core.exchange_client")
_market_data_mod    = _import_strict("core.market_data_feed")
_shared_state_mod   = _import_strict("core.shared_state")
_execution_mod      = _import_strict("core.execution_manager")
_risk_mod           = _import_strict("core.risk_manager")
_tpsl_mod           = _import_strict("core.tp_sl_engine")
_recovery_mod       = _import_strict("core.recovery_engine")
_watchdog_mod       = _import_strict("core.watchdog")
_heartbeat_mod      = _import_strict("core.heartbeat")
_symbol_mgr_mod     = _import_strict("core.symbol_manager")
_governor_mod       = _import_strict("core.capital_symbol_governor")
_uure_mod           = _import_strict("core.universe_rotation_engine")
_csl_mod            = _import_strict("core.component_status_logger")

# Meta/agents (canonical modules live in core for this repo)
_agent_mgr_mod      = _import_strict("core.agent_manager")
_meta_ctrl_mod      = None  # Direct import needed due to file/package conflict
# StrategyManager remains in core to avoid drift.
_strategy_mgr_mod   = _import_strict("core.strategy_manager")

# Optional modules (no fallbacks, but optional to import)
_perf_mod           = _import_optional("core.performance_monitor")
_alert_mod          = _import_optional("core.alert_system")
_comp_mod           = _import_optional("core.compounding_engine")
_vol_mod            = _import_optional("core.volatility_regime")
_portfolio_mod      = _import_optional("portfolio.balancer")
_pnl_mod            = _import_optional("utils.pnl_calculator")
_perf_eval_mod      = _import_optional("core.performance_evaluator")
_liq_orch_mod       = _import_optional("core.liquidation_orchestrator")
_cash_router_mod    = _import_optional("core.cash_router")
_liq_agent_mod      = _import_optional("agents.liquidation_agent")
_adaptive_capital_mod = _import_optional("core.adaptive_capital_engine")
_truth_auditor_mod  = _import_optional("core.exchange_truth_auditor")
_dust_monitor_mod   = _import_optional("core.dust_monitor")
_wallet_scanner_mod = _import_optional("agents.wallet_scanner_agent")
_dashboard_mod      = _import_optional("dashboard_server")
_cap_alloc_mod      = _import_optional("core.capital_allocator")
_ptg_mod            = _import_optional("core.profit_target_engine")

#
# === SECTION: Small Utilities ===
# NOTE: _maybe_await is implemented as an instance helper on AppContext to
# avoid module-level duplication across components (exchange_client, market_data_feed,
# recovery_engine, etc.). References inside AppContext use self._maybe_await(...).

# NOTE: symbol list conversion is implemented on AppContext as a staticmethod
# to keep AppContext-related helpers grouped and avoid top-level duplication.

# === SECTION: Structured Logging ===
def log_structured_error(
    error: Exception,
    context: Optional[Dict[str, Any]] = None,
    logger: Optional[logging.Logger] = None,
    *,
    component: str = "AppContext",
    phase: Optional[str] = None,
    module: Optional[str] = None,
    event: str = "exception",
    severity: str = "ERROR",
    include_trace: bool = True,
) -> None:
    lg = logger or LOGGER
    payload: Dict[str, Any] = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "event": event,
        "severity": severity,
        "component": component,
        "phase": phase,
        "module": module,
        "exc_type": type(error).__name__,
        "exc_msg": str(error),
    }
    if include_trace:
        payload["traceback"] = "".join(traceback.format_exception(type(error), error, error.__traceback__))
    if context:
        safe_ctx = {}
        for k, v in context.items():
            try:
                json.dumps(v)
                safe_ctx[k] = v
            except Exception:
                safe_ctx[k] = str(v)
        payload["context"] = safe_ctx
    try:
        lg.error(json.dumps(payload))
    except Exception:
        lg.error("[structured-error-failed] %s: %s", type(error).__name__, error)


def log_structured_info(
    message: str,
    context: Optional[Dict[str, Any]] = None,
    logger: Optional[logging.Logger] = None,
    *,
    component: str = "AppContext",
    phase: Optional[str] = None,
    module: Optional[str] = None,
    event: str = "info",
    severity: str = "INFO",
) -> None:
    lg = logger or LOGGER
    payload: Dict[str, Any] = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "event": event,
        "severity": severity,
        "component": component,
        "phase": phase,
        "module": module,
        "message": message,
    }
    if context:
        try:
            json.dumps(context)
            payload["context"] = context
        except Exception:
            payload["context"] = {k: str(v) for k, v in context.items()}
    lg.info(json.dumps(payload))

__all__ = ["AppContext", "log_structured_error", "log_structured_info"]


# === SECTION: Construction Helpers ===
# Minimal, resilient constructor helper for AppContext bootstrap
def _try_construct(cls: Optional[type], **candidate_kwargs) -> Optional[Any]:
    """
    Minimal, resilient constructor helper used by AppContext during component bootstrap.
    Attempts to instantiate `cls` using a filtered subset of keyword arguments derived
    from the target __init__ signature. Falls back to a few common tuples.
    
    Silent on failure: Optional components are gracefully skipped if construction fails.
    This is expected behavior, not an error.
    """
    if not cls:
        return None
        
    # Primary attempt: signature-aware filtering
    try:
        sig = inspect.signature(cls.__init__)
        allowed = {k for k in sig.parameters.keys() if k != "self"}
        kwargs = {k: v for k, v in candidate_kwargs.items() if k in allowed and v is not None}
        instance = cls(**kwargs)
        return instance
    except Exception as e_primary:
        # Silently continue to fallbacks (optional components)
        pass
    # Fallback attempts with common kwarg subsets
    fallbacks = [
        ("config", "app", "shared_state", "symbols", "exchange_client", "database_manager", "execution_manager", "market_data_feed"),
        ("config", "app", "shared_state"),
        ("config", "app"),
        ("config",),
        (),
    ]
    for keys in fallbacks:
        try:
            kwargs = {k: candidate_kwargs[k] for k in keys if k in candidate_kwargs and candidate_kwargs[k] is not None}
            instance = cls(**kwargs)
            return instance
        except Exception as e_fallback:
            continue
    
    # Silent failure: optional components are skipped without fanfare
    return None

# === SECTION: AppContext Class Definition ===
class AppContext:
    # === SECTION: Summary Fire-and-Forget (DRY) ===
    def _summary_ff(self, event: str, **kvs) -> None:
        """Safely schedule _emit_summary(event, **kvs) as a background task."""
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._emit_summary(event, **kvs))
        except RuntimeError:
            # No running loop yet → skip silently
            return
        except Exception:
            return
    # === SECTION: Lightweight helpers (moved out of __init__ for clarity) ===
    def _loop_time(self) -> float:
        """Return a monotonic loop time when available, otherwise fall back to time.monotonic."""
        try:
            return asyncio.get_running_loop().time()
        except RuntimeError:
            # No running loop yet — use monotonic clock as a safe fallback.
            import time as _time
            return _time.monotonic()

    def _cfg_bool(self, *names: str, default: bool = False) -> bool:
        """Lookup boolean-ish config values from attributes, dict keys, or env vars.

        Returns the first matching value found, converted to bool. Falls back to `default`.
        """
        def _parse_bool_value(v) -> Optional[bool]:
            """Convert a raw config value to bool, returning None if indeterminate."""
            if isinstance(v, bool):
                return v
            if isinstance(v, str):
                s = v.strip().lower()
                if s in ("1", "true", "yes", "y", "on"):
                    return True
                if s in ("0", "false", "no", "n", "off"):
                    return False
            if v is not None:
                return bool(v)
            return None

        for n in names:
            # attribute on config object
            try:
                if hasattr(self.config, n):
                    result = _parse_bool_value(getattr(self.config, n))
                    if result is not None:
                        return result
            except Exception:
                pass
            # mapping-style config dict
            try:
                if isinstance(self.config, dict) and n in self.config:
                    result = _parse_bool_value(self.config[n])
                    if result is not None:
                        return result
            except Exception:
                pass
            # env var
            try:
                v = os.getenv(n, None)
                if v is not None:
                    result = _parse_bool_value(v)
                    if result is not None:
                        return result
            except Exception:
                pass
        return default

    def _cfg_float(self, key: str, default: Union[float, int] = 0.0) -> float:
        try:
            v = self._cfg(key, default)
            return float(v)
        except Exception:
            return float(default)

    def _cfg_int(self, key: str, default: int = 0) -> int:
        try:
            v = self._cfg(key, default)
            return int(float(v))
        except Exception:
            return int(default)
    # === SECTION: Component Lists (DRY) ===
    # === SECTION: DRY Utilities (calls & attrs) ===
    def _try_call(self, obj: Any, method_names: Iterable[str], *args, **kwargs) -> bool:
        """
        Best-effort synchronous call of the first available method in method_names on obj.
        If the method returns a coroutine, it is NOT awaited here (caller is sync).
        Returns True if a method was found and invoked, False otherwise.
        """
        if not obj:
            return False
        for name in method_names or []:
            try:
                fn = getattr(obj, name, None)
                if callable(fn):
                    res = fn(*args, **kwargs)
                    # do not await here; sync context
                    _ = res  # explicit ignore
                    return True
            except Exception:
                self._dbg("_try_call failed for %r.%s", obj, name)
                continue
        return False

    async def _try_call_async(self, obj: Any, method_names: Iterable[str], *args, **kwargs) -> bool:
        """
        Best-effort asynchronous call of the first available method in method_names on obj.
        Awaits if the method returns a coroutine. Returns True on successful invocation.
        """
        if not obj:
            return False
        for name in method_names or []:
            try:
                fn = getattr(obj, name, None)
                if callable(fn):
                    res = fn(*args, **kwargs)
                    if asyncio.iscoroutine(res):
                        await res
                    return True
            except Exception:
                self._dbg("_try_call_async failed for %r.%s", obj, name)
                continue
        return False

    async def _maybe_await(self, v: Any) -> Any:
        """
        Instance-level helper mirroring the small utility used across components.
        Awaits the value if it's a coroutine, otherwise returns it directly.
        This keeps AppContext self-contained and avoids module-level helper duplication.
        """
        try:
            if asyncio.iscoroutine(v):
                return await v
            return v
        except Exception:
            # Be resilient: if awaiting fails, return the raw value
            return v

    @staticmethod
    def _symbols_list_to_dict(symbols: Iterable[str]) -> Dict[str, Dict[str, Any]]:
        """Convert an iterable of symbol strings into the canonical accepted-symbols dict.

        Returns {SYMBOL: {'enabled': True, 'meta': {}}} for each symbol.
        Implemented as a staticmethod so it can be used from instance methods without
        depending on instance state and to keep helpers grouped on AppContext.
        """
        return {str(s).upper(): {"enabled": True, "meta": {}} for s in (symbols or [])}

    def _set_attr_if_missing(self, obj: Any, name: str, value: Any) -> None:
        """
        Assign attribute 'name' on obj only if it's currently missing or None.
        """
        try:
            if obj is None:
                return
            if getattr(obj, name, None) is None:
                setattr(obj, name, value)
        except Exception:
            self._dbg("_set_attr_if_missing failed for %r.%s", obj, name)

    # === SECTION: DRY Debug Helper ===
    def _dbg(self, msg: str, *args) -> None:
        """Safe debug logger that always includes exc_info=True and never raises."""
        try:
            self.logger.debug(msg, *args, exc_info=True)
        except Exception:
            pass

    # === SECTION: DRY Task Scheduler (fire-and-forget) ===
    def _ff(self, aw: Awaitable, *, name: str | None = None):
        """
        Safely schedule a coroutine as a background task. Returns the created task or None.
        Does not track in _tasks (use _spawn for tracked tasks). Never raises.
        """
        try:
            loop = asyncio.get_running_loop()
            if name:
                return loop.create_task(aw, name=name)
            return loop.create_task(aw)
        except RuntimeError:
            return None  # No running loop yet
        except Exception:
            return None
    def _components_for_shared_state(self) -> List[Any]:
        """
        Components that should receive shared_state on injection/swap.
        """
        return [
            self.symbol_manager,
            self.capital_symbol_governor,
            self.universe_rotation_engine,
            self.market_data_feed,
            self.execution_manager,
            self.strategy_manager,
            self.agent_manager,
            self.risk_manager,
            self.liquidation_agent,
            self.liquidation_orchestrator,
            self.cash_router,
            self.meta_controller,
            self.tp_sl_engine,
            self.performance_monitor,
            self.alert_system,
            self.watchdog,
            self.heartbeat,
            self.compounding_engine,
            self.recovery_engine,
            self.volatility_regime,
            self.performance_evaluator,
            self.portfolio_balancer,
            self.dust_monitor,
            self.exchange_truth_auditor,
        ]

    def _components_for_exchange_client(self) -> List[Any]:
        """
        Components that need an exchange client reference.
        """
        return [
            self.symbol_manager,
            self.market_data_feed,
            self.execution_manager,
            self.strategy_manager,
            self.agent_manager,
            self.risk_manager,
            self.liquidation_agent,
            self.liquidation_orchestrator,
            self.cash_router,
            self.exchange_truth_auditor,
        ]

    def _components_for_shutdown(self) -> List[Any]:
        """
        Ordered components for graceful shutdown (controllers → data/infra → liquidity).
        """
        return [
            self.meta_controller,
            self.agent_manager,
            self.strategy_manager,
            self.universe_rotation_engine,
            self.tp_sl_engine,
            self.execution_manager,
            self.market_data_feed,
            self.symbol_manager,
            self.risk_manager,
            self.performance_monitor,
            self.alert_system,
            self.watchdog,
            self.heartbeat,
            self.compounding_engine,
            self.recovery_engine,
            self.volatility_regime,
            self.performance_evaluator,
            self.portfolio_balancer,
            self.liquidation_orchestrator,
            self.liquidation_agent,
            self.cash_router,
            self.exchange_truth_auditor,
        ]
    # === SECTION: Liquidity Result Finalization (DRY helper) ===
    async def _finalize_liq_result(self, via: str, res: Any, reason: str) -> None:
        """
        Consolidated handling for liquidity orchestration results:
          - Parse result dict/bool
          - Log a concise line
          - Emit SUMMARY LIQUIDITY_RESULT
          - Notify SharedState capital gate (capital.updated)
          - Track persistent failures and emit DEGRADED health when threshold hit

        Expected res shape (dict-like):
          {ok: bool, freed: float, remaining: float, actions: list|int, status: str, error_code: str, reason: str}
        Falls back gracefully when keys are missing or res is a bool.
        """
        try:
            ok = bool(res.get("ok")) if isinstance(res, dict) else bool(res)
            freed = float(res.get("freed", 0.0)) if isinstance(res, dict) else 0.0
            remaining = float(res.get("remaining", 0.0)) if isinstance(res, dict) else 0.0
            actions_len = 0
            if isinstance(res, dict):
                actions = res.get("actions", [])
                try:
                    actions_len = int(len(actions)) if actions is not None else 0
                except Exception:
                    actions_len = 0
            status = ""
            if isinstance(res, dict):
                status = str(res.get("status", "done" if ok else "failed"))
            else:
                status = "done" if ok else "failed"
            error_code = str(res.get("error_code", "")) if isinstance(res, dict) else ""
            # log
            try:
                self.logger.info("[Liquidity:%s] result ok=%s freed=%.6f remaining=%.6f actions=%d status=%s",
                                 via, ok, freed, remaining, actions_len, status)
            except Exception:
                pass
            # summary
            try:
                await self._emit_summary(
                    "LIQUIDITY_RESULT",
                    via=via,
                    ok=ok,
                    freed=freed,
                    remaining=remaining,
                    actions=actions_len,
                    reason=(res.get("reason") if isinstance(res, dict) and res.get("reason") is not None else str(reason)),
                    status=status,
                    error_code=error_code,
                )
            except Exception:
                self.logger.debug("_finalize_liq_result: emit_summary failed", exc_info=True)
            # capital gate notification
            try:
                if hasattr(self.shared_state, "emit_event"):
                    ev = self.shared_state.emit_event("capital.updated", {"source": via, "ok": ok, "freed": freed, "remaining": remaining})
                    if asyncio.iscoroutine(ev):
                        await ev
            except Exception:
                self.logger.debug("_finalize_liq_result: emit capital.updated failed", exc_info=True)
            # persistent failure tracking
            try:
                symbol = ""
                # try extract from last _liq_need_state with max last_ts
                if self._liq_need_state:
                    symbol = max(self._liq_need_state.items(), key=lambda kv: float(kv[1].get("last_ts", 0.0)))[0]
                if ok:
                    if symbol:
                        self._liq_failures[symbol] = 0
                else:
                    if symbol:
                        n = 1 + int(self._liq_failures.get(symbol, 0))
                        self._liq_failures[symbol] = n
                        thresh = int(self._cfg("LIQ_ORCH_FAILS_DEGRADED", 3))
                        if n >= max(1, thresh):
                            await self._emit_health_status("DEGRADED", {
                                "issues": ["LiquidityOrchestrationUnsuccessful"],
                                "symbol": symbol,
                                "fail_count": n
                            })
            except Exception:
                self.logger.debug("_finalize_liq_result: persistent failure tracking failed", exc_info=True)
        except Exception:
            self.logger.debug("_finalize_liq_result failed", exc_info=True)

    # === SECTION: Liquidity Enable Helper (DRY) ===
    def _liq_enabled(self) -> bool:
        """Unified accessor for LIQ_ORCH_ENABLE with default=True."""
        try:
            return bool(self._cfg_bool("LIQ_ORCH_ENABLE", default=True))
        except Exception:
            return True

    # === SECTION: Liquidity Helper (Throttled Summary Emission + Orchestration State) ===
    async def _maybe_emit_liquidity_needed(
        self,
        *,
        symbol: str,
        required_usdt: float,
        free_usdt: float,
        gap_usdt: float,
        reason: str,
    ) -> None:
        """
        Emit LIQUIDITY_NEEDED at most once per cooldown window per symbol.
        Persist a per-symbol deficit state and (optionally) trigger orchestration
        when the gap persists long enough or is sufficiently large.

        Configs (with defaults):
          - LIQUIDITY_NOTICE_COOLDOWN_SEC: int = 120
          - LIQ_ORCH_ENABLE: bool = True
          - LIQ_ORCH_MIN_GAP_USDT: float = 1.50
          - LIQ_ORCH_CONSECUTIVE_TICKS: int = 2
          - LIQ_ORCH_DEBOUNCE_SEC: int = 240
        """
        try:
            cooldown = int(self._cfg("LIQUIDITY_NOTICE_COOLDOWN_SEC", 120) or 120)
        except Exception:
            cooldown = 120

        try:
            now = self._loop_time()
        except Exception:
            now = 0.0

        last = float(self._liquidity_notice_ts.get(symbol, 0.0))

        # Initialize / update per-symbol state
        st = self._liq_need_state.setdefault(symbol, {
            "last_ts": 0.0,
            "last_gap": 0.0,
            "consec": 0,           # use int counter (ticks)
            "orch_last_ts": 0.0,
        })
        st["last_ts"] = now
        st["last_gap"] = float(max(0.0, gap_usdt))

        # If the gap has been resolved, reset counters and return
        if gap_usdt <= 0:
            st["consec"] = 0
            st["last_gap"] = 0.0
            return

        # Common thresholds (computed once)
        try:
            min_gap = float(self._cfg("LIQ_ORCH_MIN_GAP_USDT", 1.50) or 1.50)
        except Exception:
            min_gap = 1.50
        try:
            need_ticks = int(self._cfg("LIQ_ORCH_CONSECUTIVE_TICKS", 2) or 2)
        except Exception:
            need_ticks = 2
        try:
            deb_sec = int(self._cfg("LIQ_ORCH_DEBOUNCE_SEC", 240) or 240)
        except Exception:
            deb_sec = 240

        # Suppress notices/orchestration for micro gaps below actionable threshold
        if float(gap_usdt) < float(min_gap):
            st["consec"] = 0
            st["last_gap"] = float(gap_usdt)
            return

        # Throttled SUMMARY emission (only for actionable gaps)
        if (now - last) >= max(1, cooldown):
            self._liquidity_notice_ts[symbol] = now
            await self._emit_summary(
                "LIQUIDITY_NEEDED",
                symbol=symbol,
                required_usdt=float(f"{required_usdt:.6f}"),
                free_usdt=float(f"{free_usdt:.6f}"),
                gap_usdt=float(f"{gap_usdt:.6f}"),
                reason=str(reason),
            )

        st["consec"] = int(st.get("consec", 0)) + 1

        # If no orchestration/agent is available, emit a DEGRADED health signal with throttling and return
        liq_support_present = bool(getattr(self, "liquidation_orchestrator", None) or getattr(self, "liquidation_agent", None) or hasattr(self.shared_state, "emit_event"))
        if not liq_support_present:
            try:
                if float(gap_usdt) >= min_gap and st["consec"] >= need_ticks:
                    now_ts = now
                    mute_until = float(self._liq_degraded_until.get(symbol, 0.0))
                    if now_ts >= mute_until:
                        self._liq_degraded_until[symbol] = now_ts + max(30, deb_sec)
                        try:
                            await self._emit_summary(
                                "LIQUIDITY_ORCH_MISSING",
                                symbol=symbol,
                                required_usdt=float(f"{required_usdt:.6f}"),
                                free_usdt=float(f"{free_usdt:.6f}"),
                                gap_usdt=float(f"{gap_usdt:.6f}"),
                                reason="NO_ORCHESTRATOR_OR_AGENT",
                            )
                        except Exception:
                            pass
                        try:
                            await self._emit_health_status("DEGRADED", {
                                "issues": ["NAVNotReady"],
                                "liquidity": {symbol: {"gap_usdt": float(f"{gap_usdt:.6f}"), "consec": st["consec"]}},
                                "note": "No liquidity orchestrator/agent available; consider enabling P8_liquidation components."
                            })
                        except Exception:
                            pass
            except Exception:
                self.logger.debug("liquidity missing-orchestrator handler failed", exc_info=True)
            # Do not attempt orchestration when support is missing
            return

        # Orchestration guardrails
        enable = self._liq_enabled()
        if not enable:
            return
        # Fire only if: large enough gap AND persisted enough AND outside debounce window
        try:
            if float(gap_usdt) >= min_gap and int(st.get("consec", 0)) >= need_ticks:
                if (now - float(st.get("orch_last_ts", 0.0))) >= max(30, deb_sec):
                    st["orch_last_ts"] = now
                    try:
                        await self.orchestrate_liquidity(
                            symbol=symbol,
                            required_usdt=float(required_usdt),
                            free_usdt=float(free_usdt),
                            gap_usdt=float(gap_usdt),
                            reason=str(reason),
                        )
                    except Exception:
                        self.logger.debug("orchestrate_liquidity call failed", exc_info=True)
        except Exception:
            self.logger.debug("orchestration trigger guard failed", exc_info=True)

    # === SECTION: Shared State Propagation ===
    def set_shared_state(self, shared_state: Any) -> None:
        """
        Inject or replace the SharedState instance and propagate it to all known components.
        This safely rewires components that expose `set_shared_state()` and falls back to
        assigning the `shared_state` attribute when needed.
        """
        self.shared_state = shared_state
        # Bind ComponentStatusLogger for mirroring
        try:
            if _csl_mod:
                _csl_mod.bind_shared_state(shared_state)
        except Exception:
            self.logger.debug("ComponentStatusLogger binding failed in set_shared_state", exc_info=True)

        targets = self._components_for_shared_state()
        for comp in targets:
            if not comp:
                continue
            try:
                if not self._try_call(comp, ("set_shared_state",), shared_state):
                    try:
                        setattr(comp, "shared_state", shared_state)
                    except Exception:
                        self._set_attr_if_missing(comp, "shared_state", shared_state)
            except Exception:
                self.logger.debug("shared_state propagation failed for %r", comp, exc_info=True)
        # Subscribe to wallet-scan completion if shared_state supports subscriptions
        try:
            sub = None
            if hasattr(self.shared_state, "subscribe") and callable(getattr(self.shared_state, "subscribe")):
                # Common signature: subscribe(topic, callback)
                sub = self.shared_state.subscribe("events.summary", self._on_summary_walletscan)
            elif hasattr(self.shared_state, "on_event") and callable(getattr(self.shared_state, "on_event")):
                # Alternative signature: on_event(topic, callback)
                sub = self.shared_state.on_event("events.summary", self._on_summary_walletscan)
            # Ignore the returned subscription handle if any; best-effort wiring
        except Exception:
            self.logger.debug("walletscan subscriber wiring failed", exc_info=True)
        # Schedule a delayed resync on shared_state injection to catch late universe population.
        # Use _ff (fire-and-forget) which safely handles missing event loops — never use
        # asyncio.run() from a sync method as it fails when a loop is already running.
        try:
            self._ff(self._delayed_resync_liq_symbols(float(self._cfg("WALLETSCAN_RESYNC_DELAY_SEC", 30.0))))
        except Exception:
            self.logger.debug("schedule delayed resync on shared_state set failed", exc_info=True)
        # Refresh LiquidationAgent symbol universe when shared_state is injected/swapped
        try:
            self._ff(self._ensure_universe_bootstrap())
            self._ff(self._sync_liq_agent_symbols_once())
        except Exception:
            self.logger.debug("set_shared_state: universe sync + liq agent symbol refresh failed", exc_info=True)

    def _enforce_shared_state_identity(self, emit_summary: bool = False) -> Dict[str, Any]:
        """
        Ensure core components point to the single canonical SharedState instance.
        Returns an identity snapshot for diagnostics.
        """
        canonical = self.shared_state
        canonical_id = id(canonical) if canonical is not None else None
        targets = {
            "meta_controller": self.meta_controller,
            "tp_sl_engine": self.tp_sl_engine,
            "execution_manager": self.execution_manager,
            "risk_manager": self.risk_manager,
        }

        snapshot: Dict[str, Any] = {"canonical_shared_state_id": canonical_id}
        rewired: List[str] = []

        for name, comp in targets.items():
            comp_ss = getattr(comp, "shared_state", None) if comp else None
            comp_id = id(comp_ss) if comp_ss is not None else None
            match = (comp_ss is canonical) if (comp is not None and canonical is not None) else False

            if comp and canonical is not None and not match:
                try:
                    self._try_call(comp, ("set_shared_state",), canonical)
                except Exception:
                    self.logger.debug("shared_state set_shared_state() failed on %s", name, exc_info=True)
                try:
                    setattr(comp, "shared_state", canonical)
                    rewired.append(name)
                    comp_ss = getattr(comp, "shared_state", None)
                    comp_id = id(comp_ss) if comp_ss is not None else None
                    match = (comp_ss is canonical)
                except Exception:
                    self.logger.debug("shared_state force-assign failed on %s", name, exc_info=True)

            snapshot[f"{name}_shared_state_id"] = comp_id
            snapshot[f"{name}_shared_state_match"] = bool(match)

        snapshot["rewired_components"] = rewired
        snapshot["all_core_match"] = all(
            snapshot.get(f"{name}_shared_state_match", False)
            for name in targets.keys()
            if targets[name] is not None
        )

        if rewired:
            self.logger.warning(
                "[SharedStateIdentity] Rewired mismatched components to canonical SharedState: %s",
                rewired,
            )
        else:
            self.logger.info(
                "[SharedStateIdentity] Core components aligned: %s",
                {k: v for k, v in snapshot.items() if k.endswith("_match") or k == "all_core_match"},
            )

        if emit_summary:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._emit_summary("SHARED_STATE_IDENTITY", **snapshot))
            except RuntimeError:
                pass  # No running loop yet
            except Exception:
                pass

        try:
            strict_identity = bool(self._cfg_bool("STRICT_SHARED_STATE_IDENTITY", default=False))
        except Exception:
            strict_identity = False
        if strict_identity and not bool(snapshot.get("all_core_match", False)):
            raise RuntimeError(f"SharedState identity mismatch detected: {snapshot}")

        return snapshot
    """
    Central wiring orchestrator for Octivault Trader (P9). Handles phased init & teardown.
    """

    # === SECTION: Exchange Readiness Gate ===
    def _assert_execution_manager_path_guard(self) -> None:
        """
        Hard startup invariant: ExchangeClient must enforce ExecutionManager-only order path.
        """
        ec = self.exchange_client
        if ec is None:
            raise RuntimeError("ExchangeClientNotReady")

        guard_flag = getattr(ec, "ENFORCE_EXECUTION_MANAGER_PATH", None)
        if guard_flag is None:
            guard_flag = getattr(ec, "_enforce_execution_manager_path", None)

        if bool(guard_flag) is not True:
            raise AssertionError(
                "ExchangeClient.ENFORCE_EXECUTION_MANAGER_PATH must be True at startup"
            )

    async def _gate_exchange_ready(self) -> None:
        """
        Hard gate: ensure an ExchangeClient exists, its public session is started,
        and exchangeInfo cache is warmed before proceeding to P4/P5.
        Raise RuntimeError if not satisfied.
        """
        await self._ensure_exchange_public_ready()
        if self.exchange_client is None:
            raise RuntimeError("ExchangeClientNotReady")
        # Propagate client to already-built components
        self._propagate_exchange_client()
        # Boot-time assertion: fail fast if execution-path guard is disabled.
        self._assert_execution_manager_path_guard()

    # === SECTION: Attempt Fetch Balances ===
    async def _attempt_fetch_balances(self) -> None:
        """
        Best-effort: fetch balances once at startup if API keys are configured.
        Populates SharedState to clear BalancesNotReady/NAVNotReady gates early.
        """
        try:
            ec = self.exchange_client
            if not ec:
                return
            has_keys = bool(getattr(ec, "api_key", None)) and bool(getattr(ec, "api_secret", None))
            if not has_keys:
                return
            get_bal = getattr(ec, "get_balances", None) or getattr(ec, "get_spot_balances", None)
            if not callable(get_bal):
                return
            bals = get_bal()
            bals = await bals if asyncio.iscoroutine(bals) else bals
            if bals and self.shared_state and hasattr(self.shared_state, "update_balances"):
                res = self.shared_state.update_balances(bals)
                if asyncio.iscoroutine(res):
                    await res
        except Exception:
            self.logger.debug("attempt_fetch_balances failed", exc_info=True)

    # === SECTION: Constructor & Initialization ===
    def __init__(
        self,
        config: Any = None,  # Config instance or dict-like
        logger: Optional[logging.Logger] = None,
        exchange_client: Optional[Any] = None,
        shared_state: Optional[Any] = None,
        up_to_phase: Optional[int] = None,  # back-compat: older scripts passed this to ctor
        **kwargs: Any,  # back-compat sink for stale constructor kwargs
    ) -> None:
        # Safety fix: ensure no_remainder_below_quote is always initialized
        self.no_remainder_below_quote = 0.0
        self.config = config or {}
        
        # Set default timeframe configurations if not already present
        # Multi-timeframe hierarchy: 1h = brain (regime), 5m = hands (execution)
        if not isinstance(self.config, dict):
            # If config is an object, ensure it has these attributes
            if not hasattr(self.config, "VOLATILITY_REGIME_TIMEFRAME"):
                self.config.VOLATILITY_REGIME_TIMEFRAME = "1h"  # Brain: slow, strategic
            if not hasattr(self.config, "ohlcv_timeframes"):
                self.config.ohlcv_timeframes = ["5m", "1h"]  # Hands: 5m execution, Brain: 1h regime
        else:
            # If config is a dict, set defaults
            self.config.setdefault("VOLATILITY_REGIME_TIMEFRAME", "1h")  # Brain: slow, strategic
            self.config.setdefault("ohlcv_timeframes", ["5m", "1h"])  # Hands: 5m execution, Brain: 1h regime
        
        self.logger = logger or logging.getLogger("AppContext")
        if not self.logger.handlers:
            h = logging.StreamHandler()
            fmt = logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
            h.setFormatter(fmt)
            self.logger.addHandler(h)
            # Optional file handler (enable via config.LOG_FILE or APP_LOG_FILE env)
            try:
                log_path = None
                try:
                    log_path = self._cfg("LOG_FILE", None)
                except Exception:
                    log_path = None
                if not log_path:
                    log_path = os.getenv("APP_LOG_FILE")
                if log_path:
                    fh = logging.FileHandler(str(log_path))
                    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s"))
                    self.logger.addHandler(fh)
            except Exception:
                self.logger.debug("failed to attach file handler", exc_info=True)
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False  # prevent duplicate logs via root logger

        # Announce strict imports mode (no fallbacks)
        try:
            asyncio.get_running_loop()  # only emit async summary if a loop is already running
            self._summary_ff("STRICT_IMPORTS_ENABLED", enabled=True)
        except RuntimeError:
            # No running loop yet — log synchronously.
            try:
                self.logger.info("SUMMARY %s", {"component": "AppContext", "event": "STRICT_IMPORTS_ENABLED", "enabled": True})
            except Exception:
                pass
        except Exception:
            pass

        # Optional: quiet third-party verbose logs (e.g., TensorFlow C++ messages)
        try:
            quiet_tf = bool(self._cfg("QUIET_TF_LOGS", False))
            if quiet_tf:
                # 0 = all logs, 1 = filter INFO, 2 = filter WARNING, 3 = filter ERROR
                os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", str(self._cfg("TF_CPP_MIN_LOG_LEVEL", 2)))
                # Limit BLAS threads on constrained hosts (optional)
                if self._cfg("LIMIT_BLAS_THREADS", True):
                    os.environ.setdefault("OMP_NUM_THREADS", str(self._cfg("OMP_NUM_THREADS", 1)))
                    os.environ.setdefault("MKL_NUM_THREADS", str(self._cfg("MKL_NUM_THREADS", 1)))
        except Exception:
            pass

        # External (may be injected)
        self.exchange_client = exchange_client
        self.shared_state = shared_state
        self._ctor_up_to_phase = up_to_phase

        if kwargs:
            try:
                self.logger.warning("[AppContext] Ignoring unknown constructor kwargs: %s", sorted(kwargs.keys()))
            except Exception:
                pass

        # Components (optional; can be injected externally before init)
        self.symbol_manager: Any = None
        self.capital_symbol_governor: Optional[Any] = None
        self.universe_rotation_engine: Optional[Any] = None
        self.market_data_feed: Optional[Any] = None
        self.agent_manager: Optional[Any] = None
        self.risk_manager: Optional[Any] = None
        self.execution_manager: Optional[Any] = None
        self.meta_controller: Optional[Any] = None
        self.tp_sl_engine: Optional[Any] = None
        self.performance_monitor: Optional[Any] = None
        self.alert_system: Optional[Any] = None
        self.watchdog: Optional[Any] = None
        self.heartbeat: Optional[Any] = None
        self.compounding_engine: Optional[Any] = None
        self.recovery_engine: Optional[Any] = None
        self.volatility_regime: Optional[Any] = None
        self.strategy_manager: Optional[Any] = None
        self.liquidation_agent: Optional[Any] = None
        self.performance_evaluator: Optional[Any] = None
        self.portfolio_balancer: Optional[Any] = None
        self.liquidation_orchestrator: Optional[Any] = None
        self.cash_router: Optional[Any] = None  # optional, if separate component is used
        self.dashboard_server: Optional[Any] = None
        self.capital_allocator: Optional[Any] = None # P9 Wealth
        self.profit_target_engine: Optional[Any] = None # P9 Wealth
        self.dust_monitor: Optional[Any] = None  # Phase G: Real-time dust monitoring
        self.adaptive_capital_engine: Optional[Any] = None  # Adaptive sizing engine
        self.exchange_truth_auditor: Optional[Any] = None  # P3 exchange truth reconciliation

        # Persistent execution journal (crash-safe JSONL)
        from core.trade_journal import TradeJournal
        self.trade_journal: TradeJournal = TradeJournal()
        self.session_id: str = datetime.now(timezone.utc).isoformat()

        # Back-compat aliases removed.

        # Optional objects for RecoveryEngine
        self.database_manager: Optional[Any] = None
        self.pnl_calculator: Optional[Any] = None
        self.sstools: Optional[Any] = None

        # Runtime bookkeeping
        self._tasks: List[asyncio.Task] = []
        self._tasks_map: Dict[str, asyncio.Task] = {}
        self._io_started: bool = False
        self._agents_started: bool = False
        self._p6_startables: list = []
        # Background affordability scout (round-robin over symbols)
        self._scout_task: Optional[asyncio.Task] = None
        self._scout_index: int = 0
        # Background universe rotation task (periodic UURE)
        self._uure_task: Optional[asyncio.Task] = None
        # Per-symbol throttle map for liquidity events
        self._liquidity_notice_ts: Dict[str, float] = {}
        self._liq_need_state: Dict[str, Dict[str, float]] = {}  # per-symbol persistence for liquidity orchestration
        self._liq_degraded_until: Dict[str, float] = {}  # per-symbol mute window for DEGRADED health notifications when no orchestration is available
        self._liq_inflight: Dict[str, bool] = {}
        self._liq_failures: Dict[str, int] = {}

        # Concurrency / lifecycle guards
        self._init_lock: asyncio.Lock = asyncio.Lock()
        self._init_completed: bool = False
        self._init_highest_phase: int = 0  # highest phase successfully completed
        self._tpsl_started: bool = False

        # Idempotent guard for public bootstrap
        self._public_ready_once: bool = False

        # === SECTION: Config & Loop Helpers ===
        # NOTE: lightweight config and loop helpers are implemented as class
        # methods (see _loop_time, _cfg_bool, _cfg_float, _cfg_int) and no
        # longer defined as nested functions to keep __init__ concise.
        

        # NOTE: _maybe_emit_liquidity_needed and _liq_enabled are proper class methods
        # (see below class body) — no longer defined as closures here.
        # === SECTION: Liquidity Plane (Single Construction Point) ===
        try:
            # Optional CashRouter (if module present)
            CashRouter = getattr(_cash_router_mod, "CashRouter", None) if "_cash_router_mod" in globals() else None
            if CashRouter:
                self.cash_router = _try_construct(
                    CashRouter,
                    config=self.config,
                    logger=self.logger,
                    app=self,
                    shared_state=self.shared_state,
                    execution_manager=self.execution_manager,
                )
        except Exception:
            self.logger.debug("CashRouter construct failed", exc_info=True)

        try:
            LiquidationAgent = getattr(_liq_agent_mod, "LiquidationAgent", None) if "_liq_agent_mod" in globals() else None
            if LiquidationAgent:
                self.liquidation_agent = _try_construct(
                    LiquidationAgent,
                    config=self.config,
                    logger=self.logger,
                    app=self,
                    shared_state=self.shared_state,
                    execution_manager=self.execution_manager,
                )
                # Optional one-time wiring if agent supports cash_router attribute
                try:
                    if self.cash_router is not None and getattr(self.liquidation_agent, "cash_router", None) is None:
                        setattr(self.liquidation_agent, "cash_router", self.cash_router)
                except Exception:
                    self.logger.debug("wire cash_router into LiquidationAgent failed", exc_info=True)
        except Exception:
            self.logger.debug("LiquidationAgent construct failed", exc_info=True)

        try:
            LiquidationOrchestrator = getattr(_liq_orch_mod, "LiquidationOrchestrator", None) if "_liq_orch_mod" in globals() else None
            if LiquidationOrchestrator:
                self.liquidation_orchestrator = _try_construct(
                    LiquidationOrchestrator,
                    config=self.config,
                    logger=self.logger,
                    app=self,
                    shared_state=self.shared_state,
                    execution_manager=self.execution_manager,
                )
        except Exception:
            self.logger.debug("LiquidationOrchestrator construct failed", exc_info=True)


    # === SECTION: Generic Helpers ===
    def _spawn(self, name: str, aw: Awaitable[Any]):
        t = asyncio.create_task(aw, name=name)
        self._tasks.append(t)
        self._tasks_map[name] = t
        return t

    # === SECTION: Exchange Client Propagation ===
    def _propagate_exchange_client(self) -> None:
        """
        Ensure all components that need ExchangeClient receive it (method or attr),
        even if they were constructed before the client was created/started.
        Also sets the 'ex' attribute for components (e.g., CashRouter) that expect it.
        """
        if not self.exchange_client:
            return

        targets = self._components_for_exchange_client()
        for comp in targets:
            if not comp:
                continue
            try:
                if not self._try_call(comp, ("set_exchange_client",), self.exchange_client):
                    # Assign common attribute names used across components
                    self._set_attr_if_missing(comp, "exchange_client", self.exchange_client)
                    # Some components (e.g., CashRouter) use 'ex' internally
                    self._set_attr_if_missing(comp, "ex", self.exchange_client)
            except Exception:
                self.logger.debug("exchange_client propagation failed for %r", comp, exc_info=True)

    # === SECTION: Liquidity Orchestration ===
    async def orchestrate_liquidity(self, *, symbol: str, required_usdt: float, free_usdt: float, gap_usdt: float, reason: str):
        """
        Deterministic liquidity orchestration. Uses a single selected mode and fails fast if
        the required component/entrypoint is missing.
        Modes: cash_router | orchestrator | agent | event_bus
        """
        # In-flight guard per symbol (prevents overlapping orchestration for the same asset)
        _sym_norm = str(symbol).upper().strip()
        try:
            if self._liq_inflight.get(_sym_norm):
                await self._emit_summary("LIQUIDITY_SUPPRESSED", symbol=_sym_norm, reason="inflight")
                return {"ok": False, "status": "suppressed", "reason": "inflight"}
            self._liq_inflight[_sym_norm] = True
        except Exception:
            pass
        # Normalize inputs
        try:
            symbol = str(symbol).upper().strip()
            required_usdt = float(required_usdt)
            free_usdt = float(free_usdt)
            gap_usdt = float(gap_usdt)
            reason = str(reason)
        except Exception:
            self.logger.debug("input normalization failed", exc_info=True)
        try:
            payload = {
                "symbol": symbol,
                "required_usdt": float(required_usdt),
                "free_usdt": float(free_usdt),
                "gap_usdt": float(gap_usdt),
                "reason": str(reason),
            }
            mode = str(getattr(self.config, "LIQUIDITY_ORCHESTRATION_MODE", "agent") or "agent").lower()

            if mode == "cash_router":
                cr = getattr(self, "cash_router", None)
                if not (cr and hasattr(cr, "ensure_free_usdt") and callable(getattr(cr, "ensure_free_usdt"))):
                    raise RuntimeError("CashRouterRequiredMissing")
                try:
                    headroom = float(self._cfg("CASH_ROUTER_USDT_BUFFER", 0.10) or 0.10)
                except Exception:
                    headroom = 0.10
                target = max(float(required_usdt), float(free_usdt) + float(gap_usdt)) + headroom
                res = cr.ensure_free_usdt(target, reason=str(reason))
                res = await res if asyncio.iscoroutine(res) else res
                await self._finalize_liq_result("cash_router", res, reason)
                await self._emit_summary("LIQUIDITY_TRIGGERED", via="cash_router", method="ensure_free_usdt", target_usdt=float(f"{target:.6f}"), **payload)
                return res

            if mode == "orchestrator":
                orch = getattr(self, "liquidation_orchestrator", None)
                if not (orch and hasattr(orch, "trigger_liquidity") and callable(getattr(orch, "trigger_liquidity"))):
                    raise RuntimeError("LiquidationOrchestratorRequiredMissing")
                res = orch.trigger_liquidity(**payload)
                res = await res if asyncio.iscoroutine(res) else res
                await self._finalize_liq_result("liquidation_orchestrator", res, reason)
                await self._emit_summary("LIQUIDITY_TRIGGERED", via="liquidation_orchestrator", method="trigger_liquidity", **payload)
                return res

            if mode == "agent":
                agent = getattr(self, "liquidation_agent", None)
                if not (agent and hasattr(agent, "free_up_quote") and callable(getattr(agent, "free_up_quote"))):
                    raise RuntimeError("LiquidationAgentRequiredMissing")
                res = agent.free_up_quote(**payload)
                res = await res if asyncio.iscoroutine(res) else res
                await self._finalize_liq_result("liquidation_agent", res, reason)
                await self._emit_summary("LIQUIDITY_TRIGGERED", via="liquidation_agent", method="free_up_quote", **payload)
                return res

            if mode == "event_bus":
                if not hasattr(self.shared_state, "emit_event"):
                    raise RuntimeError("EventBusRequiredMissing")
                v = self.shared_state.emit_event("Liquidity.Orchestrate", payload)
                if asyncio.iscoroutine(v):
                    await v
                await self._emit_summary("LIQUIDITY_TRIGGERED", via="event_bus", **payload)
                # Safety warning: emit one-time summary if no local consumer exists
                try:
                    if not (self.liquidation_orchestrator or self.liquidation_agent):
                        await self._emit_summary("LIQUIDITY_EVENTBUS_NO_LOCAL_CONSUMER", symbol=symbol)
                except Exception:
                    self.logger.debug("event_bus safety warning failed", exc_info=True)
                return {"ok": True, "via": "event_bus"}

            raise RuntimeError(f"Unknown orchestration mode: {mode}")
        finally:
            try:
                self._liq_inflight[_sym_norm] = False
            except Exception:
                pass

    # === SECTION: Liquidity Mode Validation ===
    def validate_liquidity_mode(self) -> None:
        """
        Validate that the configured liquidity orchestration mode has the required interface.
        Modes and required callables:
          - cash_router    -> cash_router.ensure_free_usdt(...)
          - orchestrator   -> liquidation_orchestrator.trigger_liquidity(...)
          - agent          -> liquidation_agent.free_up_quote(...)
          - event_bus      -> shared_state.emit_event(...)
        Raises:
            RuntimeError if the selected mode is unknown or the interface is missing.
        Emits:
            SUMMARY {event: 'LIQUIDITY_MODE_VALIDATED' | 'LIQUIDITY_MODE_INVALID'}
        """
        mode = str(getattr(self.config, "LIQUIDITY_ORCHESTRATION_MODE", "agent") or "agent").lower()

        ok = False
        required_iface = ""

        if mode == "cash_router":
            cr = getattr(self, "cash_router", None)
            ok = bool(cr and callable(getattr(cr, "ensure_free_usdt", None)))
            required_iface = "CashRouter.ensure_free_usdt"
        elif mode == "orchestrator":
            orch = getattr(self, "liquidation_orchestrator", None)
            ok = bool(orch and callable(getattr(orch, "trigger_liquidity", None)))
            required_iface = "LiquidationOrchestrator.trigger_liquidity"
        elif mode == "agent":
            ag = getattr(self, "liquidation_agent", None)
            ok = bool(ag and callable(getattr(ag, "free_up_quote", None)))
            required_iface = "LiquidationAgent.free_up_quote"
        elif mode == "event_bus":
            ok = bool(hasattr(self.shared_state, "emit_event") and callable(getattr(self.shared_state, "emit_event", None)))
            required_iface = "SharedState.emit_event"
        else:
            raise RuntimeError(f"Unknown orchestration mode: {mode}")

        if not ok:
            self._summary_ff("LIQUIDITY_MODE_INVALID", mode=mode, required_iface=required_iface)
            raise RuntimeError(f"Liquidity orchestration mode '{mode}' requires {required_iface}")

        self._summary_ff("LIQUIDITY_MODE_VALIDATED", mode=mode)

    # === SECTION: Async Step Helper ===
    async def _step(self, name: str, coro: Awaitable[Any]):
        try:
            return await coro
        except Exception as e:
            log_structured_error(
                e,
                context={"step": name},
                logger=self.logger,
                component="AppContext",
                phase=name,
                event="startup_step_exception",
            )
            raise

    # === SECTION: Config Value Accessor ===
    def _cfg(self, key: str, default=None):
        try:
            # 1. Check SharedState for live/dynamic overrides
            if hasattr(self, "shared_state") and hasattr(self.shared_state, "dynamic_config"):
                val = self.shared_state.dynamic_config.get(key)
                if val is not None:
                    return val
        except Exception:
            pass

        try:
            if hasattr(self.config, key):
                return getattr(self.config, key)
        except Exception:
            pass
        try:
            if isinstance(self.config, dict) and key in self.config:
                return self.config[key]
        except Exception:
            pass
        v = os.getenv(key)
        return v if v is not None else default

    # === SECTION: Start Timeout Helper ===
    def _start_timeout_sec(self, phase_key: Optional[str] = None) -> float:
        """Startup timeout for component start(); supports per-phase override like P4_MARKET_DATA_START_TIMEOUT_SEC."""
        try:
            if phase_key:
                # Convert e.g. "P4_market_data" → "P4_MARKET_DATA_START_TIMEOUT_SEC"
                key = f"{phase_key.upper()}_START_TIMEOUT_SEC"
                v = self._cfg(key, None)
                if v is not None:
                    return float(v)
        except Exception:
            pass
        # Friendly default for first-time market data warmups
        try:
            if phase_key == "P4_market_data":
                return float(self._cfg("P4_MARKET_DATA_START_TIMEOUT_SEC", 90.0))
            if phase_key == "P8_dashboard_server":
                return 1.0  # Immediate backgrounding
        except Exception:
            pass
        try:
            return float(self._cfg("START_TIMEOUT_SEC", 30.0))
        except Exception:
            return 30.0

    # === SECTION: Summary Emission ===
    async def _emit_summary(self, event: str, **kvs):
        """
        Emit a one-line SUMMARY log and forward to shared_state events bus (topic=events.summary).
        """
        # Route LIQUIDITY_NEEDED through throttled helper to enforce min-gap/debounce globally
        if event == "LIQUIDITY_NEEDED":
            try:
                await self._maybe_emit_liquidity_needed(
                    symbol=str(kvs.get("symbol")),
                    required_usdt=float(kvs.get("required_usdt")),
                    free_usdt=float(kvs.get("free_usdt")),
                    gap_usdt=float(kvs.get("gap_usdt")),
                    reason=str(kvs.get("reason", "UNKNOWN")),
                )
            except Exception:
                self.logger.debug("LIQUIDITY_NEEDED reroute failed", exc_info=True)
            return
        try:
            line = {"component": "AppContext", "event": event, **kvs}
            self.logger.info("SUMMARY %s", line)
            if hasattr(self.shared_state, "emit_event"):
                v = self.shared_state.emit_event("events.summary", line)
                if asyncio.iscoroutine(v):
                    await v
        except Exception:
            # never raise from observability helpers
            pass

    # === SECTION: Health Status Emission ===
    async def _emit_health_status(self, level: str, details: Optional[dict] = None):
        """
        Emit P9 HealthStatus-style event via shared_state (topic='events.health.status').
        level: "OK" | "DEGRADED" | "ERROR" | "STARTING" | "SHUTDOWN"
        """
        payload = {
            "component": "AppContext",
            "level": level,
            "details": details or {},
            "ts": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }
        try:
            if hasattr(self.shared_state, "emit_event"):
                v = self.shared_state.emit_event("events.health.status", payload)
                if asyncio.iscoroutine(v):
                    await v
        except Exception:
            pass

    # === SECTION: Phase Summary Emitters (DRY) ===
    def _phase_emit(self, event: str, **kvs) -> None:
        """Fire-and-forget wrapper around _emit_summary for PHASE_* events."""
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._emit_summary(event, **kvs))
        except RuntimeError:
            return  # No running loop yet
        except Exception:
            pass

    def _phase_start(self, phase: str) -> None:
        self._phase_emit("PHASE_START", phase=phase)

    def _phase_done(self, phase: str) -> None:
        self._phase_emit("PHASE_DONE", phase=phase)

    def _phase_timeout(self, phase: str, timeout_sec: float) -> None:
        self._phase_emit("PHASE_TIMEOUT", phase=phase, timeout_sec=timeout_sec)

    def _phase_error(self, phase: str, error: Exception | str) -> None:
        self._phase_emit("PHASE_ERROR", phase=phase, error=str(error))

    def _phase_skip(self, phase: str, reason: str) -> None:
        self._phase_emit("PHASE_SKIP", phase=phase, status="SKIPPED", reason=reason)

    # === SECTION: Liquidity Snapshot ===
    def _liquidity_snapshot(self) -> Dict[str, Dict[str, float]]:
        """
        Return a compact view of outstanding per-symbol liquidity gaps observed recently.
        Only symbols with a positive last_gap are included.
        Shape: {SYMBOL: {"gap_usdt": float, "consec": int, "last_ts": float}}
        """
        snap: Dict[str, Dict[str, float]] = {}
        try:
            for sym, st in (self._liq_need_state or {}).items():
                try:
                    gap = float(st.get("last_gap", 0.0) or 0.0)
                    if gap > 0.0:
                        snap[str(sym).upper()] = {
                            "gap_usdt": float(f"{gap:.6f}"),
                            "consec": int(st.get("consec", 0) or 0),
                            "last_ts": float(st.get("last_ts", 0.0) or 0.0),
                        }
                except Exception:
                    continue
        except Exception:
            pass
        return snap

    # === SECTION: Dust Metrics Snapshot ===
    def _dust_metrics_snapshot(self) -> Dict[str, Any]:
        """Return a sanitized view of SharedState dust telemetry for observability."""
        snapshot: Dict[str, Any] = {
            "registry_size": 0,
            "origin_breakdown": {},
            "strategy_pct": 0.0,
            "external_pct": 0.0,
            "has_external_trash": False,
        }
        ss = getattr(self, "shared_state", None)
        if not ss:
            return snapshot
        try:
            metrics = dict(getattr(ss, "metrics", {}) or {})
            registry_size = metrics.get("dust_registry_size")
            if registry_size is None:
                try:
                    registry_size = len(getattr(ss, "dust_registry", {}) or {})
                except Exception:
                    registry_size = 0
            snapshot["registry_size"] = int(registry_size or 0)

            breakdown_raw = dict(metrics.get("dust_origin_breakdown", {}) or {})
            breakdown: Dict[str, int] = {}
            for origin, count in breakdown_raw.items():
                try:
                    breakdown[str(origin)] = int(count)
                except Exception:
                    continue
            snapshot["origin_breakdown"] = breakdown

            total = sum(breakdown.values())
            if total > 0:
                snapshot["strategy_pct"] = round((breakdown.get("strategy_portfolio", 0) / total) * 100.0, 2)
                snapshot["external_pct"] = round((breakdown.get("external_untracked", 0) / total) * 100.0, 2)
                snapshot["has_external_trash"] = breakdown.get("external_untracked", 0) > 0
            else:
                snapshot["has_external_trash"] = False

            # TIER 2: Include policy conflict metrics
            try:
                conflicts = dict(metrics.get("policy_conflicts", {}) or {})
                snapshot["policy_conflicts"] = conflicts
            except Exception:
                snapshot["policy_conflicts"] = {}

            # TIER 2: Include cold-bootstrap duration
            try:
                if hasattr(ss, "get_cold_bootstrap_duration_sec"):
                    snapshot["bootstrap_duration_sec"] = float(ss.get_cold_bootstrap_duration_sec())
                else:
                    snapshot["bootstrap_duration_sec"] = 0.0
            except Exception:
                snapshot["bootstrap_duration_sec"] = 0.0
        except Exception:
            self.logger.debug("dust metrics snapshot failed", exc_info=True)
        return snapshot

    # === SECTION: Start With Timeout ===
    async def _start_with_timeout(self, phase_key: str, obj: Any):
        """
        Run obj.start() with a timeout. If it exceeds the timeout, keep running in the
        background and log completion when done. Uses a shielded task so wait timeout
        does not cancel the actual start() coroutine.
        Also supports common alternative entrypoints if .start() is absent: start_async(), run(), run_async().
        """
        if not obj:
            return
        # Special-case: P4_market_data can expose a warmup-then-loop contract.
        # If present, run warmup with timeout, then spawn the long-lived loop in background,
        # and emit PHASE_DONE upon warmup completion (Option 1).
        if phase_key == "P4_market_data":
            warmup_fn = None
            loop_fn = None
            # Discover warmup and loop entrypoints if available
            for nm in ("start_warmup", "warmup", "start_and_return"):
                fn = getattr(obj, nm, None)
                if callable(fn):
                    warmup_fn = fn
                    break
            for nm in ("run_loop", "run", "run_async", "start_loop"):
                fn = getattr(obj, nm, None)
                if callable(fn):
                    loop_fn = fn
                    break
            if warmup_fn and loop_fn:
                timeout = self._start_timeout_sec(phase_key)
                log_structured_info(f"[{phase_key}] warmup() beginning", component="AppContext", phase=phase_key)
                self._phase_start(phase_key)
                try:
                    warmup_task = asyncio.create_task(warmup_fn(), name=f"warmup:{phase_key}")
                    try:
                        await asyncio.wait_for(asyncio.shield(warmup_task), timeout=timeout)
                        log_structured_info(f"[{phase_key}] warmup() completed", component="AppContext", phase=phase_key)
                        try:
                            self._phase_done(phase_key)
                        except Exception:
                            pass
                    except asyncio.TimeoutError:
                        self.logger.warning("[%s] warmup() timed out at %.1fs — continuing in background", phase_key, timeout)
                        try:
                            self._phase_timeout(phase_key, timeout)
                        except Exception:
                            pass
                    except Exception as e:
                        log_structured_error(e, context={"phase": phase_key, "when": "warmup_wait"},
                                             logger=self.logger, component="AppContext",
                                             phase=phase_key, event="component_start_exception")
                        with contextlib.suppress(Exception):
                            await warmup_task
                        raise
                except Exception as e:
                    log_structured_error(e, context={"phase": phase_key, "when": "warmup_create_task"},
                                         logger=self.logger, component="AppContext",
                                         phase=phase_key, event="component_start_exception")
                    raise
                # Spawn the long-lived loop in the background (non-blocking)
                try:
                    loop_task = asyncio.create_task(loop_fn(), name=f"loop:{phase_key}")
                    # Optional: attach a done-callback to log errors from the background loop
                    def _loop_done_cb(t: asyncio.Task):
                        try:
                            exc = t.exception()
                        except asyncio.CancelledError:
                            self.logger.warning("[%s] loop task was cancelled", phase_key)
                            return
                        except Exception as e:
                            log_structured_error(e, context={"phase": phase_key, "when": "loop_done_cb-extract"},
                                                 logger=self.logger, component="AppContext",
                                                 phase=phase_key, event="component_loop_exception")
                            return
                        if exc is not None:
                            log_structured_error(exc, context={"phase": phase_key, "when": "loop_done"},
                                                 logger=self.logger, component="AppContext",
                                                 phase=phase_key, event="component_loop_exception")
                    loop_task.add_done_callback(_loop_done_cb)
                except Exception as e:
                    log_structured_error(e, context={"phase": phase_key, "when": "loop_create_task"},
                                         logger=self.logger, component="AppContext",
                                         phase=phase_key, event="component_start_exception")
                # Return after scheduling background loop; skip the generic start path.
                return
        # Resolve a usable start function
        start_fn = None
        for nm in ("start", "start_async", "run", "run_async"):
            fn = getattr(obj, nm, None)
            if callable(fn):
                start_fn = fn
                break

        if start_fn is None:
            # Emit a standardized SKIP summary for observability
            self._phase_skip(phase_key, "NoStartMethod")
            return
        timeout = self._start_timeout_sec(phase_key)
        log_structured_info(f"[{phase_key}] start() beginning", component="AppContext", phase=phase_key)
        # Phase start summary
        self._phase_start(phase_key)

        try:
            res = start_fn()
            if not asyncio.iscoroutine(res):
                # Synchronous start completed immediately
                log_structured_info(f"[{phase_key}] start() completed (sync)", component="AppContext", phase=phase_key)
                try:
                    self._phase_done(phase_key)
                except Exception:
                    pass
                return

            start_task = asyncio.create_task(res, name=f"start:{phase_key}")

            def _done_cb(t: asyncio.Task):
                try:
                    exc = t.exception()
                except asyncio.CancelledError:
                    self.logger.warning("[%s] start() task was cancelled", phase_key)
                    return
                except Exception as e:
                    log_structured_error(e, context={"phase": phase_key, "when": "done_cb-extract"},
                                         logger=self.logger, component="AppContext",
                                         phase=phase_key, event="component_start_exception")
                    return

                if exc is None:
                    log_structured_info(f"[{phase_key}] start() completed", component="AppContext", phase=phase_key)
                    try:
                        self._phase_done(phase_key)
                    except Exception:
                        pass
                else:
                    log_structured_error(exc, context={"phase": phase_key, "when": "start_done"},
                                         logger=self.logger, component="AppContext",
                                         phase=phase_key, event="component_start_exception")
                    try:
                        self._phase_error(phase_key, exc)
                    except Exception:
                        pass

            start_task.add_done_callback(_done_cb)

            try:
                await asyncio.wait_for(asyncio.shield(start_task), timeout=timeout)
                log_structured_info(f"[{phase_key}] start() returned", component="AppContext", phase=phase_key)
            except asyncio.TimeoutError:
                self.logger.warning("[%s] start() timed out at %.1fs — continuing in background", phase_key, timeout)
                try:
                    self._phase_timeout(phase_key, timeout)
                except Exception:
                    pass
            except Exception as e:
                log_structured_error(e, context={"phase": phase_key, "when": "start_wait"},
                                     logger=self.logger, component="AppContext",
                                     phase=phase_key, event="component_start_exception")
                with contextlib.suppress(Exception):
                    await start_task
                raise
        except Exception as e:
            log_structured_error(e, context={"phase": phase_key, "when": "start_create_task"},
                                 logger=self.logger, component="AppContext",
                                 phase=phase_key, event="component_start_exception")
            raise

    # === SECTION: Enforce Single Liquidity Mode ===
    def enforce_single_liquidity_mode(self) -> None:
        """
        Keep exactly ONE liquidity component based on LIQUIDITY_ORCHESTRATION_MODE.
        Others are nulled to avoid ambiguous paths.
        Modes: cash_router | orchestrator | agent | event_bus
        """
        mode = str(getattr(self.config, "LIQUIDITY_ORCHESTRATION_MODE", "agent") or "agent").lower()

        keep = {"cash_router": False, "orchestrator": False, "agent": False, "event_bus": False}
        if mode in keep:
            keep[mode] = True
        else:
            raise RuntimeError(f"Unknown orchestration mode: {mode}")

        if not keep["cash_router"]:
            self.cash_router = None
        if not keep["orchestrator"]:
            self.liquidation_orchestrator = None
        if not keep["agent"]:
            self.liquidation_agent = None
        # event_bus relies on shared_state only; nothing to keep/null specifically.

        self._summary_ff("LIQUIDITY_PLANE_SELECTED", mode=mode)

    # === SECTION: Start Liquidity Components If Enabled ===
    async def _start_liq_components_if_enabled(self):
        """
        Preferentially start liquidity components when enabled.
        Order: CashRouter (if any) → LiquidationOrchestrator → LiquidationAgent.
        Wires orchestrator completion callback to refresh MetaController's CashRouter.
        """
        # Validate mode and enforce a single component up front (no duplicates)
        try:
            self.validate_liquidity_mode()
            self.enforce_single_liquidity_mode()
        except Exception as e:
            log_structured_error(e, context={"phase": "P8_liquidity", "where": "validate/enforce"}, logger=self.logger, component="AppContext", phase="P8_liquidity", event="liquidity_mode_invalid")
            # If invalid, do not attempt to start any liquidity components
            return
        if not self._liq_enabled():
            return
        try:
            if self.cash_router:
                await self._start_with_timeout("P8_cash_router", self.cash_router)
        except Exception:
            self.logger.debug("cash_router start failed", exc_info=True)
        orch = getattr(self, "liquidation_orchestrator", None)
        if orch:
            await self._start_with_timeout("P8_liquidation_orchestrator", orch)

        # PerformanceWatcher (Dynamic Tuning) - Start after liquidity components
        pw = getattr(self, "performance_watcher", None)
        if pw and hasattr(pw, "start"):
            try:
                await pw.start()
                self.logger.info("PerformanceWatcher started.")
            except Exception as e:
                self.logger.warning("PerformanceWatcher failed to start: %s", e)
            try:
                if hasattr(orch, "on_completed") and self.meta_controller:
                    def _liq_done_cb(*_a, **_kw):
                        try:
                            if hasattr(self.meta_controller, "refresh_cash_router"):
                                self.meta_controller.refresh_cash_router()
                        except Exception:
                            self.logger.debug("liq_done_cb failed", exc_info=True)
                    orch.on_completed(_liq_done_cb)
            except Exception:
                self.logger.debug("wire liq orchestrator on_completed failed", exc_info=True)
        agent = getattr(self, "liquidation_agent", None)
        if agent:
            await self._start_with_timeout("P8_liquidation_agent", agent)
            try:
                await self._ensure_universe_bootstrap()
            except Exception:
                self.logger.debug("ensure_universe_bootstrap after agent start failed", exc_info=True)
            await self._sync_liq_agent_symbols_once()
            # Schedule a one-shot delayed resync to capture late wallet-scan results
            try:
                asyncio.create_task(self._delayed_resync_liq_symbols(float(self._cfg("WALLETSCAN_RESYNC_DELAY_SEC", 30.0))))
            except Exception:
                self.logger.debug("schedule delayed resync failed", exc_info=True)

    # === SECTION: Start P6 Controls ===
    async def _start_p6_controls(self):
        """
        Start P6 control-plane components (RiskManager first).
        Components must implement start(); each is wrapped with per-phase timeout.
        """
        items = list(getattr(self, "_p6_startables", []) or [])
        # Ensure RiskManager (if present) is first in order
        try:
            items.sort(key=lambda c: 0 if c.__class__.__name__ == "RiskManager" else 1)
        except Exception:
            pass
        for comp in items:
            phase_key = getattr(comp, "_phase_key", "P6_control")
            try:
                await self._start_with_timeout(phase_key, comp)
            except Exception:
                # Non-fatal to allow the rest of P6 to come up
                self.logger.debug("P6 component failed to start: %r", comp, exc_info=True)

    # === SECTION: Wait Until Ready ===
    async def _wait_until_ready(self, gates: List[str], timeout_sec: int, poll_sec: float = 2.0) -> dict:
        """
        Wait until selected readiness gates are clear or timeout. Returns the final snapshot.
        gates: any of ["market_data","universe","execution","capital","exchange","startup_sanity"]
        """
        want = {g.strip().lower() for g in (gates or []) if g}
        deadline = self._loop_time() + max(0, int(timeout_sec))

        def _blocked(snap: dict) -> List[str]:
            issues = set(snap.get("issues", []))
            blocks: List[str] = []
            if "market_data" in want and "MarketDataNotReady" in issues:
                blocks.append("market_data")
            if "universe" in want and "SymbolsUniverseEmpty" in issues:
                blocks.append("universe")
            if "execution" in want and "ExecutionManagerNotReady" in issues:
                blocks.append("execution")
            if "capital" in want and ({"BalancesNotReady", "NAVNotReady", "LiquidityShortfall"} & issues):    
                blocks.append("capital")
            if "exchange" in want and "ExchangeClientNotReady" in issues:
                blocks.append("exchange")
            if "startup_sanity" in want and ({"FiltersCoverageLow", "FreeQuoteBelowFloor"} & issues):
                blocks.append("startup_sanity")
            return blocks

        snap = await self._ops_plane_snapshot()
        blocked = _blocked(snap)
        while blocked and self._loop_time() < deadline:
            self.logger.info("[Init] waiting on gates: %s (poll %.1fs)", blocked, poll_sec)
            await asyncio.sleep(poll_sec)
            snap = await self._ops_plane_snapshot()
            blocked = _blocked(snap)

        if blocked:
            self.logger.warning("[Init] readiness gate timed out; still blocked: %s", blocked)
        else:
            self.logger.info("[Init] readiness gates cleared")
            # Start the background affordability scout now that gates are clear
            try:
                self._start_affordability_scout()
            except Exception:
                self.logger.debug("failed to start affordability scout after gates clear", exc_info=True)
            
            # 🔥 FIX: Seed initial symbols for UURE before loop starts
            # UURE needs candidates to score, but discovery may be slow at startup
            # This prevents the pre-scoring gate from failing on first cycle
            try:
                if self.shared_state:
                    current = await self.shared_state.get_accepted_symbols()
                    if not current or len(current) < 3:
                        self.logger.info("[Init] Seeding initial universe for UURE (discovery in progress)...")
                        
                        seed_symbols = {
                            "BTCUSDT": {"status": "TRADING", "notional": 10},
                            "ETHUSDT": {"status": "TRADING", "notional": 10},
                            "BNBUSDT": {"status": "TRADING", "notional": 10},
                            "SOLUSDT": {"status": "TRADING", "notional": 10},
                            "ADAUSDT": {"status": "TRADING", "notional": 10},
                        }
                        
                        await self.shared_state.set_accepted_symbols(seed_symbols)
                        self.logger.info(f"[Init] Seeded {len(seed_symbols)} symbols for UURE startup")
            except Exception:
                self.logger.debug("failed to seed UURE symbols", exc_info=True)
            
            # Start the background UURE loop now that gates are clear
            try:
                self._start_uure_loop()
            except Exception:
                self.logger.debug("failed to start UURE loop after gates clear", exc_info=True)

        return snap

    # ═══════════════════════════════════════════════════════════════════════════════════
    # SECTION: Restart Mode Detection (CRITICAL FIX #3)
    # ═══════════════════════════════════════════════════════════════════════════════════
    async def _detect_restart_mode(self) -> bool:
        """
        CRITICAL FIX #3: Detect if this is a restart with existing positions.
        
        Correctly identifies restart vs cold-start to prevent treating
        "existing positions" as errors that need liquidation.
        
        Return:
            True = This is a RESTART (existing positions/portfolio/intents found)
            False = This is COLD START (fresh initialization, no prior state)
        
        This enables proper restart-aware behavior throughout the system.
        """
        try:
            if not self.shared_state:
                return False
            
            # Check 1: Existing positions in portfolio
            try:
                positions = await self.shared_state.get_positions()
                if positions and len(positions) > 0:
                    position_list = [f"{sym}:{qty}" for sym, qty in positions.items() if qty > 0]
                    self.logger.warning(
                        "[AppContext:RestartDetect] RESTART MODE: Found %d existing positions: %s. "
                        "Portfolio will be OBSERVED and MANAGED per strategy, not auto-liquidated.",
                        len(position_list), ", ".join(position_list[:5])  # Show first 5
                    )
                    return True
            except Exception as e:
                self.logger.debug("[RestartDetect] Position check failed: %s", e)
            
            # Check 2: Pending accumulation intents
            try:
                intents = getattr(self.shared_state, "_pending_position_intents", {})
                if intents and len(intents) > 0:
                    intent_list = [f"{k[0]}/{k[1]}" for k in intents.keys()]
                    self.logger.warning(
                        "[AppContext:RestartDetect] RESTART MODE: Found %d pending accumulation intents: %s. "
                        "Resuming accumulation for existing positions.",
                        len(intent_list), ", ".join(intent_list[:5])
                    )
                    return True
            except Exception as e:
                self.logger.debug("[RestartDetect] Intent check failed: %s", e)
            
            # Check 3: Prior execution history
            try:
                metrics = getattr(self.shared_state, "metrics", {})
                total_trades = metrics.get("total_trades_executed", 0)
                first_trade_at = metrics.get("first_trade_at")
                
                if total_trades > 0 or first_trade_at is not None:
                    self.logger.warning(
                        "[AppContext:RestartDetect] RESTART MODE: Prior trading history detected "
                        "(total_trades=%d, first_trade_at=%s). "
                        "System will continue in NORMAL operation mode (not cold-bootstrap).",
                        total_trades, first_trade_at
                    )
                    return True
            except Exception as e:
                self.logger.debug("[RestartDetect] History check failed: %s", e)
            
            # Check 4: Balances/NAV history
            try:
                nav_history = getattr(self.shared_state, "nav_history", [])
                if nav_history and len(nav_history) > 1:  # More than just current snapshot
                    self.logger.info(
                        "[AppContext:RestartDetect] RESTART MODE: NAV history found (%d snapshots). "
                        "Prior trading session detected.",
                        len(nav_history)
                    )
                    return True
            except Exception as e:
                self.logger.debug("[RestartDetect] NAV history check failed: %s", e)
            
            # Check 5: Database / persistence file exists (any DB means NOT first-ever launch)
            try:
                db_path = self._cfg("DB_PATH", self._cfg("DATABASE_PATH", None))
                snapshot_path = self._cfg("SNAPSHOT_PATH", self._cfg("STATE_SNAPSHOT_FILE", None))
                if db_path and os.path.exists(str(db_path)):
                    self.logger.warning(
                        "[AppContext:RestartDetect] RESTART MODE: Database file exists at %s. "
                        "This is not a first-ever launch.",
                        db_path,
                    )
                    return True
                if snapshot_path and os.path.exists(str(snapshot_path)):
                    self.logger.warning(
                        "[AppContext:RestartDetect] RESTART MODE: State snapshot exists at %s. "
                        "This is not a first-ever launch.",
                        snapshot_path,
                    )
                    return True
            except Exception as e:
                self.logger.debug("[RestartDetect] DB/snapshot check failed: %s", e)

            # No restart indicators found
            self.logger.info(
                "[AppContext:RestartDetect] COLD_START MODE: No existing positions, intents, history, or DB found. "
                "True first-ever launch detected."
            )
            return False
            
        except Exception as e:
            self.logger.error("[AppContext:RestartDetect] Fatal detection error: %s. Assuming cold-start.", e, exc_info=True)
            return False

    # === SECTION: Universe Helpers ===
    # === SECTION: Universe Helpers (DRY) ===
    def _sm_symbols(self) -> List[str]:
        """
        Return uppercase, order-preserving unique list of symbols from SymbolManager using common accessors.
        Accessor preference order: get_active_symbols → get_symbols → list_symbols → symbols (attr).
        Returns [] if SymbolManager not available or exposes none of these.
        """
        try:
            sm = getattr(self, "symbol_manager", None)
            if not sm:
                return []
            candidates = ("get_active_symbols", "get_symbols", "list_symbols", "symbols")
            syms_iter = None
            for nm in candidates:
                obj = getattr(sm, nm, None)
                res = None
                if callable(obj):
                    res = obj()
                    if asyncio.iscoroutine(res):
                        # do not await here; the caller of this helper is sync; best-effort
                        try:
                            # snapshot synchronous fallback if coroutine leaks accidentally
                            return []
                        except Exception:
                            return []
                    syms_iter = res
                    break
                elif obj is not None:
                    syms_iter = obj
                    break
            if not syms_iter:
                return []
            # Normalize to uppercase list and enforce order-preserving uniqueness
            out: List[str] = []
            seen = set()
            for s in syms_iter:
                u = str(s).upper()
                if u not in seen:
                    out.append(u)
                    seen.add(u)
            return out
        except Exception:
            self.logger.debug("_sm_symbols failed", exc_info=True)
            return []

    def _config_seed_symbols(self) -> List[str]:
        """
        Read seed symbols from config.SYMBOLS or config.SEED_SYMBOLS.
        Uppercase, order-preserving unique. Returns [] if none.
        
        Safely handles:
        - List config: ["BTCUSDT", "ETHUSDT"]
        - Single string: "BTCUSDT"
        - Comma-separated string: "BTCUSDT,ETHUSDT"
        """
        syms_raw: List[str] = []

        try:
            raw = self._cfg("SYMBOLS", []) or []
            if isinstance(raw, str):
                # Support comma-separated or single symbol
                syms_raw = [s.strip() for s in raw.split(",") if s.strip()]
            else:
                syms_raw = list(raw)
        except Exception:
            syms_raw = []

        if not syms_raw:
            try:
                raw = self._cfg("SEED_SYMBOLS", []) or []
                if isinstance(raw, str):
                    # Support comma-separated or single symbol
                    syms_raw = [s.strip() for s in raw.split(",") if s.strip()]
                else:
                    syms_raw = list(raw)
            except Exception:
                syms_raw = []

        try:
            out: List[str] = []
            seen = set()
            for s in syms_raw:
                u = str(s).upper()
                if u not in seen:
                    out.append(u)
                    seen.add(u)
            return out
        except Exception:
            return []

    # === SECTION: WalletScan Resync Helpers ===
    async def _delayed_resync_liq_symbols(self, delay_sec: float = 10.0) -> None:
        """One-shot delayed resync of universe → LiquidationAgent symbols."""
        self.logger.info("[WalletScanResync] delayed resync firing (delay=%.1fs)", delay_sec)
        try:
            await asyncio.sleep(max(0.0, float(delay_sec)))
            await self._ensure_universe_bootstrap()
            await self._sync_liq_agent_symbols_once()
        except Exception:
            self.logger.debug("delayed resync failed", exc_info=True)

    async def _on_summary_walletscan(self, evt: dict) -> None:
        """
        Subscriber callback for events.summary to detect wallet-scan completion and resync symbols.
        Triggers on a variety of event/message/component/phase heuristics for wallet scan completion.
        """
        try:
            if not isinstance(evt, dict):
                return
            ev_raw = str(evt.get("event", "")).strip()
            msg_lower = str(evt.get("message", "")).lower()
            comp_lower = str(evt.get("component", "")).lower()
            phase_lower = str(evt.get("phase", "")).lower()
            ev_lower = ev_raw.lower()

            hit = False
            # Direct named completion (case-insensitive)
            if ev_lower == "p3_wallet_scan_completed":
                hit = True
            # Generic PHASE_DONE for wallet-scan phases
            elif ev_lower == "phasedone" or ev_lower == "phase_done":
                if phase_lower.startswith("p3_wallet"):
                    hit = True
            # Heuristic: free-form message text
            elif "wallet scan completed" in msg_lower or "one-shot wallet scan completed" in msg_lower:
                hit = True
            # Heuristic: WalletScannerAgent emitted a completion-ish event
            elif "walletscanneragent" in comp_lower and ev_lower in {"accepted", "done", "completed"}:
                hit = True

            if not hit:
                return

            # Universe → LiquidationAgent refresh
            await self._ensure_universe_bootstrap()
            await self._sync_liq_agent_symbols_once()

            # Schedule a short follow-up resync; shared_state may update moments later.
            try:
                self.logger.info("[WalletScanResync] scheduling short follow-up in 5.0s")
                asyncio.create_task(self._delayed_resync_liq_symbols(5.0))
            except Exception:
                self.logger.debug("walletscan follow-up resync schedule failed", exc_info=True)
        except Exception:
            self.logger.debug("on_summary_walletscan failed", exc_info=True)

    # === SECTION: Adaptive Capital Engine Background Monitor ===
    async def _run_adaptive_capital_monitor(self):
        """
        Background task for AdaptiveCapitalEngine monitoring.
        Periodically evaluates adaptive sizing decisions and logs them.
        """
        try:
            self.logger.info("[AdaptiveCapitalEngine] Monitor starting")
            
            # Initial health report
            await self._emit_health_status("OK", {
                "component": "AdaptiveCapitalEngine",
                "status": "initialized",
                "enabled": bool(getattr(self.adaptive_capital_engine, 'enabled', False))
            })
            
            while True:
                try:
                    # Wait 5 minutes between evaluations
                    await asyncio.sleep(300)
                    
                    if not self.adaptive_capital_engine or not getattr(self.adaptive_capital_engine, 'enabled', False):
                        continue
                    
                    # Get current market conditions for evaluation
                    nav = 0.0
                    free_capital = 0.0
                    volatility_pct = self._cfg_float("ACE_DEFAULT_VOLATILITY_PCT", 0.015)
                    drawdown_pct = 0.0
                    fee_bps = self._cfg_float("ACE_DEFAULT_FEE_BPS", 10.0)
                    slippage_bps = self._cfg_float("ACE_DEFAULT_SLIPPAGE_BPS", 5.0)
                    min_notional = self._cfg_float("ACE_DEFAULT_MIN_NOTIONAL", 30.0)
                    slot_utilization = 0.0
                    throughput_per_hour = 0.0
                    target_throughput_per_hour = self._cfg_float("MAX_TRADES_PER_HOUR", 4.0)
                    
                    try:
                        if self.shared_state:
                            nav = float(await self.shared_state.get_nav_quote() or 0.0)
                            free_capital = float(await self.shared_state.get_spendable_balance("USDT") or 0.0)
                            
                            # Get drawdown from metrics
                            metrics = getattr(self.shared_state, 'metrics', {}) or {}
                            drawdown_pct = float(metrics.get('drawdown_pct', 0.0) or 0.0)
                            
                            # Get throughput from recent trades
                            recent_trades = getattr(self.shared_state, 'recent_trades', []) or []
                            if recent_trades:
                                # Count trades in last hour
                                now = asyncio.get_running_loop().time()
                                hour_ago = now - 3600
                                recent_count = sum(1 for t in recent_trades[-50:] if getattr(t, 'ts', 0) > hour_ago)
                                throughput_per_hour = recent_count
                    except Exception as e:
                        self.logger.debug("[AdaptiveCapitalEngine] Failed to gather market data: %s", e)

                    # Get trade history for performance analysis
                    trade_history = []
                    try:
                        if self.shared_state:
                            # Get recent trades from shared_state
                            recent_trades = getattr(self.shared_state, 'recent_trades', []) or []
                            trade_history = recent_trades[-50:]  # Last 50 trades
                    except Exception as e:
                        self.logger.debug("[AdaptiveCapitalEngine] Failed to get trade history: %s", e)
                    
                    # Evaluate adaptive sizing
                    decision = self.adaptive_capital_engine.evaluate(
                        symbol="ADAPTIVE_GLOBAL",  # Global evaluation
                        nav=nav,
                        free_capital=free_capital,
                        base_risk_fraction=0.10,  # 10% base risk
                        volatility_pct=volatility_pct,
                        drawdown_pct=drawdown_pct,
                        fee_bps=fee_bps,
                        slippage_bps=slippage_bps,
                        min_notional=min_notional,
                        slot_utilization=slot_utilization,
                        throughput_per_hour=throughput_per_hour,
                        target_throughput_per_hour=target_throughput_per_hour,
                        trade_history=trade_history,
                        now_ts=None
                    )
                    
                    # Log the decision
                    self.logger.info(
                        "[AdaptiveCapitalEngine] Evaluation: risk=%.3f, min_quote=%.2f, win_rate=%.3f, "
                        "avg_r=%.4f, reasons=%s",
                        decision.risk_fraction,
                        decision.min_trade_quote,
                        decision.win_rate,
                        decision.avg_r_multiple,
                        ",".join(decision.reasons) if decision.reasons else "none"
                    )
                    
                    # Emit health status with current adaptive metrics
                    await self._emit_health_status("OK", {
                        "component": "AdaptiveCapitalEngine",
                        "risk_fraction": float(f"{decision.risk_fraction:.3f}"),
                        "min_trade_quote": float(f"{decision.min_trade_quote:.2f}"),
                        "win_rate": float(f"{decision.win_rate:.3f}"),
                        "avg_r_multiple": float(f"{decision.avg_r_multiple:.4f}"),
                        "reasons": decision.reasons,
                        "last_evaluation": asyncio.get_running_loop().time()
                    })

                except Exception as e:
                    self.logger.warning("[AdaptiveCapitalEngine] Evaluation failed: %s", e)
                    await self._emit_health_status("DEGRADED", {
                        "component": "AdaptiveCapitalEngine",
                        "error": str(e),
                        "last_error_ts": asyncio.get_running_loop().time()
                    })
                    await asyncio.sleep(60)  # Wait 1 minute before retrying
                    
        except asyncio.CancelledError:
            self.logger.info("[AdaptiveCapitalEngine] Monitor cancelled")
            await self._emit_health_status("SHUTDOWN", {
                "component": "AdaptiveCapitalEngine",
                "status": "shutdown"
            })
        except Exception as e:
            self.logger.error("[AdaptiveCapitalEngine] Monitor crashed: %s", e)
            await self._emit_health_status("ERROR", {
                "component": "AdaptiveCapitalEngine",
                "error": str(e),
                "crashed": True
            })

    # === SECTION: Shutdown ===
    async def shutdown(self, save_snapshot: bool = False) -> None:
        """
        Graceful teardown used by main_phased.py.
        - Cancels background tasks (affordability scout).
        - Calls stop()/shutdown()/close() on known components if present.
        - Optionally asks SharedState/RecoveryEngine to persist a snapshot.
        Never raises; logs errors at DEBUG to keep shutdown robust.
        
        Enhanced with timeout protection to prevent hangs during cleanup.
        """
        # Stop background scout with timeout
        try:
            await asyncio.wait_for(self._stop_affordability_scout(), timeout=5.0)
        except asyncio.TimeoutError:
            self.logger.debug("shutdown: affordability scout stop timed out")
        except Exception:
            self.logger.debug("shutdown: stop scout failed", exc_info=True)

        # Stop background UURE loop with timeout
        try:
            await asyncio.wait_for(self._stop_uure_loop(), timeout=5.0)
        except asyncio.TimeoutError:
            self.logger.debug("shutdown: UURE loop stop timed out")
        except Exception:
            self.logger.debug("shutdown: stop UURE loop failed", exc_info=True)

        # Ordered stop: higher-level controllers first
        comps = self._components_for_shutdown()
        for c in comps:
            if not c:
                continue
            try:
                await asyncio.wait_for(
                    self._try_call_async(c, ("stop", "shutdown", "close")),
                    timeout=5.0
                )
            except asyncio.TimeoutError:
                self.logger.debug("shutdown: component stop timed out for %r", c)
            except Exception:
                self.logger.debug("shutdown: component stop failed for %r", c, exc_info=True)

        # Exchange client last (network connections) with longer timeout
        try:
            ec = getattr(self, "exchange_client", None)
            if ec:
                await asyncio.wait_for(
                    self._try_call_async(ec, ("stop", "shutdown", "close")),
                    timeout=10.0
                )
        except asyncio.TimeoutError:
            self.logger.debug("shutdown: exchange client stop timed out")
        except Exception:
            self.logger.debug("shutdown: exchange client stop failed", exc_info=True)

        # Optional snapshot
        if save_snapshot:
            try:
                if self.shared_state and hasattr(self.shared_state, "save_snapshot"):
                    v = self.shared_state.save_snapshot()
                    if asyncio.iscoroutine(v):
                        await asyncio.wait_for(v, timeout=5.0)
                elif self.recovery_engine and hasattr(self.recovery_engine, "save_snapshot"):
                    v = self.recovery_engine.save_snapshot()
                    if asyncio.iscoroutine(v):
                        await asyncio.wait_for(v, timeout=5.0)
            except asyncio.TimeoutError:
                self.logger.debug("shutdown: snapshot operation timed out")
            except Exception:
                self.logger.debug("shutdown: snapshot failed", exc_info=True)

        # Cancel any stray tasks we spawned
        try:
            for t in list(self._tasks or []):
                if t and not t.done():
                    t.cancel()
            # Best-effort join with timeout
            with contextlib.suppress(Exception):
                await asyncio.wait_for(
                    asyncio.gather(*[t for t in self._tasks if t], return_exceptions=True),
                    timeout=5.0
                )
        except asyncio.TimeoutError:
            self.logger.debug("shutdown: task gathering timed out")
        except Exception:
            self.logger.debug("shutdown: tasks join failed", exc_info=True)

        try:
            await self._emit_summary("SHUTDOWN_DONE", ok=True)
        except Exception:
            pass
    # === SECTION: Accepted Symbols Dictionary ===
    async def _get_accepted_symbols_dict(self) -> Dict[str, Dict[str, Any]]:
        # Prefer shared_state provider
        try:
            if self.shared_state and hasattr(self.shared_state, "get_accepted_symbols"):
                r = self.shared_state.get_accepted_symbols()
                r = await r if asyncio.iscoroutine(r) else r
                if isinstance(r, dict) and r:
                    return {str(k).upper(): (v if isinstance(v, dict) else {"enabled": True, "meta": {}}) for k, v in r.items()}
        except Exception:
            self.logger.debug("get_accepted_symbols failed", exc_info=True)
        # Fallback: SymbolManager (active/universe symbols) via DRY helper
        try:
            syms_list = self._sm_symbols()
            if syms_list:
                return self._symbols_list_to_dict(syms_list)
        except Exception:
            self.logger.debug("symbol_manager symbols fallback failed", exc_info=True)
        # Fallback: config seed symbols (SYMBOLS/SEED_SYMBOLS) via DRY helper
        try:
            syms = self._config_seed_symbols()
            return self._symbols_list_to_dict(syms)
        except Exception:
            return {}

    # === SECTION: Sync Liquidation Agent Symbols Once ===
    async def _sync_liq_agent_symbols_once(self) -> None:
        """
        Push the full accepted symbols list into LiquidationAgent (if present).
        Least-impact liquidation decisions require global context.
        Emits a SUMMARY line on success.
        """
        try:
            agent = getattr(self, "liquidation_agent", None)
            if not agent:
                return
            if not (hasattr(agent, "set_symbols") and callable(getattr(agent, "set_symbols"))):
                return
            symbols_dict = await self._get_accepted_symbols_dict()
            all_syms = [str(s).upper() for s in (symbols_dict or {}).keys()]
            if not all_syms:
                return
            maybe = agent.set_symbols(all_syms)
            if asyncio.iscoroutine(maybe):
                await maybe
            try:
                await self._emit_summary("LIQ_AGENT_SYMBOLS_SYNCED", count=len(all_syms))
            except Exception:
                pass
            self.logger.info("[LiquidationAgent] set_symbols() pushed %d symbols", len(all_syms))
        except Exception:
            self.logger.debug("sync_liq_agent_symbols_once failed", exc_info=True)

    # === SECTION: Ensure Universe Bootstrap ===
    async def _ensure_universe_bootstrap(self):
        """If shared_state has empty universe, bootstrap from config.SYMBOLS (no hardcoded defaults)."""
        if not self.shared_state:
            return
        try:
            getter = getattr(self.shared_state, "get_accepted_symbols", None)
            setter = getattr(self.shared_state, "set_accepted_symbols", None)
            if not callable(getter) or not callable(setter):
                return
            current = getter()
            current = await current if asyncio.iscoroutine(current) else current
            if isinstance(current, dict) and current:
                return  # already populated
            cfg_syms = self._config_seed_symbols()
            if cfg_syms:
                await self._maybe_await(setter(self._symbols_list_to_dict(cfg_syms)))
                self.logger.info("[P3] Universe bootstrapped from config.SYMBOLS (%d symbols)", len(cfg_syms))
                # Push updated universe to LiquidationAgent for low-impact SELL planning
                try:
                    await self._sync_liq_agent_symbols_once()
                except Exception:
                    self.logger.debug("universe bootstrap: liq agent symbols refresh failed", exc_info=True)
        except Exception:
            self.logger.debug("universe bootstrap failed", exc_info=True)

    # === SECTION: Trade Quantity Helper ===
    async def get_trade_quantity(self, symbol: str, current_price: float) -> float:
        """
        Calculates a safe trade quantity for a BUY order, dynamically sizing based on
        equity (Compounding) while respecting MAX_PER_TRADE_USDT limits and
        MIN_NOTIONAL constraints (Dust Prevention).
        Returns 0.0 if the trade is unsafe or too small.
        """
        try:
            if current_price <= 0:
                return 0.0
            
            # --- 1. Determine Dynamic Trade Size (Compounding) ---
            # Default: Fixed $15 or Configured Max
            max_spend = float(self._cfg("MAX_PER_TRADE_USDT", 15.0))
            
            # Compounding: Use % of Total Equity if enabled (e.g., 2.0%)
            risk_pct = float(self._cfg("RISK_PER_TRADE_PCT", 0.0)) # 0.0 = Disabled
            if risk_pct > 0:
                try:
                    # Get Total Equity safely
                    equity = 0.0
                    pw = getattr(self, "performance_watcher", None)
                    if pw and hasattr(pw, "_get_total_equity"):
                        equity = await pw._get_total_equity()
                    elif hasattr(self.shared_state, "get_total_dashboard_equity"):
                        equity = await self.shared_state.get_total_dashboard_equity()
                    
                    if equity > 0:
                        compounded_size = equity * (risk_pct / 100.0)
                        # Use the smaller of (Compounded Size, Hard Max Cap) to manage risk
                        # If user wants purely compounding without cap, they can set MAX_PER_TRADE_USDT very high.
                        target_usdt = min(compounded_size, max_spend)

                        # Ensure we don't go below the configured floor just because equity is small.
                        _compound_floor = self._cfg_float("COMPOUND_SIZE_FLOOR_USDT", 11.0)
                        target_usdt = max(target_usdt, _compound_floor)
                    else:
                        target_usdt = max_spend
                except Exception:
                    target_usdt = max_spend
            else:
                target_usdt = max_spend

            # --- 2. Calculate Quantity ---
            qty = target_usdt / float(current_price)
            
            # --- 3. Check against Exit-Feasibility Floor (Safety & Dust) ---
            min_notional = self._cfg_float("MIN_NOTIONAL_FLOOR_USDT", 5.0)  # operator-configurable fallback
            _floor_multiplier = self._cfg_float("MIN_NOTIONAL_FLOOR_MULTIPLIER", 1.1)
            safe_floor = min_notional * _floor_multiplier

            try:
                if hasattr(self.shared_state, "compute_min_entry_quote"):
                    base_quote = float(getattr(self.config, "DEFAULT_PLANNED_QUOTE", 0.0) or 0.0)
                    safe_floor = await self.shared_state.compute_min_entry_quote(
                        symbol,
                        default_quote=base_quote,
                        price=float(current_price),
                    )
            except Exception:
                # Fall back to minNotional if exit floor isn't available
                em = getattr(self, "execution_manager", None)
                if em and hasattr(em, "ensure_symbol_filters_ready"):
                    try:
                        nf_res = em.ensure_symbol_filters_ready(symbol)
                        nf = await nf_res if asyncio.iscoroutine(nf_res) else nf_res
                        if hasattr(em, "_extract_filter_vals"):
                            _, _, _, _, mn = em._extract_filter_vals(nf)
                            if mn and float(mn) > 0:
                                min_notional = float(mn)
                    except Exception:
                        pass
                safe_floor = min_notional * _floor_multiplier
            
            notional = qty * current_price
            if notional < safe_floor:
                self.logger.info("[%s] Trade quantity %.6f ($%.2f) < safe min notional ($%.2f). Skipping.", symbol, qty, notional, safe_floor)
                return 0.0

            return qty
        except Exception as e:
            self.logger.error("get_trade_quantity failed for %s: %s", symbol, e)
            return 0.0

    # === SECTION: Ensure Exchange Public Ready ===
    async def _ensure_exchange_public_ready(self):
        """
        Ensure an ExchangeClient exists and can serve unsigned/public endpoints (e.g., exchangeInfo)
        before P4 starts. Safe to call multiple times.
        """
        if getattr(self, "_public_ready_once", False):
            return
        try:
            # Prefer an existing instance; otherwise build from module-level helpers
            if self.exchange_client is None and _exchange_mod is not None:
                # Try factory (singleton) first
                _factory = getattr(_exchange_mod, "get_global_exchange_client", None)
                if callable(_factory):
                    self.exchange_client = _factory(
                        config=self.config,
                        logger=self.logger,
                        app=self,
                        shared_state=self.shared_state,
                    )
            if self.exchange_client is None and _exchange_mod is not None:
                # Fallback: explicit public bootstrap
                _ensure_pub = getattr(_exchange_mod, "ensure_public_bootstrap", None)
                if callable(_ensure_pub):
                    self.exchange_client = await _ensure_pub(
                        config=self.config,
                        logger=self.logger,
                        app=self,
                        shared_state=self.shared_state,
                    )

            # Make sure the public session is up (unsigned GETs will work)
            if self.exchange_client and hasattr(self.exchange_client, "_ensure_started_public"):
                maybe = self.exchange_client._ensure_started_public()
                if asyncio.iscoroutine(maybe):
                    await maybe

            # Warm the exchange info cache so early consumers (SymbolManager/MDF) won't 404 on None
            if self.exchange_client and hasattr(self.exchange_client, "get_exchange_info"):
                info = self.exchange_client.get_exchange_info()
                if asyncio.iscoroutine(info):
                    await info
            self._public_ready_once = True
            try:
                await self._emit_summary("PUBLIC_READY", ready=True)
            except Exception:
                pass
        except Exception:
            self.logger.debug("ensure_exchange_public_ready failed", exc_info=True)

    # === SECTION: Ensure Exchange Signed Ready ===
    async def _ensure_exchange_signed_ready(self) -> bool:
        """
        Attempt to ensure ExchangeClient is running in signed mode (API keys present & start() succeeded).
        Emits PHASE_SKIP summaries if we must remain public-only. Safe and idempotent.
        Returns True if signed mode is confirmed; False otherwise.
        """
        try:
            ec = self.exchange_client
            if not ec:
                self.logger.warning("[AppContext] No exchange_client instance; cannot enter signed mode.")
                try:
                    await self._emit_summary("PHASE_SKIP", phase="P4_market_data", status="SKIPPED", reason="ExchangeMissing")
                except Exception:
                    pass
                return False

            has_keys = bool(getattr(ec, "api_key", None)) and bool(getattr(ec, "api_secret", None))
            if not has_keys:
                self.logger.warning("[AppContext] API keys not detected; staying in public-only mode.")
                try:
                    await self._emit_summary("PHASE_SKIP", phase="P4_market_data", status="SKIPPED", reason="ExchangeNotSigned")
                except Exception:
                    pass
                return False

            # Idempotent signed start
            try:
                if hasattr(ec, "start"):
                    res = ec.start()
                    if asyncio.iscoroutine(res):
                        await res
                self.logger.info("[AppContext] ExchangeClient is signed and ready.")

                # Eagerly fetch balances so BalancesNotReady/NAVNotReady gates clear early
                try:
                    await self._attempt_fetch_balances()
                except Exception:
                    self.logger.debug("attempt_fetch_balances after signed start failed", exc_info=True)

                return True
            except Exception as e:
                self.logger.warning("[AppContext] Exchange signed start failed: %s", e, exc_info=True)
                try:
                    await self._emit_summary(
                        "PHASE_SKIP",
                        phase="P4_market_data",
                        status="SKIPPED",
                        reason="ExchangeStartFailed",
                        error=str(e),
                    )
                except Exception:
                    pass
                return False
        except Exception:
            self.logger.debug("_ensure_exchange_signed_ready unexpected error", exc_info=True)
            return False

    # === SECTION: Runtime Mode Announcement ===
    def _announce_runtime_mode(self):
        try:
            mode = "live"
            is_testnet = self._cfg_bool("TESTNET_MODE", "testnet", default=False)
            is_paper = self._cfg_bool("PAPER_MODE", "paper_trade", default=False)
            signal_only = self._cfg_bool("SIGNAL_ONLY", "signal_only_mode", default=False)
            if is_paper:
                mode = "paper"
            if signal_only:
                mode = "signal_only"
            self.logger.info(
                "Runtime mode: %s (testnet=%s, paper=%s, signal_only=%s)",
                mode, is_testnet, is_paper, signal_only,
            )
            if hasattr(self.shared_state, "emit_event"):
                try:
                    res = self.shared_state.emit_event("RuntimeModeChanged", {
                        "mode": mode,
                        "testnet": bool(is_testnet),
                        "paper": bool(is_paper),
                        "signal_only": bool(signal_only),
                    })
                    if asyncio.iscoroutine(res):
                        try:
                            loop = asyncio.get_running_loop()
                            loop.create_task(res)
                        except RuntimeError:
                            pass
                except Exception:
                    pass
        except Exception:
            self.logger.debug("announce_runtime_mode failed", exc_info=True)

    # === SECTION: Execution Probe & Readiness ===
    async def _dry_probe_execution(self):
        """
        Quick, non-fatal affordability probe to verify execution path and symbol filters.
        Prefers SharedState.affordability_snapshot(); falls back to ExecutionManager.can_afford_market_buy().
        """
        try:
            # Ensure filters cache is warmed if the client supports it
            try:
                if self.exchange_client and hasattr(self.exchange_client, "ensure_symbol_filters_ready"):
                    maybe = self.exchange_client.ensure_symbol_filters_ready()
                    if asyncio.iscoroutine(maybe):
                        await maybe
            except Exception:
                pass

            syms = await self._get_accepted_symbols_dict()
            if not syms:
                return
            candidates = [s for s in syms.keys() if s.endswith("USDT")]
            symbol = candidates[0] if candidates else next(iter(syms.keys()), None)
            if not symbol:
                return

            # Determine a planned quote bounded by configured floor; optionally cap to free_now
            # Determine a planned quote bounded by configured floor (hard floor unless explicitly allowed).
            # Optional behavior: cap to free_now if EXEC_PROBE_CAP_TO_FREE=true.
            planned_quote = float(self._cfg("DEFAULT_PLANNED_QUOTE", self._cfg("MIN_ENTRY_QUOTE_USDT", 11.0)))
            configured_floor = float(self._cfg("EXECUTION_MIN_NOTIONAL_QUOTE", self._cfg("MIN_ORDER_USDT", 11.0)))
            allow_subfloor = bool(self._cfg_bool("EXEC_PROBE_ALLOW_SUBFLOOR", default=False))
            cap_to_free = bool(self._cfg_bool("EXEC_PROBE_CAP_TO_FREE", default=False))
            free_now = 0.0
            if self.shared_state:
                try:
                    fu = getattr(self.shared_state, "free_usdt", None)
                    if callable(fu):
                        r = fu()
                        free_now = float(await r) if asyncio.iscoroutine(r) else float(r)
                    else:
                        free_now = float(getattr(self.shared_state, "balances", {}).get("USDT", {}).get("free", 0.0))
                except Exception:
                    free_now = 0.0

            planned_quote = float(planned_quote)

            # Enforce hard floor unless explicitly allowed to probe subfloor
            if not allow_subfloor:
                planned_quote = max(configured_floor, planned_quote)

            # Optionally cap to free_now; never violate the floor unless allow_subfloor=True
            if cap_to_free and free_now and free_now > 0:
                if allow_subfloor:
                    planned_quote = min(planned_quote, free_now)
                else:
                    planned_quote = max(configured_floor, min(planned_quote, free_now))

            # Prefer SharedState.affordability_snapshot if available
            snapshot = None
            if self.shared_state and hasattr(self.shared_state, "affordability_snapshot"):
                try:
                    sn = self.shared_state.affordability_snapshot(symbol=symbol, planned_quote=planned_quote)
                    snapshot = await sn if asyncio.iscoroutine(sn) else sn
                except Exception:
                    snapshot = None

            if isinstance(snapshot, dict) and "AffordabilityProbe" in snapshot:
                ap = snapshot["AffordabilityProbe"]
                ok = bool(ap.get("ok"))
                amt = float(ap.get("amount", 0.0))
                code = str(ap.get("code", "UNKNOWN"))
                self.logger.info("[ExecutionProbe] %s quote≈%s → (%s, %s, '%s')", symbol, planned_quote, ok, amt, code)
                return

            # Fallback to ExecutionManager probe
            if self.execution_manager and hasattr(self.execution_manager, "can_afford_market_buy"):
                res = self.execution_manager.can_afford_market_buy(symbol, planned_quote)
                res = await res if asyncio.iscoroutine(res) else res
                try:
                    ok, amt, code = res
                except Exception:
                    ok, amt, code = (bool(res), 0.0, "UNKNOWN")
                self.logger.info("[ExecutionProbe] %s quote≈%s → (%s, %s, '%s')", symbol, planned_quote, ok, amt, code)
        except Exception:
            self.logger.debug("dry-probe failed (non-fatal)", exc_info=True)


    # === SECTION: Affordability Scout Loop ===
    async def _affordability_scout_loop(self) -> None:
        """
        Background affordability scout:
          - Round-robin over accepted symbols (optionally USDT-quoted only)
          - Uses ExecutionManager.can_afford_market_buy()
          - Emits throttled LIQUIDITY_NEEDED via _maybe_emit_liquidity_needed
        Config (defaults):
          AFFORD_SCOUT_ENABLE=True
          AFFORD_SCOUT_INTERVAL_SEC=15
          AFFORD_SCOUT_JITTER_PCT=0.10
          AFFORD_SCOUT_ONLY_USDT=True
        """
        try:
            if not self._cfg_bool("AFFORD_SCOUT_ENABLE", default=True):
                return
        except Exception:
            pass

        lg = self.logger
        try:
            interval = float(self._cfg("AFFORD_SCOUT_INTERVAL_SEC", 15) or 15.0)
        except Exception:
            interval = 15.0
        try:
            jitter_pct = float(self._cfg("AFFORD_SCOUT_JITTER_PCT", 0.10) or 0.10)
        except Exception:
            jitter_pct = 0.10
        only_usdt = self._cfg_bool("AFFORD_SCOUT_ONLY_USDT", default=True)

        async def _sleep():
            try:
                import random
                j = 1.0 + random.uniform(-abs(jitter_pct), abs(jitter_pct))
                await asyncio.sleep(max(1.0, interval * j))
            except Exception:
                await asyncio.sleep(interval)

        lg.info("[AffordScout] started (interval≈%ss, jitter≈%s%%, only_usdt=%s)", interval, int(jitter_pct*100), only_usdt)

        while True:
            try:
                # Preconditions
                if not (self.execution_manager and hasattr(self.execution_manager, "can_afford_market_buy")):
                    await _sleep(); continue

                syms_dict = await self._get_accepted_symbols_dict()
                if not isinstance(syms_dict, dict) or not syms_dict:
                    await _sleep(); continue
                symbols = list(syms_dict.keys())
                if only_usdt:
                    symbols = [s for s in symbols if str(s).upper().endswith("USDT")]
                if not symbols:
                    await _sleep(); continue

                # Round-robin index
                idx = int(getattr(self, "_scout_index", 0) or 0) % len(symbols)
                symbol = symbols[idx]
                self._scout_index = (idx + 1) % len(symbols)

                # Warm per-symbol filters if supported
                try:
                    if self.exchange_client and hasattr(self.exchange_client, "ensure_symbol_filters_ready"):
                        maybe = self.exchange_client.ensure_symbol_filters_ready(symbol)
                        if asyncio.iscoroutine(maybe):
                            await maybe
                except Exception:
                    pass

                # Plan quote using floors
                try:
                    cfg_floor = float(self._cfg("EXECUTION_MIN_NOTIONAL_QUOTE", self._cfg("MIN_ORDER_USDT", 11.0)))
                except Exception:
                    cfg_floor = 11.0
                try:
                    planned_quote = float(self._cfg("DEFAULT_PLANNED_QUOTE", self._cfg("MIN_ENTRY_QUOTE_USDT", cfg_floor)))
                except Exception:
                    planned_quote = cfg_floor
                planned_quote = max(planned_quote, cfg_floor)

                # Free cash snapshot
                free_now = 0.0
                try:
                    if self.shared_state:
                        fu = getattr(self.shared_state, "free_usdt", None)
                        if callable(fu):
                            r = fu(); free_now = float(await r) if asyncio.iscoroutine(r) else float(r)
                        else:
                            free_now = float(getattr(self.shared_state, "balances", {}).get("USDT", {}).get("free", 0.0))
                except Exception:
                    free_now = 0.0

                # Probe execution path
                res = self.execution_manager.can_afford_market_buy(symbol, planned_quote)
                res = await res if asyncio.iscoroutine(res) else res
                try:
                    ok, amt, code = res
                except Exception:
                    ok, amt, code = (bool(res), 0.0, "UNKNOWN")

                # Normalize absolute requirement
                required_min_quote = None
                if str(code) == "QUOTE_LT_MIN_NOTIONAL":
                    m = float(amt or 0.0)
                    if m <= 0.0: m = cfg_floor
                    required_min_quote = max(m, cfg_floor)
                    await self._adapt_min_notional_floor(symbol, float(required_min_quote), reason="scout")
                elif str(code) in ("INSUFFICIENT_QUOTE", "INSUFFICIENT_FUNDS"):
                    required_min_quote = max(planned_quote + float(amt or 0.0), cfg_floor)

                # Emit need if gap exists
                if not ok and required_min_quote is not None:
                    gap = max(0.0, float(required_min_quote) - float(free_now or 0.0))
                    if gap > 0.0:
                        await self._maybe_emit_liquidity_needed(
                            symbol=str(symbol),
                            required_usdt=float(f"{required_min_quote:.6f}"),
                            free_usdt=float(f"{free_now:.6f}"),
                            gap_usdt=float(f"{gap:.6f}"),
                            reason=str(code),
                        )
                else:
                    st = self._liq_need_state.get(str(symbol).upper())
                    if st:
                        st["consec"] = 0
                        st["last_gap"] = 0.0

            except asyncio.CancelledError:
                lg.info("[AffordScout] cancelled.")
                raise
            except Exception:
                self.logger.debug("affordability scout iteration failed", exc_info=True)
            finally:
                await _sleep()

    # === SECTION: Start Affordability Scout ===
    def _start_affordability_scout(self) -> None:
        """
        Idempotently start the background affordability scout loop (if enabled).
        """
        # Idempotence guard: avoid duplicate background scouts
        if self._scout_task and not self._scout_task.done():
            return
        try:
            if not self._cfg_bool("AFFORD_SCOUT_ENABLE", default=True):
                return
        except Exception:
            pass
        try:
            loop = asyncio.get_running_loop()
            self._scout_task = loop.create_task(self._affordability_scout_loop(), name="affordability_scout")
        except RuntimeError:
            self.logger.debug("failed to start affordability scout: no running loop")
        except Exception:
            self.logger.debug("failed to start affordability scout", exc_info=True)

    # === SECTION: Stop Affordability Scout ===
    async def _stop_affordability_scout(self) -> None:
        """
        Stop the background affordability scout if running.
        """
        t = getattr(self, "_scout_task", None)
        if t:
            try:
                t.cancel()
                with contextlib.suppress(Exception):
                    await t
            finally:
                self._scout_task = None

    # === SECTION: Universe Rotation Engine Background Loop ===
    async def _uure_loop(self) -> None:
        """
        Background Universe Rotation Engine loop:
          - Runs immediately once at startup (critical for universe population)
          - Then periodically calls UURE every UURE_INTERVAL_SEC seconds (default: 300, i.e., 5 minutes)
          - Enterprise best practice: immediate + periodic (not sleep-first)
          - Graceful error handling with debug logging
        Config (defaults):
          UURE_ENABLE=True
          UURE_INTERVAL_SEC=300 (5 minutes)
        """
        try:
            if not self._cfg_bool("UURE_ENABLE", default=True):
                return
        except Exception:
            pass

        lg = self.logger
        try:
            interval = float(self._cfg("UURE_INTERVAL_SEC", 300) or 300.0)
        except Exception:
            interval = 300.0

        lg.info("[UURE] background loop started (immediate + periodic every %ss)", interval)

        async def _execute_rotation():
            """Execute universe rotation with comprehensive error handling."""
            try:
                # Preconditions
                if not self.universe_rotation_engine:
                    lg.debug("[UURE] engine not ready, skipping")
                    return

                if not hasattr(self.universe_rotation_engine, "compute_and_apply_universe"):
                    lg.debug("[UURE] engine missing compute_and_apply_universe method, skipping")
                    return

                # Call UURE to compute and apply universe
                lg.debug("[UURE] invoking compute_and_apply_universe()")
                result = self.universe_rotation_engine.compute_and_apply_universe()
                if asyncio.iscoroutine(result):
                    result = await result

                # Log result
                if isinstance(result, dict):
                    rotation = result.get("rotation", {}) or {}
                    lg.info("[UURE] rotation result: added=%d, removed=%d, kept=%d",
                            len(rotation.get("added", [])),
                            len(rotation.get("removed", [])),
                            len(rotation.get("kept", [])))
                else:
                    lg.debug("[UURE] rotation result: %s", result)

                # Emit summary
                try:
                    rotation = result.get("rotation", {}) if isinstance(result, dict) else {}
                    await self._emit_summary(
                        "UNIVERSE_ROTATION",
                        added=len(rotation.get("added", [])),
                        removed=len(rotation.get("removed", [])),
                        kept=len(rotation.get("kept", [])),
                    )
                except Exception:
                    lg.debug("[UURE] emit_summary failed", exc_info=True)

            except asyncio.CancelledError:
                raise
            except Exception:
                lg.debug("[UURE] rotation execution failed", exc_info=True)

        # Run immediately once at startup (critical for universe population)
        lg.debug("[UURE] running immediate execution at startup")
        try:
            await _execute_rotation()
        except asyncio.CancelledError:
            lg.info("[UURE] loop cancelled during immediate execution.")
            raise
        except Exception:
            lg.debug("[UURE] immediate execution failed", exc_info=True)

        # Then periodic loop every interval
        while True:
            try:
                # Wait before each periodic iteration
                await asyncio.sleep(interval)
                # Execute periodic rotation
                await _execute_rotation()

            except asyncio.CancelledError:
                lg.info("[UURE] loop cancelled.")
                raise
            except Exception:
                lg.debug("[UURE] loop iteration failed", exc_info=True)

    # === SECTION: Start UURE Background Loop ===
    def _start_uure_loop(self) -> None:
        """
        Idempotently start the background UURE loop (if enabled).
        """
        # Idempotence guard: avoid duplicate background loops
        if self._uure_task and not self._uure_task.done():
            return
        try:
            if not self._cfg_bool("UURE_ENABLE", default=True):
                return
        except Exception:
            pass
        try:
            loop = asyncio.get_running_loop()
            self._uure_task = loop.create_task(self._uure_loop(), name="uure_rotation")
        except RuntimeError:
            self.logger.debug("failed to start UURE loop: no running loop")
        except Exception:
            self.logger.debug("failed to start UURE loop", exc_info=True)

    # === SECTION: Stop UURE Background Loop ===
    async def _stop_uure_loop(self) -> None:
        """
        Stop the background UURE loop if running.
        """
        t = getattr(self, "_uure_task", None)
        if t:
            try:
                t.cancel()
                with contextlib.suppress(Exception):
                    await t
            finally:
                self._uure_task = None

    async def _is_market_data_ready(self) -> bool:
        try:
            if hasattr(self.shared_state, "is_market_data_ready"):
                r = self.shared_state.is_market_data_ready()
                return (await r) if asyncio.iscoroutine(r) else bool(r)
        except Exception:
            pass
        return True

    async def _has_nonempty_universe(self) -> bool:
        try:
            d = await self._get_accepted_symbols_dict()
            return bool(d)
        except Exception:
            return False

    async def _is_execution_ready(self) -> bool:
        return bool(self.execution_manager and callable(getattr(self.execution_manager, "execute_trade", None)))

    # === SECTION: Ops Plane Snapshot (Startup Sanity) ===
    async def _ops_plane_snapshot(self) -> dict:
        """
        Returns: {ready: bool, issues: List[str], detail: dict}
        Issues may include: MarketDataNotReady, SymbolsUniverseEmpty, ExecutionManagerNotReady,
        BalancesNotReady, NAVNotReady, ExchangeClientNotReady, MinNotionalTooHighForConfiguredQuote,
        FiltersCoverageLow, FreeQuoteBelowFloor
        """
        issues = []
        detail: Dict[str, Any] = {}
        aff_ok_flag = False

        # --- A) ExchangeClient readiness ---
        if self.exchange_client is None:
            issues.append("ExchangeClientNotReady")

        # --- B) Market data readiness ---
        try:
            md_ready = await self._is_market_data_ready()
            if not md_ready:
                issues.append("MarketDataNotReady")
        except Exception:
            issues.append("MarketDataNotReady")

        # --- C) Symbols universe ---
        try:
            has_universe = await self._has_nonempty_universe()
            if not has_universe:
                issues.append("SymbolsUniverseEmpty")
        except Exception:
            issues.append("SymbolsUniverseEmpty")

        # --- D) Execution manager readiness ---
        try:
            exec_ready = await self._is_execution_ready()
            if not exec_ready:
                issues.append("ExecutionManagerNotReady")
        except Exception:
            issues.append("ExecutionManagerNotReady")

        # --- E) Balances / NAV readiness ---
        free_usdt_now = 0.0
        try:
            if self.shared_state:
                fu = getattr(self.shared_state, "free_usdt", None)
                if callable(fu):
                    _v = fu()
                    free_usdt_now = float(await _v) if asyncio.iscoroutine(_v) else float(_v)
                else:
                    free_usdt_now = float(getattr(self.shared_state, "balances", {}).get("USDT", {}).get("free", 0.0))
        except Exception:
            free_usdt_now = 0.0

        try:
            balances_ready = bool(getattr(self.shared_state, "balances_ready", False)) or free_usdt_now > 0
            if not balances_ready:
                issues.append("BalancesNotReady")
        except Exception:
            issues.append("BalancesNotReady")

        try:
            nav = 0.0
            if self.shared_state and hasattr(self.shared_state, "get_nav_quote"):
                nav_v = self.shared_state.get_nav_quote()
                nav = float(await nav_v) if asyncio.iscoroutine(nav_v) else float(nav_v or 0.0)
            if nav <= 0 and free_usdt_now <= 0:
                if "BalancesNotReady" not in issues:
                    issues.append("NAVNotReady")
        except Exception:
            pass

        # Capital gate epsilon (tolerance to avoid penny-level stalls)
        try:
            eps = float(self._cfg("CAPITAL_GATE_EPS_USDT", 0.10) or 0.10)
        except Exception:
            eps = 0.10

        # --- E2) Startup sanity: filters coverage & free-quote floor ---
        try:
            coverage_pct, free_floor = await self._compute_startup_sanity_requirements()
            detail["FiltersCoveragePct"] = float(f"{coverage_pct:.1f}")
            detail["FreeQuoteFloor"] = float(f"{free_floor:.2f}")
            min_coverage = float(self._cfg("STARTUP_MIN_FILTERS_COVERAGE_PCT", 50.0))
            if coverage_pct < min_coverage:
                issues.append("FiltersCoverageLow")
            if free_floor > 0 and (free_usdt_now + eps) < free_floor:
                issues.append("FreeQuoteBelowFloor")
        except Exception:
            self.logger.debug("startup sanity requirements check failed", exc_info=True)

        # --- F) Affordability probe (Path A: SharedState snapshot; Path B: ExecutionManager fallback) ---
        try:
            # Determine a candidate symbol (prefer USDT-quoted) and a safe planned quote
            syms = await self._get_accepted_symbols_dict()
            symbol = None
            if isinstance(syms, dict) and syms:
                usdt_syms = [s for s in syms.keys() if str(s).upper().endswith("USDT")]
                symbol = usdt_syms[0] if usdt_syms else next(iter(syms.keys()))

            # Planned quote: align with MetaController defaults and respect configured floors
            try:
                cfg_floor = float(self._cfg("EXECUTION_MIN_NOTIONAL_QUOTE", self._cfg("MIN_ORDER_USDT", 25.0)))
            except Exception:
                cfg_floor = 25.0
            try:
                base_quote = float(self._cfg("DEFAULT_PLANNED_QUOTE", self._cfg("MIN_ENTRY_QUOTE_USDT", cfg_floor)))
            except Exception:
                base_quote = cfg_floor
            planned_quote = max(cfg_floor, base_quote)
            detail["PlannedQuoteUsed"] = float(planned_quote)
            detail["MinNotionalFloor"] = float(cfg_floor)

            # Path A: SharedState.affordability_snapshot()
            ap: dict = {}
            used_snapshot = False
            if symbol and self.shared_state and hasattr(self.shared_state, "affordability_snapshot"):
                try:
                    snap = self.shared_state.affordability_snapshot(symbol=symbol, planned_quote=planned_quote)
                    snap = await snap if asyncio.iscoroutine(snap) else snap
                    if isinstance(snap, dict) and "AffordabilityProbe" in snap:
                        ap = dict(snap.get("AffordabilityProbe") or {})
                        used_snapshot = True
                except Exception:
                    ap = {}

            # Path B: ExecutionManager.can_afford_market_buy() fallback
            if not used_snapshot and symbol and self.execution_manager and hasattr(self.execution_manager, "can_afford_market_buy"):
                try:
                    res = self.execution_manager.can_afford_market_buy(symbol, planned_quote)
                    res = await res if asyncio.iscoroutine(res) else res
                    try:
                        ok, amount, code = res
                    except Exception:
                        ok, amount, code = (bool(res), 0.0, "UNKNOWN")
                    ap = {
                        "symbol": str(symbol),
                        "ok": bool(ok),
                        "amount": float(amount or 0.0),
                        "code": str(code or "UNKNOWN"),
                        "planned_quote": float(planned_quote),
                    }
                    # Normalize required_min_quote where applicable and adapt floors
                    if ap["code"] == "QUOTE_LT_MIN_NOTIONAL":
                        req = max(float(ap["amount"] or 0.0), cfg_floor)
                        ap["required_min_quote"] = float(req)
                        try:
                            await self._adapt_min_notional_floor(symbol, float(req), reason="ops_probe")
                        except Exception:
                            pass
                    elif ap["code"] in ("INSUFFICIENT_QUOTE", "INSUFFICIENT_FUNDS"):
                        req = max(float(planned_quote) + float(ap["amount"] or 0.0), cfg_floor)
                        ap["required_min_quote"] = float(req)
                except Exception:
                    ap = {}

            if ap:
                detail["AffordabilityProbe"] = ap
                aff_ok_flag = bool(ap.get("ok", False))
            else:
                detail["AffordabilityProbe"] = {"error": "probe_failed"}
        except Exception:
            detail["AffordabilityProbe"] = {"error": "probe_failed"}

        # Attach recent per-symbol liquidity gaps to detail
        try:
            detail["Liquidity"] = self._liquidity_snapshot()
        except Exception:
            pass

        # Attach dust origin telemetry for Tier-1 monitoring
        try:
            detail["Dust"] = self._dust_metrics_snapshot()
        except Exception:
            detail["Dust"] = {"error": "dust_snapshot_failed"}

        # Strict Rule 2: ZERO_QTY_AFTER_ROUNDING -> Readiness = FALSE
        probe = detail.get("AffordabilityProbe", {})
        if probe.get("code") == "ZERO_QTY_AFTER_ROUNDING":
            issues.append("ZeroQuantityExecution")
            detail["affordability_failure"] = "Amount rounded to zero (Rule 2)"
        elif probe.get("ok") and float(probe.get("amount", 0.0)) > 0:
            # This would be an internal inconsistency in EM (ok=True but gap > 0)
            self.logger.warning("EM internal inconsistency: ok=True but gap=%s", probe.get("amount"))

        # De-duplicate any accumulated issues
        issues = list(dict.fromkeys(issues))

        return {
            "ready": (len(issues) == 0),
            "issues": issues,
            "detail": detail,
        }
            
    async def _compute_startup_sanity_requirements(self) -> Tuple[float, float]:
        """
        Returns (filters_coverage_pct, min_free_quote_floor_usdt).
        Coverage = % of accepted symbols that have valid filters (and optionally recent price).
        Floor = STARTUP.min_free_quote_factor × max(Tier-A minNotional).
        Tiering heuristic: treat accepted symbols as Tier-A by default (defensive).
        """
        try:
            syms_dict = await self._get_accepted_symbols_dict()
            symbols = list(syms_dict.keys())
            # If exchange client can ensure filters are ready, do so first
            if self.exchange_client and hasattr(self.exchange_client, "ensure_symbol_filters_ready"):
                maybe = self.exchange_client.ensure_symbol_filters_ready()
                if asyncio.iscoroutine(maybe):
                    await maybe
            # If no symbols, return full coverage (100%) and 0.0 as floor
            if not symbols:
                return 100.0, 0.0
            have = 0
            max_min_notional = 0.0
            for s in symbols:
                f_min = None
                try:
                    if self.exchange_client and hasattr(self.exchange_client, "get_symbol_filters"):
                        filt = self.exchange_client.get_symbol_filters(s)
                        filt = await filt if asyncio.iscoroutine(filt) else filt
                        if isinstance(filt, dict):
                            # Support both normalized and raw Binance filter maps
                            if "min_notional" in filt:
                                f_min = float(filt.get("min_notional", 0.0))
                            elif "MIN_NOTIONAL" in filt and isinstance(filt["MIN_NOTIONAL"], dict):
                                f_min = float(filt["MIN_NOTIONAL"].get("minNotional", 0.0))
                            else:
                                f_min = float(filt.get("minNotional", 0.0))
                        else:
                            # allow tuple/obj styles: (min_notional, step, tick) or attr-like
                            try:
                                f_min = float(getattr(filt, "minNotional", 0.0))
                            except Exception:
                                f_min = None
                except Exception:
                    f_min = None
                if f_min and f_min > 0:
                    have += 1
                    if f_min > max_min_notional:
                        max_min_notional = f_min
            coverage_pct = (have / max(1, len(symbols))) * 100.0
            floor_factor = float(self._cfg("STARTUP.min_free_quote_factor", 1.2))
            free_floor = float(max_min_notional) * floor_factor if max_min_notional else 0.0
            return coverage_pct, free_floor
        except Exception:
            return 0.0, 0.0

    async def _adapt_min_notional_floor(self, symbol: str, observed_needed_quote: float, reason: str = "probe"):
        """
        Raise the unified min-notional floor to at least the absolute min quote required.
        'observed_needed_quote' is treated as an ABSOLUTE requirement (not a delta).
        """
        try:
            # Point 2: Freeze floor once intent exists (Mandatory Fix)
            if self.shared_state and hasattr(self.shared_state, "get_pending_intent"):
                intent = self.shared_state.get_pending_intent(symbol, "BUY")
                if intent and intent.state == "ACCUMULATING":
                    # Already in accumulation; do NOT raise the hurdle further.
                    return
            sym_min = None
            if self.exchange_client and hasattr(self.exchange_client, "get_min_notional"):
                r = self.exchange_client.get_min_notional(symbol)
                sym_min = await r if asyncio.iscoroutine(r) else r
            sym_min = float(sym_min or 0.0)

            absolute_needed = float(observed_needed_quote or 0.0)
            cushion = float(getattr(self.config, "EXECUTION_CUSHION_MULTIPLIER", 1.02))
            floor_now = float(self._cfg("MIN_ORDER_USDT", 11.0))

            target_floor = max(floor_now, sym_min * cushion, absolute_needed)

            # (Removed: capping target_floor to free cash)

            # Stabilize rounding for USDT-like quotes
            target_floor = float(f"{target_floor:.2f}")

            if target_floor > floor_now:
                # Update the canonical floor
                try:
                    setattr(self.config, "MIN_ORDER_USDT", target_floor)
                except Exception:
                    pass

                # Keep related knobs in sync
                for k in ("EXEC_PROBE_QUOTE", "EXECUTION_MIN_NOTIONAL_QUOTE", "QUOTE_MIN_NOTIONAL"):
                    try:
                        setattr(self.config, k, target_floor)
                    except Exception:
                        pass
                for name in ("DUST_MIN_QUOTE_USDT", "LIQ_ORCH_MIN_USDT_FLOOR"):
                    try:
                        if float(getattr(self.config, name, 0.0) or 0.0) < target_floor:
                            setattr(self.config, name, target_floor)
                    except Exception:
                        pass

                if hasattr(self.shared_state, "emit_event"):
                    evt = self.shared_state.emit_event("RuntimeMinNotionalRaised", {
                        "symbol": symbol,
                        "old_floor": floor_now,
                        "new_floor": target_floor,
                        "reason": reason,
                        "sym_min": sym_min,
                        "cushion": cushion,
                    })
                    if asyncio.iscoroutine(evt):
                        asyncio.create_task(evt)

                self.logger.info(
                    "Raised unified min-notional floor from %.2f -> %.2f (sym=%s, reason=%s)",
                    floor_now, target_floor, symbol, reason,
                )
        except Exception:
            self.logger.debug("adaptive min-notional bump failed", exc_info=True)

    # ----------------------------- Build components -----------------------------
    def _ensure_components_built(self):
        """
        Best-effort construction for standard components if not injected.
        Uses permissive constructors: tries (config, logger, app, shared_state, exchange_client) subsets.
        """
        def _get_cls(mod, name: str):
            return getattr(mod, name, None) if mod else None

        def _missing_deps(**deps: Any) -> List[str]:
            return [name for name, value in deps.items() if value is None]

        # ExchangeClient (before others that depend on it)
        # Auto-select API keys based on BINANCE_TESTNET flag
        def _is_testnet_env() -> bool:
            for key in ("BINANCE_TESTNET", "TESTNET_MODE"):
                v = os.environ.get(key, "").strip().lower()
                if v in ("1", "true", "yes", "on"):
                    return True
            return False

        if _is_testnet_env():
            BINANCE_API_KEY = os.environ.get('BINANCE_TESTNET_API_KEY') or os.environ.get('BINANCE_API_KEY')
            BINANCE_API_SECRET = os.environ.get('BINANCE_TESTNET_API_SECRET') or os.environ.get('BINANCE_API_SECRET')
        else:
            BINANCE_API_KEY = os.environ.get('BINANCE_API_KEY')
            BINANCE_API_SECRET = os.environ.get('BINANCE_API_SECRET')

        if self.exchange_client is None:
            ExchangeClient = _get_cls(_exchange_mod, "ExchangeClient")
            self.exchange_client = _try_construct(
                ExchangeClient,
                config=self.config,
                logger=self.logger,
                app=self,
                shared_state=self.shared_state,
                api_key=BINANCE_API_KEY,
                api_secret=BINANCE_API_SECRET
            )
            if self.exchange_client is None and ExchangeClient is not None:
                try:
                    self.logger.warning(
                        "[Bootstrap] ExchangeClient _try_construct returned None; attempting direct construction for diagnostics..."
                    )
                    self.exchange_client = ExchangeClient(
                        config=self.config,
                        logger=self.logger,
                        app=self,
                        shared_state=self.shared_state,
                        api_key=BINANCE_API_KEY,
                        api_secret=BINANCE_API_SECRET,
                    )
                except Exception as e_direct:
                    self.logger.error(
                        "[Bootstrap] ExchangeClient direct construction failed: %s: %s",
                        type(e_direct).__name__,
                        e_direct,
                        exc_info=True,
                    )
        # Fallback: use module-level factory if direct construction failed
        if self.exchange_client is None and _exchange_mod is not None:
            _factory = getattr(_exchange_mod, "get_global_exchange_client", None)
            if callable(_factory):
                try:
                    self.exchange_client = _factory(
                        config=self.config,
                        logger=self.logger,
                        app=self,
                        shared_state=self.shared_state,
                    )
                except Exception as e_factory:
                    self.logger.error(
                        "[Bootstrap] ExchangeClient factory construction failed: %s: %s",
                        type(e_factory).__name__,
                        e_factory,
                        exc_info=True,
                    )

        if self.exchange_client is None:
            self.logger.error(
                "[Bootstrap] ExchangeClient unavailable; downstream components requiring signed/order-capable exchange access may be skipped."
            )

        # SharedState
        if self.shared_state is None:
            SharedState = _get_cls(_shared_state_mod, "SharedState") or _get_cls(_shared_state_mod, "State")
            
            # CRITICAL: Try to construct SharedState with detailed error logging
            try:
                self.shared_state = _try_construct(SharedState, config=self.config, logger=self.logger, app=self, exchange_client=self.exchange_client)
            except Exception as e:
                error_msg = f"[AppContext:CRITICAL] SharedState construction raised exception: {type(e).__name__}: {e}"
                self.logger.error(error_msg, exc_info=True)
                raise RuntimeError(error_msg) from e
            
            # If _try_construct returns None (silent failure), attempt direct construction to get real error
            if self.shared_state is None:
                try:
                    self.logger.warning("[AppContext] _try_construct returned None, attempting direct construction for error details...")
                    # SharedState only accepts: config, database_manager, exchange_client
                    self.shared_state = SharedState(config=self.config, exchange_client=self.exchange_client)
                except Exception as e:
                    error_msg = f"[AppContext:CRITICAL] SharedState direct construction failed: {type(e).__name__}: {e}"
                    self.logger.error(error_msg, exc_info=True)
                    raise RuntimeError(error_msg) from e
                
                if self.shared_state is None:
                    error_msg = "[AppContext:CRITICAL] SharedState construction failed (returned None). This is a fatal error - cannot proceed without SharedState."
                    self.logger.error(error_msg)
                    raise RuntimeError(error_msg)
            
            # Bind CSL immediately after construction
            try:
                if self.shared_state and _csl_mod:
                    _csl_mod.bind_shared_state(self.shared_state)
            except Exception:
                self.logger.debug("ComponentStatusLogger binding failed in _ensure_components_built", exc_info=True)

        # ExchangeTruthAuditor (governance-only reconciliation layer)
        # 🔧 MODE-BASED DESIGN: Determine TruthAuditor mode based on trading_mode
        # DISABLED: shadow mode (no exchange truth to reconcile)
        # CONTINUOUS: live mode (startup + passive monitoring)
        if self.exchange_truth_auditor is None:
            trading_mode = str(getattr(self.config, "trading_mode", "live") or "live").lower()
            
            # Infer TruthAuditor mode from trading_mode
            if trading_mode == "shadow":
                auditor_mode = "disabled"
            else:
                auditor_mode = "continuous"  # Default for live/paper
            
            ExchangeTruthAuditor = _get_cls(_truth_auditor_mod, "ExchangeTruthAuditor")
            if ExchangeTruthAuditor:
                self.exchange_truth_auditor = _try_construct(
                    ExchangeTruthAuditor,
                    config=self.config,
                    logger=self.logger,
                    app=self,
                    shared_state=self.shared_state,
                    exchange_client=self.exchange_client,
                    mode=auditor_mode,
                )
                if self.exchange_truth_auditor is None:
                    try:
                        self.logger.warning(
                            "[Bootstrap] ExchangeTruthAuditor _try_construct returned None; attempting direct construction..."
                        )
                        self.exchange_truth_auditor = ExchangeTruthAuditor(
                            config=self.config,
                            logger=self.logger,
                            shared_state=self.shared_state,
                            exchange_client=self.exchange_client,
                            app=self,
                            mode=auditor_mode,
                        )
                    except Exception as e_direct:
                            self.logger.error(
                                "[Bootstrap] ExchangeTruthAuditor direct construction failed: %s: %s",
                                type(e_direct).__name__,
                                e_direct,
                                exc_info=True,
                            )

        # Prepare candidate kwargs for all subsequent components
        ck = {
            "config": self.config,
            "logger": self.logger,
            "app": self,
            "shared_state": self.shared_state,
            "exchange_client": self.exchange_client,
            "market_data_feed": self.market_data_feed,
            "market_data": self.market_data_feed,  # some expect market_data
            "execution_manager": self.execution_manager,
            "symbol_manager": self.symbol_manager,
            "risk_manager": self.risk_manager,
            "tp_sl_engine": self.tp_sl_engine,
            "recovery_engine": self.recovery_engine,
            "cash_router": self.cash_router,
            "database_manager": self.database_manager,
            "pnl_calculator": self.pnl_calculator,
            "performance_evaluator": self.performance_evaluator,
        }

        # (Optional) Cash Router
        if self.cash_router is None:
            CashRouter = _get_cls(_cash_router_mod, "CashRouter")
            if CashRouter:
                self.cash_router = _try_construct(
                    CashRouter,
                    config=self.config,
                    logger=self.logger,
                    app=self,
                    shared_state=self.shared_state,
                    execution_manager=self.execution_manager,
                )

        # SymbolManager
        if self.symbol_manager is None:
            SymbolManager = _get_cls(_symbol_mgr_mod, "SymbolManager")
            self.symbol_manager = _try_construct(SymbolManager, config=self.config, logger=self.logger, app=self, shared_state=self.shared_state, exchange_client=self.exchange_client)

        # CapitalSymbolGovernor
        if self.capital_symbol_governor is None:
            CapitalSymbolGovernor = _get_cls(_governor_mod, "CapitalSymbolGovernor")
            self.capital_symbol_governor = _try_construct(CapitalSymbolGovernor, config=self.config, logger=self.logger, shared_state=self.shared_state)

        # UniverseRotationEngine
        if self.universe_rotation_engine is None:
            UniverseRotationEngine = _get_cls(_uure_mod, "UniverseRotationEngine")
            self.universe_rotation_engine = _try_construct(
                UniverseRotationEngine,
                shared_state=self.shared_state,
                capital_governor=self.capital_symbol_governor,
                config=self.config,
                execution_manager=self.execution_manager,
                meta_controller=self.meta_controller,
                logger=self.logger,
            )

        # MarketDataFeed
        if self.market_data_feed is None:
            MarketDataFeed = _get_cls(_market_data_mod, "MarketDataFeed")
            self.market_data_feed = _try_construct(MarketDataFeed, config=self.config, logger=self.logger, app=self, shared_state=self.shared_state, exchange_client=self.exchange_client)

        # ExecutionManager
        if self.execution_manager is None:
            ExecutionManager = _get_cls(_execution_mod, "ExecutionManager")
            self.execution_manager = _try_construct(ExecutionManager, config=self.config, logger=self.logger, app=self, shared_state=self.shared_state, exchange_client=self.exchange_client)
            if self.execution_manager is None and ExecutionManager is not None:
                missing = _missing_deps(shared_state=self.shared_state, exchange_client=self.exchange_client)
                if missing:
                    self.logger.error(
                        "[Bootstrap] ExecutionManager skipped: missing dependencies (%s).",
                        ", ".join(missing),
                    )
                else:
                    try:
                        self.logger.warning(
                            "[Bootstrap] ExecutionManager _try_construct returned None; attempting direct construction for diagnostics..."
                        )
                        self.execution_manager = ExecutionManager(
                            config=self.config,
                            shared_state=self.shared_state,
                            exchange_client=self.exchange_client,
                        )
                    except Exception as e_direct:
                        self.logger.error(
                            "[Bootstrap] ExecutionManager direct construction failed: %s: %s",
                            type(e_direct).__name__,
                            e_direct,
                            exc_info=True,
                        )

        # Inject TradeJournal + session_id into ExecutionManager
        if self.execution_manager:
            self.execution_manager.trade_journal = self.trade_journal
            self.execution_manager.session_id = self.session_id

        # Wire ExecutionManager into ExchangeTruthAuditor for full SELL lifecycle finalization
        if self.exchange_truth_auditor and self.execution_manager:
            if hasattr(self.exchange_truth_auditor, "set_execution_manager"):
                self.exchange_truth_auditor.set_execution_manager(self.execution_manager)

        # AdaptiveCapitalEngine - after execution manager is ready
        if self.adaptive_capital_engine is None:
            AdaptiveCapitalEngine = _get_cls(_adaptive_capital_mod, "AdaptiveCapitalEngine")
            if AdaptiveCapitalEngine:
                # AdaptiveCapitalEngine only needs config and logger
                self.adaptive_capital_engine = AdaptiveCapitalEngine(config=self.config, logger=self.logger)
                self.logger.info("[Bootstrap] AdaptiveCapitalEngine constructed successfully")
            else:
                self.logger.warning("[Bootstrap] AdaptiveCapitalEngine class not found in module")

        # MetaController - Direct import due to file/package naming conflict
        if self.meta_controller is None:
            missing = _missing_deps(
                shared_state=self.shared_state,
                exchange_client=self.exchange_client,
                execution_manager=self.execution_manager,
            )
            if missing:
                self.logger.error(
                    "[Bootstrap] MetaController skipped: missing dependencies (%s).",
                    ", ".join(missing),
                )
            else:
                try:
                    from core.meta_controller import MetaController

                    # NOTE: MetaController creates its own logger internally, don't pass logger parameter
                    self.meta_controller = _try_construct(
                        MetaController,
                        config=self.config,
                        app=self,
                        shared_state=self.shared_state,
                        execution_manager=self.execution_manager,
                        exchange_client=self.exchange_client,
                        cash_router=self.cash_router,
                        adaptive_capital_engine=self.adaptive_capital_engine,
                    )
                    # If _try_construct returns None, attempt a direct construction for visibility
                    if self.meta_controller is None:
                        try:
                            self.logger.warning("[Bootstrap] MetaController _try_construct returned None; attempting direct construction for diagnostics...")
                            self.meta_controller = MetaController(
                                shared_state=self.shared_state,
                                exchange_client=self.exchange_client,
                                execution_manager=self.execution_manager,
                                config=self.config,
                                adaptive_capital_engine=self.adaptive_capital_engine,
                            )
                        except Exception as e_direct:
                            self.logger.error("[Bootstrap] MetaController direct construction failed: %s", e_direct, exc_info=True)
                            self.meta_controller = None
                except Exception as e:
                    self.logger.error("[Bootstrap] MetaController construction failed: %s", e, exc_info=True)
                    self.meta_controller = None

        # Ensure UURE has late-bound runtime deps needed for eviction/liquidation.
        try:
            if self.universe_rotation_engine:
                if hasattr(self.universe_rotation_engine, "wire_runtime_dependencies"):
                    self.universe_rotation_engine.wire_runtime_dependencies(
                        capital_governor=self.capital_symbol_governor,
                        execution_manager=self.execution_manager,
                        meta_controller=self.meta_controller,
                    )
                else:
                    if self.capital_symbol_governor is not None:
                        setattr(self.universe_rotation_engine, "governor", self.capital_symbol_governor)
                    if self.execution_manager is not None:
                        setattr(self.universe_rotation_engine, "exec", self.execution_manager)
                    if self.meta_controller is not None:
                        setattr(self.universe_rotation_engine, "mc", self.meta_controller)
        except Exception:
            self.logger.debug("UURE runtime dependency wiring failed", exc_info=True)

        # Strategy/Agents/Risk
        if self.strategy_manager is None:
            StrategyManager = _get_cls(_strategy_mgr_mod, "StrategyManager")
            self.strategy_manager = _try_construct(StrategyManager, config=self.config, logger=self.logger, app=self, shared_state=self.shared_state, exchange_client=self.exchange_client)

        if self.agent_manager is None:
            AgentManager = _get_cls(_agent_mgr_mod, "AgentManager")
            self.agent_manager = _try_construct(AgentManager, config=self.config, logger=self.logger, app=self, shared_state=self.shared_state, execution_manager=self.execution_manager, exchange_client=self.exchange_client, market_data=self.market_data_feed, symbol_manager=self.symbol_manager, meta_controller=self.meta_controller)
        
        # 🔥 CRITICAL FIX: Ensure MetaController is injected into AgentManager
        # This was missing, causing signals to never reach the decision pipeline
        if self.agent_manager and self.meta_controller:
            self.agent_manager.meta_controller = self.meta_controller
            self.logger.info("[Bootstrap] ✅ Injected MetaController into AgentManager - signal pipeline connected!")

        # 🔥 CRITICAL FIX: Register discovery agents with AgentManager
        # This was missing, preventing discovery agents (IPOChaser, WalletScanner, etc.) from running
        if self.agent_manager:
            try:
                # Try importing from agent_registry first
                try:
                    from core.agent_registry import register_all_discovery_agents as _reg_discovery, register_all_strategy_agents as _reg_strategy
                    _reg_discovery(self.agent_manager, self)
                    _reg_strategy(self.agent_manager, self)
                except ImportError:
                    # Fall back to local implementations if import fails
                    self.logger.warning("[Bootstrap] ⚠️ core.agent_registry import failed, using local agent registration functions")
                    register_all_discovery_agents(self.agent_manager, self)
                    register_all_strategy_agents(self.agent_manager, self)
                
                self.logger.info("[Bootstrap] ✅ Registered discovery agents with AgentManager (IPOChaser, WalletScanner, SymbolScreener, LiquidationAgent)")
                self.logger.info("[Bootstrap] ✅ Registered strategy agents with AgentManager (MLForecaster, DipSniper, TrendHunter, SwingTradeHunter, NewsReactor)")
            except Exception as e:
                self.logger.error("[Bootstrap] ❌ Failed to register agents: %s", e, exc_info=True)

        if self.risk_manager is None:
            RiskManager = _get_cls(_risk_mod, "RiskManager")
            self.risk_manager = _try_construct(RiskManager, config=self.config, logger=self.logger, app=self, shared_state=self.shared_state, exchange_client=self.exchange_client)

        # Portfolio accounting
        if self.pnl_calculator is None and _pnl_mod:
            PnLCalculator = _get_cls(_pnl_mod, "PnLCalculator")
            self.pnl_calculator = _try_construct(
                PnLCalculator,
                config=self.config,
                logger=self.logger,
                app=self,
                shared_state=self.shared_state,
                exchange_client=self.exchange_client,
            )
            if self.pnl_calculator is None and PnLCalculator is not None:
                try:
                    self.logger.warning(
                        "[Bootstrap] PnLCalculator _try_construct returned None; attempting direct construction for diagnostics..."
                    )
                    self.pnl_calculator = PnLCalculator(
                        shared_state=self.shared_state,
                        config=self.config,
                        exchange_client=self.exchange_client,
                    )
                except Exception as e_direct:
                    self.logger.error(
                        "[Bootstrap] PnLCalculator direct construction failed: %s: %s",
                        type(e_direct).__name__,
                        e_direct,
                        exc_info=True,
                    )

        # Protective services
        if self.heartbeat is None:
            Heartbeat = _get_cls(_heartbeat_mod, "Heartbeat")
            self.heartbeat = _try_construct(Heartbeat, config=self.config, logger=self.logger, app=self, shared_state=self.shared_state)
        if self.watchdog is None:
            Watchdog = _get_cls(_watchdog_mod, "Watchdog")
            self.watchdog = _try_construct(Watchdog, config=self.config, logger=self.logger, app=self, shared_state=self.shared_state)
        if self.alert_system is None:
            AlertSystem = _get_cls(_alert_mod, "AlertSystem")
            self.alert_system = _try_construct(AlertSystem, config=self.config, logger=self.logger, app=self, shared_state=self.shared_state)
        if self.tp_sl_engine is None:
            TPSLEngine = _get_cls(_tpsl_mod, "TPSLEngine")
            missing = _missing_deps(shared_state=self.shared_state, execution_manager=self.execution_manager)
            if missing:
                self.logger.error(
                    "[Bootstrap] TPSLEngine skipped: missing dependencies (%s).",
                    ", ".join(missing),
                )
            else:
                self.tp_sl_engine = _try_construct(
                    TPSLEngine,
                    config=self.config,
                    logger=self.logger,
                    app=self,
                    shared_state=self.shared_state,
                    execution_manager=self.execution_manager,
                )
                if self.tp_sl_engine is None and TPSLEngine is not None:
                    try:
                        self.logger.warning(
                            "[Bootstrap] TPSLEngine _try_construct returned None; attempting direct construction for diagnostics..."
                        )
                        self.tp_sl_engine = TPSLEngine(
                            shared_state=self.shared_state,
                            config=self.config,
                            execution_manager=self.execution_manager,
                        )
                    except Exception as e_direct:
                        self.logger.error(
                            "[Bootstrap] TPSLEngine direct construction failed: %s: %s",
                            type(e_direct).__name__,
                            e_direct,
                            exc_info=True,
                        )
            if self.tp_sl_engine is None:
                self.logger.error("[Bootstrap] TPSLEngine construction failed or returned None")
            else:
                self.logger.info("[Bootstrap] TPSLEngine constructed: %s", type(self.tp_sl_engine).__name__)
        # Inject TradeJournal + session_id into TPSLEngine
        if self.tp_sl_engine:
            self.tp_sl_engine.trade_journal = self.trade_journal
            self.tp_sl_engine.session_id = self.session_id

        # Wire TP/SL engine into core execution path (mandatory exits)
        try:
            if self.tp_sl_engine and self.execution_manager:
                if hasattr(self.execution_manager, "set_tp_sl_engine"):
                    self.execution_manager.set_tp_sl_engine(self.tp_sl_engine)
                else:
                    self._set_attr_if_missing(self.execution_manager, "tp_sl_engine", self.tp_sl_engine)
            if self.tp_sl_engine and self.meta_controller:
                self._set_attr_if_missing(self.meta_controller, "tp_sl_engine", self.tp_sl_engine)
        except Exception:
            self.logger.debug("TP/SL wiring into execution path failed", exc_info=True)

        # Analytics / portfolio / orchestration
        if self.performance_monitor is None:
            PerformanceMonitor = _get_cls(_perf_mod, "PerformanceMonitor")
            self.performance_monitor = _try_construct(PerformanceMonitor, config=self.config, cfg=self.config, logger=self.logger, app=self, shared_state=self.shared_state, execution_manager=self.execution_manager)
        if self.compounding_engine is None:
            CompoundingEngine = _get_cls(_comp_mod, "CompoundingEngine")
            self.compounding_engine = _try_construct(CompoundingEngine, config=self.config, logger=self.logger, app=self, shared_state=self.shared_state, execution_manager=self.execution_manager, exchange_client=self.exchange_client)
        if self.volatility_regime is None:
            VolatilityRegimeDetector = _get_cls(_vol_mod, "VolatilityRegimeDetector")
            # Pass symbols from symbol_manager if available
            symbols = [s.symbol for s in getattr(self.symbol_manager, "symbols", [])] if hasattr(self, "symbol_manager") and self.symbol_manager else []
            self.volatility_regime = _try_construct(VolatilityRegimeDetector, config=self.config, logger=self.logger, app=self, shared_state=self.shared_state, market_data_feed=self.market_data_feed, symbols=symbols if symbols else None)
        if self.portfolio_balancer is None:
            PortfolioBalancer = _get_cls(_portfolio_mod, "PortfolioBalancer")
            self.portfolio_balancer = _try_construct(PortfolioBalancer, config=self.config, logger=self.logger, app=self, shared_state=self.shared_state, execution_manager=self.execution_manager, exchange_client=self.exchange_client)
        if self.liquidation_agent is None:
            LiquidationAgent = _get_cls(_liq_agent_mod, "LiquidationAgent")
            self.liquidation_agent = _try_construct(LiquidationAgent, config=self.config, logger=self.logger, app=self, shared_state=self.shared_state, execution_manager=self.execution_manager, symbol_manager=self.symbol_manager, cash_router=self.cash_router, risk_manager=self.risk_manager)
        if self.liquidation_orchestrator is None:
            LiquidationOrchestrator = _get_cls(_liq_orch_mod, "LiquidationOrchestrator")
            self.liquidation_orchestrator = _try_construct(
                LiquidationOrchestrator, 
                config=self.config, 
                logger=self.logger, 
                app=self, 
                shared_state=self.shared_state, 
                execution_manager=self.execution_manager,
                liquidation_agent=self.liquidation_agent,
                cash_router=self.cash_router,
                risk_manager=self.risk_manager,
                position_manager=getattr(self, "position_manager", None),
                meta_controller=self.meta_controller
            )
        if self.performance_evaluator is None and _perf_eval_mod:
            self.performance_evaluator = _try_construct(getattr(_perf_eval_mod, "PerformanceEvaluator", None), config=self.config, logger=self.logger, shared_state=self.shared_state)
        
        if self.dashboard_server is None and _dashboard_mod and self.shared_state is not None:
            DS = getattr(_dashboard_mod, "DashboardServer", None)
            if DS:
                self.dashboard_server = DS(shared_state=self.shared_state, host=self._cfg("DASHBOARD_HOST", "0.0.0.0"), port=int(self._cfg("DASHBOARD_PORT", 8000)))

        # Recovery bits
        if self.recovery_engine is None:
            RecoveryEngine = _get_cls(_recovery_mod, "RecoveryEngine")
            self.recovery_engine = _try_construct(
                RecoveryEngine,
                config=self.config,
                logger=self.logger,
                app=self,
                shared_state=self.shared_state,
                exchange_client=self.exchange_client,
                execution_manager=self.execution_manager,
            )

        if self.profit_target_engine is None:
            ProfitTargetEngine = _get_cls(_ptg_mod, "ProfitTargetEngine")
            self.profit_target_engine = _try_construct(ProfitTargetEngine, config=self.config, logger=self.logger, app=self, shared_state=self.shared_state)
            # P9: Wire guard to SharedState (only if both profit_target_engine and shared_state exist)
            if (self.profit_target_engine and self.shared_state and 
                hasattr(self.profit_target_engine, "check_global_compliance") and 
                hasattr(self.shared_state, "set_profit_guard")):
                self.shared_state.set_profit_guard(self.profit_target_engine.check_global_compliance)

        if self.capital_allocator is None:
            CapitalAllocator = _get_cls(_cap_alloc_mod, "CapitalAllocator")
            self.capital_allocator = _try_construct(
                CapitalAllocator,
                config=self.config,
                logger=self.logger,
                app=self,
                shared_state=self.shared_state,
                strategy_manager=self.strategy_manager,
                agent_manager=self.agent_manager,  # P9: Add agent_manager for agent discovery
                execution_manager=self.execution_manager,  # P9: Add execution_manager for validation
                liquidation_agent=self.liquidation_agent,  # P9: Add liquidation_agent for opportunistic cap
                risk_manager=self.risk_manager,
                profit_target_engine=self.profit_target_engine
            )

        # Phase G: Dust Monitor (Real-time monitoring of dust positions)
        if self.dust_monitor is None:
            DustMonitor = _get_cls(_dust_monitor_mod, "DustMonitor")
            self.dust_monitor = _try_construct(DustMonitor, shared_state=self.shared_state, config=self.config, logger=self.logger)

        # Maintain back-compat aliases
        self.liq_agent = self.liquidation_agent
        self.liq_orch = self.liquidation_orchestrator

        if self.meta_controller and self.liquidation_orchestrator:
            # P9: Wire orchestration for opportunistic liquidations
            if hasattr(self.meta_controller, "set_liquidation_orchestrator"):
                self.meta_controller.set_liquidation_orchestrator(self.liquidation_orchestrator)

        if self.meta_controller and self.compounding_engine:
            # P9: Wire compounding for loop summary reporting
            if hasattr(self.meta_controller, "set_compounding_engine"):
                self.meta_controller.set_compounding_engine(self.compounding_engine)

        # Enforce single SharedState identity across core execution components.
        try:
            self._enforce_shared_state_identity(emit_summary=True)
        except Exception:
            self.logger.debug("shared_state identity enforcement failed", exc_info=True)

        dust_metrics = self._dust_metrics_snapshot()

        self.logger.info("[BootInventory] built: exch=%s shared=%s mdf=%s exec=%s meta=%s agents=%s risk=%s dashboard=%s cap_alloc=%s dust_monitor=%s",
                         bool(self.exchange_client), bool(self.shared_state), bool(self.market_data_feed),
                         bool(self.execution_manager), bool(self.meta_controller),
                         bool(self.agent_manager), bool(self.risk_manager), bool(self.dashboard_server),
                         bool(self.capital_allocator), bool(self.dust_monitor))
        self._summary_ff(
            "BOOT_INVENTORY",
            exch=bool(self.exchange_client),
            shared=bool(self.shared_state),
            mdf=bool(self.market_data_feed),
            exec=bool(self.execution_manager),
            meta=bool(self.meta_controller),
            agents=bool(self.agent_manager),
            risk=bool(self.risk_manager),
            liq_agent=bool(self.liquidation_agent),
            liq_orch=bool(self.liquidation_orchestrator),
            cap_alloc=bool(self.capital_allocator),
            dust_monitor=bool(self.dust_monitor),
            dust_registry_size=dust_metrics.get("registry_size"),
            dust_origin_breakdown=dust_metrics.get("origin_breakdown"),
            dust_external_pct=dust_metrics.get("external_pct"),
        )

    @property
    def has_liq_agent(self) -> bool:
        return bool(self.liquidation_agent)

    @property
    def has_liq_orch(self) -> bool:
        return bool(self.liquidation_orchestrator)

    async def _periodic_readiness_log(self, every_sec: int = 30):
        while True:
            try:
                snap = await self._ops_plane_snapshot()
                ready = bool(snap.get("ready"))
                issues = list(snap.get("issues") or [])
                detail = dict(snap.get("detail") or {})

                # Attach compact per-symbol liquidity gaps
                try:
                    detail["Liquidity"] = self._liquidity_snapshot()
                except Exception:
                    pass

                # Health levels: infra vs. capital/liquidity
                infra_blockers = {"MarketDataNotReady", "ExecutionManagerNotReady", "ExchangeClientNotReady", "SymbolsUniverseEmpty"}
                capital_blockers = {"NAVNotReady", "MinNotionalTooHighForConfiguredQuote", "FreeQuoteBelowFloor"}

                issues_set = set(issues)
                if ready:
                    health_level = "OK"
                elif issues_set & infra_blockers:
                    health_level = "ERROR"       # core systems not up
                elif issues_set & capital_blockers:
                    health_level = "DEGRADED"    # running, but can’t trade yet
                else:
                    health_level = "DEGRADED"

                # Log one informative line
                self.logger.info("[Readiness] ready=%s issues=%s detail=%s", ready, issues, detail)

                # Emit summary + health
                try:
                    await self._emit_summary(
                        "READINESS_TICK",
                        ready=ready,
                        issues=issues,
                        dust=detail.get("Dust", {}),
                    )
                    await self._emit_health_status(
                        health_level,
                        {
                            "issues": issues,
                            "liquidity": detail.get("Liquidity", {}),
                            "dust": detail.get("Dust", {}),
                        },
                    )
                except Exception:
                    pass
            except Exception:
                self.logger.debug("periodic readiness log failed", exc_info=True)
            await asyncio.sleep(every_sec)

    # ----------------------------- Phased init (P3→P9) ------------------------
    async def initialize_all(self, up_to_phase: int = 9):
        """Run phased initialization. Phases are best-effort; optional comps may be absent."""
        # Prevent concurrent init runs
        async with self._init_lock:
            if self._init_completed and up_to_phase <= self._init_highest_phase:
                self.logger.info(
                    "[Init] initialize_all already completed up to P%s; skipping (requested up_to_phase=%s)",
                    self._init_highest_phase, up_to_phase,
                )
                try:
                    await self._emit_summary("INIT_DUPLICATE_CALL_SKIPPED", up_to_phase=up_to_phase,
                                             highest_phase=self._init_highest_phase)
                except Exception:
                    pass
                return
            log_structured_info("[Init] Starting phased initialization up to P%d" % up_to_phase, component="AppContext")
            await self._emit_summary("INIT_START", up_to_phase=up_to_phase)
            await self._emit_health_status("STARTING", {"up_to_phase": up_to_phase})
            # Early ceiling reporter — if capped at/below P3, emit explicit summaries and return
            if up_to_phase <= 3:
                try:
                    await self._emit_summary("PHASE_CEILING", up_to_phase=up_to_phase)
                    for phase in ("P4_market_data","P5_execution","P6_stack","P7_protective","P8_analytics","P9_run"):
                        await self._emit_summary("PHASE_SKIP", phase=phase, status="SKIPPED", reason="PhaseDisabled")
                except Exception:
                    self.logger.debug("emit PHASE_CEILING/PHASE_SKIP failed", exc_info=True)
                self.logger.info("[AppContext] Initialization ended at requested ceiling: P%d", up_to_phase)
                return

            # Build any missing components first
            self._ensure_components_built()
            # Optional readiness gating configuration
            try:
                wait_ready_secs = int(self._cfg("WAIT_READY_SECS", 0))  # 0 = don't block
            except Exception:
                wait_ready_secs = 0
            gate_on = str(self._cfg("GATE_READY_ON", "") or "").strip()
            gate_list = [g.strip() for g in gate_on.split(",") if g.strip()] if gate_on else []
            # Default gates if WAIT_READY_SECS>0 but GATE_READY_ON unset
            if wait_ready_secs > 0 and not gate_list:
                gate_list = ["market_data", "execution", "capital", "exchange", "startup_sanity"]
            if wait_ready_secs > 0:
                self.logger.info("[AppContext] Ready-gating wait_ready_secs=%s gates=%s", wait_ready_secs, ",".join(gate_list))

            # P3: Hard gate exchange public readiness + balances, then universe bootstrap
            if up_to_phase >= 3:
                await self._step("P3_exchange_gate", self._gate_exchange_ready())
                await self._step("P3_balances_probe", self._attempt_fetch_balances())
                await self._step("P3_universe_bootstrap", self._ensure_universe_bootstrap())

            # P3.5: Exchange client (must be started before P4 so MDF can hit public/private endpoints)
            if up_to_phase >= 3 and self.exchange_client and hasattr(self.exchange_client, "start"):
                await self._start_with_timeout("P3_exchange", self.exchange_client)
                # Ensure everyone sees the client after start()
                self._propagate_exchange_client()

                # P3.55: Ensure exchange public-ready before any component tries symbol info
                try:
                    await self._ensure_exchange_public_ready()
                    # propagate again in case public bootstrap swapped client internals
                    self._propagate_exchange_client()
                    try:
                        await self._emit_summary("P3_EXCHANGE_PUBLIC_READY", ready=True)
                    except Exception:
                        pass
                except Exception:
                    self.logger.debug("P3.55 ensure_exchange_public_ready failed (non-fatal)", exc_info=True)

                # P3.58: Exchange truth reconciliation loop (governance-only)
                # 🔧 BOUNDARY CHECK: Only run in LIVE mode
                # Shadow mode must be pure simulation without exchange reconciliation
                trading_mode = str(getattr(self.shared_state, "trading_mode", "live") or "live").lower()
                if trading_mode == "live":
                    if self.exchange_truth_auditor and any(
                        hasattr(self.exchange_truth_auditor, nm) for nm in ("start", "start_async", "run", "run_async")
                    ):
                        await self._start_with_timeout("P3_truth_auditor", self.exchange_truth_auditor)
                    else:
                        try:
                            await self._emit_summary(
                                "PHASE_SKIP",
                                phase="P3_truth_auditor",
                                status="SKIPPED",
                                reason=("ComponentMissing" if not self.exchange_truth_auditor else "NoStartMethod"),
                            )
                        except Exception:
                            pass
                else:
                    self.logger.info("[Bootstrap] Skipping TruthAuditor in shadow mode")

            # P3.6: SharedState background tasks (balances/NAV/events)
            if up_to_phase >= 3 and self.shared_state and hasattr(self.shared_state, "start_background_tasks"):
                try:
                    # Use same startup timeout budget; do not treat as fatal
                    await asyncio.wait_for(self.shared_state.start_background_tasks(), timeout=self._start_timeout_sec())
                    self.logger.info("[P3_shared_state] background tasks started")
                except Exception as e:
                    self.logger.warning("[P3_shared_state] background tasks failed: %s", e)
            
            # ═══════════════════════════════════════════════════════════════════════════════════
            # P3.62: RESTART MODE DETECTION (CRITICAL FIX #3)
            # Detect if this is a restart with existing portfolio or pending intents
            # This guards against treating "existing positions" as errors on restart
            # ═══════════════════════════════════════════════════════════════════════════════════
            is_restart = await self._detect_restart_mode()

            # Propagate restart detection into SharedState so downstream components
            # (MetaController, ExecutionLogic) can check it without re-running detection.
            try:
                if self.shared_state:
                    self.shared_state._is_restart = is_restart
            except Exception:
                self.logger.debug("failed to set _is_restart on shared_state", exc_info=True)

            # Single source of truth: trading_mode controls live/shadow globally
            is_live_mode = (str(getattr(self.config, "trading_mode", "live")).lower() == "live")

            # P3.63: SHADOW MODE VIRTUAL PORTFOLIO INITIALIZATION
            # Before trading starts: Initialize virtual portfolio from real balances
            # This is the CRITICAL MISSING PIECE - without this, virtual_balances stays at 0
            if not is_live_mode and self.shared_state:
                try:
                    await self.shared_state.init_virtual_portfolio_from_real_snapshot()
                    self.logger.info("[P3_shadow_mode] Virtual portfolio initialized from real snapshot")
                except Exception as e:
                    self.logger.error("[P3_shadow_mode] Failed to initialize virtual portfolio: %s", e, exc_info=True)

            if is_restart or is_live_mode:
                self.logger.warning(
                    "[AppContext:StartupPolicy] PURE RECONCILIATION startup (restart=%s, live_mode=%s). "
                    "No forced entries. No seed trades. No capital overrides. No confidence bypasses. "
                    "Existing positions will be OBSERVED and MANAGED per strategy.",
                    is_restart, is_live_mode,
                )
                # Force-disable bootstrap seed so MetaController cannot fire it
                try:
                    setattr(self.config, "BOOTSTRAP_SEED_ENABLED", False)
                    setattr(self.config, "COLD_BOOTSTRAP_ENABLED", False)
                except Exception:
                    pass
                try:
                    await self._emit_summary("STARTUP_POLICY", mode="RECONCILIATION_ONLY",
                                             is_restart=is_restart, is_live_mode=is_live_mode,
                                             bootstrap_seed_enabled=False, cold_bootstrap_enabled=False)
                except Exception:
                    pass
            else:
                self.logger.info(
                    "[AppContext:ColdStart] System operating in COLD_START mode (no prior trading history). "
                    "Bootstrap seed will only fire if COLD_BOOTSTRAP_ENABLED=True and BOOTSTRAP_SEED_ENABLED=True."
                )
            
            # P3.65: Ensure SymbolManager is fully wired before any symbol proposals (e.g., WalletScanner)
            try:
                if self.symbol_manager:
                    # Make sure the exchange client is injected everywhere first
                    self._propagate_exchange_client()

                    # Keep shared_state in sync in case it was constructed earlier
                    if hasattr(self.symbol_manager, "set_shared_state"):
                        self.symbol_manager.set_shared_state(self.shared_state)

                    # Ensure the exchange client can serve unsigned GETs during early phases
                    try:
                        if self.exchange_client and hasattr(self.exchange_client, "_ensure_started_public"):
                            maybe = self.exchange_client._ensure_started_public()
                            if asyncio.iscoroutine(maybe):
                                await maybe
                    except Exception:
                        self.logger.debug("public bootstrap on symbol_manager wiring failed", exc_info=True)

                    # Only warm exchange info cache if we actually have a client now
                    if self.exchange_client and hasattr(self.symbol_manager, "_ensure_exchange_info"):
                        await self.symbol_manager._ensure_exchange_info(force=True)
                    elif hasattr(self.symbol_manager, "_ensure_exchange_info"):
                        # Avoid noisy warning path inside SymbolManager when client is not available yet
                        self.logger.info("[P3.65] Skipped _ensure_exchange_info because exchange_client is not available yet")
            except Exception:
                self.logger.debug("symbol_manager wiring/warmup failed", exc_info=True)

            # P3.7: Seed universe from wallet balances (one-shot) if available
            try:
                # Only run if universe is still empty
                has_universe = await self._has_nonempty_universe()
                if (not has_universe) and _wallet_scanner_mod and self.exchange_client and self.symbol_manager:
                    WSA = getattr(_wallet_scanner_mod, "WalletScannerAgent", None)
                    if WSA:
                        scanner = WSA(
                            shared_state=self.shared_state,
                            config=self.config,
                            exchange_client=self.exchange_client,
                            symbol_manager=self.symbol_manager,
                            interval=int(getattr(self.config, "WALLET_SCANNER_INTERVAL", 1800)),
                            min_balance_threshold=float(getattr(self.config, "WALLET_SCANNER_MIN_BAL", 0.0)),
                        )
                        try:
                            await asyncio.wait_for(scanner.run_once(), timeout=25)
                            self.logger.info("[P3_wallet_scan] one-shot wallet scan completed")
                            # Emit an explicit completion SUMMARY for wallet scan to trigger resync listeners
                            try:
                                accepted_count = None
                                try:
                                    if self.shared_state and hasattr(self.shared_state, "get_accepted_symbols"):
                                        r = self.shared_state.get_accepted_symbols()
                                        r = await r if asyncio.iscoroutine(r) else r
                                        if isinstance(r, dict):
                                            accepted_count = len(r)
                                except Exception:
                                    accepted_count = None

                                if accepted_count is None:
                                    asyncio.create_task(self._emit_summary("P3_WALLET_SCAN_COMPLETED"))
                                else:
                                    asyncio.create_task(self._emit_summary("P3_WALLET_SCAN_COMPLETED", accepted=accepted_count))
                            except Exception:
                                self.logger.debug("emit P3_WALLET_SCAN_COMPLETED failed", exc_info=True)
                        except asyncio.TimeoutError:
                            self.logger.warning("[P3_wallet_scan] one-shot scan timed out; continuing")
            except Exception:
                self.logger.debug("wallet scan bootstrap failed", exc_info=True)

            # P3.9: (kept for clarity; idempotent)
            try:
                await self._ensure_exchange_public_ready()
                self._propagate_exchange_client()
            except Exception:
                self.logger.debug("P3.9 ensure_exchange_public_ready failed (non-fatal)", exc_info=True)

            # P3.92: Attempt to elevate to signed mode (so P4+ can proceed). Non-fatal if keys are absent.
            try:
                signed_ok = await self._ensure_exchange_signed_ready()
                if signed_ok:
                    self._propagate_exchange_client()
            except Exception:
                self.logger.debug("P3.92 ensure_exchange_signed_ready failed (non-fatal)", exc_info=True)

            # Rebuild optional components that depend on a signed ExchangeClient (idempotent, constructs only if None)
            try:
                self._ensure_components_built()
                self._propagate_exchange_client()
            except Exception:
                self.logger.debug("post-signed components ensure failed (non-fatal)", exc_info=True)

            # Visibility: what happens next (confirms MDF presence and startability)
            try:
                self.logger.info("[AppContext] NextPhaseCheck: up_to_phase=%s mdf=%s has_mdf_start=%s",
                                 up_to_phase, bool(self.market_data_feed),
                                 bool(self.market_data_feed and (hasattr(self.market_data_feed, 'start') or hasattr(self.market_data_feed, 'start_async') or hasattr(self.market_data_feed, 'run') or hasattr(self.market_data_feed, 'run_async'))))
            except Exception:
                pass

            try:
                # Propagate the client into already-constructed components that may have been built earlier
                for comp_name in ("symbol_manager", "market_data_feed", "execution_manager", "strategy_manager", "agent_manager", "risk_manager"):
                    comp = getattr(self, comp_name, None)
                    if comp:
                        set_ec = getattr(comp, "set_exchange_client", None)
                        if callable(set_ec):
                            try:
                                res = set_ec(self.exchange_client)
                                if asyncio.iscoroutine(res):
                                    await res
                            except Exception:
                                self.logger.debug("set_exchange_client failed on %s", comp_name, exc_info=True)
                        else:
                            try:
                                setattr(comp, "exchange_client", self.exchange_client)
                            except Exception:
                                pass
                # Ensure public session is up for unsigned endpoints (no-op if already done)
                if self.exchange_client and hasattr(self.exchange_client, "_ensure_started_public"):
                    maybe = self.exchange_client._ensure_started_public()
                    if asyncio.iscoroutine(maybe):
                        await maybe
            except Exception:
                self.logger.debug("P3.95 exchange client propagation failed (non-fatal)", exc_info=True)

            # P4: Market data feed (HARD GATE — failure blocks P5+ if gate not met)
            _p4_gate_ok = True
            try:
                if up_to_phase >= 4 and self.market_data_feed and any(hasattr(self.market_data_feed, nm) for nm in ("start","start_async","run","run_async")):
                    # Start MDF (supports start/start_async/run/run_async) and apply timeout handling
                    await self._start_with_timeout("P4_market_data", self.market_data_feed)
                    # NOTE: _start_with_timeout already spawns mdf.run() as a background task.
                    try:
                        md_timeout = int(self._cfg("P4_MARKET_DATA_READY_TIMEOUT_SEC", 180))
                    except Exception:
                        md_timeout = 180
                    snap = await self._wait_until_ready(gates=["exchange","universe","market_data"], timeout_sec=md_timeout, poll_sec=2.0)
                    if "SymbolsUniverseEmpty" in snap.get("issues", []) or "MarketDataNotReady" in snap.get("issues", []):
                        _p4_gate_ok = False
                        log_structured_error(
                            RuntimeError("P4GateNotSatisfied"),
                            context={"where": "initialize_all:P4", "issues": snap.get("issues", [])},
                            logger=self.logger, component="AppContext", phase="INIT", event="p4_gate_failed",
                        )
                        try:
                            await self._emit_summary("INIT_EXCEPTION", error="P4GateNotSatisfied", issues=snap.get("issues", []))
                        except Exception:
                            pass
                else:
                    # Explain why we skipped P4
                    reason = "PhaseDisabled" if up_to_phase < 4 else ("ComponentMissing" if not self.market_data_feed else "NoStartMethod")
                    try:
                        await self._emit_summary("PHASE_SKIP", phase="P4_market_data", status="SKIPPED", reason=reason)
                    except Exception:
                        pass
                # Guarantee at least one outcome log for P4
                try:
                    await self._emit_summary("P4_MARKET_DATA_DECISION",
                                             present=bool(self.market_data_feed),
                                             has_start=bool(self.market_data_feed and any(hasattr(self.market_data_feed, nm) for nm in ("start","start_async","run","run_async"))),
                                             requested=(up_to_phase >= 4))
                except Exception:
                    pass
            except Exception as e:
                _p4_gate_ok = False
                log_structured_error(e, context={"where": "initialize_all:P4"}, logger=self.logger,
                                     component="AppContext", phase="INIT", event="init_exception")
                try:
                    await self._emit_summary("INIT_EXCEPTION", error=str(e))
                except Exception:
                    pass

            # P5: Execution manager warmup — only proceeds if P4 gate passed
            if not _p4_gate_ok:
                self.logger.error("[AppContext] P4 gate failed — aborting P5+ startup to avoid trading with bad market data.")
                return
            try:
                if up_to_phase >= 5 and self.execution_manager and any(hasattr(self.execution_manager, nm) for nm in ("start","start_async","run","run_async")):
                    # Allow execution warmup if balances are reported ready OR we can observe non-zero free USDT.
                    balances_ready_flag = bool(getattr(self.shared_state, "balances_ready", False))
                    free_probe = 0.0
                    try:
                        if self.shared_state:
                            fu = getattr(self.shared_state, "free_usdt", None)
                            if callable(fu):
                                _v = fu()
                                free_probe = float(await _v) if asyncio.iscoroutine(_v) else float(_v)
                            else:
                                free_probe = float(getattr(self.shared_state, "balances", {}).get("USDT", {}).get("free", 0.0))
                    except Exception:
                        free_probe = 0.0
                    balances_ok = balances_ready_flag or (free_probe > 0.0)

                    pre_ok = all([
                        self.exchange_client is not None,
                        await self._has_nonempty_universe(),
                        balances_ok,
                    ])
                    if not pre_ok:
                        await self._emit_summary("P5_EXECUTION_SKIPPED", reason="prerequisites_not_met")
                    else:
                        await self._start_with_timeout("P5_execution", self.execution_manager)
                        try:
                            await self._emit_summary("P5_EXECUTION_STARTED")
                        except Exception:
                            pass
                        try:
                            self.logger.info("[P5_execution] warmup complete — symbol filters and balances should now be available to ExecutionManager.")
                        except Exception:
                            pass

                        # Start AdaptiveCapitalEngine background task after execution manager is ready
                        if self.adaptive_capital_engine:
                            try:
                                adaptive_task = asyncio.create_task(
                                    self._run_adaptive_capital_monitor(),
                                    name="adaptive_capital_monitor"
                                )
                                self._tasks.append(adaptive_task)
                                self._tasks_map["adaptive_capital_monitor"] = adaptive_task
                                self.logger.info("[AdaptiveCapitalEngine] Background monitoring task started")
                            except Exception as e:
                                self.logger.warning("[AdaptiveCapitalEngine] Failed to start background task: %s", e)
                else:
                    try:
                        await self._emit_summary("PHASE_SKIP", phase="P5_execution", status="SKIPPED",
                                                 reason=("ComponentMissing" if not self.execution_manager else "NoStartMethod"))
                    except Exception:
                        pass
            except Exception as e:
                log_structured_error(e, context={"where": "initialize_all:P5"}, logger=self.logger,
                                     component="AppContext", phase="INIT", event="init_exception")
                try:
                    await self._emit_summary("INIT_EXCEPTION", error=str(e))
                except Exception:
                    pass
            # P6: Strategy/agents/risk/meta
            if up_to_phase >= 6:
                # Pre-register MetaController to avoid startup races being classified as "no-report".
                if self.meta_controller and hasattr(self.shared_state, "register_component"):
                    try:
                        await self.shared_state.register_component("MetaController")
                        await self.shared_state.update_component_status("MetaController", "Initialized", "Constructed")
                    except Exception:
                        self.logger.debug("Failed to register MetaController component status", exc_info=True)

                if self.meta_controller:
                    # [DIAGNOSTIC] Log available methods if start is missing
                    if not hasattr(self.meta_controller, "start"):
                        try:
                            self.logger.warning("[P6_meta_controller] MetaController instance found but 'start' attribute missing. Methods: %s", 
                                                [m for m in dir(self.meta_controller) if not m.startswith("_")])
                        except Exception:
                            self.logger.warning("[P6_meta_controller] MetaController found but 'start' missing (introspection failed).")

                    # [CRITICAL] Authoritative wallet sync before MetaController starts
                    self.logger.warning("[BOOT] Authoritative wallet sync (exchange is source of truth)")
                    if hasattr(self.shared_state, "authoritative_wallet_sync"):
                        await self.shared_state.authoritative_wallet_sync()
                    else:
                        await self.shared_state.hard_reset_capital_state()
                    self.logger.warning("[BOOT] Authoritative wallet sync complete")
                    
                    await self._start_with_timeout("P6_meta_controller", self.meta_controller)
                    
                    # Sanity Check as requested
                    if not getattr(self.meta_controller, "_running", False):
                        raise RuntimeError("MetaController failed to enter running state; aborting downstream phases.")
                    self.logger.info("[AppContext] MetaController running state declared VALID. Proceeding to AgentManager.")

                else:
                    try:
                        await self._emit_summary("PHASE_SKIP", phase="P6_meta_controller", status="SKIPPED",
                                                 reason=("ComponentMissing" if not self.meta_controller else "NoStartMethod"))
                    except Exception:
                        pass

                _p6_start_methods = ("start", "start_async", "run", "run_async")
                if self.strategy_manager and any(hasattr(self.strategy_manager, nm) for nm in _p6_start_methods):
                    await self._start_with_timeout("P6_strategy", self.strategy_manager)
                else:
                    try:
                        await self._emit_summary("PHASE_SKIP", phase="P6_strategy", status="SKIPPED",
                                                 reason=("ComponentMissing" if not self.strategy_manager else "NoStartMethod"))
                    except Exception:
                        pass

                # 🔥 CRITICAL RACE CONDITION FIX
                # Wait for strategies to finish registering agents before starting AgentManager.
                # This ensures the pipeline order: StrategyManager → register agents → AgentManager → schedule agents
                # Without this wait, AgentManager starts with empty agents={} and runs with no strategy agents.
                if self.strategy_manager and self.agent_manager:
                    LOGGER.info("[P6_Race] Waiting for agents to be registered by StrategyManager...")
                    for _ in range(30):
                        if getattr(self.agent_manager, "agents", None):
                            if len(self.agent_manager.agents) > 0:
                                LOGGER.info("[P6_Race] ✅ Agents registered! Found %d agents. Starting AgentManager.", len(self.agent_manager.agents))
                                break
                        await asyncio.sleep(1)
                    else:
                        LOGGER.warning("[P6_Race] ⚠️ Timeout waiting for agents. Proceeding anyway (agents may register during startup).")

                if self.agent_manager and any(hasattr(self.agent_manager, nm) for nm in _p6_start_methods):
                    await self._start_with_timeout("P6_agent_manager", self.agent_manager)
                else:
                    try:
                        await self._emit_summary("PHASE_SKIP", phase="P6_agent_manager", status="SKIPPED",
                                                 reason=("ComponentMissing" if not self.agent_manager else "NoStartMethod"))
                    except Exception:
                        pass

                if self.risk_manager and any(hasattr(self.risk_manager, nm) for nm in _p6_start_methods):
                    await self._start_with_timeout("P6_risk_manager", self.risk_manager)
                else:
                    try:
                        await self._emit_summary("PHASE_SKIP", phase="P6_risk_manager", status="SKIPPED",
                                                 reason=("ComponentMissing" if not self.risk_manager else "NoStartMethod"))
                    except Exception:
                        pass

                self.logger.info("[StartInventory] P4=%s P5=%s P6(meta=%s,agents=%s,risk=%s)",
                                 hasattr(self.market_data_feed, "start"),
                                 hasattr(self.execution_manager, "start"),
                                 hasattr(self.meta_controller, "start"),
                                 hasattr(self.agent_manager, "start"),
                                 hasattr(self.risk_manager, "start"))
                try:
                    await self._emit_summary("P6_STACK_STARTED")
                except Exception:
                    pass
            else:
                try:
                    await self._emit_summary("PHASE_SKIP", phase="P6_stack", status="SKIPPED", reason="PhaseDisabled")
                except Exception:
                    pass

            # P7: Protective services
            if up_to_phase >= 7:
                # CRITICAL FIX: Register components with shared_state BEFORE they start
                # This prevents health gate from seeing "no-report" status
                component_registrations = {
                    "pnl_calculator": ("PnLCalculator", self.pnl_calculator),
                    "tp_sl_engine": ("TPSLEngine", self.tp_sl_engine),
                }
                for comp_name, (display_name, obj) in component_registrations.items():
                    if obj and hasattr(self.shared_state, "register_component"):
                        try:
                            await self.shared_state.register_component(display_name)
                            await self.shared_state.update_component_status(display_name, "Initializing")
                        except Exception:
                            self.logger.debug(f"Failed to register {display_name} component status", exc_info=True)
                
                for name, phase in (
                    ("pnl_calculator", "P7_pnl_calculator"),
                    ("heartbeat", "P7_heartbeat"),
                    ("watchdog", "P7_watchdog"),
                    ("alert_system", "P7_alert_system"),
                    ("tp_sl_engine", "P7_tp_sl_engine"),
                    ("dust_monitor", "P7_dust_monitor"),
                ):
                    obj = getattr(self, name, None)
                    if name == "tp_sl_engine":
                        # Enforce single TPSL instance
                        if self._tpsl_started:
                            self.logger.info("[AppContext] TPSLEngine already started, skipping duplicate start")
                            continue
                        self._tpsl_started = True
                    if obj and hasattr(obj, "start"):
                        await self._start_with_timeout(phase, obj)
                        # Update status after successful start
                        if name == "tp_sl_engine" and hasattr(self.shared_state, "update_component_status"):
                            try:
                                await self.shared_state.update_component_status("TPSLEngine", "Running")
                            except Exception:
                                self.logger.debug("Failed to update TPSLEngine status", exc_info=True)
                        elif name == "pnl_calculator" and hasattr(self.shared_state, "update_component_status"):
                            try:
                                await self.shared_state.update_component_status("PnLCalculator", "Running")
                            except Exception:
                                self.logger.debug("Failed to update PnLCalculator status", exc_info=True)
                    else:
                        try:
                            await self._emit_summary("PHASE_SKIP", phase=phase, status="SKIPPED",
                                                     reason=("ComponentMissing" if not obj else "NoStartMethod"))
                        except Exception:
                            pass
                try:
                    await self._emit_summary("P7_PROTECTIVE_STARTED")
                except Exception:
                    pass
            else:
                try:
                    await self._emit_summary("PHASE_SKIP", phase="P7_protective", status="SKIPPED", reason="PhaseDisabled")
                except Exception:
                    pass

            # P8: Analytics / portfolio / orchestration
            if up_to_phase >= 8:
                # CRITICAL FIX: Register PerformanceEvaluator with shared_state BEFORE it starts
                if self.performance_evaluator and hasattr(self.shared_state, "register_component"):
                    try:
                        await self.shared_state.register_component("PerformanceEvaluator")
                        await self.shared_state.update_component_status("PerformanceEvaluator", "Initializing")
                    except Exception:
                        self.logger.debug("Failed to register PerformanceEvaluator component status", exc_info=True)
                
                for name, phase in (
                    ("performance_monitor", "P8_performance_monitor"),
                    ("compounding_engine", "P8_compounding_engine"),
                    ("volatility_regime", "P8_volatility_regime"),
                    ("portfolio_balancer", "P8_portfolio_balancer"),
                    ("liquidation_agent", "P8_liquidation_agent"),
                    ("liquidation_orchestrator", "P8_liquidation_orchestrator"),
                    ("performance_evaluator", "P8_performance_evaluator"),
                    ("dashboard_server", "P8_dashboard_server"),
                    ("capital_allocator", "P8_capital_allocator"),
                    ("profit_target_engine", "P8_profit_target_engine"),
                ):
                    obj = getattr(self, name, None)
                    if obj and hasattr(obj, "start"):
                        if name == "dashboard_server":
                            if not getattr(self.config, "DASHBOARD_ENABLED", True):
                                self.logger.warning("[P8_dashboard_server] Disabled by config; skipping start")
                                try:
                                    await self._emit_summary("PHASE_SKIP", phase=phase, status="SKIPPED", reason="DisabledByConfig")
                                except Exception:
                                    pass
                                continue
                        if name == "capital_allocator":
                            ca_cfg = getattr(self.config, "CAPITAL_ALLOCATOR", {}) or {}
                            if not bool(ca_cfg.get("ENABLED", True)):
                                self.logger.warning("[P8_capital_allocator] Disabled by config; skipping start")
                                try:
                                    await self._emit_summary("PHASE_SKIP", phase=phase, status="SKIPPED", reason="DisabledByConfig")
                                except Exception:
                                    pass
                                continue
                        await self._start_with_timeout(phase, obj)
                        # Update status after successful start
                        if name == "performance_evaluator" and hasattr(self.shared_state, "update_component_status"):
                            try:
                                await self.shared_state.update_component_status("PerformanceEvaluator", "Running")
                            except Exception:
                                self.logger.debug("Failed to update PerformanceEvaluator status", exc_info=True)
                    else:
                        try:
                            await self._emit_summary("PHASE_SKIP", phase=phase, status="SKIPPED",
                                                     reason=("ComponentMissing" if not obj else "NoStartMethod"))
                        except Exception:
                            pass

                # Optional: wire Liquidity→CashRouter refresh if both sides expose hooks
                try:
                    if self.liquidation_orchestrator and hasattr(self.liquidation_orchestrator, "on_completed") and self.meta_controller:
                        def _liquidity_done_cb(*_a, **_kw):
                            try:
                                if hasattr(self.meta_controller, "refresh_cash_router"):
                                    self.meta_controller.refresh_cash_router()
                            except Exception:
                                self.logger.debug("liquidity_done_cb failed", exc_info=True)
                        self.liquidation_orchestrator.on_completed(_liquidity_done_cb)
                except Exception:
                    self.logger.debug("wiring liquidity_done callback failed", exc_info=True)
                try:
                    await self._emit_summary("P8_ANALYTICS_STARTED")
                except Exception:
                    pass
            else:
                try:
                    await self._emit_summary("PHASE_SKIP", phase="P8_analytics", status="SKIPPED", reason="PhaseDisabled")
                except Exception:
                    pass

            # P8.5: STARTUP ORCHESTRATOR - Canonical P9 sequencing gate
            # BLOCKING GATE: Coordinates reconciliation components (RecoveryEngine, ExchangeTruthAuditor, etc.)
            # in correct order BEFORE MetaController begins. Prevents race condition.
            # CRITICAL: This is an ORCHESTRATOR, not a reconciliation engine.
            # It delegates to existing components to avoid duplication.
            if up_to_phase >= 8.5:
                from core.startup_orchestrator import StartupOrchestrator
                
                self.logger.warning("[P8.5_orchestrator] ═══════════════════════════════════════════════════")
                self.logger.warning("[P8.5_orchestrator] PHASE 8.5: STARTUP ORCHESTRATOR")
                self.logger.warning("[P8.5_orchestrator] Canonical sequencing: RecoveryEngine → SharedState → Auditor → Manager → Verify")
                self.logger.warning("[P8.5_orchestrator] ═══════════════════════════════════════════════════")
                
                try:
                    # Create orchestrator with all available components
                    orchestrator = StartupOrchestrator(
                        config=self.config,
                        shared_state=self.shared_state,
                        exchange_client=self.exchange_client,
                        recovery_engine=getattr(self, 'recovery_engine', None),
                        exchange_truth_auditor=getattr(self, 'exchange_truth_auditor', None),
                        portfolio_manager=getattr(self, 'portfolio_manager', None),
                        logger=self.logger,
                    )
                    
                    # Run orchestration sequence (BLOCKING - does not return until complete)
                    await orchestrator.execute_startup_sequence()
                    
                    # Verify orchestrator completed
                    if not orchestrator.is_ready():
                        raise RuntimeError("[P8.5] Orchestrator marked incomplete despite success signal")
                    
                    # Log metrics for audit trail
                    metrics = orchestrator.get_metrics()
                    self.logger.warning(f"[P8.5_orchestrator] Orchestration metrics: {metrics}")
                    
                    self.logger.warning("[P8.5_orchestrator] ✅ Phase 8.5 complete - portfolio ready for MetaController")
                    try:
                        await self._emit_summary("P8.5_ORCHESTRATOR_COMPLETE", metrics=metrics)
                    except Exception:
                        pass
                    
                except RuntimeError as e:
                    self.logger.error(
                        f"[P8.5_orchestrator] 💥 FATAL ERROR: {e}",
                        exc_info=True
                    )
                    try:
                        await self._emit_summary("P8.5_ORCHESTRATOR_FAILED", error=str(e))
                    except Exception:
                        pass
                    raise

            # P9: Finalize, announce mode, probe, health
            if up_to_phase >= 9:
                self._announce_runtime_mode()
                await self._dry_probe_execution()
                # Allow explicit 'startup_sanity' gate to include the new checks
                snap = await (self._wait_until_ready(gate_list, wait_ready_secs) if (wait_ready_secs > 0) else self._ops_plane_snapshot())
                blocked = snap.get("issues", [])
                if blocked:
                    try:
                        await self._emit_summary("INIT_GATES_TIMEOUT" if wait_ready_secs > 0 else "INIT_ISSUES", issues=blocked)
                    except Exception:
                        pass
                if blocked and any(b in blocked for b in ("ExchangeClientNotReady","SymbolsUniverseEmpty","MarketDataNotReady")):
                    try:
                        await self._emit_summary("INIT_BLOCKED_CORE_GATES", issues=blocked)
                    except Exception:
                        pass
                # Begin periodic readiness dumps for visibility
                try:
                    self._spawn("readiness:periodic", self._periodic_readiness_log(30))
                except Exception:
                    self.logger.debug("failed to spawn periodic readiness logger", exc_info=True)
                ready = bool(snap.get("ready"))
                log_structured_info("Phased initialization complete.")
                try:
                    await self._emit_summary("INIT_PHASES_SUMMARY", requested_up_to=up_to_phase,
                                             comps={"mdf": bool(self.market_data_feed),
                                                    "exec": bool(self.execution_manager),
                                                    "meta": bool(self.meta_controller),
                                                    "agents": bool(self.agent_manager),
                                                    "risk": bool(self.risk_manager)})
                except Exception:
                    pass
                # Mark init completed and record highest phase reached
                self._init_highest_phase = max(self._init_highest_phase, up_to_phase)
                self._init_completed = True
                try:
                    await self._emit_summary("INIT_COMPLETE", ready=bool(snap.get("ready")))
                    await self._emit_health_status("OK" if bool(snap.get("ready")) else "DEGRADED", {"startup_ready": bool(snap.get("ready")), "issues": snap.get("issues", [])})
                except Exception:
                    pass


# === SECTION: Agent Registration Helpers ===
# Fallback agent registration functions (in case core.agent_registry is unavailable)

def register_all_strategy_agents(agent_manager, app_context):
    """
    Register all strategy agents with AgentManager.
    
    This is a standalone function that can be called even if core.agent_registry
    import fails. It uses AGENT_CLASS_MAP from the registry if available.
    
    Intelligently passes only the parameters each agent's __init__ accepts
    to avoid "unexpected keyword argument" errors.
    """
    import inspect
    
    try:
        from core.agent_registry import AGENT_CLASS_MAP
    except ImportError:
        LOGGER.warning("[register_all_strategy_agents] core.agent_registry not available, skipping strategy agent registration")
        return
    
    for name, cls in AGENT_CLASS_MAP.items():
        try:
            if getattr(cls, "agent_type", None) == "strategy":
                # Introspect the __init__ signature to determine valid parameters
                try:
                    sig = inspect.signature(cls.__init__)
                    valid_params = set(sig.parameters.keys()) - {'self'}
                except Exception:
                    valid_params = set()
                
                # Build kwargs only with parameters the agent accepts
                all_kwargs = {
                    "shared_state": getattr(app_context, "shared_state", None),
                    "market_data_feed": getattr(app_context, "market_data_feed", None),
                    "execution_manager": getattr(app_context, "execution_manager", None),
                    "config": getattr(app_context, "config", None),
                    "tp_sl_engine": getattr(app_context, "tp_sl_engine", None),
                    "model_manager": getattr(app_context, "model_manager", None),
                }
                
                kwargs = {k: v for k, v in all_kwargs.items() if k in valid_params or len(valid_params) == 0}
                
                # Build agent instance from class
                agent = cls(**kwargs)
                
                # register_agent() expects an agent INSTANCE with a 'name' attribute
                agent_manager.register_agent(agent)
                LOGGER.info("[register_all_strategy_agents] Registered strategy agent: %s", name)
        except Exception as e:
            LOGGER.error("[register_all_strategy_agents] Failed to register %s: %s", name, e)


def register_all_discovery_agents(agent_manager, app_context):
    """
    Register all discovery agents with AgentManager.
    
    This is a standalone function that can be called even if core.agent_registry
    import fails. It uses AGENT_CLASS_MAP from the registry if available.
    
    Intelligently passes only the parameters each agent's __init__ accepts
    to avoid "unexpected keyword argument" errors.
    """
    import inspect
    
    try:
        from core.agent_registry import AGENT_CLASS_MAP
    except ImportError:
        LOGGER.warning("[register_all_discovery_agents] core.agent_registry not available, skipping discovery agent registration")
        return
    
    for name, cls in AGENT_CLASS_MAP.items():
        try:
            if getattr(cls, "agent_type", None) == "discovery":
                # Introspect the __init__ signature to determine valid parameters
                try:
                    sig = inspect.signature(cls.__init__)
                    valid_params = set(sig.parameters.keys()) - {'self'}
                except Exception:
                    valid_params = set()
                
                # Build kwargs only with parameters the agent accepts
                all_kwargs = {
                    "shared_state": getattr(app_context, "shared_state", None),
                    "config": getattr(app_context, "config", None),
                    "market_data_feed": getattr(app_context, "market_data_feed", None),
                    "execution_manager": getattr(app_context, "execution_manager", None),
                    "exchange_client": getattr(app_context, "exchange_client", None),
                    "symbol_manager": getattr(app_context, "symbol_manager", None),
                    "tp_sl_engine": getattr(app_context, "tp_sl_engine", None),
                    "model_manager": getattr(app_context, "model_manager", None),
                }
                
                kwargs = {k: v for k, v in all_kwargs.items() if k in valid_params or len(valid_params) == 0}
                
                # Build agent instance from class
                agent = cls(**kwargs)
                
                # register_discovery_agent() expects an agent INSTANCE
                agent_manager.register_discovery_agent(agent)
                LOGGER.info("[register_all_discovery_agents] Registered discovery agent: %s", name)
        except Exception as e:
            LOGGER.error("[register_all_discovery_agents] Failed to register %s: %s", name, e)
