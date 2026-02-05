import asyncio
import logging
import time
import inspect
from typing import Dict, Tuple, Optional, Any, List

logger = logging.getLogger("Watchdog")

# Healthy status values (lowercase)
_HEALTHY = {"operational", "running", "initialized", "active", "healthy", "ok"}

def _is_healthy(s: str) -> bool:
    try:
        return str(s).strip().lower() in _HEALTHY
    except Exception:
        return False


def _is_coro_fn(fn) -> bool:
    return inspect.iscoroutinefunction(fn)


def _schedule_coro(coro) -> None:
    """Best-effort schedule of a coroutine without raising if no loop is running."""
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(coro)
    except RuntimeError:
        # No running loop (e.g., unit tests instantiating before loop); ignore.
        pass


class Watchdog:
    """
    Internal auditor:
      • checks component freshness & status on a cadence
      • concurrent per-component checks
      • monotonic grace windows via config multipliers
      • emits a single summary status per cycle
    """

    def __init__(self, check_interval_seconds: Optional[float] = None, config=None, shared_state=None, **kwargs):
        """
        P9-compatible constructor with backward-compat support.
        Accepts either the legacy positional form (check_interval_seconds, config, shared_state)
        or keyword-based calls such as:
            Watchdog(shared_state=..., interval_sec=30.0)
            Watchdog(shared_state=..., check_sec=30.0)
        # =============================
        # Globals & Utilities
        # =============================
        logger = logging.getLogger("Watchdog")

        _HEALTHY = {"operational", "running", "initialized", "active", "healthy", "ok"}

        """
        # Map legacy/new keyword aliases if provided
        # Prefer explicit keyword args over positional defaults
        if shared_state is None and "shared_state" in kwargs:
            shared_state = kwargs.pop("shared_state")
        if config is None and "config" in kwargs:
            config = kwargs.pop("config")

        # interval can come from: interval_sec > check_sec > check_interval_seconds > config > default
        interval_sec = kwargs.pop("interval_sec", None)
        check_sec = kwargs.pop("check_sec", None)
        interval_candidate = interval_sec if interval_sec is not None else (
            check_sec if check_sec is not None else check_interval_seconds
        )

        # Interval from arg or config, with sane defaults
        if interval_candidate is None:
        # =============================
        # Watchdog Class
        # =============================
            interval_candidate = float(getattr(config, "WATCHDOG_CHECK_INTERVAL", 30.0) or 30.0)

        # Tolerance: multiplier and optional cap
        tol_mult = float(getattr(config, "WATCHDOG_TOLERANCE_MULTIPLIER", 3.0) or 3.0)
        tol_cap  = getattr(config, "WATCHDOG_MAX_TOLERANCE_SEC", None)
        tolerance_time_seconds = float(interval_candidate) * tol_mult
        if tol_cap is not None:
            try:
                tolerance_time_seconds = min(float(tolerance_time_seconds), float(tol_cap))
            except Exception:
                pass

        self.check_interval: float = float(interval_candidate)
        # Back-compat alias retained
        self.check_interval_seconds: float = float(interval_candidate)
        self.tolerance_time_seconds: float = float(tolerance_time_seconds)
        self.config = config
        self.shared_state = shared_state
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(
            "✅ Watchdog initialized (interval=%.1fs, tolerance=%.1fs).",
            self.check_interval, self.tolerance_time_seconds
        )

        self._stop_event = asyncio.Event()
        self._task = None

        # Allow override from config; include both TPSLEngine spellings for safety
        default_components = [
            "MarketDataFeed",
            "ExecutionManager",
            "MetaController",
            "AgentManager",
            "RiskManager",
            "PnLCalculator",
            "PerformanceEvaluator",
            "TPSLEngine",   # canonical
            "TP_SLEngine",  # legacy alias
            "Heartbeat",
        ]
        conf_list = getattr(config, "WATCHDOG_COMPONENTS", None)
        self.monitored_components: List[str] = list(conf_list) if conf_list else default_components

        # Track last time a component was confirmed healthy
        self.last_healthy_report: Dict[str, float] = {}

        self.warn_cooldown_seconds: float = float(getattr(self.config, "WATCHDOG_WARN_COOLDOWN_SEC", 60.0) or 60.0)
        self._last_warn_time: Dict[str, float] = {}

        # Mark as initialized (best-effort; supports sync or async impls)
        try:
            self._safe_update_component_status_sync(
                "Watchdog", "Initialized", "Ready to monitor components."
            )
        except Exception:
            logger.debug("Watchdog: initial component status set failed (non-fatal).", exc_info=True)

    async def start(self):
        """
        P9 contract: start() spawns the periodic health loop once (idempotent).
        """
        if getattr(self, "_task", None) and not self._task.done():
            return
        if hasattr(self, "_stop_event"):
            self._stop_event.clear()
        self._task = asyncio.create_task(self.run(), name="ops.watchdog")
        try:
            await self._safe_update_component_status("Watchdog", "Initialized", "Ready")
        except Exception:
            logger.debug("Watchdog initial health update failed", exc_info=True)

    async def stop(self):
        """
        P9 contract: stop() requests the loop to end and waits for it.
        """
        if hasattr(self, "_stop_event"):
            self._stop_event.set()
        t = getattr(self, "_task", None)
        self._task = None
        if t:
            try:
                t.cancel()
                try:
                    await asyncio.wait_for(t, timeout=float(getattr(self.config, "STOP_JOIN_TIMEOUT_S", 5.0)))
                except asyncio.CancelledError:
                    pass
            except Exception:
                logger.debug("Watchdog stop wait failed", exc_info=True)
        try:
            await self._safe_update_component_status("Watchdog", "Stopped", "Stopped by request")
        except Exception:
            logger.debug("Watchdog final health update failed", exc_info=True)

    # -------- shared_state helpers (sync/async tolerant) --------

    def _safe_status(self, name: str) -> Dict[str, Any]:
        """
        Retrieve a component status snapshot from SharedState safely.
        Expected keys: {'timestamp': epoch_seconds, 'status': str, 'detail': str}
        """
        si: Optional[Dict[str, Any]] = None

        # Preferred: component-status snapshot (Phase 9 store: _component_status)
        try:
            snap_fn = getattr(self.shared_state, "get_component_status_snapshot", None)
            if callable(snap_fn):
                # If someone made this async in their build, don't await from sync path.
                snap = snap_fn() if not _is_coro_fn(snap_fn) else {}
                if isinstance(snap, dict):
                    si = snap.get(name)
                    if not si and name == "TP_SLEngine":
                        si = snap.get("TPSLEngine")
        except Exception:
            si = None

        # Fallback: system-health mirror (component_statuses: {name: {status, message, timestamp, ts}})
        if not si:
            try:
                mirror = getattr(self.shared_state, "component_statuses", {}) or {}
                if isinstance(mirror, dict):
                    si = mirror.get(name)
                    if not si and name == "TP_SLEngine":
                        si = mirror.get("TPSLEngine")
            except Exception:
                si = None

        # Normalize structure to {timestamp, status, detail}
        out: Dict[str, Any] = {}
        if isinstance(si, dict):
            ts = si.get("timestamp", si.get("ts", 0.0)) or 0.0
            status = si.get("status", "Unknown") or "Unknown"
            # Prefer 'detail' (component store). Otherwise map 'message' (system mirror) → detail.
            detail = si.get("detail")
            if detail is None:
                detail = si.get("message", "")
            out.update({"timestamp": float(ts), "status": str(status), "detail": str(detail)})
        else:
            out.update({"timestamp": 0.0, "status": "Unknown", "detail": ""})

        # Fallback: if no timestamp present, try alternate last-seen mirrors
        try:
            if float(out.get("timestamp", 0.0) or 0.0) == 0.0:
                fallback_ts = None
                # Preferred getter
                get_ls = getattr(self.shared_state, "get_last_seen", None)
                if callable(get_ls) and not _is_coro_fn(get_ls):
                    val = get_ls(name)
                    if isinstance(val, (int, float)):
                        fallback_ts = float(val)
                # Common dict mirrors
                if fallback_ts is None:
                    for attr in ("last_seen", "timestamps", "component_last_seen"):
                        store = getattr(self.shared_state, attr, None)
                        if isinstance(store, dict) and name in store:
                            v = store.get(name)
                            if isinstance(v, (int, float)):
                                fallback_ts = float(v)
                                break
                if fallback_ts is not None:
                    out["timestamp"] = fallback_ts
        except Exception:
            pass
        return out

    async def _safe_update_component_status(self, comp: str, status: str, detail: str):
        """
        Call shared_state.update_component_status / set_component_status (async or sync).
        Also tolerant to newer update_system_health signature (message/detail).
        """
        # Prefer dedicated component API
        fn = getattr(self.shared_state, "update_component_status", None) or \
             getattr(self.shared_state, "set_component_status", None)
        if callable(fn):
            if _is_coro_fn(fn):
                await fn(comp, status, detail)
            else:
                fn(comp, status, detail)
            return

        # Fallback to system-health API
        uh = getattr(self.shared_state, "update_system_health", None)
        if callable(uh):
            try:
                if _is_coro_fn(uh):
                    await uh(component=comp, status=status, message=detail)
                else:
                    uh(component=comp, status=status, message=detail)
            except TypeError:
                # Some builds expect 'detail='
                if _is_coro_fn(uh):
                    await uh(component=comp, status=status, detail=detail)
                else:
                    uh(component=comp, status=status, detail=detail)

    def _should_warn(self, comp: str) -> bool:
        """Return True if it's OK to emit a warning for this component (cooldown-aware)."""
        try:
            now = time.time()
            last = float(self._last_warn_time.get(comp, 0.0) or 0.0)
            if now - last >= float(self.warn_cooldown_seconds):
                self._last_warn_time[comp] = now
                return True
        except Exception:
            pass
        return False
    def _safe_update_component_status_sync(self, comp: str, status: str, detail: str):
        """
        Synchronous helper (used at init); no await context required.
        If the target API is async, schedule it on the running loop when available.
        """
        fn = getattr(self.shared_state, "update_component_status", None) or \
             getattr(self.shared_state, "set_component_status", None)
        if callable(fn):
            try:
                if _is_coro_fn(fn):
                    _schedule_coro(fn(comp, status, detail))
                else:
                    fn(comp, status, detail)
                return
            except Exception:
                logger.debug("component_status sync update failed.", exc_info=True)
        uh = getattr(self.shared_state, "update_system_health", None)
        if callable(uh):
            try:
                if _is_coro_fn(uh):
                    _schedule_coro(uh(component=comp, status=status, message=detail))
                else:
                    uh(component=comp, status=status, message=detail)
            except TypeError:
                try:
                    if _is_coro_fn(uh):
                        _schedule_coro(uh(component=comp, status=status, detail=detail))
                    else:
                        uh(component=comp, status=status, detail=detail)
                except Exception:
                    logger.debug("system_health sync update failed.", exc_info=True)
            except Exception:
                logger.debug("system_health sync update failed.", exc_info=True)

    async def _emit_system_halt(self, reason: str):
        """Emit a SYSTEM_HALT event if the API exists (sync or async)."""
        emit_event = getattr(self.shared_state, "emit_event", None)
        if callable(emit_event):
            if _is_coro_fn(emit_event):
                await emit_event("SYSTEM_HALT", {"reason": reason})
            else:
                emit_event("SYSTEM_HALT", {"reason": reason})

    # -------- core check --------

    async def _check_component_health(self, component_name: str) -> Tuple[str, bool, Optional[str]]:
        """
        Check one component and (best-effort) annotate Watchdog status when needed.
        Returns: (component_name, is_healthy, problem_label_or_None)
        """
        status_info = self._safe_status(component_name)
        now = time.time()

        last_ts = float(status_info.get("timestamp", 0.0) or 0.0)
        status  = str(status_info.get("status", "Unknown") or "Unknown")
        detail  = str(status_info.get("detail", "") or "")
        stale_s = now - last_ts if last_ts > 0 else None

        # Case 1: never reported
        if last_ts == 0.0:
            if self._should_warn(component_name):
                logger.warning("🚨 Watchdog: '%s' has not reported any status yet.", component_name)
                await self._safe_update_component_status(
                    "Watchdog", "Warning",
                    f"'{component_name}' status unknown / not yet reported."
                )
            return component_name, False, "no-report"

        # Case 2: current status string evaluation
        healthy_string = _is_healthy(status)

        # Special-case: status is "Unknown" but timestamp is present and not stale → warn, not error
        if not healthy_string and status == "Unknown" and last_ts > 0.0 and (stale_s is None or stale_s <= self.tolerance_time_seconds):
            if self._should_warn(component_name):
                logger.warning(
                    "⚠️ Watchdog: '%s' has timestamp but no status; treating as Degraded until status arrives.",
                    component_name
                )
                await self._safe_update_component_status(
                    "Watchdog", "Warning",
                    f"'{component_name}' timestamp present but no status."
                )
            return component_name, False, "no-status"

        # Case 3: freshness check
        if healthy_string:
            # record last known healthy moment
            self.last_healthy_report[component_name] = now

            if stale_s is not None and stale_s > self.tolerance_time_seconds:
                if self._should_warn(component_name):
                    logger.warning(
                        "⚠️ Watchdog: '%s' marked '%s' but last report is %.1fs old (tolerating up to %.1fs).",
                        component_name, status, stale_s, self.tolerance_time_seconds
                    )
                    await self._safe_update_component_status(
                        "Watchdog", "Warning",
                        f"'{component_name}' potentially unresponsive. Last report {stale_s:.1f}s ago."
                    )
                return component_name, False, "stale"
            return component_name, True, None

        # Non-healthy string → emit error and consider escalation
        logger.error("❌ Watchdog: '%s' reported '%s'. Detail: %s", component_name, status, detail)
        await self._safe_update_component_status("Watchdog", "Error", f"'{component_name}' reported {status}: {detail}")

        if status in {"Critical", "Critical Failure"}:
            logger.critical("🛑 Watchdog: Critical failure in '%s'. Requesting SYSTEM_HALT.", component_name)
            await self._emit_system_halt(f"Critical failure in {component_name}")

        # Case 4: prolonged absence of healthy reports
        last_ok = self.last_healthy_report.get(component_name)
        if last_ok is not None:
            no_healthy_for = now - last_ok
            if no_healthy_for > (self.tolerance_time_seconds * 2.0):
                logger.critical(
                    "💔 Watchdog: '%s' not healthy for %.1fs; may be stuck/crashed.",
                    component_name, no_healthy_for
                )
                await self._safe_update_component_status(
                    "Watchdog", "Critical",
                    f"'{component_name}' unresponsive/unhealthy for {no_healthy_for:.1f}s."
                )
        return component_name, False, status

    # -------- main loops --------

    async def run(self):
        """
        Periodic health checks with per-cycle summary emission.
        """
        logger.info("Watchdog started. Beginning periodic health checks.")
        await self._safe_update_component_status("Watchdog", "Running", "Monitoring activated.")

        try:
            while not getattr(self, "_stop_event", asyncio.Event()).is_set():
                # Concurrent per-component checks
                results = await asyncio.gather(
                    *(self._check_component_health(c) for c in self.monitored_components),
                    return_exceptions=True
                )

                overall = "Operational"
                problems: List[str] = []

                for comp, res in zip(self.monitored_components, results):
                    if isinstance(res, Exception):
                        logger.error("Watchdog: exception while checking '%s': %s", comp, res, exc_info=True)
                        overall = "Degraded"
                        problems.append(f"{comp} (check-error)")
                        continue

                    _name, healthy, problem = res
                    if not healthy:
                        overall = "Degraded"
                        problems.append(f"{_name} ({problem})")

                if overall == "Operational":
                    await self._safe_update_component_status("Watchdog", "Operational", "All monitored components are healthy.")
                    logger.debug("Watchdog: All monitored components healthy.")
                else:
                    msg = f"System health issues detected in: {', '.join(problems)}."
                    await self._safe_update_component_status("Watchdog", overall, msg)
                    logger.warning("Watchdog overall health: %s. %s", overall, msg)

                await asyncio.sleep(self.check_interval)
        except asyncio.CancelledError:
            logger.info("Watchdog loop cancelled.")
            raise

    async def run_loop(self):
        """Phase 9 compatibility: delegate to main loop."""
        await self.run()
