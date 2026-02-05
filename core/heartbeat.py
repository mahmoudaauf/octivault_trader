import asyncio
import logging
import time
from datetime import datetime
from typing import Dict, Any, Iterable, Tuple, Optional
import inspect

try:
    # optional: use your helper if present (not strictly required)
    from core.stubs import maybe_await  # noqa: F401
except Exception:
    maybe_await = None  # unused here but harmless if imported

# Best-effort import with safe fallback for demos/tests
try:
    from core.component_status_logger import ComponentStatusLogger
except ImportError:
    class ComponentStatusLogger:
        _statuses = {}
        @staticmethod
        def log_status(component_name, status, detail=""):
            ComponentStatusLogger._statuses[component_name] = {
                "status": status, "detail": detail, "timestamp": datetime.now().isoformat()
            }
        @staticmethod
        def get_all_statuses():
            return ComponentStatusLogger._statuses.copy()

def _is_awaitable(x) -> bool:
    return inspect.isawaitable(x)

async def _csl_call(method: str, *args, **kwargs):
    """Call a CSL method (sync or async) safely. No exceptions leak."""
    try:
        fn = getattr(ComponentStatusLogger, method, None)
        if not callable(fn):
            return
        res = fn(*args, **kwargs)
        if _is_awaitable(res):
            await res
    except Exception:
        logger.debug("CSL.%s failed", method, exc_info=True)

logger = logging.getLogger("Heartbeat")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logger.addHandler(_h)
logger.setLevel(logging.INFO)

class Heartbeat:
    """
    Periodically:
      • updates its own status & timestamp
      • aggregates component statuses → single SystemOverall status
      • uses monotonic drift-free cadence
      • tolerates sync/async shared_state implementations
    """

    def __init__(
        self,
        shared_state,
        config=None,
        interval_seconds: Optional[float] = None,
        components: Optional[Iterable[str]] = None,
        **_,
    ):
        self.shared_state = shared_state
        self.interval = float(
            interval_seconds
            if interval_seconds is not None
            else getattr(
                config, "HEARTBEAT_INTERVAL",
                getattr(config, "HEARTBEAT_INTERVAL_SECONDS", 15.0)
            )
        )
        self.components = set(components or [])
        self._last_overall: Optional[Tuple[str, str]] = None  # ensure first consolidated line always prints
        self._stop_event = asyncio.Event()
        self._task = None

        # Seed initial status so Watchdog sees us immediately
        self._safe_set_component_status("Heartbeat", "Initialized", f"Ready. Interval={self.interval:.1f}s")
        # Mirror to CSL (don’t await in __init__; schedule)
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(_csl_call("log_status", "Heartbeat", "Initialized", "Ready to send periodic beats."))
        except RuntimeError:
            # If constructed before loop exists (tests), skip
            pass

        logger.info("Heartbeat initialized (interval=%.1fs).", self.interval)

    # ---------------- helpers (sync/async tolerant) ----------------

    def _safe_set_component_status(self, comp: str, status: str, detail: str = ""):
        """
        Sync path: prefer update_system_health (SharedState v1.9.3+),
        but tolerate older set/update_component_status signatures.
        If the target returns a coroutine, schedule it so we don't leak
        'coroutine was never awaited' warnings.
        """
        def _spawn_if_coro(result):
            if inspect.isawaitable(result):
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    # No running loop yet (e.g., very early init); bail quietly
                    return
                loop.create_task(result)

        # Prefer new API
        fn = getattr(self.shared_state, "update_system_health", None)
        if callable(fn):
            try:
                res = None
                try:
                    # Newer signature
                    res = fn(component=comp, status=status, message=detail)
                except TypeError:
                    # Older alias for the same field
                    res = fn(component=comp, status=status, detail=detail)
                _spawn_if_coro(res)
            except Exception:
                logger.debug("set_component_status via update_system_health failed.", exc_info=True)
            return

        # Fallback to legacy component status API
        fn2 = getattr(self.shared_state, "update_component_status", None) or \
              getattr(self.shared_state, "set_component_status", None)
        if callable(fn2):
            try:
                res = fn2(comp, status, detail)
                _spawn_if_coro(res)
            except Exception:
                logger.debug("legacy set_component_status failed.", exc_info=True)

    async def _safe_update_component_status(self, comp: str, status: str, detail: str = ""):
        """Async path: prefer update_system_health (supports both message/detail)."""
        fn = getattr(self.shared_state, "update_system_health", None)
        if callable(fn):
            try:
                if asyncio.iscoroutinefunction(fn):
                    await fn(component=comp, status=status, message=detail)
                else:
                    fn(component=comp, status=status, message=detail)
            except TypeError:
                if asyncio.iscoroutinefunction(fn):
                    await fn(component=comp, status=status, detail=detail)
                else:
                    fn(component=comp, status=status, detail=detail)
            except Exception:
                logger.debug("update_system_health failed.", exc_info=True)
            return

        # Fallback to component status API
        fn2 = getattr(self.shared_state, "update_component_status", None) or \
              getattr(self.shared_state, "set_component_status", None)
        if callable(fn2):
            if asyncio.iscoroutinefunction(fn2):
                await fn2(comp, status, detail)
            else:
                fn2(comp, status, detail)

    async def _safe_update_system_health(self, component: str, status: str, message: str):
        """
        Support both signatures used across the codebase:
          - update_system_health(component, status, message=...)
          - update_system_health(component, status, detail=...)
        """
        fn = getattr(self.shared_state, "update_system_health", None)
        if not callable(fn):
            return
        try:
            if asyncio.iscoroutinefunction(fn):
                await fn(component=component, status=status, message=message)
            else:
                fn(component=component, status=status, message=message)
        except TypeError:
            if asyncio.iscoroutinefunction(fn):
                await fn(component=component, status=status, detail=message)
            else:
                fn(component=component, status=status, detail=message)
        except Exception:
            logger.debug("update_system_health failed.", exc_info=True)

    async def _safe_update_timestamp(self, component: str):
        fn = getattr(self.shared_state, "update_timestamp", None)
        if not callable(fn):
            return
        try:
            if asyncio.iscoroutinefunction(fn):
                await fn(component)
            else:
                fn(component)
        except TypeError:
            if asyncio.iscoroutinefunction(fn):
                await fn(component_name=component)
            else:
                fn(component_name=component)
        except Exception:
            logger.debug("update_timestamp failed.", exc_info=True)

    def _collect_statuses(self) -> Dict[str, Dict[str, Any]]:
        """
        Merge statuses from SharedState and ComponentStatusLogger.
        Prefer newer timestamps; downgrade very stale entries (> TTL) to Unresponsive.
        This prevents a stale 'Degraded' from poisoning the overall summary.
        """
        def _coerce_dict(v):
            return dict(v) if isinstance(v, dict) else {"status": str(v)}

        def _parse_ts(ts_val) -> float:
            # Accept float epoch seconds or ISO strings
            try:
                return float(ts_val)
            except Exception:
                pass
            try:
                # e.g., "2025-10-03T22:47:27.675123+00:00" or "2025-10-03T22:47:27Z"
                return datetime.fromisoformat(str(ts_val).replace("Z", "+00:00")).timestamp()
            except Exception:
                return 0.0

        merged: Dict[str, Dict[str, Any]] = {}

        # 1) SharedState.component_statuses (preferred live map)
        try:
            ss_map = getattr(self.shared_state, "component_statuses", None)
            if isinstance(ss_map, dict):
                for comp, info in ss_map.items():
                    merged[comp] = _coerce_dict(info)
        except Exception:
            pass

        # 2) SharedState.get_all_component_statuses() (if sync)
        try:
            get_all = getattr(self.shared_state, "get_all_component_statuses", None)
            if callable(get_all) and not asyncio.iscoroutinefunction(get_all):
                res = get_all() or {}
                for comp, info in res.items():
                    cur = merged.get(comp, {})
                    if _parse_ts((info or {}).get("timestamp")) >= _parse_ts(cur.get("timestamp")):
                        merged[comp] = _coerce_dict(info)
        except Exception:
            pass

        # 3) ComponentStatusLogger aggregate (fallback/parallel map)
        try:
            csl_all = ComponentStatusLogger.get_all_statuses() or {}
            for comp, info in csl_all.items():
                cur = merged.get(comp, {})
                if _parse_ts((info or {}).get("timestamp")) >= _parse_ts(cur.get("timestamp")):
                    merged[comp] = _coerce_dict(info)
        except Exception:
            pass

        # 4) Downgrade very stale entries so they don't show as 'Degraded' forever
        ttl = getattr(ComponentStatusLogger, "_ttl_seconds", 120)
        now = time.time()
        for comp, info in list(merged.items()):
            ts = _parse_ts(info.get("timestamp"))
            if ts and (now - ts) > float(ttl):
                info = {**info, "status": "Unresponsive", "stale": True}
                merged[comp] = info

        return merged

    @staticmethod
    def _summarize(statuses: Dict[str, Dict[str, Any]], restrict: Optional[Iterable[str]] = None) -> Tuple[str, str]:
        """
        Returns (overall_status, detail).
        Priority: Critical/Critical Failure > Error/Degraded > Unknown > Healthy.
        Includes per-component 'detail'/'message' in the Issues string for better diagnostics.
        """
        care = set(restrict or statuses.keys())
        problems = []
        any_crit = any_err = any_unknown = False

        for name, info in statuses.items():
            if name not in care:
                continue
            st = str(info.get("status", "Unknown"))
            detail = str(info.get("detail") or info.get("message") or "").strip()
            tag = f"{name}:{st}" + (f" ({detail})" if detail else "")
            if st in {"Critical", "Critical Failure"}:
                any_crit = True
                problems.append(tag)
            elif st in {"Error", "Degraded"}:
                any_err = True
                problems.append(tag)
            elif st == "Unknown":
                any_unknown = True

        if any_crit:
            return "Critical", f"Issues: {', '.join(problems)}"
        if any_err:
            return "Degraded", f"Issues: {', '.join(problems)}"
        if any_unknown and not problems:
            return "Running", "Some components are Unknown; monitoring."
        return "Operational", "All systems nominal."

    # ---------------- core beat ----------------

    async def _send_beat(self):
        statuses = self._collect_statuses()

        # If user supplied a component subset, keep it, otherwise include everything we see.
        if not self.components:
            self.components = set(statuses.keys()) or {"Heartbeat"}

        overall_status, overall_detail = self._summarize(statuses, self.components)

        # Log only if changed to reduce noise
        if (overall_status, overall_detail) != self._last_overall:
            logger.info("❤️ Heartbeat: %s — %s", overall_status, overall_detail)
            self._last_overall = (overall_status, overall_detail)

        # Update timestamps & statuses visible to Watchdog
        await self._safe_update_timestamp("Heartbeat")
        await self._safe_update_component_status("Heartbeat", "Operational", overall_detail)
        await self._safe_update_system_health("SystemOverall", overall_status, overall_detail)

        # Mirror to the UI/status board (CSL; sync or async)
        await _csl_call("log_status", "Heartbeat", "Operational", overall_detail)

    # ---------------- run loops ----------------

    async def run_loop(self):
        """
        Drift-free scheduler: beats every `interval` seconds by monotonic clock.
        """
        # Mirror to CSL (UI/status board)
        await _csl_call("log_status", "Heartbeat", "Running", "Periodic beats activated.")
        await self._safe_update_component_status("Heartbeat", "Running", "Periodic beats activated.")

        interval = max(0.5, float(self.interval))
        next_tick = time.monotonic()
        try:
            while not self._stop_event.is_set():
                await self._send_beat()
                # drift-free sleep
                next_tick += interval
                now = time.monotonic()
                if next_tick > now:
                    await asyncio.sleep(next_tick - now)
        except asyncio.CancelledError:
            logger.info("Heartbeat loop cancelled.")
            raise
        except Exception as e:
            logger.error("Heartbeat error: %s", e, exc_info=True)
            await _csl_call("log_status", "Heartbeat", "Error", f"Run loop error: {e}")
            await self._safe_update_component_status("Heartbeat", "Error", str(e))
            await self._safe_update_system_health("SystemOverall", "Degraded", f"Heartbeat failure: {e}")
            raise

    async def start(self):
        """
        P9 contract: start() creates the heartbeat task and emits an initial status.
        """
        if getattr(self, "_task", None) and not self._task.done():
            return
        self._stop_event.clear()
        self._task = asyncio.create_task(self.run_loop(), name="ops.heartbeat")
        try:
            await self._safe_update_component_status("Heartbeat", "Initialized", f"Ready. Interval={self.interval:.1f}s")
        except Exception:
            logger.debug("Heartbeat initial health update failed", exc_info=True)

    async def stop(self):
        """
        P9 contract: stop() requests the loop to end and waits for it, then emits final status.
        """
        self._stop_event.set()
        t = getattr(self, "_task", None)
        self._task = None
        if t:
            try:
                t.cancel()
                try:
                    await asyncio.wait_for(t, timeout=float(getattr(getattr(self, "config", None), "STOP_JOIN_TIMEOUT_S", 5.0)))
                except asyncio.CancelledError:
                    pass
            except Exception:
                logger.debug("Heartbeat stop wait failed", exc_info=True)
        try:
            await self._safe_update_component_status("Heartbeat", "Stopped", "Stopped by request")
            await _csl_call("log_status", "Heartbeat", "Stopped", "Stopped by request")
        except Exception:
            logger.debug("Heartbeat final health update failed", exc_info=True)

    async def run(self):  # alias
        await self.run_loop()


def spawn_heartbeat(shared_state, config=None, interval_seconds: Optional[float] = None, components: Optional[Iterable[str]] = None):
    """
    Convenience spawner: schedules hb.start() so we emit Initialized/Running properly
    and avoid un-awaited coroutine warnings.
    """
    hb = Heartbeat(shared_state, config=config, interval_seconds=interval_seconds, components=components)
    asyncio.get_event_loop().create_task(hb.start())
    return hb
