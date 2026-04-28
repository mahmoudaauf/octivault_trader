# core/utils/component_status_logger.py (canonical)
# Provides both class-based and functional APIs for component status logging.

__all__ = [
    "ComponentStatusLogger",
    "log_component_status",
    "bind_shared_state",
    "get_status",
    "get_all_statuses",
    "log_heartbeat",
    "clear_statuses",
]

import logging
import re
import asyncio
import time
import datetime
import threading
import inspect
import sys # Used for basic logging fallback if no handlers are configured
from typing import Any, Optional, Callable

class ComponentStatusLogger:
    """
    The 'Operational Status Board' for the Octivault AI Office.
    This singleton class provides a centralized mechanism for components to report
    their real-time status and for other monitoring components (like Heartbeat
    and Watchdog) to query current system health.
    """
    _instance = None # The singleton instance
    _status_cache = {} # Cache for last reported status of each component.
    _status_lock = threading.RLock() # Protect cache for cross-thread usage
    _last_emit = {} # {(comp, status, detail): last_ts} for deduplication
    _shared_state = None # Optional mirror target for statuses

    # Defaults (configurable at runtime)
    _DEFAULT_TTL_SECONDS = 120
    _DEFAULT_DEDUPE_WINDOW_SEC = 5

    # Effective values (can be changed via configure())
    _ttl_seconds = _DEFAULT_TTL_SECONDS  # Mark status stale after this many seconds
    _dedupe_window_sec = _DEFAULT_DEDUPE_WINDOW_SEC  # Throttle identical logs within this window
    @classmethod
    def configure(cls, ttl_seconds: Optional[int] = None, dedupe_window_sec: Optional[int] = None):
        """
        Optional runtime configuration of staleness TTL and dedupe window.
        If a value is None, it is left unchanged.
        """
        if ttl_seconds is not None:
            try:
                cls._ttl_seconds = int(ttl_seconds)
            except Exception:
                pass
        if dedupe_window_sec is not None:
            try:
                cls._dedupe_window_sec = int(dedupe_window_sec)
            except Exception:
                pass

    def __init__(self, logger=None):
        """
        Initializes the ComponentStatusLogger instance.
        If no logger is provided, it defaults to a named logger.
        This constructor is typically called only once via the get() method.
        """
        self.logger = logger or logging.getLogger("ComponentStatusLogger")
        # Ensure the logger has at least a basic handler if none are configured globally
        if not logging.getLogger().handlers and not self.logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        if self.logger.level == logging.NOTSET:
            self.logger.setLevel(logging.INFO)

        self.logger.info("📊 ComponentStatusLogger (Operational Status Board) initialized.")

    @classmethod
    def get(cls):
        """
        Retrieves the singleton instance of the ComponentStatusLogger.
        Ensures that only one instance exists and is properly initialized.
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # Back-compat alias used elsewhere in the codebase
    @classmethod
    def instance(cls):
        return cls.get()

    @classmethod
    def bind_shared_state(cls, shared_state):
        """
        Optional: Binds an external SharedState object to mirror component statuses.
        This method should be called once during application bootstrap.
        """
        cls._shared_state = shared_state
        cls.get().logger.info("🔗 ComponentStatusLogger bound to SharedState for mirroring.")

    @staticmethod
    def _strip_emojis(text: str) -> str:
        """
        Removes emojis from a string to prevent potential encoding issues and
        ensure consistent display across different logging environments.
        """
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags (iOS)
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "\U0001F900-\U0001F9FF"  # supplemental symbols and pictographs
            "\U0001FA00-\U0001FA6F"  # chess, etc.
            "\U0001FA70-\U0001FAFF"  # pictographs ext-A
            "]+",
            flags=re.UNICODE
        )
        return emoji_pattern.sub(r'', text)

    @classmethod
    def _is_duplicate_emit(cls, component_name: str, status: str, detail: str, now_ts: float) -> bool:
        """
        Checks if the exact status update for a component has been logged very recently
        to prevent excessive logging of identical messages.
        """
        key = (component_name, status, detail)
        last = cls._last_emit.get(key, 0.0)
        if (now_ts - last) <= cls._dedupe_window_sec:
            return True
        cls._last_emit[key] = now_ts
        return False

    def _mirror_to_shared_state(self, component: str, status: str, detail: str):
        """
        Mirror status into SharedState. Works with or without a running loop.
        Prefers component-scoped status API; only falls back if unavailable or fails.
        """
        ss = self._shared_state
        if not ss:
            return

        target: Optional[Callable[..., Any]] = None
        if hasattr(ss, "set_component_status"):
            target = getattr(ss, "set_component_status")
        elif hasattr(ss, "update_component_status"):
            target = getattr(ss, "update_component_status")

        async def _async_mirror():
            # try component-scoped first
            if target:
                try:
                    wants_ts = False
                    try:
                        sig = inspect.signature(target)
                        wants_ts = "timestamp" in sig.parameters
                    except Exception:
                        wants_ts = False
                    args = (component, status, detail)
                    kwargs = {"timestamp": time.time()} if wants_ts else {}
                    if asyncio.iscoroutinefunction(target):
                        await target(*args, **kwargs)
                    else:
                        target(*args, **kwargs)
                    return
                except Exception as e:
                    self.logger.debug("Component-status mirror failed; will fallback: %s", e, exc_info=True)

            # fallback path
            emit = getattr(ss, "emit_event", None) or getattr(ss, "emit", None)
            payload = {
                "component": component,
                "status": status,
                "message": detail,
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            }
            if callable(emit):
                try:
                    if asyncio.iscoroutinefunction(emit):
                        await emit("HealthStatus", payload)
                    else:
                        emit("HealthStatus", payload)
                    return
                except Exception as e:
                    self.logger.debug("HealthStatus emit failed; trying update_system_health: %s", e, exc_info=True)

            ush = getattr(ss, "update_system_health", None)
            if callable(ush):
                try:
                    if asyncio.iscoroutinefunction(ush):
                        await ush(component=component, status=status, message=detail)
                    else:
                        ush(component=component, status=status, message=detail)
                except TypeError:
                    if asyncio.iscoroutinefunction(ush):
                        await ush(component=component, status=status, detail=detail)
                    else:
                        ush(component=component, status=status, detail=detail)

        try:
            loop = asyncio.get_running_loop()
            loop.create_task(_async_mirror())
        except RuntimeError:
            if target and not asyncio.iscoroutinefunction(target):
                try:
                    target(component, status, detail)
                    return
                except Exception as e:
                    self.logger.debug("Sync component-status mirror failed; will fallback: %s", e, exc_info=True)
            emit = getattr(ss, "emit_event", None) or getattr(ss, "emit", None)
            payload = {
                "component": component,
                "status": status,
                "message": detail,
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            }
            if callable(emit) and not asyncio.iscoroutinefunction(emit):
                try:
                    emit("HealthStatus", payload)
                    return
                except Exception as e:
                    self.logger.debug("Sync HealthStatus emit failed; trying update_system_health: %s", e, exc_info=True)
            ush = getattr(ss, "update_system_health", None)
            if callable(ush) and not asyncio.iscoroutinefunction(ush):
                try:
                    ush(component=component, status=status, message=detail)
                except TypeError:
                    ush(component=component, status=status, detail=detail)
                except Exception as e:
                    self.logger.debug("Sync system-health mirror failed: %s", e, exc_info=True)

    # ---------- Public API (Class-level) ----------
    @classmethod
    async def heartbeat(cls, component: str, detail: str = "OK"):
        """
        Async heartbeat as per spec:
         1) logs a 'Running' status locally
         2) emits a HealthStatus event to SharedState if available
        """
        # 1) Local status log (dedup/throttle respected)
        cls.log_status(component, "Running", detail)

        # 2) Emit HealthStatus to SharedState (prefer emit_event; fallback to update_system_health)
        ss = cls._shared_state
        if not ss:
            return
        try:
            payload = {
                "component": component,
                "status": "Running",
                "message": detail,   # P9 canonical
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
            }
            emit = getattr(ss, "emit_event", None) or getattr(ss, "emit", None)
            if callable(emit):
                if asyncio.iscoroutinefunction(emit):
                    await emit("HealthStatus", payload)
                else:
                    emit("HealthStatus", payload)
                return

            # Fallback path for older builds
            ush = getattr(ss, "update_system_health", None)
            if callable(ush):
                try:
                    if asyncio.iscoroutinefunction(ush):
                        await ush(component=component, status="Running", message=detail)
                    else:
                        ush(component=component, status="Running", message=detail)
                except TypeError:
                    if asyncio.iscoroutinefunction(ush):
                        await ush(component=component, status="Running", detail=detail)
                    else:
                        ush(component=component, status="Running", detail=detail)
        except Exception as e:
            cls.get().logger.debug("Heartbeat SharedState emit failed: %s", e, exc_info=True)

    @classmethod
    def log_status(cls, component: str = None, status: str = "", detail: str = "", level: int = logging.INFO, **kwargs):
        """
        Logs the status of a component and updates the internal cache.
        Accepts both modern and legacy keyword names for the component:
        component_name | name | component.
        """
        comp = component or kwargs.get("component_name") or kwargs.get("name") or kwargs.get("component")
        if not comp:
            cls.get().logger.warning("ComponentStatusLogger.log_status called without a component name.")
            comp = "UnknownComponent"
        return cls.get()._log_status(comp, status, detail, level)

    @classmethod
    def set_component_status(cls, component: str = None, status: str = "", detail: str = "", **kwargs):
        """
        Back-compat alias for log_status. Accepts component_name|component|name.
        """
        # Resolve component name from various possible keyword arguments
        comp = component or kwargs.get("component_name") or kwargs.get("component") or kwargs.get("name")
        if not comp:
            cls.get().logger.warning(
                "ComponentStatusLogger received status update with no component name."
            )
            comp = "UnknownComponent" # Default if no component name is provided
        return cls.log_status(comp, status, detail)

    # Additional back-compat alias: some callers use set_status(...)
    @classmethod
    def set_status(cls, component: str = None, status: str = "", detail: str = "", **kwargs):
        comp = component or kwargs.get("component_name") or kwargs.get("name") or kwargs.get("component")
        if not comp:
            comp = "UnknownComponent"
        return cls.log_status(comp, status, detail)

    @classmethod
    def report_status(cls, component: str, status: str, detail: str = ""):
        """Legacy alias used in some code paths."""
        return cls.log_status(component, status, detail)

    # ---------- Internal Core Logic (Instance-level) ----------
    def _log_status(self, component: str, status: str, detail: str = "", level: int = logging.INFO):
        """
        The internal method that performs the actual logging and cache update.
        Called by the class-level log_status method.
        """
        current_timestamp = time.time()
        detail_stripped = self._strip_emojis(detail)

        is_dup = type(self)._is_duplicate_emit(component, status, detail_stripped, current_timestamp)
        msg = f"[{component}] Status: {status} | Detail: {detail_stripped}"

        try:
            if is_dup:
                self.logger.debug(msg)
            else:
                self.logger.log(level, msg)
        except Exception as e:
            # Fallback logging if the main logger has issues
            logging.getLogger("ComponentStatusLoggerFallback").warning(
                f"Status log failed for {component}: {type(e).__name__}: {e}; msg={msg}"
            )

        with type(self)._status_lock: # Access class-level lock
            type(self)._status_cache[component] = { # Access class-level cache
                "timestamp": current_timestamp,
                "status": status,
                "detail": detail_stripped
            }

        self._mirror_to_shared_state(component, status, detail_stripped)
        with type(self)._status_lock:
            return type(self)._status_cache.get(component, {
                "timestamp": current_timestamp,
                "status": status,
                "detail": detail_stripped
            })
    @classmethod
    def get_stale_components(cls, since_seconds: Optional[int] = None) -> dict:
        """
        Returns a dict {component: data} for components whose last update is older than
        'since_seconds' (default: current TTL). Does not mutate internal cache.
        """
        threshold = float(since_seconds if since_seconds is not None else cls._ttl_seconds)
        now = time.time()
        with cls._status_lock:
            snapshot = cls._status_cache.copy()
        out = {}
        for comp, data in snapshot.items():
            age = now - float(data.get("timestamp", 0))
            if age > threshold:
                out[comp] = {**data, "stale": True, "age_seconds": age}
        return out

    # Instance-level back-compat for code that does: ComponentStatusLogger.instance().set_status(...)
    def set_status(self, component: str, status: str, detail: str = ""):
        return self._log_status(component, status, detail)

    # ---------- Getters (Class-level, access shared state) ----------
    @classmethod
    def get_status(cls, component_name: str) -> dict:
        """
        Retrieves the last reported status of a specific component.
        Returns a dictionary with 'timestamp', 'status', 'detail', and 'stale'.
        If the component has not reported, returns default 'Unknown' values.
        """
        with cls._status_lock:
            data = cls._status_cache.get(component_name)
        if not data:
            return {"timestamp": 0, "status": "Unknown", "detail": "No status reported yet.", "stale": True}

        age = max(0.0, time.time() - float(data.get("timestamp", 0)))
        stale = age > cls._ttl_seconds
        if stale:
            # Return a derived view without modifying the cached data
            derived_status = "Unresponsive" if data.get("status") != "Stopped" else "Stopped"
            return {
                **data,
                "stale": True,
                "status": derived_status
            }
        return {**data, "stale": False}

    @classmethod
    def get_all_statuses(cls) -> dict:
        """
        Returns a copy of the current status cache containing all reported components,
        with each status annotated with its staleness.
        """
        with cls._status_lock:
            snapshot = cls._status_cache.copy()
        now = time.time()
        out = {}
        for k, v in snapshot.items():
            age = now - float(v.get("timestamp", 0))
            stale = age > cls._ttl_seconds
            vv = {**v, "stale": stale}
            if stale and vv.get("status") != "Stopped":
                vv["status"] = "Unresponsive" # Indicate unresponsiveness if stale and not explicitly stopped
            out[k] = vv
        return out

    # ---------- Periodic Summary (Class-level) ----------
    @classmethod
    async def log_periodic_summary(cls, interval_seconds: int = 600):
        """
        Logs a periodic summary of all component statuses. Designed to be run
        as an asyncio task, providing a regular overview of the system health.
        """
        # Ensure the logger instance is initialized before starting the loop
        logger_instance = cls.get().logger
        try:
            while True:
                await asyncio.sleep(interval_seconds)
                statuses = cls.get_all_statuses()
                if statuses:
                    lines = ["\n--- Operational Status Board Summary ---"]
                    for comp, data in statuses.items():
                        ts = float(data.get("timestamp", 0) or 0)
                        tstr = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S') if ts > 0 else "N/A"
                        stale_indicator = " (STALE)" if data.get("stale") else ""
                        lines.append(f"- {comp}: {data.get('status')}{stale_indicator} ({data.get('detail','')}) at {tstr}")
                    lines.append("--------------------------------------")
                    logger_instance.info("\n".join(lines))
                else:
                    logger_instance.info("Operational Status Board: No component statuses to report yet.")
        except asyncio.CancelledError:
            logger_instance.info("Operational Status Board summary task cancelled (shutdown).")
            raise # Re-raise to propagate cancellation

    # ---------- Quality-of-Life Helpers (Class-level) ----------
    @classmethod
    def log_heartbeat(cls, component_name: str, detail: str = "Heartbeat"):
        """
        Sync-friendly helper. If we're on an event loop, schedule the async heartbeat;
        otherwise fall back to a simple status log.
        """
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(cls.heartbeat(component_name, detail))
        except RuntimeError:
            # No running loop (e.g., synchronous context) → best-effort log only
            cls.log_status(component_name, "Running", detail)

    @classmethod
    def log_exception(cls, component_name: str, exc: Exception, detail: str = ""):
        """
        Helper method to log an 'Error' status when an exception occurs.
        It automatically includes the exception type and message in the detail.
        """
        full_detail = f"{detail} | {type(exc).__name__}: {exc}".strip()
        cls.log_status(component_name, "Error", full_detail, level=logging.ERROR)

    @classmethod
    def clear(cls):
        """
        Clears all cached component statuses and deduplication history.
        Useful for resetting the status board, for example, during testing or reinitialization.
        """
        with cls._status_lock:
            cls._status_cache.clear()
            cls._last_emit.clear()
        cls.get().logger.info("Operational Status Board cleared.")
    # ---------- Optional: quick helper to bind + announce ----------
    @classmethod
    def announce_bind(cls, shared_state, detail: str = "ComponentStatusLogger bound to SharedState"):
        """
        Convenience: bind_shared_state + optional config read + log Initialized.
        """
        cls.bind_shared_state(shared_state)
        # Best-effort: pull observability knobs from shared_state/config if available
        try:
            ttl = None
            dedupe = None
            cfg = getattr(shared_state, "config", None)
            if isinstance(cfg, dict):
                obs = cfg.get("observability", {})
                ttl = ttl or obs.get("status_ttl_s")
                dedupe = dedupe or obs.get("dedupe_s")
            get_cfg = getattr(shared_state, "get_config", None)
            if callable(get_cfg):
                try:
                    c2 = get_cfg()
                    if isinstance(c2, dict):
                        obs2 = c2.get("observability", {})
                        ttl = ttl or obs2.get("status_ttl_s")
                        dedupe = dedupe or obs2.get("dedupe_s")
                except Exception:
                    pass
            cls.configure(ttl_seconds=ttl, dedupe_window_sec=dedupe)
        except Exception:
            pass
        try:
            cls.log_status("ComponentStatusLogger", "Initialized", detail)
        except Exception:
            pass

# ---------- Functional Facade (Back-Compat & Simplicity) ----------
def log_component_status(component: str, status: str, detail: str = "", level: int = logging.INFO, **kwargs):
    """
    Functional facade for agents/components that import a simple function.
    Delegates to ComponentStatusLogger.log_status.
    """
    return ComponentStatusLogger.log_status(component=component, status=status, detail=detail, level=level, **kwargs)

def bind_shared_state(shared_state):
    """
    Functional facade to bind SharedState once at bootstrap.
    """
    return ComponentStatusLogger.bind_shared_state(shared_state)

def get_status(component_name: str) -> dict:
    """
    Functional facade to retrieve a single component status.
    """
    return ComponentStatusLogger.get_status(component_name)

def get_all_statuses() -> dict:
    """
    Functional facade to retrieve all component statuses.
    """
    return ComponentStatusLogger.get_all_statuses()

def log_heartbeat(component_name: str, detail: str = "Heartbeat"):
    """
    Functional facade to emit a heartbeat; async scheduling handled internally.
    """
    return ComponentStatusLogger.log_heartbeat(component_name, detail)

def clear_statuses():
    """
    Functional facade to clear cached statuses and dedupe history.
    """
    return ComponentStatusLogger.clear()
