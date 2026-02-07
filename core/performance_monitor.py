# core/performance_monitor.py
from __future__ import annotations

import asyncio
import time
from typing import Dict, Any, Optional

# --- Compatibility layer: support both maybe_await(value) and maybe_call(obj, method, ...) ---
try:
    # Newer helper that awaits a single value if it's awaitable
    from core.stubs import maybe_await as _maybe_await  # type: ignore
except Exception:
    async def _maybe_await(value):
        import inspect
        if inspect.isawaitable(value):
            return await value
        return value

try:
    # Older helper that calls obj.method(...) and awaits result iff needed
    from core.stubs import maybe_awaitll as _maybe_call  # type: ignore
except Exception:
    _maybe_call = None  # noqa: N816


async def _call(obj: Any, method: str, *args, **kwargs):
    """
    Call obj.method(*args, **kwargs), awaiting iff needed.
    Uses maybe_call when available; otherwise falls back to maybe_await.
    Returns None if the method doesn't exist.
    """
    if obj is None:
        return None
    fn = getattr(obj, method, None)
    if not callable(fn):
        return None
    if _maybe_call is not None:
        # Prefer the library helper if present
        return await _maybe_call(obj, method, *args, **kwargs)
    # Fallback path: invoke and maybe await the returned value
    return await _maybe_await(fn(*args, **kwargs))


def _iso(ts: Optional[float] = None) -> str:
    import datetime as _dt
    return _dt.datetime.utcfromtimestamp(ts or time.time()).isoformat(timespec="seconds") + "Z"


class PerformanceMonitor:
    """
    PerformanceMonitor (P9 compliant, online)
    - Records live KPIs and emits periodic snapshots.
    - Writes to SharedState.kpi_metrics["perf_monitor"]
    - Emits HealthStatus and PerformanceSnapshot events via sstools.emit_event(...)
    Config:
      PERF_MONITOR.INTERVAL_S (default: 30)
    """

    def __init__(self, cfg, shared_state, sstools=None, db=None, logger=None):
        self.cfg = cfg
        self.ss = shared_state
        self.sstools = sstools or shared_state  # tolerate missing sstools by falling back to SharedState
        self.db = db

        import logging
        self.log = logger or logging.getLogger(self.__class__.__name__)
        self.component_name = "PerformanceMonitor"

        # Pre-create KPI slots in SharedState
        if not hasattr(self.ss, "kpi_metrics") or not isinstance(getattr(self.ss, "kpi_metrics", None), dict):
            setattr(self.ss, "kpi_metrics", {})
        km = self.ss.kpi_metrics
        km.setdefault("perf_monitor", {})
        km.setdefault("trades", [])
        km.setdefault("nav_series", [])
        km.setdefault("per_agent", {})
        self._stop_event = asyncio.Event()
        self._task: Optional[asyncio.Task] = None
        self._profitability_last_ts: float = 0.0
        self._profitability_interval_sec: float = float(
            getattr(self.cfg, "PROFITABILITY_STATUS_INTERVAL_SEC", 3600) or 3600
        )

    # ----------------------------
    # Public recording API
    # ----------------------------
    def record_trade(self, exec_result: Dict[str, Any]):
        """
        Caller pushes each executed trade result here (typically from ExecutionManager hooks).
        Expected keys: symbol, side, executedQty, avgPrice, pnl_delta, agent, tag, ts, win
        """
        try:
            km = self.ss.kpi_metrics
            km.setdefault("trades", []).append(exec_result)
            agent = exec_result.get("agent") or "unknown"
            per_agent = km.setdefault("per_agent", {})
            per_agent.setdefault(agent, []).append({
                "pnl": float(exec_result.get("pnl_delta", 0.0) or 0.0),
                "win": bool(exec_result.get("win", False)),
                "ts": exec_result.get("ts", time.time())
            })
        except Exception:
            self.log.debug("record_trade failed", exc_info=True)

    # ----------------------------
    # Internal helpers
    # ----------------------------
    def _snapshot_now(self) -> Dict[str, Any]:
        try:
            nav = None
            if hasattr(self.sstools, "nav_quote"):
                nav = self.sstools.nav_quote()
            sym_count = len(getattr(self.ss, "accepted_symbols", {}) or {})
            pos_count = len(getattr(self.ss, "positions", {}) or {})
            balances = getattr(self.ss, "balances", {}) or {}

            snap = {
                "ts": _iso(),
                "nav_quote": nav,
                "symbols": sym_count,
                "positions": pos_count,
                "balances_keys": list(balances.keys()),
            }

            # Append NAV point
            try:
                nav_f = float(nav) if nav is not None else 0.0
            except Exception:
                nav_f = 0.0
            self.ss.kpi_metrics.setdefault("nav_series", []).append((time.time(), nav_f))

            # Update the perf_monitor block
            self.ss.kpi_metrics.setdefault("perf_monitor", {}).update(snap)
            return snap
        except Exception:
            self.log.debug("snapshot_now failed", exc_info=True)
            return {"ts": _iso(), "error": "snapshot-failed"}

    async def _emit_health(self, status: str, message: str):
        payload = {
            "component": self.component_name,
            "status": status,
            "message": message,
            "timestamp": _iso()
        }
        try:
            await _call(self.sstools, "emit_event", "HealthStatus", payload)
        except Exception:
            self.log.debug("HealthStatus emit failed", exc_info=True)

    async def _emit_snapshot_event(self, snapshot: Dict[str, Any]):
        try:
            await _call(
                self.sstools,
                "emit_event",
                "PerformanceSnapshot",
                {"component": self.component_name, **snapshot}
            )
        except Exception:
            self.log.debug("PerformanceSnapshot emit failed", exc_info=True)

    def _profitability_snapshot(self) -> Dict[str, Any]:
        now = time.time()
        window_sec = float(
            getattr(self.cfg, "PROFITABILITY_STATUS_WINDOW_SEC", self._profitability_interval_sec)
            or self._profitability_interval_sec
        )
        cutoff = now - max(1.0, window_sec)

        trades = []
        try:
            for t in list(getattr(self.ss, "trade_history", []) or []):
                ts = float(t.get("ts", 0.0) or 0.0)
                if ts >= cutoff and str(t.get("side", "")).upper() == "SELL":
                    trades.append(t)
        except Exception:
            trades = []

        pnl_values = [float(t.get("realized_delta", 0.0) or 0.0) for t in trades]
        wins = [v for v in pnl_values if v > 0]
        losses = [v for v in pnl_values if v < 0]
        trades_count = len(pnl_values)
        win_rate = (len(wins) / trades_count) if trades_count else 0.0
        avg_win = (sum(wins) / len(wins)) if wins else 0.0
        avg_loss = (sum(losses) / len(losses)) if losses else 0.0

        fee_total = 0.0
        try:
            fee_total = sum(float(t.get("fee_quote", 0.0) or 0.0) for t in trades)
        except Exception:
            fee_total = 0.0

        gross_abs_pnl = sum(abs(v) for v in pnl_values)
        fee_ratio = (fee_total / gross_abs_pnl) if gross_abs_pnl > 0 else 0.0
        expectancy = (win_rate * avg_win) + ((1.0 - win_rate) * avg_loss) if trades_count else 0.0

        return {
            "trades": trades_count,
            "win_rate": round(win_rate, 6),
            "avg_win": round(avg_win, 6),
            "avg_loss": round(avg_loss, 6),
            "fee_ratio": round(fee_ratio, 6),
            "expectancy": round(expectancy, 6),
            "window_sec": int(window_sec),
        }

    async def _maybe_emit_profitability_status(self) -> None:
        now = time.time()
        if (now - self._profitability_last_ts) < max(1.0, self._profitability_interval_sec):
            return
        self._profitability_last_ts = now
        payload = {"ts": _iso(now), **self._profitability_snapshot()}
        try:
            self.log.info("ProfitabilityStatus %s", payload)
        except Exception:
            pass
        try:
            await _call(self.sstools, "emit_event", "ProfitabilityStatus", payload)
        except Exception:
            self.log.debug("ProfitabilityStatus emit failed", exc_info=True)

    # ----------------------------
    # Run loop
    # ----------------------------
    async def run_forever(self):
        # Config-safe interval lookup: PERF_MONITOR.INTERVAL_S or default 30
        interval = 30
        try:
            section = getattr(self.cfg, "PERF_MONITOR", None)
            interval = int(getattr(section, "INTERVAL_S", 30)) if section is not None else 30
            if interval <= 0:
                interval = 30
        except Exception:
            interval = 30

        await self._emit_health("Running", f"Starting @ {interval}s")

        try:
            while not self._stop_event.is_set():
                try:
                    snap = self._snapshot_now()
                    await self._emit_snapshot_event(snap)
                    await self._emit_health("Running", "OK")
                    await self._maybe_emit_profitability_status()
                except Exception as e:
                    await self._emit_health("Error", f"monitor error: {e!r}")
                    self.log.exception("PerformanceMonitor cycle failed")

                await asyncio.sleep(interval)
        except asyncio.CancelledError:
            await self._emit_health("Running", "Loop cancelled")
            raise

    # P9 lifecycle adapter: expose a standard entrypoint so AppContext can schedule us natively.
    async def start(self):
        """
        P9 contract: start() spawns the monitoring loop once (idempotent).
        """
        if getattr(self, "_task", None) and not self._task.done():
            return
        self._stop_event.clear()
        self._task = asyncio.create_task(self.run_forever(), name="ops.performance_monitor")
        try:
            await self._emit_health("Initialized", "Ready")
        except Exception:
            self.log.debug("PerformanceMonitor initial health update failed", exc_info=True)

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
                self.log.debug("PerformanceMonitor stop wait failed", exc_info=True)
        try:
            await self._emit_health("Stopped", "Stopped by request")
        except Exception:
            self.log.debug("PerformanceMonitor final health update failed", exc_info=True)
