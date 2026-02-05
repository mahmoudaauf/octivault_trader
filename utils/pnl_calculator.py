import asyncio
import logging
import time
from datetime import datetime
import inspect
import random
import json
from typing import Dict, Any, Optional
from collections import deque
from core.component_status_logger import ComponentStatusLogger

logger = logging.getLogger("PnLCalculator")
logger.setLevel(logging.INFO)

OPEN_STATUSES = {"OPEN", "ACTIVE", "RUNNING", "IN_POSITION"}

class PnLCalculator:
    def __init__(self, shared_state, config, exchange_client=None, **kwargs):
        self.shared_state = shared_state
        self.config = config
        # allow direct injection OR borrow from SharedState if present
        self.exchange_client = exchange_client or getattr(shared_state, "exchange_client", None)
        self.evaluation_interval = float(getattr(config, 'PNL_EVALUATION_INTERVAL', 5))  # seconds
        logger.info("PnLCalculator initialized.")

        # Rolling realized window cache baseline
        self._last_realized_seen = float(getattr(self.shared_state, "realized_pnl", 0.0))
        # P9 lifecycle
        self._stop_event = asyncio.Event()
        self._task = None
        self._health_task = None

    # ---------- tiny helpers ----------

    async def _maybe_call(self, obj, method: str, *args, **kwargs):
        """
        Calls obj.<method>(*args, **kwargs) and awaits if awaitable.
        Returns None if method doesn't exist.
        """
        fn = getattr(obj, method, None)
        if not callable(fn):
            return None
        try:
            res = fn(*args, **kwargs)
        except TypeError as e:
            # Hotfix: some call sites pass only (component, status) but API expects (component, status, detail)
            msg = str(e)
            if method == "update_component_status" and "missing 1 required positional argument: 'detail'" in msg:
                try:
                    res = fn(*args, "", **kwargs)  # append empty detail
                except Exception:
                    raise
            else:
                raise
        if inspect.isawaitable(res):
            return await res
        return res

    @staticmethod
    def _is_open(trade: Dict[str, Any]) -> bool:
        status = (str(trade.get("status") or trade.get("trade_status") or "")).upper()
        return status in OPEN_STATUSES

    async def _get_active_trades_snapshot(self) -> Dict[str, Dict[str, Any]]:
        """
        Unified way to obtain open/active trades without relying on a specific internal attribute.
        Tries:
          1) shared_state.get_active_trades_snapshot()  [preferred]
          2) shared_state.get_positions_snapshot() and filter OPEN/ACTIVE
          3) getattr(shared_state, "positions", {}) and filter
        """
        # 1) Preferred explicit API
        if hasattr(self.shared_state, "get_active_trades_snapshot"):
            try:
                trades = await self.shared_state.get_active_trades_snapshot()
                if isinstance(trades, dict):
                    return trades
            except Exception as e:
                logger.debug(f"[PnLCalculator] get_active_trades_snapshot failed: {e}")

        # 2) Positions snapshot
        if hasattr(self.shared_state, "get_positions_snapshot"):
            try:
                got = self.shared_state.get_positions_snapshot()
                pos = await got if inspect.isawaitable(got) else got
                if isinstance(pos, dict):
                    return {k: v for k, v in pos.items() if isinstance(v, dict) and self._is_open(v)}
            except Exception as e:
                logger.debug(f"[PnLCalculator] get_positions_snapshot failed: {e}")

        # 3) Fallback to internal store (best-effort)
        try:
            pos = getattr(self.shared_state, "positions", {}) or {}
            if isinstance(pos, dict):
                return {k: v for k, v in pos.items() if isinstance(v, dict) and self._is_open(v)}
        except Exception:
            pass

        return {}

    async def _get_latest_price(self, symbol: str) -> Optional[float]:
        """
        Tries several price sources in order:
          1) shared_state.get_latest_price_safe(symbol)
          2) shared_state.get_latest_price(symbol)
          3) shared_state.safe_price(symbol)
          4) exchange_client.get_current_price(symbol) if available
          5) last 1m candle close from shared_state.get_market_data(symbol, "1m")
        """
        # 1
        if hasattr(self.shared_state, "get_latest_price_safe"):
            try:
                p = await self.shared_state.get_latest_price_safe(symbol)
                if p and p > 0:
                    return float(p)
            except Exception:
                pass

        # 2
        if hasattr(self.shared_state, "get_latest_price"):
            try:
                p = await self.shared_state.get_latest_price(symbol)
                if p and p > 0:
                    return float(p)
            except Exception:
                pass

        # 3) Try SharedState.safe_price (symbol-scoped) as a light fallback
        if hasattr(self.shared_state, "safe_price"):
            try:
                p = await self.shared_state.safe_price(symbol, default=0.0)
                if p and p > 0:
                    return float(p)
            except Exception:
                pass

        # 4
        if hasattr(self, "exchange_client") and getattr(self, "exchange_client"):
            try:
                p = await self.exchange_client.get_current_price(symbol)
                if p and p > 0:
                    return float(p)
            except Exception:
                pass

        # 5) Last 1m candle close as a final fallback (async accessor in SharedState)
        try:
            candles = None
            if hasattr(self.shared_state, "get_market_data"):
                candles = await self.shared_state.get_market_data(symbol, "1m")
            if candles:
                last = candles[-1]
                if isinstance(last, dict):
                    # accept either verbose 'close' or canonical short key 'c'
                    val = last.get("close")
                    if val is None:
                        val = last.get("c")
                    if val is not None:
                        return float(val)
                elif isinstance(last, (list, tuple)) and len(last) > 4:
                    return float(last[4])
        except Exception:
            pass

        return None

    def _readiness_snapshot(self) -> Dict[str, bool]:
        """Best-effort readiness snapshot from SharedState for gating decisions."""
        snap: Dict[str, bool] = {
            "market_data_ready": True,
            "balances_ready": True,
            "ops_plane_ready": True,
        }
        try:
            getter = getattr(self.shared_state, "get_readiness_snapshot", None)
            if callable(getter):
                got = getter()
                if inspect.isawaitable(got):
                    # tolerate both sync/async
                    try:
                        loop = asyncio.get_event_loop()
                        if loop.is_running():
                            pass
                    except Exception:
                        pass
                if isinstance(got, dict):
                    for k in list(snap.keys()):
                        if k in got:
                            snap[k] = bool(got[k])
        except Exception:
            pass
        return snap

    # ---------- core calculations ----------

    async def _calculate_unrealized_pnl(self):
        """
        Calculates unrealized PnL over open positions.
        Writes per-trade `current_pnl` when possible and updates shared_state.unrealized_pnl.
        """
        total_unrealized_pnl = 0.0

        # Prefer snapshot methods that manage their own locking
        trades = await self._get_active_trades_snapshot()

        for key, trade in (trades or {}).items():
            # Accept both shapes: key may be order_id or the symbol itself
            symbol = trade.get("symbol") or trade.get("asset") or key
            if not symbol:
                continue

            # Extract fields with tolerant aliases
            entry_price = trade.get("entry_price", trade.get("price"))
            qty = trade.get("quantity", trade.get("qty") or trade.get("size"))
            direction = (trade.get("direction") or "").lower()
            side = (trade.get("side") or "").upper()

            if entry_price is None or qty in (None, 0, 0.0):
                logger.debug(f"[PnLCalculator] Incomplete trade data for {symbol}: entry={entry_price}, qty={qty} -> skip")
                continue

            current_price = await self._get_latest_price(symbol)
            if not current_price or current_price <= 0:
                logger.debug(f"[PnLCalculator] No current price for {symbol} -> skip")
                continue

            # Long/Short handling (direction or side)
            if direction == "long" or side == "BUY":
                pnl = (float(current_price) - float(entry_price)) * float(qty)
            elif direction == "short" or side == "SELL":
                pnl = (float(entry_price) - float(current_price)) * float(qty)
            else:
                pnl = 0.0
                logger.debug(f"[PnLCalculator] Unknown side/direction for {symbol} (dir='{direction}', side='{side}') -> pnl=0")

            total_unrealized_pnl += pnl
            # Best-effort write-back to trade dict (non-fatal if it fails)
            try:
                trade["current_pnl"] = pnl
            except Exception:
                pass

        # Store result atomically if SharedState exposes a safe setter, else set attribute
        if hasattr(self.shared_state, "set_unrealized_pnl"):
            try:
                await self.shared_state.set_unrealized_pnl(total_unrealized_pnl)
            except Exception:
                setattr(self.shared_state, "unrealized_pnl", total_unrealized_pnl)
        else:
            setattr(self.shared_state, "unrealized_pnl", total_unrealized_pnl)

    async def _calculate_total_portfolio_value(self):
        """
        Triggers a portfolio snapshot from SharedState and updates system health upon success.
        """
        try:
            snap = None
            if hasattr(self.shared_state, "get_portfolio_snapshot"):
                snap = await self.shared_state.get_portfolio_snapshot()
            if isinstance(snap, dict):
                total_value = float(snap.get("nav", 0.0))
                unrealized_pnl = float(snap.get("unrealized_pnl", 0.0))
                setattr(self.shared_state, "total_value", total_value)
                setattr(self.shared_state, "unrealized_pnl", unrealized_pnl)
            else:
                total_value = float(getattr(self.shared_state, "total_value", 0.0))
                unrealized_pnl = float(getattr(self.shared_state, "unrealized_pnl", 0.0))
            realized_pnl = float(getattr(self.shared_state, "realized_pnl", 0.0))
            await self._maybe_call(
                self.shared_state, "update_system_health",
                component="PnLCalculator",
                status="Operational",
                message="Portfolio value and unrealized PnL calculated successfully."
            )
        except Exception as e:
            logger.debug(f"[PnLCalculator] portfolio snapshot failed: {e}")

    async def _after_valuation_bookkeeping(self):
        """Write total_equity and feed 60m realized pnl rolling window."""
        # 1) total_equity (canonical: total_value + realized + unrealized)
        try:
            total_equity = float(self.shared_state.get_total_equity())
        except Exception:
            tv  = float(getattr(self.shared_state, "total_value", 0.0))
            rp  = float(getattr(self.shared_state, "realized_pnl", 0.0))
            up  = float(getattr(self.shared_state, "unrealized_pnl", 0.0))
            total_equity = tv + rp + up
        setattr(self.shared_state, "total_equity", total_equity)

        # 2) rolling 60m realized PnL feed (append deltas)
        now = time.time()
        cur_realized = float(getattr(self.shared_state, "realized_pnl", 0.0))
        delta = cur_realized - self._last_realized_seen
        self._last_realized_seen = cur_realized

        if abs(delta) > 0.0:
            # Prefer a public API if available; otherwise fall back to internal deque
            if hasattr(self.shared_state, "append_realized_pnl_delta"):
                try:
                    await self._maybe_call(self.shared_state, "append_realized_pnl_delta", now, float(delta))
                except Exception:
                    pass
            else:
                try:
                    self.shared_state._realized_pnl.append((now, float(delta)))
                except Exception:
                    self.shared_state._realized_pnl = deque(maxlen=4096)
                    self.shared_state._realized_pnl.append((now, float(delta)))

        # Optional O(1) convenience cache
        try:
            last60 = await self._maybe_call(self.shared_state, "get_rolling_realized_pnl", 60)
            if last60 is not None:
                setattr(self.shared_state, "pnl_realized_60m", float(last60))
        except Exception as e:
            logger.debug(f"[PnLCalculator] Failed to update pnl_realized_60m: {e}")

    # ---------- run loops ----------

    async def start(self):
        """
        P9 contract: start() spawns the main loop and health loop (idempotent).
        """
        if getattr(self, "_task", None) and not self._task.done():
            return
        self._stop_event.clear()
        # Spawn health loop first (so Watchdog sees us early)
        if not getattr(self, "_health_task", None) or self._health_task.done():
            self._health_task = asyncio.create_task(self.report_health_loop(), name="pnl_calculator.health")
        # Touch timestamp immediately so Watchdog sees us before first tick
        try:
            await self._maybe_call(self.shared_state, "update_timestamp", "PnLCalculator")
        except Exception:
            pass
        ComponentStatusLogger.log_status("PnLCalculator", "Initialized", "OK")
        # Main loop
        self._task = asyncio.create_task(self.run(), name="pnl_calculator.run")
        try:
            await self._maybe_call(
                self.shared_state, "update_component_status",
                "PnLCalculator", "Initialized", "OK"
            )
        except Exception:
            logger.debug("PnLCalculator initial health update failed", exc_info=True)

    async def stop(self):
        """
        P9 contract: stop() requests both loops to end and waits for them.
        """
        self._stop_event.set()
        t_main = getattr(self, "_task", None)
        t_hb = getattr(self, "_health_task", None)
        self._task = None
        self._health_task = None

        # cancel and await tasks
        for t in (t_main, t_hb):
            if not t:
                continue
            try:
                t.cancel()
                try:
                    await asyncio.wait_for(t, timeout=float(getattr(self.config, "STOP_JOIN_TIMEOUT_S", 5.0)))
                except asyncio.CancelledError:
                    pass
            except Exception:
                logger.debug("PnLCalculator stop wait failed", exc_info=True)

        try:
            await self._maybe_call(
                self.shared_state, "update_component_status",
                "PnLCalculator", "Stopped", "Stopped by request"
            )
        except Exception:
            logger.debug("PnLCalculator final health update failed", exc_info=True)

    async def start(self):
        """Standard lifecycle start."""
        if self._stop_event.is_set():
            self._stop_event.clear()
        loop = asyncio.get_running_loop()
        self._task = loop.create_task(self.run())
        return self._task

    async def run(self):
        """
        Main loop: periodic unrealized PnL + portfolio valuation + bookkeeping.
        """
        await self._maybe_call(self.shared_state, "register_component", "PnLCalculator")

        logger.info("✅ PnLCalculator is running.")
        ComponentStatusLogger.log_status(
            component="PnLCalculator",
            status="Running",
            detail="Started and reporting health."
        )

        await self._maybe_call(
            self.shared_state, "update_system_health",
            component="PnLCalculator",
            status="Running",
            message="Tracking PnL across active trades."
        )

        while not self._stop_event.is_set():
            # Readiness gating (market data + balances + ops plane)
            gated_reasons = []
            try:
                snap = self._readiness_snapshot()
                if not snap.get("market_data_ready", True):
                    gated_reasons.append("MarketDataReady")
                if not snap.get("balances_ready", True):
                    gated_reasons.append("BalancesReady")
                # if not snap.get("ops_plane_ready", True):
                #     gated_reasons.append("OpsPlaneReady")
            except Exception:
                snap = {}
                gated_reasons = []

            if gated_reasons:
                try:
                    await self._maybe_call(self.shared_state, "update_component_status", "PnLCalculator", "Degraded", f"Gated: waiting for {', '.join(gated_reasons)}")
                except Exception:
                    pass
                try:
                    await self._maybe_call(
                        self.shared_state, "update_system_health",
                        component="PnLCalculator",
                        status="Degraded",
                        message=f"Waiting for {', '.join(gated_reasons)}"
                    )
                except Exception:
                    pass
                await asyncio.sleep(max(0.5, float(self.evaluation_interval) / 2.0))
                continue
            cycle_t0 = time.time()
            try:
                await self._calculate_unrealized_pnl()
                await self._calculate_total_portfolio_value()
                await self._after_valuation_bookkeeping()
                await self._maybe_call(self.shared_state, "update_component_status", "PnLCalculator", "Operational", "Valuation OK")

                # Emit a PortfolioSnapshot event (contract-friendly)
                try:
                    snapshot = {
                        "total_value": float(getattr(self.shared_state, "total_value", 0.0)),
                        "realized_pnl": float(getattr(self.shared_state, "realized_pnl", 0.0)),
                        "unrealized_pnl": float(getattr(self.shared_state, "unrealized_pnl", 0.0)),
                        "total_equity": float(getattr(self.shared_state, "total_equity", 0.0)),
                        "ts": time.time(),
                    }
                    if hasattr(self.shared_state, "emit_event"):
                        await self._maybe_call(self.shared_state, "emit_event", "PortfolioSnapshot", snapshot)
                except Exception:
                    pass

                # Structured log for observability
                try:
                    duration_ms = int((time.time() - cycle_t0) * 1000)
                    log_evt = {
                        "ts": time.time(),
                        "component": "PnLCalculator",
                        "event": "valuation_cycle",
                        "status": "ok",
                        "total_value": snapshot.get("total_value", 0.0),
                        "realized_pnl": snapshot.get("realized_pnl", 0.0),
                        "unrealized_pnl": snapshot.get("unrealized_pnl", 0.0),
                        "total_equity": snapshot.get("total_equity", 0.0),
                        "duration_ms": duration_ms,
                    }
                    logger.info(json.dumps(log_evt))
                except Exception:
                    pass

            except Exception as e:
                logger.exception("❌ PnLCalculator encountered an error.")
                await self._maybe_call(self.shared_state, "update_component_status", "PnLCalculator", "Error", "Exception during valuation")
                await self._maybe_call(
                    self.shared_state, "update_system_health",
                    component="PnLCalculator",
                    status="Error",
                    message=f"PnLCalculator encountered an error: {e}"
                )

            # Jitter to avoid phase alignment; warn if cycle exceeds interval
            duration = time.time() - cycle_t0
            interval = float(self.evaluation_interval)
            if duration > interval * 1.5:
                warn = f"valuation cycle slow: {duration:.2f}s > {interval:.2f}s"
                try:
                    # Keep component status Operational but include the warning detail
                    await self._maybe_call(
                        self.shared_state, "update_component_status",
                        "PnLCalculator", "Operational", warn
                    )
                except Exception:
                    pass
                try:
                    # Surface to system health as an informational/healthy note
                    await self._maybe_call(
                        self.shared_state, "update_system_health",
                        component="PnLCalculator",
                        status="Healthy",
                        message=warn
                    )
                except Exception:
                    pass
            # ±5% jitter
            jitter = interval * (random.uniform(-0.05, 0.05))
            await asyncio.sleep(max(0.0, interval + jitter))

    async def run_loop(self):
        """Wrapper for Phase 9 compatibility."""
        await self.run()

    async def report_health_loop(self):
        """
        Reports the health of the PnLCalculator component periodically.
        """
        while not self._stop_event.is_set():
            try:
                await self._maybe_call(self.shared_state, "update_timestamp", "PnLCalculator")
                # Canonical heartbeat + health emissions
                await ComponentStatusLogger.heartbeat("PnLCalculator", "OK")
                await self._maybe_call(
                    self.shared_state,
                    "update_system_health",
                    component="PnLCalculator",
                    status="Running",
                    message="Heartbeat OK"
                )
                _emit_health(self.shared_state, "OK", "PnLCalculator heartbeat OK")
            except Exception as e:
                logger.warning(f"⚠️ PnLCalculator health update failed: {e}")
            await asyncio.sleep(10)  # Must be <= 30s



# ===== P9 Spec Helpers (added) =====
def _iso_now():
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def _emit_health(ss, status: str, message: str):
    """
    P9 unified HealthStatus event.
    status → level: one of OK | DEGRADED | ERROR
    """
    try:
        if ss and hasattr(ss, "emit_event"):
            try:
                res = ss.emit_event("HealthStatus", {
                    "component": "PnLCalculator",
                    "level": status,
                    "details": {"message": message},
                    "ts": _iso_now()
                })
                if inspect.isawaitable(res):
                    asyncio.create_task(res)
            except Exception:
                pass
    except Exception:
        pass

def _emit_realized_pnl(ss, pnl_delta: float, nav_quote: float = None, symbol: str = None):
    try:
        payload = {
            "pnl_delta": float(pnl_delta or 0.0),
            "timestamp": _iso_now()
        }
        if symbol:
            payload["symbol"] = symbol
        if nav_quote is not None:
            payload["nav_quote"] = float(nav_quote)
        if ss and hasattr(ss, "emit_event"):
            try:
                res = ss.emit_event("RealizedPnlUpdated", payload)
                if inspect.isawaitable(res):
                    asyncio.create_task(res)
            except Exception:
                pass
    except Exception:
        pass



# ===== P9 Emission Wrappers (added) =====
def _wrap_emit_realized_pnl_on_methods(cls, shared_state_attr: str = "shared_state"):
    target_methods = ["on_fill", "handle_fill", "record_fill", "apply_fill", "update_on_fill"]
    for name in target_methods:
        if hasattr(cls, name) and not hasattr(cls, f"_{name}_raw"):
            setattr(cls, f"_{name}_raw", getattr(cls, name))
            def _make_wrapper(mname):
                def _wrapped(self, *a, **kw):
                    res = getattr(cls, f"_{mname}_raw")(self, *a, **kw)
                    ss = getattr(self, shared_state_attr, None)
                    pnl_delta = None
                    symbol = None
                    try:
                        if isinstance(res, dict):
                            pnl_delta = res.get("pnl_delta") or res.get("realized_pnl_delta")
                            symbol = res.get("symbol")
                        elif isinstance(res, (int, float)):
                            pnl_delta = float(res)
                        for obj in list(a) + list(kw.values()):
                            if isinstance(obj, dict):
                                if "pnl_delta" in obj and pnl_delta is None:
                                    pnl_delta = obj.get("pnl_delta")
                                if "symbol" in obj and symbol is None:
                                    symbol = obj.get("symbol")
                    except Exception:
                        pass
                    nav = None
                    try:
                        if ss and hasattr(ss, "get_nav_quote"):
                            nav = float(ss.get_nav_quote())
                    except Exception:
                        nav = None
                    if pnl_delta is not None:
                        _emit_realized_pnl(ss, pnl_delta, nav_quote=nav, symbol=symbol)
                        _emit_health(ss, "OK", f"RealizedPnlUpdated delta={pnl_delta}")
                    return res
                return _wrapped
            setattr(cls, name, _make_wrapper(name))
    return cls


# Apply P9 emission wrapper
PnLCalculator = _wrap_emit_realized_pnl_on_methods(PnLCalculator)
