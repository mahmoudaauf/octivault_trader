"""
L4 — Safety Order Manager
=========================
Exchange-native protection layer that complements the in-process TPSLEngine.

While `TPSLEngine` performs *software* polling-based exits (which fail when the
bot crashes), `SafetyOrderManager` parks **real OCO orders** on the exchange so
that protection survives bot restarts, network loss, or process crashes.

Responsibilities
----------------
1. On startup: enumerate open positions and place an OCO (TP + STOP_LOSS_LIMIT)
   for any that lack one.
2. Periodically re-scan and re-arm positions whose protection was filled or
   cancelled.
3. Honour symbol filters (PRICE_FILTER tickSize, LOT_SIZE stepSize, NOTIONAL).
4. Tag every order with `clientOrderId="safety_<symbol>_<ts>"` so that the
   manager can identify and cancel its own orders without touching others.

Configuration (read from `config`, all optional, sensible defaults):
    SAFETY_ORDERS_ENABLED            (bool, default True)
    SAFETY_ORDER_TP_PCT              (float, default 0.015 = +1.5%)
    SAFETY_ORDER_SL_PCT              (float, default 0.030 = -3.0%)
    SAFETY_ORDER_SL_LIMIT_BUFFER     (float, default 0.003 = 0.3% below stop)
    SAFETY_ORDER_MIN_NOTIONAL_USDT   (float, default 5.0)
    SAFETY_ORDER_RECHECK_INTERVAL    (float seconds, default 300 = 5 min)
    SAFETY_ORDER_AUTO_ARM_ON_STARTUP (bool, default True)
    SAFETY_ORDER_DRY_RUN             (bool, default False – plan only, no submit)
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from decimal import Decimal, ROUND_DOWN, ROUND_UP, getcontext
from inspect import iscoroutine
from typing import Any, Dict, List, Optional, Tuple

getcontext().prec = 28

CLIENT_ID_PREFIX = "safety_"


def _f(v: Decimal) -> str:
    """Format a Decimal as plain (non-scientific) string, no trailing zeros."""
    s = format(v.normalize(), "f")
    if s.startswith("."):
        s = "0" + s
    if s.startswith("-."):
        s = "-0" + s[1:]
    return s


class SafetyOrderManager:
    """
    Places exchange-native OCO sell orders on Binance SPOT for currently held
    positions, providing hardware stop-loss + take-profit protection that
    survives bot restarts, crashes, and connectivity loss.
    """

    COMPONENT_NAME = "SafetyOrderManager"

    def __init__(
        self,
        shared_state,
        config,
        exchange_client,
        execution_manager=None,
        logger: Optional[logging.Logger] = None,
        **_kwargs,
    ):
        self.shared_state = shared_state
        self.config = config
        self.exchange_client = exchange_client
        self.execution_manager = execution_manager
        self.logger = logger or logging.getLogger(self.COMPONENT_NAME)

        # Lifecycle
        self._stop_event = asyncio.Event()
        self._task: Optional[asyncio.Task] = None
        self._heartbeat_task: Optional[asyncio.Task] = None

        # Config snapshot
        # OPTION B FIX (2026-05-05): Honor SAFETY_ORDERS_ENABLED env var directly.
        # Config class doesn't define this attr, so getattr() always returned True
        # default. Now we check os.environ first, then config attr, then default True.
        import os as _os
        _env_flag = _os.environ.get("SAFETY_ORDERS_ENABLED", "").strip().lower()
        if _env_flag in ("false", "0", "no", "off"):
            self._enabled = False
        elif _env_flag in ("true", "1", "yes", "on"):
            self._enabled = True
        else:
            self._enabled = bool(getattr(config, "SAFETY_ORDERS_ENABLED", True))
        
        self._tp_pct = float(getattr(config, "SAFETY_ORDER_TP_PCT", 0.015))
        self._sl_pct = float(getattr(config, "SAFETY_ORDER_SL_PCT", 0.030))
        self._sl_limit_buffer = float(
            getattr(config, "SAFETY_ORDER_SL_LIMIT_BUFFER", 0.003)
        )
        self._min_notional = float(
            getattr(config, "SAFETY_ORDER_MIN_NOTIONAL_USDT", 5.0)
        )
        self._recheck_interval = float(
            getattr(config, "SAFETY_ORDER_RECHECK_INTERVAL", 300.0)
        )
        self._auto_arm_on_startup = bool(
            getattr(config, "SAFETY_ORDER_AUTO_ARM_ON_STARTUP", True)
        )
        self._dry_run = bool(getattr(config, "SAFETY_ORDER_DRY_RUN", False))

        # State
        self._last_arm_attempt: Dict[str, float] = {}
        self._symbol_filters: Dict[str, Dict[str, Any]] = {}

    # ─── Lifecycle ─────────────────────────────────────────────────────────
    async def start(self) -> None:
        if self._task and not self._task.done():
            return
        if not self._enabled:
            self.logger.info(
                "[SafetyOrderManager] disabled via SAFETY_ORDERS_ENABLED=False"
            )
            await self._safe_status_update("Disabled", "Disabled by config")
            return

        self._stop_event.clear()
        self.logger.info(
            "[SafetyOrderManager] starting (tp=+%.2f%% sl=-%.2f%% recheck=%.0fs dry_run=%s)",
            self._tp_pct * 100, self._sl_pct * 100,
            self._recheck_interval, self._dry_run,
        )
        await self._safe_status_update("Starting", "Initializing")

        # Heartbeat first (Watchdog gate)
        if not self._heartbeat_task or self._heartbeat_task.done():
            self._heartbeat_task = asyncio.create_task(
                self._heartbeat_loop(), name=f"{self.COMPONENT_NAME}:heartbeat"
            )

        # One-shot auto-arm
        if self._auto_arm_on_startup:
            try:
                armed, skipped = await self.arm_all_positions()
                self.logger.info(
                    "[SafetyOrderManager] startup auto-arm: armed=%d skipped=%d",
                    armed, skipped,
                )
            except Exception:
                self.logger.error(
                    "[SafetyOrderManager] auto-arm on startup failed",
                    exc_info=True,
                )

        self._task = asyncio.create_task(self._run(), name=self.COMPONENT_NAME)
        await self._safe_status_update("Operational", "Active / Monitoring")

    async def stop(self) -> None:
        self._stop_event.set()
        for t in (self._heartbeat_task, self._task):
            if t and not t.done():
                t.cancel()
                try:
                    await asyncio.wait_for(t, timeout=5.0)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    pass
        self._heartbeat_task = None
        self._task = None
        try:
            await self._safe_status_update("Stopped", "Stopped by request")
        except Exception:
            pass

    # ─── Loops ─────────────────────────────────────────────────────────────
    async def _run(self) -> None:
        """Periodic re-arm loop — re-checks every SAFETY_ORDER_RECHECK_INTERVAL."""
        while not self._stop_event.is_set():
            try:
                await asyncio.sleep(self._recheck_interval)
                if self._stop_event.is_set():
                    break
                armed, skipped = await self.arm_all_positions()
                if armed:
                    self.logger.info(
                        "[SafetyOrderManager] periodic re-arm: armed=%d skipped=%d",
                        armed, skipped,
                    )
            except asyncio.CancelledError:
                break
            except Exception:
                self.logger.error(
                    "[SafetyOrderManager] periodic loop error", exc_info=True
                )
                await asyncio.sleep(30)

    async def _heartbeat_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                await self._safe_status_update("Operational", "Heartbeat: Active")
            except Exception:
                pass
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=60)
            except asyncio.TimeoutError:
                continue

    # ─── Public API ────────────────────────────────────────────────────────
    async def arm_all_positions(self) -> Tuple[int, int]:
        """
        Discover unprotected positions and place OCO orders.
        Returns (armed_count, skipped_count).
        """
        # OPTION B FIX: Double-check disabled state
        if not self._enabled:
            print(f"⚠️  [SafetyOrderManager.arm_all_positions] Called but DISABLED (enabled={self._enabled}). Returning 0,0.", flush=True)
            return 0, 0
            
        positions = await self._discover_positions()
        if not positions:
            return 0, 0

        protected = await self._symbols_with_safety_orders()
        armed = 0
        skipped = 0
        for pos in positions:
            sym = pos["symbol"]
            if sym in protected:
                skipped += 1
                continue
            try:
                ok = await self._arm_one(pos)
                if ok:
                    armed += 1
                else:
                    skipped += 1
            except Exception as e:
                self.logger.error(
                    "[SafetyOrderManager] arm %s failed: %s", sym, e, exc_info=False
                )
                skipped += 1
        return armed, skipped

    async def cancel_all_safety_orders(self) -> int:
        """Cancel every order on the account whose clientOrderId begins with safety_."""
        try:
            orders = await self.exchange_client.get_open_orders()
        except Exception as e:
            self.logger.error("[SafetyOrderManager] get_open_orders failed: %s", e)
            return 0
        cancelled = 0
        for o in orders or []:
            cid = str(o.get("clientOrderId", ""))
            ocid = str(o.get("origClientOrderId", ""))
            if not (cid.startswith(CLIENT_ID_PREFIX) or ocid.startswith(CLIENT_ID_PREFIX)):
                continue
            try:
                await self.exchange_client._request(
                    "DELETE",
                    "/api/v3/order",
                    {"symbol": o["symbol"], "orderId": o["orderId"]},
                    signed=True,
                    api="spot_api",
                )
                self.logger.info(
                    "[SafetyOrderManager] cancelled %s order %s",
                    o["symbol"], o["orderId"],
                )
                cancelled += 1
            except Exception as e:
                self.logger.warning(
                    "[SafetyOrderManager] cancel %s/%s failed: %s",
                    o.get("symbol"), o.get("orderId"), e,
                )
        return cancelled

    # ─── Discovery ─────────────────────────────────────────────────────────
    async def _discover_positions(self) -> List[Dict[str, Any]]:
        """Return tradeable positions worth ≥ min_notional with current price + qty."""
        try:
            balances = await self.exchange_client.get_account_balances()
        except Exception as e:
            self.logger.error("[SafetyOrderManager] balances fetch failed: %s", e)
            return []

        positions: List[Dict[str, Any]] = []
        # SharedState canonical positions snapshot (preferred for entry price)
        ss_positions = getattr(self.shared_state, "positions", {}) or {}

        for asset, bal in (balances or {}).items():
            if asset in ("USDT", "BFUSD", "USDC", "BUSD"):
                continue
            if isinstance(bal, dict):
                free = float(bal.get("free", 0) or 0)
                locked = float(bal.get("locked", 0) or 0)
            else:
                free = float(bal or 0)
                locked = 0.0
            total = free + locked
            if total <= 0 or free <= 0:
                continue

            symbol = f"{asset}USDT"
            try:
                price = float(await self.exchange_client.get_price(symbol))
            except Exception:
                price = 0.0
            if price <= 0:
                continue

            value = total * price
            if value < self._min_notional:
                continue

            # Entry price preference: shared_state.positions → known fallback → current price
            ss_pos = ss_positions.get(symbol, {}) or {}
            entry = float(
                ss_pos.get("avg_price")
                or ss_pos.get("entry_price")
                or ss_pos.get("entry")
                or 0.0
            )
            if entry <= 0:
                entry = price  # arm relative to now if no known entry

            positions.append({
                "symbol": symbol,
                "asset": asset,
                "free_qty": free,
                "locked_qty": locked,
                "total_qty": total,
                "price_now": price,
                "value_usdt": value,
                "entry_price": entry,
            })
        positions.sort(key=lambda r: -r["value_usdt"])
        return positions

    async def _symbols_with_safety_orders(self) -> set:
        """Return the set of symbols that already have a sell-side stop-loss order."""
        try:
            orders = await self.exchange_client.get_open_orders()
        except Exception as e:
            self.logger.warning("[SafetyOrderManager] get_open_orders failed: %s", e)
            return set()
        protected = set()
        for o in orders or []:
            if o.get("side") != "SELL":
                continue
            otype = o.get("type", "")
            if otype in ("STOP_LOSS_LIMIT", "STOP_LOSS"):
                protected.add(o["symbol"])
        return protected

    async def _get_filters(self, symbol: str) -> Optional[Dict[str, Any]]:
        if symbol in self._symbol_filters:
            return self._symbol_filters[symbol]
        try:
            info = await self.exchange_client.get_symbol_info(symbol)
        except Exception as e:
            self.logger.warning("[SafetyOrderManager] symbol_info %s failed: %s", symbol, e)
            return None
        if not info:
            return None
        flt = {f["filterType"]: f for f in info.get("filters", [])}
        filters = {
            "tickSize": Decimal(str(flt.get("PRICE_FILTER", {}).get("tickSize", "0.01"))),
            "stepSize": Decimal(str(flt.get("LOT_SIZE", {}).get("stepSize", "0.0001"))),
            "minQty": Decimal(str(flt.get("LOT_SIZE", {}).get("minQty", "0"))),
            "minNotional": Decimal(str(
                flt.get("NOTIONAL", flt.get("MIN_NOTIONAL", {})).get("minNotional", "5")
            )),
            "ocoAllowed": bool(info.get("ocoAllowed", True)),
        }
        self._symbol_filters[symbol] = filters
        return filters

    # ─── Arming ────────────────────────────────────────────────────────────
    @staticmethod
    def _round_step(value: Decimal, step: Decimal, mode=ROUND_DOWN) -> Decimal:
        if step == 0:
            return value
        return (value / step).quantize(Decimal("1"), rounding=mode) * step

    def _build_oco_plan(
        self,
        pos: Dict[str, Any],
        filt: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        sym = pos["symbol"]
        entry = Decimal(str(pos["entry_price"]))
        price_now = Decimal(str(pos["price_now"]))

        tp_raw = entry * (Decimal("1") + Decimal(str(self._tp_pct)))
        sl_raw = entry * (Decimal("1") - Decimal(str(self._sl_pct)))

        # Defensive: SL must sit below current price; TP must sit above.
        if sl_raw >= price_now:
            sl_raw = price_now * Decimal("0.99")
        if tp_raw <= price_now:
            tp_raw = price_now * Decimal("1.008")

        tick = filt["tickSize"]
        step = filt["stepSize"]

        tp_price = self._round_step(tp_raw, tick, ROUND_UP)
        sl_stop = self._round_step(sl_raw, tick, ROUND_DOWN)
        sl_limit = self._round_step(
            sl_stop * (Decimal("1") - Decimal(str(self._sl_limit_buffer))),
            tick, ROUND_DOWN,
        )

        qty = self._round_step(Decimal(str(pos["free_qty"])), step, ROUND_DOWN)
        if qty < filt["minQty"]:
            return None
        notional = qty * price_now
        if notional < max(filt["minNotional"], Decimal(str(self._min_notional))):
            return None

        return {
            "symbol": sym,
            "qty": _f(qty),
            "tp_price": _f(tp_price),
            "sl_stop": _f(sl_stop),
            "sl_limit": _f(sl_limit),
            "client_order_id": f"{CLIENT_ID_PREFIX}{sym}_{int(time.time())}",
        }

    async def _arm_one(self, pos: Dict[str, Any]) -> bool:
        # OPTION B FIX: Guard against disabled state
        if not self._enabled:
            return False
            
        sym = pos["symbol"]
        # Debounce
        last = self._last_arm_attempt.get(sym, 0.0)
        if (time.time() - last) < 30:
            return False
        self._last_arm_attempt[sym] = time.time()

        filt = await self._get_filters(sym)
        if not filt:
            self.logger.warning("[SafetyOrderManager] %s no filters; skipping", sym)
            return False
        if not filt["ocoAllowed"]:
            self.logger.warning("[SafetyOrderManager] %s OCO not allowed; skipping", sym)
            return False

        plan = self._build_oco_plan(pos, filt)
        if not plan:
            self.logger.info(
                "[SafetyOrderManager] %s plan invalid (qty/notional below filters); skipping", sym
            )
            return False

        if self._dry_run:
            self.logger.info(
                "[SafetyOrderManager] DRY_RUN %s qty=%s tp=%s sl_stop=%s sl_limit=%s",
                plan["symbol"], plan["qty"], plan["tp_price"],
                plan["sl_stop"], plan["sl_limit"],
            )
            return True

        params = {
            "symbol": plan["symbol"],
            "side": "SELL",
            "quantity": plan["qty"],
            "price": plan["tp_price"],            # TP limit leg
            "stopPrice": plan["sl_stop"],         # Stop trigger
            "stopLimitPrice": plan["sl_limit"],   # Stop limit leg
            "stopLimitTimeInForce": "GTC",
            "newOrderRespType": "RESULT",
            "listClientOrderId": plan["client_order_id"],
        }
        try:
            resp = await self.exchange_client._request(
                "POST", "/api/v3/order/oco", params, signed=True, api="spot_api"
            )
        except Exception as e:
            self.logger.error(
                "[SafetyOrderManager] OCO POST %s failed: %s", sym, e
            )
            return False

        list_id = (resp or {}).get("orderListId", "?")
        self.logger.info(
            "[SafetyOrderManager] ✅ ARMED %s qty=%s tp=%s sl=%s (orderListId=%s)",
            plan["symbol"], plan["qty"], plan["tp_price"],
            plan["sl_stop"], list_id,
        )
        # Best-effort journal record
        await self._journal_event("SAFETY_OCO_ARMED", plan, list_id)
        return True

    async def _journal_event(
        self, event: str, plan: Dict[str, Any], list_id: Any
    ) -> None:
        try:
            tj = getattr(self.shared_state, "trade_journal", None) or getattr(
                self, "trade_journal", None
            )
            if tj is None:
                return
            payload = {
                "event": event,
                "ts": time.time(),
                "symbol": plan["symbol"],
                "qty": plan["qty"],
                "tp_price": plan["tp_price"],
                "sl_stop": plan["sl_stop"],
                "sl_limit": plan["sl_limit"],
                "client_order_id": plan["client_order_id"],
                "order_list_id": list_id,
            }
            if hasattr(tj, "log_event"):
                res = tj.log_event(payload)
                if iscoroutine(res):
                    await res
            elif hasattr(tj, "write"):
                tj.write(json.dumps(payload) + "\n")
        except Exception:
            pass

    # ─── Status helper ─────────────────────────────────────────────────────
    async def _safe_status_update(self, status: str, message: str) -> None:
        ts = time.time()
        try:
            statuses = getattr(self.shared_state, "component_statuses", None)
            if isinstance(statuses, dict):
                statuses[self.COMPONENT_NAME] = {
                    "status": status, "message": message,
                    "timestamp": ts, "ts": ts,
                }
            last_seen = getattr(self.shared_state, "component_last_seen", None)
            if isinstance(last_seen, dict):
                last_seen[self.COMPONENT_NAME] = ts
        except Exception:
            pass
        try:
            uh = getattr(self.shared_state, "update_system_health", None)
            if callable(uh):
                res = uh(component=self.COMPONENT_NAME, status=status, message=message)
                if iscoroutine(res):
                    await asyncio.wait_for(res, timeout=1.0)
        except Exception:
            pass
        try:
            cs = getattr(self.shared_state, "update_component_status", None)
            if callable(cs):
                res = cs(self.COMPONENT_NAME, status, message)
                if iscoroutine(res):
                    await asyncio.wait_for(res, timeout=1.0)
        except Exception:
            pass
