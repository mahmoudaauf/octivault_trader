# core/position_manager.py
# PositionManager — reconcile open positions & fills with SharedState (P9-aligned)
# Fixes:
#  - Await all async event emissions (no 'coroutine was never awaited')
#  - Robust sync → async bridging via maybe_call
#  - Warming-up health state until gates are green
#  - Non-fatal reconciliation loop

import asyncio
import logging
import time
from typing import Any, Dict, Optional

from core.stubs import maybe_await  # safe sync/async invocation

logger = logging.getLogger("PositionManager")


class PositionManager:
    """
    Responsibilities:
      - Periodically fetch positions from the exchange
      - Normalize and upsert into SharedState
      - Emit audits/health events (PositionAudit, HealthStatus)

    Contracts / Events:
      - HealthStatus: {"source": "PositionManager", "status": "...", "message": "..."}
      - PositionAudit: {"source": "PositionManager", "active": N, "closed": M, "timestamp": ...}
    """

    def __init__(self, config: Any, shared_state: Any, exchange_client: Any):
        self.config = config
        self.ss = shared_state
        self.ex = exchange_client

        self.interval_s = int(getattr(config, "POSITION_SYNC_INTERVAL_S", 20))
        self._running = False
        self._lock = asyncio.Lock()

        self.base_ccy = getattr(config, "BASE_CURRENCY", "USDT")

        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("PositionManager initialized (interval=%ss).", self.interval_s)

    # ---------------------------
    # Public lifecycle
    # ---------------------------
    async def start(self):
        if self._running:
            return
        self._running = True
        self.logger.info("🎯 Starting PositionManager task...")
        try:
            while self._running:
                try:
                    async with self._lock:
                        await self._tick_once()
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    self.logger.error("PositionManager tick error: %s", e, exc_info=True)
                    await self._emit_health("Error", f"Tick failed: {e}")
                await asyncio.sleep(self.interval_s)
        finally:
            self._running = False
            self.logger.info("PositionManager stopped.")

    async def stop(self):
        self._running = False

    # ---------------------------
    # Core logic
    # ---------------------------
    async def _tick_once(self):
        # Warm-up: don't spam errors before P5 market data & symbols are set
        if not await self._is_ready():
            await self._emit_health("WarmingUp", "Awaiting accepted symbols + market data")
            return

        # Fetch open positions from the exchange (best-effort)
        try:
            raw = await self._get_open_positions_safe()
        except Exception as e:
            await self._emit_health("Error", f"get_open_positions failed: {e}")
            return

        # Normalize into a consistent dict keyed by symbol (e.g., BTCUSDT)
        bulk = await self._normalize_positions(raw or {})

        # Reconcile shrink vs previous snapshot (mark vanished symbols as closed/flat)
        prev = await self._get_prev_positions_snapshot()
        prev_syms = set(prev.keys()) if isinstance(prev, dict) else set()
        curr_syms = set(bulk.keys())
        vanished = prev_syms - curr_syms
        for sym in vanished:
            bulk[sym] = {
                "symbol": sym,
                "quantity": 0.0, # Modified to use "quantity"
                "side": "LONG",
                "avg_price": None,
                "entry_ts": None,
                "mark_price": None,
                "notional": 0.0,
            }

        # Write-through to SharedState (bulk if available)
        wrote_bulk = False
        if hasattr(self.ss, "update_positions"):
            try:
                res = self.ss.update_positions(bulk)
                if asyncio.iscoroutine(res):
                    await res
                wrote_bulk = True
            except Exception:
                wrote_bulk = False

        if not wrote_bulk:
            # Fallback per-symbol upserts
            tasks = []
            for sym, rec in bulk.items():
                if hasattr(self.ss, "update_position"):
                    tasks.append(self.ss.update_position(sym, rec))
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

        # Reflect has-open-positions flag into SharedState (best-effort)
        try:
            if hasattr(self.ss, "set_has_open_positions"):
                await maybe_call(self.ss, "set_has_open_positions", bool(curr_syms))
        except Exception:
            pass

        # Persist the latest snapshot for other components (optional)
        try:
            if hasattr(self.ss, "set_last_positions_snapshot"):
                await maybe_call(self.ss, "set_last_positions_snapshot", bulk)
        except Exception:
            pass

        # Also refresh wallet/balance snapshot so LiquidationAgent & Executors can act
        await self._update_wallet_snapshot()

        # Emit audit + health
        await self._emit_audit(active=len(curr_syms), closed=len(vanished))
        await self._emit_health("Running", f"Reconciled {len(curr_syms)} active, {len(vanished)} closed")

    # ---------------------------
    # Helpers
    # ---------------------------
    async def _is_ready(self) -> bool:
        try:
            have_symbols = False
            if hasattr(self.ss, "accepted_symbols"):
                syms = getattr(self.ss, "accepted_symbols") or {}
                have_symbols = bool(syms)

            have_market = False
            ev = getattr(self.ss, "market_data_ready_event", None)
            if ev and hasattr(ev, "is_set"):
                have_market = ev.is_set()

            return bool(have_symbols and have_market)
        except Exception:
            return False

    async def _get_open_positions_safe(self) -> Dict[str, Any]:
        """
        ExchangeClient.get_open_positions() should return a mapping like:
          {"<SYMBOL>": {"qty": 0.5, "avg_price": 64200.0, ...}, ...}
        but we tolerate broker-specific shapes and let _normalize_positions handle it.
        """
        if not self.ex or not hasattr(self.ex, "get_open_positions"):
            return {}
        res = self.ex.get_open_positions()
        return await res if asyncio.iscoroutine(res) else (res or {})

    async def _normalize_positions(self, raw: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """
        Normalize varied shapes into:
                "symbol": "<SYMBOL>",
                "quantity": float,
                "side": "LONG"|"SHORT",
                "avg_price": float|None,
                "entry_ts": float|None,
                "mark_price": float|None,
                "notional": float
            }, ... }
        """
        out: Dict[str, Dict[str, Any]] = {}

        async def _get_mark_price(sym: str) -> float:
            # Prefer SharedState.safe price, fallback to ExchangeClient
            px = None
            if hasattr(self.ss, "get_latest_price_safe"):
                try:
                    px = await maybe_call(self.ss, "get_latest_price_safe", sym) # Modified to use maybe_call
                except Exception:
                    px = None
            if px is None and hasattr(self.ex, "get_current_price"):
                try:
                    v = self.ex.get_current_price(sym)
                    px = (await v) if asyncio.iscoroutine(v) else v
                except Exception:
                    px = None
            try:
                return float(px or 0.0)
            except Exception:
                return 0.0

        for k, rec in (raw or {}).items():
            try:
                rec = rec or {}
                # Resolve symbol name robustly (handles BTC, BTC/USDT, BTCUSDT)
                raw_sym = (rec.get("symbol") or k or "").upper()
                if raw_sym.endswith(self.base_ccy):
                    sym = raw_sym.replace("/", "")
                elif "/" in raw_sym:
                    # e.g. BTC/USDT -> BTCUSDT
                    parts = raw_sym.split("/")
                    sym = (parts[0] + parts[-1]).upper()
                else:
                    sym = f"{raw_sym}{self.base_ccy}" if raw_sym else k

                # qty
                if "qty" in rec:
                    qty = float(rec.get("qty") or 0.0)
                else:
                    # balances-like structure {free, locked}
                    free = float(rec.get("free") or 0.0)
                    locked = float(rec.get("locked") or 0.0)
                    qty = free + locked

                # Treat dust as flat
                try:
                    dust_eps = float(getattr(self.config, "POSITION_DUST_EPS", 1e-12))
                except Exception:
                    dust_eps = 1e-12
                if abs(qty) <= dust_eps:
                    continue

                if qty <= 0.0:
                    continue

                # side
                side = str(rec.get("side") or ("LONG" if qty > 0 else "SHORT")).upper()
                if side not in ("LONG", "SHORT"):
                    side = "LONG" if qty > 0 else "SHORT"

                # avg/entry price (support multiple keys)
                avg_price = (
                    rec.get("avg_price")
                    or rec.get("entry_price")
                    or rec.get("entryPrice")
                    or rec.get("avgPrice")
                    or None
                )
                avg_price = float(avg_price) if (avg_price is not None and float(avg_price) > 0.0) else None

                # timestamps (optional)
                entry_ts = rec.get("entry_ts") or rec.get("timestamp") or rec.get("time") or None
                entry_ts = float(entry_ts) if entry_ts is not None else None

                # mark price & notional
                mark_price = await _get_mark_price(sym)
                notional = abs(qty) * (mark_price if mark_price > 0 else 0.0)

                out[sym] = {
                    "symbol": sym,
                    "quantity": float(qty), # Modified to use "quantity"
                    "side": side,
                    "avg_price": avg_price,
                    "entry_ts": entry_ts,
                    "mark_price": float(mark_price) if mark_price > 0 else None,
                    "notional": float(notional),
                }
            except Exception:
                # Skip malformed records rather than failing the whole tick
                continue

        return out

    async def _get_prev_positions_snapshot(self) -> Dict[str, Any]:
        if hasattr(self.ss, "get_positions_snapshot"):
            try:
                snap = self.ss.get_positions_snapshot()
                return (await snap) if asyncio.iscoroutine(snap) else (snap or {})
            except Exception:
                return {}
        return {}

    async def _update_wallet_snapshot(self):
        """Best-effort wallet refresh + events.
        Looks for ExchangeClient.get_balances() -> {"USDT": {"free": x, "locked": y}, ...}
        Updates SharedState if the corresponding methods exist, and emits a WalletSnapshot event.
        """
        balances: Dict[str, Any] = {}
        try:
            if self.ex and hasattr(self.ex, "get_balances"):
                res = self.ex.get_balances()
                balances = await res if asyncio.iscoroutine(res) else (res or {})
        except Exception:
            balances = {}

        # Try to update SharedState with raw balances
        try:
            if balances and hasattr(self.ss, "update_wallet"):
                await maybe_call(self.ss, "update_wallet", balances)
        except Exception:
            pass

        # Extract base currency free/locked if present
        base_info = (balances or {}).get(self.base_ccy, {}) if isinstance(balances, dict) else {}
        try:
            base_free = float(base_info.get("free") or 0.0)
        except Exception:
            base_free = 0.0
        try:
            base_locked = float(base_info.get("locked") or 0.0)
        except Exception:
            base_locked = 0.0

        # Provide convenient mirrors for other components (optional)
        try:
            if hasattr(self.ss, "set_base_balance"):
                await maybe_call(self.ss, "set_base_balance", self.base_ccy, base_free, base_locked)
        except Exception:
            pass

        # Emit a structured wallet snapshot event
        payload = {
            "source": "PositionManager",
            "base": self.base_ccy,
            "base_free": base_free,
            "base_locked": base_locked,
            "timestamp": time.time(),
        }
        if balances:
            payload["assets"] = {k: {"free": (v or {}).get("free", 0.0), "locked": (v or {}).get("locked", 0.0)} for k, v in balances.items() if isinstance(v, dict)}

        if self.ss and hasattr(self.ss, "emit_event"):
            await maybe_call(self.ss, "emit_event", "WalletSnapshot", payload)

        # Emit a low-balance hint if applicable (non-fatal)
        try:
            min_free = float(getattr(self.config, "MIN_BASE_FREE_FOR_ORDERS", 5.0))
        except Exception:
            min_free = 5.0
        if base_free < min_free:
            warn_payload = {
                "source": "PositionManager",
                "status": "LowBalance",
                "message": f"{self.base_ccy} free {base_free:.8f} < threshold {min_free}",
                "threshold": min_free,
                "base": self.base_ccy,
                "timestamp": time.time(),
            }
            if self.ss and hasattr(self.ss, "emit_event"):
                await maybe_call(self.ss, "emit_event", "HealthStatus", warn_payload)

    # ---------------------------
    # Event emitters (awaited)
    # ---------------------------
    async def _emit_health(self, status: str, message: str):
        payload = {
            "source": "PositionManager",
            "status": status,
            "message": message,
            "timestamp": time.time(),
        }
        if self.ss and hasattr(self.ss, "emit_event"):
            await maybe_call(self.ss, "emit_event", "HealthStatus", payload)
        else:
            self.logger.debug("HealthStatus (no SS): %s", payload)

    async def _emit_audit(self, active: int, closed: int):
        payload = {
            "source": "PositionManager",
            "active": int(active),
            "closed": int(closed),
            "timestamp": time.time(),
        }
        if self.ss and hasattr(self.ss, "emit_event"):
            # This was previously causing: coroutine 'emit_event' was never awaited
            await maybe_call(self.ss, "emit_event", "PositionAudit", payload)
        else:
            self.logger.debug("PositionAudit (no SS): %s", payload)
