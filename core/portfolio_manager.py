__all__ = ["PortfolioManager"]

import logging
import asyncio
import time
import json
from datetime import datetime
from typing import Dict, Any, Optional
from decimal import Decimal, ROUND_DOWN, InvalidOperation
from core.stubs import maybe_await, maybe_call

logger = logging.getLogger("PortfolioManager")

# --- Global Helpers ---
STABLECOIN_1to1 = {"USDT", "FDUSD", "TUSD", "BUSD", "USDC"}  # extend as needed

def _safe_emit_event(ss, name: str, payload: dict) -> None:
    """
    Emit a SharedState event from sync contexts without 'never awaited' warnings.
    If there's a running loop, schedule it; otherwise run it to completion.
    """
    try:
        if not ss or not hasattr(ss, "emit_event"):
            return
        loop = asyncio.get_running_loop()
        loop.create_task(ss.emit_event(name, payload))
    except RuntimeError:
        # No running loop → run inline once
        asyncio.run(ss.emit_event(name, payload))

def _d(val, q="0.00000001"):
    """
    Safely convert a value to a Decimal, or return a default Decimal(q) on error.
    """
    try:
        return Decimal(str(val))
    except Exception:
        return Decimal(q)

def _iso_now() -> str:
    """Return current timestamp in ISO format."""
    return datetime.utcnow().isoformat()

def _emit_health(ss: Any, status: str, message: str):
    """Emit a health status event safely from sync contexts."""
    try:
        _safe_emit_event(ss, "HealthStatus", {
            "source": "PortfolioManager", "status": status, "message": message
        })
    except Exception:
        pass


# --- PortfolioManager Class ---
class PortfolioManager:
    def __init__(self, config, shared_state, exchange_client, database_manager, notification_manager=None):
        self.config = config
        self.shared_state = shared_state
        self.exchange_client = exchange_client
        self.database_manager = database_manager
        self.notification_manager = notification_manager

        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("PortfolioManager (Treasury & Asset Management) initialized.")

        self.last_portfolio_audit_time: Optional[float] = None
        self._run_lock = asyncio.Lock()               # prevent overlapping loops
        # Lazy initialization via properties below
        self._last_alert_ts: float = 0.0              # throttle alerting
        self._base_ccy = getattr(config, "BASE_CURRENCY", "USDT")

    # ------------- Dust classification -------------
    async def _is_dust(self, asset: str, amount: Decimal, price: Optional[Decimal]) -> bool:
        """
        اعتبر الرصيد Dust إذا كانت قيمته الاسمية أقل من حد التداول (minNotional).
        - للستيبلكوينز 1:1 (USDT/USDC/FDUSD/BUSD/TUSD): نستخدم عتبة ثابتة صغيرة من config.
        - لباقي الأصول: نقارن (amount * price) بـ minNotional من exchange_client.get_symbol_info،
          مع fallback إلى config.MIN_NOTIONAL_USDT (افتراضياً 5.0).
        - Fail-safe: أي حالة غير مؤكدة تُعتبر Dust لحماية النظام.
        """
        try:
            asset = (asset or "").upper()
            if not asset or amount is None:
                return True
            if amount <= Decimal("0"):
                return True

            # Stablecoins: 1:1 → استخدم حد صغير من الإعدادات
            if asset in STABLECOIN_1to1:
                stbl_thr = self.dust_threshold_stables
                return amount < stbl_thr

            # للأصول غير المستقرة نحتاج سعر مقابل العملة الأساسية
            if price is None or price <= Decimal("0"):
                return True  # تحفظياً اعتبره Dust

            notional = amount * price

            # حاول جلب minNotional من معلومات الرمز (نمط Binance)
            min_notional = None
            symbol = f"{asset}{self._base_ccy}"
            get_info = getattr(self.exchange_client, "get_symbol_info", None)
            if callable(get_info):
                try:
                    info = await get_info(symbol)
                    if info:
                        # شكل شائع في Binance
                        filters = info.get("filters") if isinstance(info, dict) else None
                        if isinstance(filters, list):
                            for f in filters:
                                ftype = (f.get("filterType") or "").upper()
                                if ftype in {"MIN_NOTIONAL", "NOTIONAL"}:
                                    mn = f.get("minNotional") or f.get("notional")
                                    if mn is not None:
                                        try:
                                            min_notional = Decimal(str(mn))
                                        except Exception:
                                            pass
                                            break
                        # بعض التغليفات قد تضع القيمة مباشرة
                        if min_notional is None and "minNotional" in info:
                            try:
                                min_notional = Decimal(str(info["minNotional"]))
                            except Exception:
                                pass
                except Exception:
                    # تجاهل أخطاء الميتاداتا واستعمل fallback
                    pass

            if min_notional is None:
                min_notional = self.min_notional_usdt

            return notional < min_notional
        except Exception:
            # أي فشل في التصنيف → Dust كتحفّظ
            return True

    # ===== Internal helpers (I/O) =====
    async def _get_price(self, symbol: str) -> float:
        """
        Get a best-effort latest price for a symbol (e.g., BTCUSDT).
        Order of preference:
          1) SharedState.get_latest_price_safe (if available)
          2) ExchangeClient.get_current_price (async)
        Returns 0.0 on failure.
        """
        ss = self.shared_state
        px = None

        # Try SharedState first (may be sync)
        if ss and hasattr(ss, "get_latest_price_safe"):
            try:
                px = ss.get_latest_price_safe(symbol)
            except Exception:
                px = None

        # Fallback to ExchangeClient
        if px is None and self.exchange_client and hasattr(self.exchange_client, "get_current_price"):
            try:
                res = self.exchange_client.get_current_price(symbol)
                px = (await res) if asyncio.iscoroutine(res) else res
            except Exception:
                px = None

        try:
            return float(px or 0.0)
        except Exception:
            return 0.0

    async def _fetch_balances(self) -> dict:
        """
        Prefer ExchangeClient balances; fallback to SharedState snapshot.
        """
        if self.exchange_client and hasattr(self.exchange_client, "get_balances"):
            try:
                return await self.exchange_client.get_balances()
            except Exception as e:
                self.logger.warning("get_balances error: %s", e)

        ss = self.shared_state
        if ss and hasattr(ss, "get_balance_snapshot"):
            try:
                return ss.get_balance_snapshot()
            except Exception:
                pass
        return {}

    async def _fetch_positions(self) -> dict:
        """
        Fetch positions from SharedState snapshot (local cache, NOT exchange).
        
        ARCHITECTURE:
          - SOURCE: SharedState (kept in sync by PositionManager)
          - PURPOSE: Portfolio analysis, NAV computation, UI display
          - COST: Cheap (in-memory lookup)
          - LATENCY: Milliseconds
        
        NOTE: For live exchange reconciliation, see _fetch_and_update_open_positions().
        That method fetches directly from exchange and updates SharedState.
        This method reads the cached snapshot for efficiency.
        """
        ss = self.shared_state
        if ss and hasattr(ss, "get_positions_snapshot"):
            try:
                maybe_positions = ss.get_positions_snapshot()
                # Allow async or sync
                if asyncio.iscoroutine(maybe_positions):
                    return await maybe_positions
                return maybe_positions or {}
            except Exception:
                pass
        return {}

    # ===== Calculations =====
    async def _compute_nav(self, balances: Dict[str, Dict[str, float]]) -> float:
        """
        Robust NAV with:
          - 1:1 stablecoin handling
          - price fallback path
          - dust classification
          - detailed breakdown stored into shared_state for SystemSummary
        """
        balances = balances or {}
        # Note: prices are fetched on-demand via _get_price() with proper fallback chain

        base = self._base_ccy
        equity = _d("0")
        # SharedState stores {free, locked}; derive total
        base_info = balances.get(base, {}) if balances else {}
        cash = _d((base_info.get("free", 0.0) or 0.0)) + _d((base_info.get("locked", 0.0) or 0.0))

        breakdown = {
            "base_cash_total": str(cash),
            "stablecoins": {},
            "assets": {},
            "dust_total": "0",
            "tradable_equity": "0",
            "blocked_equity": "0",
            "missing_price_assets": [],
        }

        equity += cash
        dust_total = _d("0")
        tradable_assets_value = _d("0")

        for asset, info in balances.items():
            free_d = _d(info.get("free", 0.0) or 0.0)
            locked_d = _d(info.get("locked", 0.0) or 0.0)
            total = free_d + locked_d
            if total <= 0 or asset == base:
                continue

            # Stablecoins: treat as 1:1 to base
            if asset in STABLECOIN_1to1:
                equity += total  # 1:1
                breakdown["stablecoins"][asset] = str(total)
                # Treat stablecoin as tradable cash, not dust
                tradable_assets_value += total
                continue

            symbol = f"{asset}{base}"
            px = await self._get_price(symbol)
            if px == 0:
                breakdown["missing_price_assets"].append(asset)
                self.logger.debug("No price for %s; skipping NAV contribution.", symbol)
                continue
            
            price_d = _d(px)

            value = total * price_d
            is_dust = await self._is_dust(asset, total, price_d)

            # accumulate totals
            equity += value
            breakdown["assets"][asset] = {
                "qty": str(total),
                "price": str(price_d),
                "value": str(value),
                "dust": bool(is_dust),
            }
            if is_dust:
                dust_total += value
            else:
                tradable_assets_value += value

        breakdown["dust_total"] = str(dust_total)
        # Tradable equity = base cash + stablecoins + non-dust assets
        tradable_equity = (equity - dust_total)
        breakdown["tradable_equity"] = str(tradable_equity)

        # Optional "blocked_equity" concept: everything not immediately tradable
        breakdown["blocked_equity"] = str(dust_total)

        # Store a rounded float to legacy field + keep the detailed breakdown
        nav_float = float(equity.quantize(_d("0.001")))
        try:
            if hasattr(self.shared_state, "set_portfolio_breakdown"):
                maybe = self.shared_state.set_portfolio_breakdown(breakdown)
                if asyncio.iscoroutine(maybe):
                    await maybe
            else:
                # fallback attribute; your SystemSummary can read this
                self.shared_state._portfolio_breakdown = breakdown
        except Exception:
            self.logger.debug("Failed to persist portfolio breakdown.", exc_info=True)

        # Optional: info-level summary once per loop
        self.logger.info(
            "[NAV] cash=%s %s | tradable=%s | dust=%s | equity=%s",
            cash, base, breakdown["tradable_equity"], breakdown["dust_total"], nav_float
        )
        return nav_float

    # ===== Public lifecycle =====
    async def initialize_portfolio_audit(self):
        """One-time boot audit (balances → positions+discover → snapshot)."""
        self.logger.info("Starting initial portfolio audit...")

        # run_once covers all the logic needed
        await self.run_once()

        self.last_portfolio_audit_time = time.time()
        self.logger.info("✅ Initial portfolio audit completed.")

    async def start_periodic_update(self, interval_seconds: int = 300):
        """Legacy periodic loop (kept for compatibility)."""
        await self.run(interval_seconds)

    async def run(self, interval_seconds: int = 300):
        """
        Main loop (balances+positions+snapshot). Fixed-rate scheduling, no overlap,
        jittered sleeps, and throttled alerts.
        """
        self.logger.info("📊 PortfolioManager run loop started (interval=%ss).", interval_seconds)
        while True:
            loop_start = time.time()
            try:
                # prevent overlapping cycles if previous one is still running
                async with self._run_lock:
                    await self.run_once()
            except asyncio.CancelledError:
                self.logger.info("PortfolioManager run loop cancelled.")
                break
            except Exception as e:
                self.logger.error("❌ Error in PortfolioManager run loop: %s", e, exc_info=True)
                await self._maybe_alert(f"PortfolioManager run error: {e}")

            # fixed-rate with small jitter to avoid thundering herd
            elapsed = time.time() - loop_start
            target_sleep = max(0.0, interval_seconds - elapsed)
            jitter = min(0.2 * interval_seconds, 1.0)  # up to 1s max jitter
            await asyncio.sleep(target_sleep + (jitter * 0.5))  # cheap, stable jitter

    # ===== One-shot (optional external callers) =====
    async def run_once(self) -> dict:
        """
        Fetch balances & positions, compute NAV, update SharedState, and emit a PortfolioSnapshot.
        This is a single-tick variant useful for tests or ad-hoc calls.
        """
        ss = self.shared_state

        # Fetch fresh data
        balances = await self._fetch_balances()
        positions = await self._fetch_positions()

        # Push balances/positions to SharedState if APIs exist
        if ss and hasattr(ss, "update_balances"):
            try:
                await maybe_call(ss.update_balances, balances)
            except Exception:
                self.logger.debug("update_balances failed (non-fatal).", exc_info=True)

        if ss and hasattr(ss, "update_positions"):
            try:
                await maybe_call(ss.update_positions, positions)
            except Exception:
                self.logger.debug("update_positions failed (non-fatal).", exc_info=True)

        # Compute NAV (prefer SharedState helper if present)
        nav = None
        if ss and hasattr(ss, "get_nav_quote"):
            try:
                maybe_nav = ss.get_nav_quote()
                nav = float(maybe_nav if maybe_nav is not None else 0.0)
            except Exception:
                nav = None

        if nav is None:
            nav = await self._compute_nav(balances or {})

        # Emit standardized PortfolioSnapshot (evt)
        payload = {
            "timestamp": _iso_now(),
            "nav_quote": float(nav or 0.0),
            "balances": balances or {},
            "positions": positions or {},
        }
        try:
            if ss and hasattr(ss, "emit_event"):
                await maybe_call(ss.emit_event, "PortfolioSnapshot", payload)
        except Exception:
            self.logger.debug("emit_event(PortfolioSnapshot) failed (non-fatal).", exc_info=True)

        _emit_health(ss, "Running", "Snapshot updated")
        self.logger.info(
            "Portfolio snapshot: NAV=%.2f balances=%d positions=%d",
            payload["nav_quote"], len(balances or {}), len(positions or {})
        )
        return payload
    
    # ===== Internal helpers (I/O) =====
    async def _fetch_and_update_open_positions(self):
        """
        Fetch positions directly from EXCHANGE and update SharedState.
        
        ARCHITECTURE:
          - SOURCE: Exchange API (authoritative, live data)
          - PURPOSE: Reconciliation, gap detection, recovery
          - COST: API call (network roundtrip)
          - LATENCY: 100-500ms
        
        NOTE: This is a synchronization method, NOT used by run_once().
        Used separately for periodic reconciliation or recovery if needed.
        For normal portfolio updates, see _fetch_positions() (reads SharedState cache).
        
        Accepts two common formats:
          A) {"<SYMBOL>": {"qty": 0.5}, ...}
          B) {"BTC": {"free":..., "locked":...}, ...}  → maps to symbol with qty
        """
        raw = await self.exchange_client.get_open_positions()
        raw = raw or {}

        # Normalize → bulk map: { "<SYMBOL>": {qty, side, avg_price, entry_ts, mark_price, notional} }
        bulk: Dict[str, Dict[str, Any]] = {}

        async def _normalize(sym: str, rec: Dict[str, Any]) -> Optional[Dict[str, Any]]:
            """
            Robust position normalization from varied broker/exchange shapes.
            Required out: qty (Decimal), side (LONG/SHORT), avg_price, mark_price, notional.
            """
            try:
                # qty
                if "qty" in rec:
                    qty = _d(rec.get("qty", "0"))
                else:
                    # balances-like record → free+locked
                    free = _d(rec.get("free", "0"))
                    locked = _d(rec.get("locked", "0"))
                    qty = free + locked

                if qty <= Decimal("0"):
                    return None

                # avg/entry price (support several keys)
                avg_price = _d(
                    rec.get("avg_price")
                    or rec.get("entry_price")
                    or rec.get("entryPrice")
                    or rec.get("avgPrice")
                    or "0"
                )

                # side
                side = str(rec.get("side") or ("LONG" if qty > 0 else "SHORT")).upper()
                if side not in ("LONG", "SHORT"):
                    side = "LONG" if qty > 0 else "SHORT"

                # mark price (best-effort)
                mark_price = await self._get_price(sym)
                mp = _d(mark_price)
                notional = (qty * mp) if mp > 0 else Decimal("0")

                # entry timestamp (optional)
                entry_ts = rec.get("entry_ts") or rec.get("timestamp") or rec.get("time") or None

                return {
                    "symbol": sym,
                    "qty": float(qty),
                    "side": side,
                    "avg_price": float(avg_price) if avg_price > 0 else None,
                    "entry_ts": float(entry_ts) if entry_ts else None,
                    "mark_price": float(mp) if mp > 0 else None,
                    "notional": float(notional),
                }
            except Exception:
                return None

        # Build normalized view
        for k, v in raw.items():
            if k.endswith(self._base_ccy):
                sym, rec = k, (v or {})
            else:
                sym = f"{k}{self._base_ccy}"
                rec = v or {}
            norm = await _normalize(sym, rec)
            if norm:
                bulk[sym] = norm

        # Optional: reconcile shrink (close symbols that disappeared upstream)
        prev = {}
        if hasattr(self.shared_state, "get_positions_snapshot"):
            try:
                prev = self.shared_state.get_positions_snapshot() or {}
            except Exception:
                prev = {}
        prev_syms = set(prev.keys()) if isinstance(prev, dict) else set()
        curr_syms = set(bulk.keys())
        missing = prev_syms - curr_syms
        for sym in missing:
            # mark as flat to clear out stale positions
            bulk[sym] = {
                "symbol": sym,
                "qty": 0.0,
                "side": "LONG",
                "avg_price": None,
                "entry_ts": None,
                "mark_price": None,
                "notional": 0.0,
            }

        # Write-through: prefer bulk API, fallback to per-symbol
        wrote_bulk = False
        if hasattr(self.shared_state, "update_positions"):
            try:
                res = self.shared_state.update_positions(bulk)
                if asyncio.iscoroutine(res):
                    await res
                wrote_bulk = True
            except Exception:
                wrote_bulk = False

        if not wrote_bulk:
            tasks = [self.shared_state.update_position(sym, rec) for sym, rec in bulk.items()]
            await asyncio.gather(*tasks, return_exceptions=True)

        self.logger.debug("Open positions reconciled: %d active, %d closed.", len(curr_syms), len(missing))

    async def _save_portfolio_snapshot(self):
        try:
            balances = self.shared_state.get_balance_snapshot() if hasattr(self.shared_state, "get_balance_snapshot") else {}
            # Prefer the same source the UI/summary uses for consistency
            if hasattr(self.shared_state, "get_positions_snapshot"):
                open_positions = self.shared_state.get_positions_snapshot() or {}
            elif hasattr(self.shared_state, "get_all_open_trades"):
                open_positions = self.shared_state.get_all_open_trades() or {}
            else:
                open_positions = {}

            base = self._base_ccy
            base_bal = balances.get(base, {}) if balances else {}
            free = float(base_bal.get("free", 0.0) or 0.0)
            locked = float(base_bal.get("locked", 0.0) or 0.0)
            total = free + locked
            available = float(base_bal.get("spendable", free))  # use spendable if present

            net_worth = float(getattr(self.shared_state, "_total_value", 0.0) or 0.0)
            pnl_current_session = float(getattr(self.shared_state, "realized_pnl", 0.0) or 0.0)

            holdings_json = json.dumps(
                {"balances": balances, "positions": open_positions},
                separators=(",", ":"),
                ensure_ascii=False,
                default=float,
            )

            query = (
                "INSERT INTO portfolio_snapshots "
                "(timestamp, total_balance, available_balance, locked_balance, net_worth, pnl, holdings_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)"
            )
            params = (
                float(time.time()),
                total,
                available,
                locked,
                net_worth,
                pnl_current_session,
                holdings_json,
            )

            await self.database_manager.insert_row(query, params)
            self.logger.debug("Portfolio snapshot saved.")
        except Exception as e:
            self.logger.error("Error saving portfolio snapshot: %s", e, exc_info=True)
            await self._maybe_alert(f"Portfolio Snapshot Save Error: {e}", level="WARNING")

    # ------------- Accessors (compat) -------------

    def get_current_portfolio_value(self) -> float:
        return float(getattr(self.shared_state, "_total_value", 0.0) or 0.0)

    def get_available_balance(self, asset: str) -> float:
        bal = self.shared_state.balances.get(str(asset).upper(), {})
        return float(bal.get("free", 0.0) or 0.0)

    def get_total_balance(self, asset: str) -> float:
        bal = self.shared_state.balances.get(str(asset).upper(), {})
        return float((bal.get("free", 0.0) or 0.0) + (bal.get("locked", 0.0) or 0.0))

    def get_locked_balance(self, asset: str) -> float:
        bal = self.shared_state.balances.get(str(asset).upper(), {})
        return float(bal.get("locked", 0.0) or 0.0)

    # ------------- Alerts -------------

    async def _maybe_alert(self, message: str, level: str = "ERROR"):
        """
        Throttle alerts to avoid spam during transient outages.
        """
        now = time.time()
        if (now - self._last_alert_ts) < self._alert_cooldown:
            return
        self._last_alert_ts = now

        nm = self.notification_manager
        if nm and getattr(nm, "enabled", False) and hasattr(nm, "send_alert"):
            try:
                await nm.send_alert(message, level=level)
            except Exception:
                # never fail the loop on alert errors
                self.logger.debug("Notification send failed.", exc_info=True)
    # ------------- dynamic properties -------------
    def _cfg(self, key: str, default: Any = None) -> Any:
        # 1. Check SharedState for live/dynamic overrides
        if hasattr(self.shared_state, "dynamic_config"):
            val = self.shared_state.dynamic_config.get(key)
            if val is not None:
                return val

        # 2. Fallback to static config
        if isinstance(self.config, dict):
            return self.config.get(key, default)
        return getattr(self.config, key, default)

    @property
    def _alert_cooldown(self) -> int:
        return int(self._cfg("PORTFOLIO_ALERT_COOLDOWN", 300))

    @property
    def dust_threshold_stables(self) -> Decimal:
        val = self._cfg("DUST_THRESHOLD_STABLES", Decimal("0.5"))
        try:
            return Decimal(str(val))
        except (InvalidOperation, TypeError):
            return Decimal("0.5")

    @property
    def min_notional_usdt(self) -> Decimal:
        val = self._cfg("MIN_NOTIONAL_USDT", 5.0)
        try:
            return Decimal(str(val))
        except (InvalidOperation, TypeError):
            return Decimal("5")

    @property
    def loop_interval(self) -> int:
        return int(self._cfg("PORTFOLIO_MANAGER_INTERVAL", 300))
