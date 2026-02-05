# core/cash_router.py
from __future__ import annotations
import asyncio
import logging
import math
from typing import Any, Dict, List, Optional, Tuple, Set

class CashRouter:
    """
    Routes idle/non-core balances into USDT to satisfy temporary liquidity needs,
    without liquidating core risk positions. All actions are best-effort and
    respect exchange filters/fees.

    Config flags:
      - CR_ENABLE: bool (default True)
      - CR_SWEEP_DUST_MIN: float USDT (default 1.0)
      - CR_ENABLE_REDEEM_STABLES: bool (default True)
      - CR_STABLE_SYMBOLS: list[str] e.g. ["FDUSD","BUSD","USDC"]
      - CR_QUOTE: str (default "USDT")
      - CR_PROTECTED_ASSETS: list[str] (assets we must never touch, e.g. ["BTC","ETH"])
      - CR_MAX_ACTIONS: int (limit dust/stable actions per run, default 8)
      - CR_PRICE_SLIPPAGE_BPS: int (downside slippage safety for sells, default 25)
      - CR_FEE_BPS: int (estimated fee bps for market sells/convert, default 10)
    """

    def __init__(self, config: Any, logger: Optional[logging.Logger] = None,
                 app: Optional[Any] = None, shared_state: Optional[Any] = None,
                 exchange_client: Optional[Any] = None) -> None:
        self.config = config
        self.logger = logger or logging.getLogger("CashRouter")
        self.app = app
        self.shared_state = shared_state
        self.ex = exchange_client
        # single-flight guard to avoid concurrent frees racing
        self._lock = asyncio.Lock()

        if not self.logger.handlers:
            import sys as _sys
            h = logging.StreamHandler(_sys.stdout)
            fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            h.setFormatter(fmt)
            self.logger.addHandler(h)
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False

        self.quote = getattr(config, "CR_QUOTE", "USDT") or "USDT"

    # ---------- config util ----------
    def _cfg_bool(self, name: str, default: bool = False) -> bool:
        try:
            v = getattr(self.config, name, None)
            if isinstance(v, bool): return v
            if isinstance(v, str): return v.strip().lower() in ("1","true","yes","y","on")
        except Exception:
            pass
        return default

    def _cfg_float(self, name: str, default: float = 0.0) -> float:
        try:
            return float(getattr(self.config, name, default))
        except Exception:
            return float(default)

    def _cfg_int(self, name: str, default: int = 0) -> int:
        try:
            return int(getattr(self.config, name, default))
        except Exception:
            return int(default)

    def _cfg_list(self, name: str, default: Optional[List[str]] = None) -> List[str]:
        try:
            v = getattr(self.config, name, None)
            if isinstance(v, (list, tuple, set)):
                return [str(x) for x in v]
            if isinstance(v, str):
                return [s.strip() for s in v.split(',') if s.strip()]
        except Exception:
            pass
        return list(default or [])

    def _protected_assets(self) -> Set[str]:
        try:
            vals = getattr(self.config, "CR_PROTECTED_ASSETS", []) or []
            if isinstance(vals, str):
                vals = [v.strip() for v in vals.split(",") if v.strip()]
            return {str(v).upper() for v in vals}
        except Exception:
            return set()

    # ---------- account helpers ----------
    async def _free_usdt(self) -> float:
        """
        Fast path for free USDT from shared_state (if exposed), otherwise via balances.
        """
        try:
            if self.shared_state and hasattr(self.shared_state, "free_usdt"):
                r = self.shared_state.free_usdt()
                return float(await r) if asyncio.iscoroutine(r) else float(r)
            if self.shared_state:
                return float(getattr(self.shared_state, "balances", {}).get(self.quote, {}).get("free", 0.0))
        except Exception:
            pass
        return 0.0

    async def _balances(self) -> Dict[str, Dict[str, float]]:
        bals = {}
        if not self.ex:
            return bals
        try:
            if hasattr(self.ex, "get_balances"):
                res = self.ex.get_balances()
                res = await res if asyncio.iscoroutine(res) else res
                if isinstance(res, dict):
                    # normalize to {"ASSET":{"free":float,"locked":float}}
                    bals = {k:strdict(v) for k,v in res.items()}  # type: ignore[name-defined]
        except Exception:
            self.logger.debug("balances fetch failed", exc_info=True)
        return bals

    # ---------- market/symbol helpers ----------
    async def _get_book(self, symbol: str) -> Dict[str, float]:
        """
        Returns {"bid": float|0.0, "ask": float|0.0, "spread_bps": float|None}
        """
        bid = ask = 0.0
        try:
            if self.ex and hasattr(self.ex, "get_order_book_ticker"):
                r = self.ex.get_order_book_ticker(symbol)
                r = await r if asyncio.iscoroutine(r) else r
                if isinstance(r, dict):
                    bid = float(r.get("bidPrice") or r.get("bid") or 0.0)
                    ask = float(r.get("askPrice") or r.get("ask") or 0.0)
        except Exception:
            self.logger.debug("order book fetch failed for %s", symbol, exc_info=True)
        spread_bps = None
        try:
            if bid > 0.0 and ask >= bid:
                spread_bps = (ask - bid) / ask * 10000.0
        except Exception:
            spread_bps = None
        return {"bid": bid, "ask": ask, "spread_bps": (float(spread_bps) if spread_bps is not None else None)}
    async def _get_price(self, symbol: str) -> Optional[float]:
        price = None
        try:
            if hasattr(self.ex, "get_symbol_price"):
                r = self.ex.get_symbol_price(symbol)
                price = await r if asyncio.iscoroutine(r) else r
        except Exception:
            price = None
        return float(price) if price is not None else None

    async def _get_filters(self, symbol: str) -> Dict[str, Any]:
        """
        Expected keys from exchange:
          - min_notional: float or None
          - lot_step: float (stepSize) or None
          - min_qty: float or None
          - max_qty: float or None
        """
        out: Dict[str, Any] = {}
        try:
            if hasattr(self.ex, "get_symbol_filters"):
                r = self.ex.get_symbol_filters(symbol)
                r = await r if asyncio.iscoroutine(r) else r
                if isinstance(r, dict):
                    out = r
        except Exception:
            self.logger.debug("filters fetch failed for %s", symbol, exc_info=True)
        return out

    async def _min_exit_quote(self, symbol: str, price: Optional[float], min_notional: float) -> float:
        """Return exit-feasibility floor for a symbol, falling back to min_notional."""
        try:
            if self.shared_state and hasattr(self.shared_state, "compute_symbol_exit_floor"):
                exit_info = await self.shared_state.compute_symbol_exit_floor(symbol, price=price)
                return float(exit_info.get("min_exit_quote", 0.0) or 0.0)
        except Exception:
            pass
        return float(min_notional or 0.0)

    @staticmethod
    def _round_step(qty: float, step: Optional[float]) -> float:
        if not step or step <= 0:
            return float(qty)
        # floor to step size to avoid rejection
        return math.floor(qty / step) * step

    async def _market_sell(self, symbol: str, qty: float, *, tag: str = "balancer") -> Any:
        """
        Wrapper over exchange.market_sell that prefers passing an audit tag, and falls back
        to a tag-less call for exchanges that don't accept it.
        """
        if not hasattr(self.ex, "market_sell"):
            return None
        try:
            r = self.ex.market_sell(symbol, qty, tag=tag)
            return (await r) if asyncio.iscoroutine(r) else r
        except TypeError:
            # Some clients don't accept tag kwarg
            r = self.ex.market_sell(symbol, qty)
            return (await r) if asyncio.iscoroutine(r) else r

    # ---------- core actions ----------
    async def sweep_dust(self, *, min_quote: Optional[float] = None, want_usdt: Optional[float] = None) -> Dict[str, Any]:
        """
        Convert small, non-protected free balances into the quote asset respecting
        price, min_notional and lot step filters. Can optionally throttle by a dust max
        and sell only what's needed to reach want_usdt. Returns an estimate of freed USDT.
        """
        if not self._cfg_bool("CR_ENABLE", True):
            return {"ok": False, "freed": 0.0, "reason": "disabled"}

        min_quote = float(min_quote if min_quote is not None else self._cfg_float("CR_SWEEP_DUST_MIN", 1.0))
        max_actions = max(1, self._cfg_int("CR_MAX_ACTIONS", 8))
        slippage_bps = max(0, self._cfg_int("CR_PRICE_SLIPPAGE_BPS", 15))
        fee_bps = max(0, self._cfg_int("CR_FEE_BPS", 10))
        protected = self._protected_assets()

        # harmonize floors with global settings (if provided)
        min_order_usdt = self._cfg_float("MIN_ORDER_USDT", 0.0)
        quote_min_notional = self._cfg_float("QUOTE_MIN_NOTIONAL", 0.0)
        min_quote = max(min_quote, min_order_usdt, quote_min_notional)

        # spread cap (prefer LIQUIDATION_SPREAD_CAP_BPS, fallback CR_SPREAD_CAP_BPS)
        spread_cap_bps = self._cfg_float("LIQUIDATION_SPREAD_CAP_BPS", self._cfg_float("CR_SPREAD_CAP_BPS", 12.0))

        dust_max = self._cfg_float("CR_DUST_MAX_USDT", 25.0)
        remaining_need = float(want_usdt) if want_usdt is not None else float('inf')

        if not self.ex:
            return {"ok": False, "freed": 0.0, "reason": "no_exchange_client"}

        freed = 0.0
        actions: List[Dict[str, Any]] = []
        try:
            bals = await self._balances()
            # Prefer smallest notional first to consolidate efficiently
            items = []
            for asset, row in bals.items():
                if not isinstance(row, dict):
                    continue
                asset_up = str(asset).upper()
                if asset_up in (self.quote, None) or asset_up in protected:
                    continue
                free_amt = float(row.get("free", 0.0) or 0.0)
                if free_amt <= 0:
                    continue
                items.append((asset_up, free_amt))

            # gather prices and books in sequence to keep it simple & robust
            priced: List[Tuple[str, float, Optional[float], Dict[str, float]]] = []
            for asset, amt in items:
                sym = f"{asset}{self.quote}"
                p = await self._get_price(sym)
                book = await self._get_book(sym)
                priced.append((asset, amt, p, book))
            priced.sort(key=lambda x: (x[1] * (x[2] or 0.0)))

            for asset, free_amt, price, book in priced:
                if len(actions) >= max_actions:
                    break
                if not price:
                    continue

                # enforce spread cap if we have a book
                bid = float(book.get("bid") or 0.0)
                ask = float(book.get("ask") or 0.0)
                spread_bps = book.get("spread_bps")
                if spread_bps is not None and spread_bps > spread_cap_bps:
                    # skip if market is too wide
                    continue

                # conservative execution price: prefer bid, else slippage on last
                if bid > 0.0:
                    exec_price = bid * (1.0 - slippage_bps / 10_000.0)
                else:
                    exec_price = float(price) * (1.0 - slippage_bps / 10_000.0)

                est_quote = free_amt * exec_price
                if est_quote < min_quote:
                    continue

                if est_quote > dust_max and math.isinf(remaining_need):
                    # when not targeting a specific amount, treat only small notional (dust)
                    continue

                sym = f"{asset}{self.quote}"
                filt = await self._get_filters(sym)
                min_notional = float(filt.get("min_notional") or 0.0)
                step = float(filt.get("lot_step") or 0.0)
                min_qty = float(filt.get("min_qty") or 0.0)
                max_qty = float(filt.get("max_qty") or 0.0)
                min_required = await self._min_exit_quote(sym, exec_price, min_notional)
                # Enforce exit-feasibility floor at conservative price
                if min_required and est_quote < min_required:
                    continue

                # cap sale by remaining need
                max_qty_by_need = free_amt
                if math.isfinite(remaining_need) and remaining_need > 0 and exec_price > 0:
                    max_qty_by_need = min(max_qty_by_need, remaining_need / exec_price)
                qty = self._round_step(max_qty_by_need, step)
                if (min_qty and qty < min_qty) or (max_qty and qty > max_qty) or qty <= 0:
                    continue

                # try to sell; ignore failures gracefully
                ok = False
                try:
                    res = await self._market_sell(sym, qty, tag="balancer")
                    ok = bool(res)
                except Exception:
                    ok = False

                if ok:
                    # net freed after fee estimate
                    gross = qty * exec_price
                    net = gross * (1.0 - fee_bps / 10_000.0)
                    freed += net
                    actions.append({
                        "action": "sweep_dust",
                        "asset": asset,
                        "symbol": sym,
                        "qty": float(f"{qty:.12f}"),
                        "exec_price": float(f"{exec_price:.10f}"),
                        "net_quote": float(f"{net:.6f}"),
                        "spread_bps": (float(spread_bps) if spread_bps is not None else None)
                    })
                    self.logger.info("[Liquidity] action sweep_dust %s qty=%.8f exec=%.8f net=%.6f spread_bps=%s",
                                     sym, qty, exec_price, net,
                                     (None if spread_bps is None else float(spread_bps)))
                    if math.isfinite(remaining_need) and remaining_need > 0 and exec_price > 0:
                        remaining_need = max(0.0, remaining_need - net)
                    if math.isfinite(remaining_need) and remaining_need <= 0.0:
                        break
        except Exception:
            self.logger.debug("sweep_dust failed", exc_info=True)

        return {"ok": (freed > 0.0), "freed": float(f"{freed:.6f}"), "actions": actions, "reason": "dust_sweep"}

    async def free_dust(self, *, min_quote: Optional[float] = None, want_usdt: Optional[float] = None) -> Dict[str, Any]:
        return await self.sweep_dust(min_quote=min_quote, want_usdt=want_usdt)

    async def redeem_wrapped_stables(self, *, want_usdt: Optional[float] = None) -> Dict[str, Any]:
        """
        Convert non-quote stable balances to the quote stable using either a native
        convert endpoint (preferred) or a market sell fallback. Optionally, sell only up to want_usdt.
        """
        if not self._cfg_bool("CR_ENABLE_REDEEM_STABLES", True):
            return {"ok": False, "freed": 0.0, "reason": "disabled"}
        stables = getattr(self.config, "CR_STABLE_SYMBOLS", ["FDUSD","BUSD","USDC"]) or []
        protected = self._protected_assets()
        if not self.ex:
            return {"ok": False, "freed": 0.0, "reason": "no_exchange_client"}

        freed = 0.0
        actions: List[Dict[str, Any]] = []
        fee_bps = max(0, self._cfg_int("CR_FEE_BPS", 10))
        slippage_bps = max(0, self._cfg_int("CR_PRICE_SLIPPAGE_BPS", 15))
        remaining_need = float(want_usdt) if want_usdt is not None else float('inf')

        try:
            bals = await self._balances()
            for asset, row in (bals.items() if isinstance(bals, dict) else []):
                asset_up = str(asset).upper()
                if asset_up == self.quote or asset_up not in stables or asset_up in protected:
                    continue
                amt = float(row.get("free", 0.0) or 0.0)
                if amt <= 0:
                    continue

                to_convert_amt = amt
                if math.isfinite(remaining_need) and remaining_need > 0:
                    # Ensure enough gross is converted so NET after fee meets remaining_need
                    fee = max(0.0, self._cfg_int("CR_FEE_BPS", 10)) / 10_000.0
                    to_convert_amt = min(amt, remaining_need / max(1e-12, (1.0 - fee)))
                if to_convert_amt <= 0:
                    continue

                sym = f"{asset_up}{self.quote}"
                # spread cap (prefer LIQUIDATION_SPREAD_CAP_BPS, fallback CR_SPREAD_CAP_BPS)
                spread_cap_bps = self._cfg_float("LIQUIDATION_SPREAD_CAP_BPS", self._cfg_float("CR_SPREAD_CAP_BPS", 12.0))
                ok = False
                net = 0.0
                try:
                    if hasattr(self.ex, "convert_stable"):
                        # preferred: fee-efficient convert endpoint
                        r = self.ex.convert_stable(asset_up, self.quote, to_convert_amt)
                        r = await r if asyncio.iscoroutine(r) else r
                        ok = bool(r)
                        if ok:
                            net = to_convert_amt * (1.0 - fee_bps / 10_000.0)
                    elif hasattr(self.ex, "market_sell"):
                        # check spread; if too wide, skip sell fallback
                        book = await self._get_book(sym)
                        s_bps = book.get("spread_bps")
                        if s_bps is not None and s_bps > spread_cap_bps:
                            ok = False
                        else:
                            # sell amount up to to_convert_amt, respecting lot step if available
                            filt = await self._get_filters(sym)
                            step = float(filt.get("lot_step") or 0.0)
                            min_qty = float(filt.get("min_qty") or 0.0)
                            max_qty = float(filt.get("max_qty") or 0.0)
                            min_notional = float(filt.get("min_notional") or 0.0)
                            # conservative exec price: use bid minus slippage; default to 1.0 if no bid (stable pairs)
                            book_bid = float(book.get("bid") or 0.0) if isinstance(book, dict) else 0.0
                            exec_price = (book_bid * (1.0 - slippage_bps / 10_000.0)) if book_bid > 0.0 else 1.0
                            fee = max(0.0, self._cfg_int("CR_FEE_BPS", 10)) / 10_000.0

                            # Determine quantity so that net after fee meets remaining_need (capped by to_convert_amt)
                            if exec_price <= 0.0:
                                ok = False
                            else:
                                gross_needed = (remaining_need / max(1e-12, (1.0 - fee)))
                                qty_needed = gross_needed / exec_price
                                to_convert_amt = min(amt, qty_needed)
                                qty = self._round_step(to_convert_amt, step)

                                if (min_qty and qty < min_qty) or (max_qty and qty > max_qty) or qty <= 0:
                                    ok = False
                                else:
                                    min_required = await self._min_exit_quote(sym, exec_price, min_notional)
                                    # optional conservative notional check
                                    if min_required and (qty * exec_price) < min_required:
                                        ok = False
                                    else:
                                        r = await self._market_sell(sym, qty, tag="balancer")
                                        ok = bool(r)
                                        if ok:
                                            net = qty * exec_price * (1.0 - fee)
                except Exception:
                    ok = False

                if ok:
                    freed += net
                    actions.append({"action": "redeem_stable", "asset": asset_up, "amount": float(f"{to_convert_amt:.8f}"), "symbol": sym})
                    self.logger.info("[Liquidity] action redeem_stable %s amt=%.8f net=%.6f",
                                     sym, float(f"{to_convert_amt:.8f}"), net)
                    if math.isfinite(remaining_need) and remaining_need > 0:
                        remaining_need = max(0.0, remaining_need - net)
                    if math.isfinite(remaining_need) and remaining_need <= 0.0:
                        break
        except Exception:
            self.logger.debug("redeem_wrapped_stables failed", exc_info=True)

        return {"ok": (freed > 0.0), "freed": float(f"{freed:.6f}"), "actions": actions, "reason": "redeem_stables"}

    async def free_from_positions(self, want_usdt: float) -> Dict[str, Any]:
        """
        Sell small portions of non-protected assets to free quote, prioritizing
        the smallest notionals first. Respects min_notional, lot step, spread caps,
        slippage and fee estimates. Best-effort and stops once want_usdt is met.
        Controlled by CR_ALLOW_POSITION_FREE (default False).
        """
        if not self._cfg_bool("CR_ENABLE", True):
            return {"ok": False, "freed": 0.0, "reason": "disabled"}
        if not self._cfg_bool("CR_ALLOW_POSITION_FREE", False):
            return {"ok": False, "freed": 0.0, "reason": "position_free_disabled"}
        if not self.ex:
            return {"ok": False, "freed": 0.0, "reason": "no_exchange_client"}

        protected = self._protected_assets()
        max_actions = max(1, self._cfg_int("CR_MAX_ACTIONS", 8))
        slippage_bps = max(0, self._cfg_int("CR_PRICE_SLIPPAGE_BPS", 15))
        fee_bps = max(0, self._cfg_int("CR_FEE_BPS", 10))
        spread_cap_bps = self._cfg_float("LIQUIDATION_SPREAD_CAP_BPS", self._cfg_float("CR_SPREAD_CAP_BPS", 12.0))

        remaining = max(0.0, float(want_usdt))
        freed = 0.0
        actions: List[Dict[str, Any]] = []

        try:
            bals = await self._balances()
            # consider only sizeable balances; we'll sort by notional asc
            candidates: List[Tuple[str, float, float]] = []  # (asset, free_amt, notional)
            price_cache: Dict[str, float] = {}
            for asset, row in (bals.items() if isinstance(bals, dict) else []):
                asset_up = str(asset).upper()
                if asset_up in (self.quote, None) or asset_up in protected:
                    continue
                free_amt = float(row.get("free", 0.0) or 0.0)
                if free_amt <= 0:
                    continue
                sym = f"{asset_up}{self.quote}"
                p = price_cache.get(sym)
                if p is None:
                    p = await self._get_price(sym)
                    p = float(p) if p is not None else 0.0
                    price_cache[sym] = p
                if p <= 0.0:
                    continue
                candidates.append((asset_up, free_amt, free_amt * p))

            candidates.sort(key=lambda x: x[2])  # smallest notional first

            for asset_up, free_amt, notional in candidates:
                if remaining <= 0 or len(actions) >= max_actions:
                    break
                sym = f"{asset_up}{self.quote}"
                book = await self._get_book(sym)
                s_bps = book.get("spread_bps")
                if s_bps is not None and s_bps > spread_cap_bps:
                    continue

                bid = float(book.get("bid") or 0.0)
                if bid <= 0.0:
                    price = price_cache.get(sym) or 0.0
                else:
                    price = bid
                if price <= 0.0:
                    continue
                exec_price = price * (1.0 - slippage_bps / 10_000.0)

                filt = await self._get_filters(sym)
                step = float(filt.get("lot_step") or 0.0)
                min_qty = float(filt.get("min_qty") or 0.0)
                max_qty = float(filt.get("max_qty") or 0.0)
                min_notional = float(filt.get("min_notional") or 0.0)

                # qty limited by remaining need
                max_qty_by_need = min(free_amt, remaining / exec_price) if exec_price > 0 else 0.0
                qty = self._round_step(max_qty_by_need, step)
                if (min_qty and qty < min_qty) or (max_qty and qty > max_qty) or qty <= 0:
                    continue
                min_required = await self._min_exit_quote(sym, exec_price, min_notional)
                if min_required and (qty * exec_price) < min_required:
                    # bump to exit floor if possible
                    bump_qty = self._round_step((min_required / exec_price), step)
                    if bump_qty <= free_amt:
                        qty = bump_qty
                    else:
                        continue

                ok = False
                try:
                    res = await self._market_sell(sym, qty, tag="balancer")
                    ok = bool(res)
                except Exception:
                    ok = False

                if ok:
                    gross = qty * exec_price
                    net = gross * (1.0 - fee_bps / 10_000.0)
                    freed += net
                    remaining = max(0.0, remaining - net)
                    actions.append({
                        "action": "free_from_positions",
                        "asset": asset_up,
                        "symbol": sym,
                        "qty": float(f"{qty:.12f}"),
                        "exec_price": float(f"{exec_price:.10f}"),
                        "net_quote": float(f"{net:.6f}"),
                        "spread_bps": (float(s_bps) if s_bps is not None else None)
                    })
                    self.logger.info("[Liquidity] action positions %s qty=%.8f exec=%.8f net=%.6f spread_bps=%s",
                                     sym, qty, exec_price, net,
                                     (None if s_bps is None else float(s_bps)))
        except Exception:
            self.logger.debug("free_from_positions failed", exc_info=True)

        return {"ok": (freed >= want_usdt), "freed": float(f"{freed:.6f}"), "remaining": float(f"{max(0.0, want_usdt - freed):.6f}"), "actions": actions, "reason": "positions"}

    async def route_best_effort(self, want_usdt: float) -> Dict[str, Any]:
        """
        Try multiple strategies in order until the requested USDT is freed.
        """
        total_freed = 0.0
        all_actions: List[Dict[str, Any]] = []

        remaining = max(0.0, float(want_usdt))

        # 1) Try redeeming wrapped stables
        res = await self.redeem_wrapped_stables(want_usdt=remaining)
        total_freed += float(res.get("freed", 0.0) or 0.0)
        remaining = max(0.0, remaining - float(res.get("freed", 0.0) or 0.0))
        all_actions.extend(res.get("actions", []))
        if remaining <= 0:
            return {"ok": True, "freed": float(f"{total_freed:.6f}"), "remaining": 0.0, "actions": all_actions}

        # 2) Sweep dust (sell only what’s needed)
        res = await self.sweep_dust(want_usdt=remaining)
        total_freed += float(res.get("freed", 0.0) or 0.0)
        remaining = max(0.0, remaining - float(res.get("freed", 0.0) or 0.0))
        all_actions.extend(res.get("actions", []))
        if remaining <= 0:
            return {"ok": True, "freed": float(f"{total_freed:.6f}"), "remaining": 0.0, "actions": all_actions}

        # 3) As a guarded last resort, sell small portions of positions
        if self._cfg_bool("CR_ALLOW_POSITION_FREE", False):
            res = await self.free_from_positions(remaining)
            total_freed += float(res.get("freed", 0.0) or 0.0)
            remaining = max(0.0, remaining - float(res.get("freed", 0.0) or 0.0))
            all_actions.extend(res.get("actions", []))

        return {
            "ok": (remaining <= 0.0),
            "freed": float(f"{total_freed:.6f}"),
            "remaining": float(f"{remaining:.6f}"),
            "actions": all_actions,
        }

    async def ensure_free_usdt(self, target_usdt: float, *, reason: str = "") -> Dict[str, Any]:
        """
        Ensure at least target_usdt free in quote currency using best-effort routing.
        This function is idempotent and will do nothing if current free >= target.
        """
        if not self._cfg_bool("CR_ENABLE", True):
            return {"ok": False, "freed": 0.0, "remaining": float(target_usdt), "reason": "disabled"}
        target_usdt = float(max(0.0, target_usdt))
        if target_usdt <= 0:
            tag = "balancer"
            return {"ok": True, "freed": 0.0, "remaining": 0.0, "reason": reason or "no_gap", "tag": tag}

        async with self._lock:
            start_free = await self._free_usdt()
            if start_free >= target_usdt:
                tag = "balancer"
                return {"ok": True, "freed": 0.0, "remaining": 0.0, "reason": reason or "already_sufficient", "tag": tag}

            need = target_usdt - start_free
            routed = await self.route_best_effort(need)

            # Refresh balances so capital gate sees the change
            try:
                if self.ex and hasattr(self.ex, "get_balances"):
                    b = self.ex.get_balances()
                    b = await b if asyncio.iscoroutine(b) else b
                    if self.shared_state and hasattr(self.shared_state, "update_balances"):
                        u = self.shared_state.update_balances(b)
                        await u if asyncio.iscoroutine(u) else u
                    if self.shared_state and hasattr(self.shared_state, "emit_event"):
                        ev = self.shared_state.emit_event("balances.changed", {"source": "cash_router"})
                        if asyncio.iscoroutine(ev):
                            asyncio.create_task(ev)
            except Exception:
                self.logger.debug("post-route balance refresh failed", exc_info=True)

            end_free = await self._free_usdt()
            freed_actual = max(0.0, end_free - start_free)
            remaining = max(0.0, need - freed_actual)

            # tolerate sub-cent residuals due to rounding/fees
            epsilon = float(getattr(self.config, "CR_EPSILON_USDT", 0.01))
            if remaining <= epsilon:
                remaining = 0.0

            tag = "balancer"
            # Emit a structured result event for higher layers/grep
            try:
                if self.shared_state and hasattr(self.shared_state, "emit_event"):
                    ev3 = self.shared_state.emit_event("LIQUIDITY_RESULT", {
                        "component": "CashRouter",
                        "via": "cash_router",
                        "ok": (remaining <= 0.0),
                        "freed": float(f"{freed_actual:.6f}"),
                        "remaining": float(f"{remaining:.6f}"),
                        "actions": int(len(routed.get("actions", []))),
                        "reason": reason or "routed",
                    })
                    if asyncio.iscoroutine(ev3):
                        asyncio.create_task(ev3)
            except Exception:
                self.logger.debug("LIQUIDITY_RESULT emit failed", exc_info=True)
            return {
                "ok": (remaining <= 0.0),
                "freed": float(f"{freed_actual:.6f}"),
                "remaining": float(f"{remaining:.6f}"),
                "actions": routed.get("actions", []),
                "reason": reason or "routed",
                "tag": tag,
            }

# small helper to normalize balance rows to floats
def strdict(v: Any) -> Dict[str, float]:
    try:
        if isinstance(v, dict):
            out = {}
            for k in ("free", "locked"):
                x = v.get(k, 0.0)
                out[k] = float(x if x is not None else 0.0)
            return out
    except Exception:
        pass
    return {"free": 0.0, "locked": 0.0}
