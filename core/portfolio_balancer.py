# core/portfolio_balancer.py
import logging
import math
import asyncio
from typing import Dict, List, Tuple, Optional, Any
from types import SimpleNamespace

from utils.hyg_guards import compute_tradable_qty, enforce_min_entry_quote, enforce_fee_safety
from core.stubs import MetaPolicy, TradeIntent, ExecOrder
from core.stubs import maybe_await, maybe_call

logger = logging.getLogger("PortfolioBalancer")


def _default_pcfg():
    return SimpleNamespace(
        ENABLE_BALANCER=False,
        REBALANCE_INTERVAL_SEC=300,
        N_MAX_POSITIONS=8,
        METHOD="equal_weight",              # or "risk_parity"
        MIN_TARGET_MULTIPLIER=1.0,          # x * minNotional
        COOLDOWN_AFTER_REBALANCE_SEC=900,
        FEE_SAFETY_MULTIPLIER=5.0,
        MIN_ENTRY_QUOTE_USDT=20.0,
    )


class PortfolioBalancer:
    def __init__(self, shared_state, exchange_client, execution_manager, config, meta_controller=None, **kwargs):
        self.ss = shared_state
        self.ex = exchange_client
        self.exec = execution_manager
        self.cfg = config
        self.mc = meta_controller

        # P9 lifecycle state
        self._stop_event = asyncio.Event()
        self._task: Optional[asyncio.Task] = None

        # Component identity for health events
        self.component = "PortfolioBalancer"

        # Lazy initialization via properties below

    # -- helpers to access config safely
    # -- dynamic config helpers
    def _cfg(self, key: str, default: Any = None) -> Any:
        # 1. Check SharedState for live/dynamic overrides
        if hasattr(self.ss, "dynamic_config"):
            val = self.ss.dynamic_config.get(key)
            if val is not None:
                return val

        # 2. Fallback to static config (nested lookup)
        pcfg = getattr(self.cfg, "PORTFOLIO", {})
        if isinstance(pcfg, dict):
            val = pcfg.get(key)
        else:
            val = getattr(pcfg, key, None)
        
        if val is not None:
            return val
            
        # 3. Final fallback to top-level config or default
        if isinstance(self.cfg, dict):
            return self.cfg.get(key, default)
        return getattr(self.cfg, key, default)

    @property
    def enabled(self) -> bool:
        return bool(self._cfg("ENABLE_BALANCER", False))

    @property
    def rebalance_interval(self) -> int:
        return int(self._cfg("REBALANCE_INTERVAL_SEC", 300))

    @property
    def max_positions(self) -> int:
        return int(self._cfg("N_MAX_POSITIONS", 8))

    @property
    def method(self) -> str:
        return str(self._cfg("METHOD", "equal_weight"))

    @property
    def min_target_multiplier(self) -> float:
        return float(self._cfg("MIN_TARGET_MULTIPLIER", 1.0))

    def _exchange_cfg(self):
        ex = getattr(self.cfg, "EXCHANGE", None)
        if ex is None:
            return SimpleNamespace(TAKER_FEE_BPS=10)
        return ex

    def _norm_filters(self, f):
        if isinstance(f, list):
            by_type = {}
            for it in f:
                t = (it or {}).get("filterType") or (it or {}).get("type")
                if t:
                    by_type[t] = it
            f = by_type
        f = f or {}
        lot = f.get("LOT_SIZE", {}) or {}
        notl = f.get("NOTIONAL", {}) or f.get("MIN_NOTIONAL", {}) or {}
        if not lot and "stepSize" in f:
            lot = f
        if not notl and ("minNotional" in f or "min_notional" in f):
            notl = f
        return {
            "lot": {
                "stepSize": float(lot.get("stepSize", lot.get("step_size", "0.000001")) or "0.000001"),
                "minQty": float(lot.get("minQty", lot.get("min_qty", 0.0)) or 0.0),
            },
            "notional": {
                "minNotional": float(
                    notl.get("minNotional", notl.get("min_notional", 0.0)) or 0.0
                )
            }
        }

    async def run(self):
        await self.run_periodic()

    async def start(self):
        """
        P9 contract: start() spawns the periodic loop once (idempotent).
        """
        if getattr(self, "_task", None) and not self._task.done():
            return
        self._stop_event.clear()
        await self._emit_health("OK", "Initialized")
        self._task = asyncio.create_task(self.run_periodic(), name="ops.portfolio_balancer")
        logger.info("[Balancer] start() launched background loop.")

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
                logger.debug("[Balancer] stop wait failed", exc_info=True)
        await self._emit_health("OK", "Stopped by request")
        logger.info("[Balancer] stopped.")

    async def run_periodic(self):
        if not self.enabled:
            logger.info("[Balancer] Disabled — exiting task cleanly.")
            await self._emit_health("DEGRADED", "Disabled")
            return
        
        await self._emit_health("OK", f"Running (interval={self.rebalance_interval}s)")
        while not self._stop_event.is_set():
            try:
                if self.enabled:
                    await self.run_once()
                    await self._emit_health("OK", "Tick OK")
                else:
                    await self._emit_health("DEGRADED", "Disabled (Dynamic)")
            except Exception as e:
                logger.exception(f"[Balancer] Crash: {e}")
                await self._emit_health("ERROR", f"Crash: {e!r}")
            await asyncio.sleep(self.rebalance_interval)

    async def _emit_health(self, level: str, message: str):
        """
        Emit P9 HealthStatus event via SharedState if available.
        level: OK | DEGRADED | ERROR
        """
        try:
            ss = self.ss
            if ss:
                await maybe_call(ss, "emit_event", "HealthStatus", {
                    "component": self.component,
                    "level": level,
                    "details": {"message": message},
                    "ts": self._iso_now(),
                })
        except Exception:
            pass

    @staticmethod
    def _iso_now():
        from datetime import datetime
        return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

    async def run_once(self):
        if not self.enabled:
            return

        # Optional: Skip on stale market data
        if getattr(self.ss, "market_data_stale", False):
            logger.info("[Balancer] Skip — market data stale.")
            return

        # 1) snapshot
        positions: Dict[str, Dict] = await maybe_call(self.ss, "get_positions_snapshot")
        accepted: Dict[str, Any] = await maybe_call(self.ss, "get_accepted_symbols")
        
        # Combine existing positions and candidate accepted symbols
        all_candidates = set(positions.keys()) | set(accepted.keys())
        if not all_candidates:
            logger.info("[Balancer] No positions and no accepted symbols — skipping.")
            return

        symbols = list(all_candidates)
        # 2) prices
        prices: Dict[str, float] = await maybe_call(self.ss, "get_all_prices")

        # 3) filters
        async def _one(sym):
            try:
                await maybe_call(self.ex, "ensure_symbol_filters_ready", sym)
                return sym, await maybe_call(self.ss, "get_symbol_filters_cached", sym)
            except Exception as e:
                logger.warning(f"[Balancer] Failed to fetch filters for {sym}: {e}")
                return sym, {}

        filters_pairs: List[Tuple[str, Dict]] = await asyncio.gather(*[_one(s) for s in symbols])
        filters: Dict[str, Dict] = {s: f or {} for s, f in filters_pairs}

        # 4) ATR (optional)
        try:
            atr = await maybe_call(self.ss, "get_indicator_snapshot", "ATR", symbols)
        except Exception:
            atr = {}

        # metrics update (may be sync)
        try:
            dust, tradable = self._classify_dust(symbols, positions, prices, filters)
            await maybe_call(self.ss, "update_portfolio_metrics",
                             dust_count=len(dust),
                             dust_quote=self._quote_sum(dust, prices))
        except Exception:
            dust, tradable = self._classify_dust(symbols, positions, prices, filters)
            pass  # non-fatal

        # 5) skip guards
        if self._should_skip(tradable, prices):
            return

        logger.info(f"[Balancer] Tradable candidates: {len(tradable)}")
        keep = await self._select_topN(tradable, atr)
        if not keep:
            logger.info("[Balancer] No keep set after scoring (TopN) — skipping.")
            return

        targets = await self._compute_targets(keep, prices, filters, atr)
        if not targets:
            logger.info("[Balancer] No targets produced — skipping.")
            return

        orders = self._diff_to_orders(tradable, prices, filters, targets)
        if not orders:
            logger.info("[Balancer] Portfolio aligned — no orders.")
            return

        # 7) batch submit (respect retries inside ExecutionManager)
        # 6) build intents
        intents = []
        for o in orders:
            symbol = o["symbol"]
            side = o["side"].upper()
            raw_qty = float(o["qty"])
            price = prices.get(symbol)

            if price is None or price <= 0.0:
                continue

            # Cooldown logic (Best effort, usually checked by Meta/Risk too)
            try:
                if hasattr(self.ss, "was_recently_rebalanced"):
                    if await maybe_call(self.ss, "was_recently_rebalanced", symbol, self._cfg("COOLDOWN_AFTER_REBALANCE_SEC", 900)):
                        continue
            except Exception:
                pass

            nf = self._norm_filters(filters.get(symbol, {}))
            qty_rounded = self._floor_step(raw_qty, nf["lot"]["stepSize"])
            if qty_rounded <= 0.0:
                continue

            trade_quote = qty_rounded * price
            
            # P9: Instead of direct execution, we build a TradeIntent
            intent = {
                "symbol": symbol,
                "action": side,
                "confidence": 1.0, # Balancer intents are high-conv policy
                "planned_qty": qty_rounded if side == "SELL" else None,
                "planned_quote": trade_quote if side == "BUY" else None,
                "agent": self.component,
                "tag": "rebalance",
                "ttl_sec": 300,
            }
            intents.append(intent)

        # 7) Submit batch to MetaController
        if intents:
            logger.info(f"[Balancer] Submitting {len(intents)} intents to MetaController.")
            try:
                if self.mc and hasattr(self.mc, "receive_intents"):
                    await maybe_call(self.mc, "receive_intents", intents)
                else:
                    # Fallback to shared_state event bus for all intents
                    for it in intents:
                        await maybe_call(self.ss, "emit_event", "TradeIntent", it)
                
                # Success bookkeeping
                await self._emit_health("OK", f"Submitted {len(intents)} intents")
            except Exception as e:
                logger.error(f"[Balancer] Failed to submit intents: {e}")
                await self._emit_health("ERROR", f"Submit failed: {e}")
        else:
            logger.info("[Balancer] No intents generated for this cycle.")

    # --- helpers: classify, select, targets, diff, rounding ---

    def _should_skip(self, tradable: Dict[str, Dict], prices: Dict[str, float]) -> bool:
        if not tradable:
            logger.info("[Balancer] No tradable positions — skipping.")
            return True
        miss = [s for s in tradable if not prices.get(s)]
        if miss and len(miss) / max(1, len(tradable)) > 0.5:
            logger.info(f"[Balancer] Too many missing prices ({len(miss)}/{len(tradable)}) — skipping.")
            return True
        return False

    def _classify_dust(self, candidates: List[str], positions: Dict[str, Dict], prices: Dict[str, float], filters: Dict[str, Dict]):
        dust, tradable = {}, {}
        for s in candidates:
            p = positions.get(s, {"current_qty": 0.0, "quantity": 0.0})
            price = prices.get(s)
            nf = self._norm_filters(filters.get(s, {}))
            step = nf["lot"]["stepSize"]
            min_notional = nf["notional"]["minNotional"]
            qty = self._floor_step(p.get("current_qty", p.get("quantity", 0.0)), step)
            
            # If price is missing, we can't trade it
            if price is None:
                continue
                
            is_dust = (qty * float(price) < min_notional)
            if is_dust and qty > 0:
                dust[s] = p
            else:
                # Even if zero qty, it's "tradable" if we might want to buy it
                tradable[s] = {**p, "qty_rounded": qty, "symbol": s}
        return dust, tradable

    async def _select_topN(self, tradable, atr):
        N = self.max_positions
        if N <= 0:
            return []
        scored = []
        for s in tradable:
            # P9 check: use safe getter
            score = 0.0
            try:
                if hasattr(self.ss, "get_unified_score"):
                    score = self.ss.get_unified_score(s)
            except Exception:
                pass
            scored.append((score, s))
        
        scored.sort(reverse=True, key=lambda x: x[0])
        top_n = [s for _, s in scored[:N]]
        
        # P9 Violation Fix: Avoid direct mutation. Emit event instead for others to consume.
        try:
             await maybe_call(self.ss, "emit_event", "BalancerTargetsUpdated", {"symbols": top_n})
        except Exception:
            pass
            
        return top_n

    async def _compute_targets(self, keep, prices, filters, atr):
        method = self.method
        min_mult = self.min_target_multiplier
        targets: Dict[str, float] = {}
        try:
            val = await maybe_call(self.ss, "get_nav_quote")
            nav_quote = float(val) if val is not None else 0.0
        except Exception:
            nav_quote = 0.0
        try:
            val = await maybe_call(self.ss, "get_target_exposure")
            exposure = float(val) if val is not None else 0.0
        except Exception:
            exposure = 0.0

        if nav_quote <= 0 or exposure <= 0 or not keep:
            logger.info(f"[Balancer] Skipping targets: nav={nav_quote}, exposure={exposure}, keep_len={len(keep)}")
            return targets

        if method == "equal_weight":
            per = (nav_quote * exposure) / max(1, len(keep))
            for s in keep:
                nf = self._norm_filters(filters.get(s, {}))
                min_notional_filter = nf["notional"]["minNotional"]
                tgt_quote = max(per, min_mult * min_notional_filter)
                targets[s] = tgt_quote
        else:  # risk_parity
            inv = []
            for s in keep:
                a = float(atr.get(s) or 1.0)
                inv.append((s, 1.0 / max(a, 1e-9)))
            denom = sum(w for _, w in inv) or 1.0
            for s, w in inv:
                nf = self._norm_filters(filters.get(s, {}))
                min_notional_filter = nf["notional"]["minNotional"]
                tgt_quote = max((w / denom) * nav_quote * exposure, min_mult * min_notional_filter)
                targets[s] = tgt_quote
        return targets

    def _diff_to_orders(self, tradable, prices, filters, targets):
        orders = []
        syms = set(list(tradable.keys()) + list(targets.keys()))
        for s in syms:
            price = prices.get(s)
            if price is None or price <= 0.0:
                continue
            nf = self._norm_filters(filters.get(s, {}))
            step = nf["lot"]["stepSize"]
            min_notional = nf["notional"]["minNotional"]

            cur_qty = float(tradable.get(s, {}).get("qty_rounded", 0.0))
            tgt_quote = float(targets.get(s, 0.0))
            tgt_qty = self._floor_step((tgt_quote / price), step) if price and step else (tgt_quote / price if price else 0.0)
            delta = tgt_qty - cur_qty
            if abs(delta) < max(step, 1e-18):
                continue
            if abs(delta) * price < min_notional:
                continue
            side = "BUY" if delta > 0 else "SELL"
            orders.append({"symbol": s, "side": side, "qty": abs(delta)})
        return orders

    @staticmethod
    def _floor_step(qty, step):
        if not step or step <= 0:
            return qty
        return math.floor(qty / step) * step

    @staticmethod
    def _quote_sum(positions, prices):
        return sum((positions[s].get("current_qty", 0.0) or 0.0) * (prices.get(s) or 0.0) for s in positions)

    @staticmethod
    def _norm(x):
        try:
            return max(-1.0, min(1.0, x / 0.05))
        except Exception:
            return 0.0



