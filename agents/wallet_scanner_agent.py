import logging
import asyncio
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple

logger = logging.getLogger("WalletScannerAgent")

class WalletScannerAgent:
    """
    Scans wallet balances and proposes base+quote symbols (default USDT).
    Safe to instantiate early (Phase 3) without full deps; it will no-op until ready.
    Additionally, it can optionally request liquidations to restore quote balance if free quote is below a configurable threshold.
    """
    agent_type = "discovery"

    def __init__(
        self,
        shared_state: Any,
        config: Any,
        exchange_client: Optional[Any] = None,
        symbol_manager: Optional[Any] = None,
        interval: int = 1800,
        min_balance_threshold: float = 0.0,
        liquidation_agent: Optional[Any] = None,
        position_manager: Optional[Any] = None,
    ):
        self.shared_state = shared_state
        self.config = config
        self.exchange_client = exchange_client
        self.symbol_manager = symbol_manager
        self.interval = int(interval)
        self.min_balance_threshold = float(min_balance_threshold)
        self.liquidation_agent = liquidation_agent
        self.position_manager = position_manager
        # thresholds for quote restoration
        self.quote_free_threshold = float(getattr(config, "QUOTE_FREE_THRESHOLD", 5.0))  # in quote units (e.g., USDT)
        self.dust_liq_min_quote = float(getattr(config, "DUST_LIQUIDATION_MIN_QUOTE", 5.0))
        self.max_liqs_per_scan = int(getattr(config, "MAX_LIQUIDATIONS_PER_SCAN", 3))

        # Risk-aware prefilter + event emit knobs
        self.max_per_trade_usdt = float(getattr(config, "MAX_PER_TRADE_USDT", 100.0))
        self.require_trading_status = bool(getattr(config, "REQUIRE_TRADING_STATUS", True))
        self.emit_accept_event = bool(getattr(config, "EMIT_ACCEPTED_SYMBOL_EVENT", True))

        self.name = "WalletScannerAgent"
        self.is_discovery_agent = True      # contract flag
        self.last_run: Optional[str] = None
        self._stop_event = asyncio.Event()
        self._task = None
        self._lock = asyncio.Lock()
        self._running = False

        logger.info(
            f"[{self.name}] Initialized "
            f"(has_exchange={self.exchange_client is not None}, "
            f"has_symbol_mgr={self.symbol_manager is not None}, "
            f"has_liq_agent={self.liquidation_agent is not None}, "
            f"has_pos_mgr={self.position_manager is not None}, "
            f"interval={self.interval}s)"
        )

    async def start(self):
        """
        P9 contract: start() should exist and spawn the periodic loop once.
        Safe to call multiple times (idempotent).
        """
        if getattr(self, "_task", None) and not self._task.done():
            return
        self._stop_event.clear()
        self._task = asyncio.create_task(self.run_loop(), name="agent.wallet_scanner")
        logger.info(f"[{self.name}] start() launched background loop.")
    
    async def stop(self):
        """
        P9 contract: stop() should exist and gracefully stop the loop.
        """
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
                logger.debug(f"[{self.name}] stop wait failed", exc_info=True)
        logger.info(f"[{self.name}] stopped.")

    # ---------------- Public loops -----------------
    async def run_loop(self):
        logger.info(f"🚀 [{self.name}] Live scan started. Interval: {self.interval}s")
        while not self._stop_event.is_set():
            try:
                await self.run_once()
            except Exception as e:
                logger.exception(f"[{self.name}] Error in run_loop: {e}")
            await asyncio.sleep(self.interval)

    async def scheduler(self):
        # alias used by some managers
        return await self.run_loop()

    async def _is_tradable_symbol(self, symbol: str) -> bool:
        """
        Best-effort check if a symbol is tradable on the connected exchange.
        Uses exchange_client helpers when available, otherwise assumes True.
        """
        ec = self.exchange_client
        if not ec:
            return False
        try:
            if hasattr(ec, "is_tradable"):
                res = ec.is_tradable(symbol)
                if asyncio.iscoroutine(res):
                    res = await res
                return bool(res)
            if hasattr(ec, "has_symbol"):
                res = ec.has_symbol(symbol)
                if asyncio.iscoroutine(res):
                    res = await res
                return bool(res)
            # fallback to True if we can't verify
            return True
        except Exception:
            logger.debug(f"[{self.name}] is_tradable check failed for {symbol}", exc_info=True)
            return False

    async def _get_price(self, symbol: str) -> Optional[float]:
        ec = self.exchange_client
        if not ec:
            return None
        try:
            if hasattr(ec, "get_ticker_price"):
                p = ec.get_ticker_price(symbol)
                if asyncio.iscoroutine(p):
                    p = await p
                return float(p) if p is not None else None
            if hasattr(ec, "get_price"):
                p = ec.get_price(symbol)
                if asyncio.iscoroutine(p):
                    p = await p
                return float(p) if p is not None else None
        except Exception:
            logger.debug(f"[{self.name}] price fetch failed for {symbol}", exc_info=True)
        return None

    async def _prefilter_symbol(self, symbol: str) -> bool:
        """Return True if symbol is acceptable: status=TRADING and MIN_NOTIONAL <= max_per_trade_usdt (best-effort)."""
        ec = self.exchange_client
        if not ec:
            return True
        try:
            if self.require_trading_status and hasattr(ec, "symbol_info"):
                info = ec.symbol_info(symbol)
                info = await info if asyncio.iscoroutine(info) else info
                if not info:
                    logger.warning(f"[{self.name}] No symbol_info for %s; skipping.", symbol)
                    return False
                status = str(info.get("status", "TRADING")).upper()
                if status != "TRADING":
                    logger.warning(f"[{self.name}] %s status is %s; skipping.", symbol, status)
                    return False
                # MIN_NOTIONAL support for dict or list formats
                min_notional = None
                filters = info.get("filters") or {}
                if isinstance(filters, dict) and "MIN_NOTIONAL" in filters:
                    try:
                        min_notional = float(filters["MIN_NOTIONAL"])
                    except Exception:
                        min_notional = None
                elif isinstance(filters, list):
                    for f in filters:
                        if isinstance(f, dict) and f.get("filterType") == "MIN_NOTIONAL":
                            try:
                                min_notional = float(f.get("minNotional", float("inf")))
                            except Exception:
                                min_notional = None
                            break
                if min_notional is not None and min_notional > self.max_per_trade_usdt:
                    logger.info(f"[{self.name}] %s MIN_NOTIONAL %.4f exceeds cap %.2f; skipping.", symbol, min_notional, self.max_per_trade_usdt)
                    return False
            return True
        except Exception:
            logger.debug(f"[{self.name}] prefilter failed for %s", symbol, exc_info=True)
            return False

    async def _request_liquidation(self, symbol: str, qty_base: float, reason: str) -> bool:
        """
        Try multiple known method names on liquidation_agent or position_manager to request liquidation.
        Returns True if any call succeeded (did not raise).
        """
        async def _try_call(obj: Any, names: List[str]) -> bool:
            for nm in names:
                if hasattr(obj, nm):
                    fn = getattr(obj, nm)
                    try:
                        res = fn(symbol=symbol, side="SELL", qty=qty_base, reason=reason)
                        if asyncio.iscoroutine(res):
                            await res
                        return True
                    except TypeError:
                        # try argument variants
                        try:
                            res = fn(symbol, "SELL", qty_base, reason)
                            if asyncio.iscoroutine(res):
                                await res
                            return True
                        except Exception:
                            logger.debug(f"[{self.name}] call variant failed for {obj}.{nm}", exc_info=True)
                    except Exception:
                        logger.debug(f"[{self.name}] liquidation call failed for {obj}.{nm}", exc_info=True)
            return False
        # Try liquidation_agent first
        if self.liquidation_agent:
            ok = await _try_call(
                self.liquidation_agent,
                ["enqueue_async", "enqueue", "request_liquidation", "suggest_liquidation", "propose", "submit"]
            )
            if ok:
                return True
        # Fallback to position_manager direct market sell
        if self.position_manager:
            ok = await _try_call(
                self.position_manager,
                ["request_liquidation", "close_position", "market_sell", "market_close", "reduce_position"]
            )
            if ok:
                return True
        return False

    async def run_once(self):
        """
        One scan: fetch balances -> build candidate symbols -> propose via SymbolManager.
        Gracefully skips if deps not ready.
        Reentrancy guard: skip if already running.
        """
        if self._running:
            logger.debug(f"[{self.name}] run_once skipped: already running")
            return
        async with self._lock:
            if self._running:
                return
            self._running = True
            try:
                logger.info(f"[{self.name}] Initiating wallet scan (min_balance_threshold={self.min_balance_threshold})")

                if not self.exchange_client:
                    logger.info(f"[{self.name}] Skipping: exchange_client not wired yet.")
                    return
                if not hasattr(self.exchange_client, "get_account_balances"):
                    logger.warning(f"[{self.name}] Skipping: exchange_client lacks get_account_balances()")
                    return

                try:
                    balances = await self.exchange_client.get_account_balances()
                    # FIX: Persist wallet universe immediately so SharedState knows about held assets
                    if balances and hasattr(self.shared_state, "update_balances"):
                        await self.shared_state.update_balances(balances)
                except Exception as e:
                    logger.error(f"[{self.name}] Failed to fetch balances: {e}", exc_info=True)
                    return

                # Determine quote & free quote amount
                quote = str(getattr(self.config, "DEFAULT_QUOTE_CURRENCY", "USDT")).upper()
                known_quotes = set(getattr(self.exchange_client, "_known_quotes", {"USDT","FDUSD","USDC","BUSD","TUSD","DAI"}))
                if hasattr(self.exchange_client, "get_known_quotes"):
                    try:
                        kq = self.exchange_client.get_known_quotes()
                        if asyncio.iscoroutine(kq):
                            kq = await kq
                        if isinstance(kq, (set, list, tuple)):
                            known_quotes = set([str(x).upper() for x in kq]) or known_quotes
                    except Exception:
                        logger.debug(f"[{self.name}] get_known_quotes() failed", exc_info=True)
                quote_info = (balances or {}).get(quote, {}) or {}
                quote_free = float(quote_info.get("free", 0.0) or 0.0)
                logger.debug(f"[{self.name}] Quote {quote} free balance: {quote_free}")

                # If free quote is below threshold, attempt to reclaim by liquidating dust assets (best-effort)
                try:
                    if quote_free < self.quote_free_threshold:
                        # Build candidate bases to liquidate (exclude quotes themselves)
                        liq_candidates: List[Tuple[str, float]] = []
                        for asset, info in (balances or {}).items():
                            if not isinstance(info, dict):
                                continue
                            base = str(asset).upper()
                            if base == quote or base in known_quotes:
                                continue
                            free_amt = float(info.get("free", 0.0) or 0.0)
                            if free_amt <= 0.0:
                                continue
                            symbol = f"{base}{quote}"
                            # ensure tradable
                            if not await self._is_tradable_symbol(symbol):
                                logger.debug(f"[{self.name}] Skip non-tradable for liquidation: {symbol}")
                                continue
                            # estimate quote value
                            price = await self._get_price(symbol)
                            if price is None:
                                continue
                            est_quote_value = price * free_amt
                            if est_quote_value >= self.dust_liq_min_quote:
                                liq_candidates.append((symbol, free_amt))
                        # Limit and request liquidations
                        liq_count = 0
                        for symbol, qty in liq_candidates[: self.max_liqs_per_scan]:
                            ok = await self._request_liquidation(symbol, qty, reason="reclaim_quote_balance")
                            if ok:
                                liq_count += 1
                                logger.info(f"[{self.name}] 📉 Requested liquidation: {symbol} qty={qty} (quote free {quote_free} &lt; {self.quote_free_threshold})")
                            await asyncio.sleep(0)  # yield
                        if liq_count == 0:
                            logger.info(f"[{self.name}] Quote below threshold but no liquidation candidates found (threshold={self.quote_free_threshold}, min_q={self.dust_liq_min_quote})")
                except Exception:
                    logger.debug(f"[{self.name}] liquidation attempt failed (non-fatal)", exc_info=True)

                # Filter non-zero > threshold
                filtered: Dict[str, float] = {}
                for asset, info in (balances or {}).items():
                    if not isinstance(info, dict):
                        continue
                    free = float(info.get("free", 0))
                    locked = float(info.get("locked", 0))
                    total = free + locked
                    if total > self.min_balance_threshold:
                        filtered[str(asset).upper()] = total

                if not filtered:
                    logger.info(f"[{self.name}] No assets above threshold.")
                    return

                logger.info(f"[{self.name}] Candidates from wallet: {list(filtered.keys())}")

                # reuse quote/known_quotes computed above
                min_vol = float(getattr(self.config, "MIN_VOLUME_THRESHOLD", 0))

                candidates = []
                idx = 0
                total = len(filtered)
                for base, _bal in filtered.items():
                    idx += 1

                    # skip self-quote / direct quotes
                    if base == quote or base in known_quotes:
                        continue

                    symbol = f"{base}{quote}"
                    
                    # Exchange status & minNotional prefilter (defensive)
                    try:
                        if not await self._prefilter_symbol(symbol):
                            continue
                    except Exception: continue

                    # tradability gate (avoid proposing delisted or paused symbols)
                    tradable = await self._is_tradable_symbol(symbol)
                    if not tradable:
                        continue

                    candidates.append(symbol)

                # Batch propose
                proposed_count = 0
                if candidates and self.symbol_manager:
                    logger.info(f"[{self.name}] Proposing batch of {len(candidates)} symbol(s)...")
                    try:
                        accepted_list = await asyncio.wait_for(
                            self.symbol_manager.propose_symbols(candidates, source=self.name),
                            timeout=30
                        )
                        
                        for s in accepted_list:
                            proposed_count += 1
                            logger.info(f"[{self.name}] ✅ Accepted: {s}")
                            # Emit events for accepted ones
                            if self.emit_accept_event and hasattr(self.shared_state, "emit_event"):
                                try:
                                    await self.shared_state.emit_event("AcceptedSymbol", {"symbol": s, "source": self.name})
                                except Exception: pass
                    except Exception as e:
                        logger.error(f"[{self.name}] Batch proposal failed: {e}")

                self.last_run = datetime.utcnow().isoformat()
                logger.info(f"[{self.name}] Done. Proposed/accepted={proposed_count}")
            finally:
                await asyncio.sleep(0)
                self._running = False

    # ---------- Optional one-shot “discovery” flavor ----------
    async def run_discovery(self):
        """Lightweight discovery: reuse run_once (so AgentManager can call it once)."""
        await self.run_once()
