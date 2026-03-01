import logging
import time
import asyncio
from typing import List, Dict, Any, Tuple, Optional
from math import inf

logger = logging.getLogger("SymbolScreener")

class SymbolScreener:

    async def _propose(self, symbol: str, *, source: str, metadata: Dict[str, Any]) -> bool:
        """
        Try SymbolManager.propose_symbol → SharedState.propose_symbol → fallback to stash.
        Always returns a boolean 'accepted' (True/False).
        """
        # Prefer SymbolManager API if present
        if self.symbol_manager and hasattr(self.symbol_manager, "propose_symbol"):
            res = self.symbol_manager.propose_symbol(symbol, source=source, **(metadata or {}))
            res = await res if asyncio.iscoroutine(res) else res
            # Normalize (True|False) or (True/False, reason)
            if isinstance(res, tuple) and res:
                return bool(res[0])
            return bool(res)

        # Fallback to SharedState API if available
        if self.shared_state and hasattr(self.shared_state, "propose_symbol"):
            res = self.shared_state.propose_symbol(symbol, source=source, metadata=metadata)
            res = await res if asyncio.iscoroutine(res) else res
            if isinstance(res, tuple) and res:
                return bool(res[0])
            return bool(res)

        # Last resort: stash into shared_state.symbol_proposals so a later
        # SymbolManager flush can pick it up
        if self.shared_state is not None:
            self.shared_state.symbol_proposals = getattr(self.shared_state, "symbol_proposals", {}) or {}
            self.shared_state.symbol_proposals[str(symbol).upper()] = {
                "symbol": str(symbol).upper(),
                "source": source,
                "metadata": dict(metadata or {}),
                "ts": time.time(),
            }
            logger.info(f"[SymbolScreener] Buffered proposal for {symbol} (no propose API available).")
            return False

        logger.warning(f"[SymbolScreener] No proposal path for {symbol}.")
        return False

    async def _prefilter_symbol(self, symbol: str) -> bool:
        """Return True if symbol should be considered (status=TRADING and MIN_NOTIONAL <= max_per_trade_usdt)."""
        try:
            if not self.exchange_client:
                return True  # nothing we can verify here
            if getattr(self, 'require_trading_status', True) and hasattr(self.exchange_client, "symbol_info"):
                info = self.exchange_client.symbol_info(symbol)
                info = await info if asyncio.iscoroutine(info) else info
                if not info:
                    logger.warning("[SymbolScreener] No symbol_info for %s; skipping.", symbol)
                    return False
                status = str(info.get("status", "TRADING")).upper()
                if status != "TRADING":
                    logger.warning("[SymbolScreener] %s status is %s; skipping.", symbol, status)
                    return False
                # MIN_NOTIONAL gate (supports dict or list formats)
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
                                min_notional = float(f.get("minNotional", inf))
                            except Exception:
                                min_notional = None
                            break
                cap = float(getattr(self, 'max_per_trade_usdt', 100.0))
                if min_notional is not None and min_notional > cap:
                    logger.info("[SymbolScreener] %s MIN_NOTIONAL %.4f exceeds cap %.2f; skipping.", symbol, min_notional, cap)
                    return False
            return True
        except Exception:
            logger.debug("[SymbolScreener] prefilter failed for %s", symbol, exc_info=True)
            return False

    agent_type = "discovery"           # Mark as discovery agent
    name = "SymbolScreener"

    def __init__(self, shared_state: Any, exchange_client: Any = None, config: Any = None, symbol_manager: Any = None, **kwargs):
        self.shared_state = shared_state
        self.exchange_client = exchange_client
        self.config = config
        self.symbol_manager = symbol_manager
        self.is_discovery_agent = True

        # optional defensive wiring if constructed early:
        if self.exchange_client is None and hasattr(shared_state, "exchange_client"):
            self.exchange_client = getattr(shared_state, "exchange_client")
        if self.symbol_manager is None and hasattr(shared_state, "symbol_manager"):
            self.symbol_manager = getattr(shared_state, "symbol_manager")

        # Lazy initialization via properties below

        logger.info("🔭 SymbolScreener (Market Scouting Unit) initialized.")
        logger.info(f"    ➔ Screening Interval: {self.screening_interval}s")
        logger.info(f"    ➔ Min 24h Quote Volume: {self.min_volume}")
        logger.info(f"    ➔ Top Volume Universe: {self.top_volume_universe_size}")
        logger.info(f"    ➔ Candidate Pool Size: {self.candidate_pool_size}")
        logger.info(f"    ➔ Min ATR%% ({self.atr_timeframe}): {self.min_atr_pct * 100.0:.2f}")
        logger.info(f"    ➔ Exclude List: {list(self.symbol_exclude_list)}")
        logger.info(f"    ➔ Screener Loop Interval: {self.screener_loop_interval}s")

        mgr = self.symbol_manager
        logger.info(
            "SymbolScreener wired SymbolManager=%s from %s; has propose_symbol=%s",
            type(mgr).__name__ if mgr else None,
            getattr(type(mgr), "__module__", "<none>") if mgr else "<none>",
            hasattr(mgr, "propose_symbol") if mgr else False,
        )

        # lifecycle & concurrency (P9 contract + anti-overlap)
        self._stop_event = asyncio.Event()
        self._task = None
        self._lock = asyncio.Lock()
        self._running = False

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
    def screening_interval(self) -> int:
        return int(self._cfg("SYMBOL_SCREENER_INTERVAL", 3600))

    @property
    def min_volume(self) -> float:
        return float(self._cfg("SYMBOL_MIN_VOLUME", 1_000_000))

    @property
    def min_percent_change(self) -> float:
        return float(self._cfg("SYMBOL_MIN_PERCENT_CHANGE", 2.0))

    @property
    def top_n_symbols(self) -> int:
        # Legacy compatibility; preferred knob is SYMBOL_CANDIDATE_POOL_SIZE.
        return int(self._cfg("SYMBOL_TOP_N", self._cfg("SYMBOL_CANDIDATE_POOL_SIZE", 50)))

    @property
    def candidate_pool_size(self) -> int:
        return max(1, int(self._cfg("SYMBOL_CANDIDATE_POOL_SIZE", self.top_n_symbols)))

    @property
    def top_volume_universe_size(self) -> int:
        base = max(1, int(self._cfg("SYMBOL_TOP_VOLUME_UNIVERSE", 50)))
        return max(base, self.candidate_pool_size)

    @property
    def min_atr_pct(self) -> float:
        raw = float(self._cfg("SYMBOL_MIN_ATR_PCT", 0.008) or 0.008)
        return (raw / 100.0) if raw > 1.0 else raw

    @property
    def atr_timeframe(self) -> str:
        return str(self._cfg("SYMBOL_ATR_TIMEFRAME", "1h") or "1h")

    @property
    def atr_period(self) -> int:
        return int(self._cfg("SYMBOL_ATR_PERIOD", 14) or 14)

    @property
    def atr_concurrency(self) -> int:
        return max(1, int(self._cfg("SYMBOL_ATR_CONCURRENCY", 8) or 8))

    @property
    def symbol_exclude_list(self) -> set:
        return set(self._cfg("SYMBOL_EXCLUDE_LIST", []))

    @property
    def base_currency(self) -> str:
        return str(self._cfg("BASE_CURRENCY", "USDT"))

    @property
    def screener_loop_interval(self) -> int:
        return int(self._cfg("SCREENER_INTERVAL_SECONDS", 1800))

    @property
    def max_per_trade_usdt(self) -> float:
        return float(self._cfg("MAX_PER_TRADE_USDT", 100.0))

    @property
    def require_trading_status(self) -> bool:
        return bool(self._cfg("REQUIRE_TRADING_STATUS", True))

    async def start(self):
        """
        P9 contract: start() spawns the periodic screening loop once (idempotent).
        """
        if getattr(self, "_task", None) and not self._task.done():
            return
        self._stop_event.clear()
        self._task = asyncio.create_task(self.run_loop(), name="agent.symbol_screener")
        logger.info("[SymbolScreener] start() launched background loop.")

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
                    await asyncio.wait_for(t, timeout=float(getattr(self.config, "STOP_JOIN_TIMEOUT_S", 5.0)))
                except asyncio.CancelledError:
                    pass
            except Exception:
                logger.debug("[SymbolScreener] stop wait failed", exc_info=True)
        logger.info("[SymbolScreener] stopped.")


    def _normalize_symbol(self, symbol: str) -> str:
        return str(symbol or "").replace("/", "").upper()

    def _is_leveraged_symbol(self, symbol: str) -> bool:
        sym = self._normalize_symbol(symbol)
        if not sym or not sym.endswith(self.base_currency):
            return False
        base = sym[: -len(self.base_currency)]
        if not base:
            return False
        suffixes = ("UP", "DOWN", "BULL", "BEAR", "2L", "2S", "3L", "3S", "4L", "4S", "5L", "5S")
        for suf in suffixes:
            if base.endswith(suf) and len(base) > (len(suf) + 1):
                return True
        return False

    async def _build_exclude_set(self) -> set:
        exclude = {self._normalize_symbol(x) for x in self.symbol_exclude_list}
        if not self.exchange_client:
            return exclude
        try:
            balances = await self.exchange_client.get_account_balances()
            for asset, bal in (balances or {}).items():
                asset_u = str(asset or "").upper()
                if not asset_u or asset_u == self.base_currency:
                    continue
                free = 0.0
                locked = 0.0
                if isinstance(bal, dict):
                    free = float(bal.get("free", 0.0) or 0.0)
                    locked = float(bal.get("locked", 0.0) or 0.0)
                else:
                    free = float(bal or 0.0)
                if free > 0.0 or locked > 0.0:
                    exclude.add(f"{asset_u}{self.base_currency}")
        except Exception:
            logger.debug("[SymbolScreener] Failed to build wallet exclude set", exc_info=True)
        return exclude

    async def _atr_pct(self, symbol: str, price: float) -> float:
        if price <= 0:
            return 0.0
        try:
            calc_atr = getattr(self.shared_state, "calc_atr", None)
            if callable(calc_atr):
                atr = calc_atr(symbol, self.atr_timeframe, self.atr_period)
                atr = await atr if asyncio.iscoroutine(atr) else atr
                atr_val = float(atr or 0.0)
                if atr_val > 0:
                    return float(atr_val) / max(float(price), 1e-9)
        except Exception:
            logger.debug("[SymbolScreener] ATR calc failed for %s", symbol, exc_info=True)
        return 0.0

    async def _evaluate_candidate(
        self, symbol: str, quote_volume: float, pct_change: float, last_price: float, sem: asyncio.Semaphore
    ) -> Optional[Dict[str, Any]]:
        async with sem:
            if not await self._prefilter_symbol(symbol):
                return None
            atr_pct = float(await self._atr_pct(symbol, last_price) or 0.0)
            return {
                "symbol": symbol,
                "quote_volume": float(quote_volume),
                "price_change_percent": float(pct_change),
                "atr_pct": atr_pct,
                "last_price": float(last_price),
            }

    async def _perform_scan(self) -> List[Dict[str, Any]]:
        """
        Volatility-driven candidate scan:
          1) Top liquid symbols by 24h quote volume
          2) Tradability prefilter (status/min notional)
          3) ATR% threshold gate (default 1h)

        Returns:
            List[Dict[str, Any]]: selected candidates with metadata.
        """
        try:
            if not self.exchange_client:
                logger.warning("[SymbolScreener] exchange_client missing; skipping scan.")
                return []

            tickers = await self.exchange_client.get_24hr_tickers()
            if not tickers:
                logger.warning("No ticker data received from exchange for symbol screening.")
                return []

            combined_exclude_set = await self._build_exclude_set()
            liquid: List[Tuple[str, float, float, float]] = []
            for ticker in tickers:
                symbol = self._normalize_symbol(ticker.get("symbol", ""))
                if not symbol or not symbol.endswith(self.base_currency):
                    continue
                if symbol in combined_exclude_set:
                    continue
                if self._is_leveraged_symbol(symbol):
                    continue
                try:
                    quote_volume = float(ticker.get("quoteVolume", 0.0) or 0.0)
                    pct_change = float(ticker.get("priceChangePercent", 0.0) or 0.0)
                    last_price = float(ticker.get("lastPrice", 0.0) or 0.0)
                except (ValueError, TypeError):
                    continue
                if quote_volume < self.min_volume or last_price <= 0:
                    continue
                liquid.append((symbol, quote_volume, pct_change, last_price))

            if not liquid:
                logger.warning("[SymbolScreener] No liquid symbols passed volume filter.")
                return []

            liquid.sort(key=lambda x: x[1], reverse=True)
            top_liquid = liquid[: self.top_volume_universe_size]

            sem = asyncio.Semaphore(self.atr_concurrency)
            tasks = [
                self._evaluate_candidate(sym, vol, pct, px, sem)
                for sym, vol, pct, px in top_liquid
            ]
            evaluated = await asyncio.gather(*tasks, return_exceptions=True)
            parsed = [x for x in evaluated if isinstance(x, dict)]
            atr_filtered = [x for x in parsed if float(x.get("atr_pct", 0.0) or 0.0) >= self.min_atr_pct]

            selected: List[Dict[str, Any]]
            if atr_filtered:
                atr_filtered.sort(
                    key=lambda x: (
                        -float(x.get("atr_pct", 0.0) or 0.0),
                        -float(x.get("quote_volume", 0.0) or 0.0),
                    )
                )
                selected = atr_filtered[: self.candidate_pool_size]
            else:
                fallback_enabled = bool(self._cfg("SYMBOL_ALLOW_ATR_FALLBACK", True))
                if not fallback_enabled:
                    logger.warning(
                        "[SymbolScreener] No symbols passed ATR%% %.2f and fallback disabled.",
                        self.min_atr_pct * 100.0,
                    )
                    return []
                parsed.sort(key=lambda x: -float(x.get("quote_volume", 0.0) or 0.0))
                selected = parsed[: self.candidate_pool_size]
                logger.warning(
                    "[SymbolScreener] No symbols passed ATR%% %.2f; using top-volume fallback (%d symbols).",
                    self.min_atr_pct * 100.0,
                    len(selected),
                )

            logger.info(
                "[SymbolScreener] Selected %d candidates from %d top-volume symbols (min_atr%%=%.2f).",
                len(selected),
                len(top_liquid),
                self.min_atr_pct * 100.0,
            )
            return selected

        except Exception as e:
            logger.error(f"❌ SymbolScreener scan error: {e}", exc_info=True)
            return []

    async def _process_and_add_symbols(self, candidates: List[Dict[str, Any]]):
        """
        Propose candidate symbols to SymbolManager / SharedState.
        """
        if candidates:
            symbols_only = [str(item.get("symbol", "")) for item in candidates]
            logger.info(f"📊 Candidate symbols found: {symbols_only}")
            accepted = 0
            for item in candidates:
                symbol = self._normalize_symbol(item.get("symbol", ""))
                if not symbol:
                    continue
                try:
                    if not await self._prefilter_symbol(symbol):
                        logger.warning(f"[SymbolScreener] ❌ Pre-filter rejected: {symbol}")
                        continue
                except Exception:
                    logger.debug("prefilter at proposal failed for %s", symbol, exc_info=True)
                    continue
                try:
                    accepted_flag = await self._propose(
                        symbol,
                        source=self.name,
                        metadata={
                            "24h_quote_volume": float(item.get("quote_volume", 0.0) or 0.0),
                            "24h_percent_change": float(item.get("price_change_percent", 0.0) or 0.0),
                            "atr_pct": float(item.get("atr_pct", 0.0) or 0.0),
                            "atr_timeframe": self.atr_timeframe,
                        },
                    )
                    if accepted_flag:
                        accepted += 1
                        logger.info(f"[SymbolScreener] ✅ Symbol accepted: {symbol}")
                    else:
                        logger.warning(f"[SymbolScreener] ❌ Symbol rejected/buffered: {symbol}")
                except Exception as e:
                    logger.error(f"Failed to propose symbol {symbol} via _propose: {e}", exc_info=True)
            logger.info(
                "[SymbolScreener] Proposal pass complete: %d/%d accepted.",
                accepted,
                len(candidates),
            )
        else:
            logger.info("No volatility candidates met criteria in this scan.")

    async def run_once(self):
        """
        One-time run wrapper for startup phase screening.
        Performs a single scan and adds eligible symbols.
        Reentrancy-guarded to avoid overlapping runs during startup scheduling.
        """
        if self._running:
            logger.debug("[SymbolScreener] run_once skipped: already running")
            return
        async with self._lock:
            if self._running:
                return
            self._running = True
            try:
                if not self.exchange_client:
                    logger.info("[SymbolScreener] Skipping: exchange_client not wired.")
                    return
                logger.info("📊 Performing one-time volatility-driven symbol screening.")
                candidates = await self._perform_scan()
                await self._process_and_add_symbols(candidates)
            finally:
                self._running = False

    async def run_loop(self):
        """
        Continuous loop for periodic symbol screening.
        """
        logger.info(f"Starting continuous run_loop for SymbolScreener with interval {self.screener_loop_interval} seconds.")
        while not self._stop_event.is_set():
            try:
                await self.run_once()
                await asyncio.sleep(self.screener_loop_interval)
            except asyncio.CancelledError:
                logger.info(f"[{self.name}] run_loop cancelled.")
                break
            except Exception as e:
                logger.exception(f"[{self.name}] Error in run_loop: {e}")
                await asyncio.sleep(10) # Sleep for a short period before retrying after an error

    async def run_discovery(self):
        """
        One-shot discovery method using volatility-first candidate construction.
        """
        try:
            logger.info("🔍 [SymbolScreener] Starting one-time volatility scan...")
            if not self.exchange_client:
                logger.warning("[SymbolScreener] exchange_client not wired.")
                return
            candidates = await self._perform_scan()
            await self._process_and_add_symbols(candidates)

        except Exception as e:
            logger.error(f"❌ Error during run_discovery: {e}", exc_info=True)


    async def start_periodic_screening(self):
        """
        Starts the continuous periodic screening process for trending symbols.
        This method is now superseded by run_loop() for continuous operation.
        It is kept for backward compatibility if needed, but run_loop() is preferred.
        """
        logger.warning("start_periodic_screening is deprecated. Please use run_loop() for continuous operation.")
        await self.run_loop() # Delegate to run_loop for consistency
