import logging
import time
import asyncio
from typing import List, Dict, Any, Tuple
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
        logger.info(f"    ➔ Min 24h Volume: {self.min_volume}")
        logger.info(f"    ➔ Min 24h % Change: {self.min_percent_change}%")
        logger.info(f"    ➔ Top N Symbols: {self.top_n_symbols}")
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
        return int(self._cfg("SYMBOL_TOP_N", 10))

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


    async def _perform_scan(self) -> List[Tuple[str, float, float, Dict[str, Any]]]:
        """
        Performs a scan of all tickers to identify trending symbols based on volume and
        percentage change, excluding symbols already in the user's wallet or on an exclude list.
        Now returns the full ticker data along with symbol, percent_change, and volume.

        Returns:
            List[Tuple[str, float, float, Dict[str, Any]]]: A list of selected trending symbols
            with their percentage change, volume, and full ticker data.
        """
        try:
            # Fetch all ticker data from the exchange
            tickers = await self.exchange_client.get_all_tickers()
            if not tickers:
                logger.warning("No ticker data received from exchange for symbol screening.")
                return []

            # Get current account balances to exclude held assets from screening
            account_balances = await self.exchange_client.get_account_balances()
            current_wallet_assets = {
                asset for asset in account_balances.keys()
                if asset != self.base_currency
            }
            # Combine user-defined exclude list with currently held assets
            combined_exclude_set = self.symbol_exclude_list.union({
                f"{asset}{self.base_currency}" for asset in current_wallet_assets
            })

            logger.debug(f"Combined exclude set for screening: {list(combined_exclude_set)}")
            candidates = []

            # Iterate through tickers to find candidates that meet criteria
            for ticker in tickers:
                symbol = ticker.get("symbol", "")
                # Ensure the symbol is a pair with the base currency
                if not symbol.endswith(self.base_currency):
                    continue
                # Exchange status & minNotional prefilter
                try:
                    ok = await self._prefilter_symbol(symbol)
                    if not ok:
                        continue
                except Exception:
                    logger.debug("prefilter_symbol error for %s", symbol, exc_info=True)
                    continue
                # Skip symbols in the combined exclude list
                if symbol in combined_exclude_set:
                    logger.debug(f"Skipping {symbol} as it is in the exclude list or currently held.")
                    continue
                try:
                    # Extract and validate price change percentage and volume
                    percent_change = float(ticker.get("priceChangePercent", 0))
                    volume = float(ticker.get("quoteVolume", 0))
                except (ValueError, TypeError):
                    logger.warning(f"Could not parse ticker data for {symbol}. Skipping.")
                    continue

                # Check if the symbol meets the min percentage change and min volume criteria
                if percent_change >= self.min_percent_change and volume >= self.min_volume:
                    candidates.append((symbol, percent_change, volume, ticker)) # Include full ticker data

            # Sort candidates by percentage change (desc) then volume (desc)
            candidates.sort(key=lambda x: (-x[1], -x[2]))
            # Select the top N symbols, keeping the full ticker data
            selected_symbols_with_data = candidates[:self.top_n_symbols]

            logger.info(f"✅ SymbolScreener selected top {len(selected_symbols_with_data)} new symbols.")
            return selected_symbols_with_data

        except Exception as e:
            logger.error(f"❌ SymbolScreener scan error: {e}", exc_info=True)
            return []

    async def _process_and_add_symbols(self, symbols_to_add_with_data: List[Tuple[str, float, float, Dict[str, Any]]]):
        """
        Helper method to log and add symbols using the symbol manager,
        now accepting full ticker data for metadata injection.
        """
        if symbols_to_add_with_data:
            symbols_only = [item[0] for item in symbols_to_add_with_data]
            logger.info(f"📊 Polling symbols found: {symbols_only}")
            for symbol, _, volume, ticker_data in symbols_to_add_with_data:
                # Prefilter again at proposal time (defensive)
                try:
                    if not await self._prefilter_symbol(symbol):
                        logger.warning(f"[SymbolScreener] ❌ Pre-filter rejected: {symbol}")
                        continue
                except Exception:
                    logger.debug("prefilter at proposal failed for %s", symbol, exc_info=True)
                    continue
                try:
                    accepted = await self._propose(
                        symbol,
                        source="SymbolScreener",
                        metadata={"24h_volume": volume}
                    )
                    if accepted:
                        logger.info(f"[SymbolScreener] ✅ Symbol accepted: {symbol}")
                    else:
                        logger.warning(f"[SymbolScreener] ❌ Symbol rejected/buffered: {symbol}")
                except Exception as e:
                    logger.error(f"Failed to propose symbol {symbol} via _propose: {e}", exc_info=True)
        else:
            logger.info("No new trending symbols met criteria in this scan.")

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
                logger.info("📊 Performing one-time symbol screening for startup.")

                # Assuming get_24hr_tickers returns a list of dictionaries with 'symbol', 'volume', 'priceChangePercent'
                tickers = await self.exchange_client.get_24hr_tickers()
                if not tickers:
                    logger.warning("No tickers received. Aborting screener.")
                    return

                filtered_by_thresholds = []
                for t in tickers:
                    symbol = t.get("symbol", "")
                    if not symbol.endswith("USDT"):  # Assuming hardcoded USDT base currency for this specific run_once
                        continue
                    try:
                        volume = float(t.get("volume", 0))
                        price_change = abs(float(t.get("priceChangePercent", 0)))
                        score = volume * price_change
                        if volume >= self.min_volume and price_change >= self.min_percent_change:  # Apply thresholds
                            filtered_by_thresholds.append({
                                "symbol": symbol,
                                "score": score,
                                "source": self.name,
                                "24h_volume": volume,
                                "24h_percent_change": price_change
                            })
                    except (ValueError, TypeError):
                        logger.warning(f"Skipping {symbol} due to parse error.")
                        continue

                final_candidates = []
                if not filtered_by_thresholds:
                    logger.warning("⚠️ No symbols passed thresholds. Falling back to best effort.")
                    # Fallback: re-populate with all tickers (that end with USDT) and their scores
                    for t in tickers:
                        symbol = t.get("symbol", "")
                        if not symbol.endswith("USDT"):
                            continue
                        try:
                            volume = float(t.get("volume", 0))
                            price_change = abs(float(t.get("priceChangePercent", 0)))
                            score = volume * price_change
                            final_candidates.append({
                                "symbol": symbol,
                                "score": score,
                                "source": self.name,
                                "24h_volume": volume,
                                "24h_percent_change": price_change
                            })
                        except (ValueError, TypeError):
                            continue
                else:
                    final_candidates = filtered_by_thresholds

                # Log Top 20 candidates by score before final selection
                logger.info("📊 Top 20 candidates by score (before final selection):")
                for c in sorted(final_candidates, key=lambda x: x["score"], reverse=True)[:20]:
                    logger.info(f"  ➤ {c['symbol']} | Vol: {c['24h_volume']:.2f} | Δ%: {c['24h_percent_change']:.2f} | Score: {c['score']:.2f}")

                # Sort by score and take top N symbols
                top_symbols_with_data = sorted(final_candidates, key=lambda x: x["score"], reverse=True)[:self.top_n_symbols]

                accepted = 0
                for s in top_symbols_with_data:
                    symbol = s["symbol"]
                    # Prefilter (status/minNotional) before proposing
                    try:
                        if not await self._prefilter_symbol(symbol):
                            logger.info(f"🚫 Rejected/Buffered: {symbol} (prefilter)")
                            continue
                    except Exception:
                        logger.debug("prefilter failed in run_once for %s", symbol, exc_info=True)
                        continue
                    logger.info(f"📤 Proposing symbol: {symbol}")
                    try:
                        accepted_flag = await asyncio.wait_for(
                            self._propose(
                                symbol,
                                source=self.name,
                                metadata={
                                    "24h_volume": s["24h_volume"],
                                    "24h_percent_change": s["24h_percent_change"]
                                }
                            ),
                            timeout=5
                        )
                        if accepted_flag:
                            accepted += 1
                            logger.info(f"✅ Accepted: {symbol}")
                        else:
                            logger.info(f"🚫 Rejected/Buffered: {symbol}")
                    except asyncio.TimeoutError:
                        logger.warning(f"⏰ Timeout proposing: {symbol}")
                    except Exception as e:
                        logger.error(f"❌ Error proposing {symbol}: {e}", exc_info=True)

                logger.info(f"✅ Screener completed. {accepted} symbols accepted.")
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
        One-shot discovery method to screen and propose top N symbols
        based on configured 24h volume and percent change thresholds.
        Used during Phase 3.
        """
        try:
            logger.info("🔍 [SymbolScreener] Starting one-time symbol screening (Phase 3)...")

            all_stats = await self.exchange_client.get_24hr_stats()
            if not all_stats:
                logger.warning("⚠️ No stats returned from exchange.")
                return

            candidates_with_score = []
            for symbol, stats in all_stats.items():
                try:
                    volume = float(stats.get("volume", 0))
                    price_change = float(stats.get("priceChangePercent", 0))
                    score = volume * abs(price_change)
                    candidates_with_score.append((symbol, volume, price_change, score))
                except Exception:
                    # Log if parsing fails for a specific symbol, but don't stop the whole process
                    logger.warning(f"Could not parse stats for {symbol} during discovery, skipping.")
                    continue

            # First, try to filter by thresholds
            threshold_met_candidates = [
                c for c in candidates_with_score
                if c[1] >= self.min_volume and abs(c[2]) >= self.min_percent_change # volume, price_change
            ]

            final_candidates_for_discovery = []
            if threshold_met_candidates:
                logger.info("Symbols met thresholds. Using filtered list for discovery.")
                # Sort by priceChangePercent (index 2), then volume (index 1)
                threshold_met_candidates.sort(key=lambda x: (x[2], x[1]), reverse=True)
                final_candidates_for_discovery = threshold_met_candidates[:self.top_n_symbols]
            else:
                logger.warning("⚠️ No symbols met thresholds. Falling back to top symbols by score for discovery.")
                # If no symbols met thresholds, sort all candidates by score
                candidates_with_score.sort(key=lambda x: x[3], reverse=True) # Sort by score (index 3)
                final_candidates_for_discovery = candidates_with_score[:self.top_n_symbols]  # Reduce to top N

            top_symbols = [s[0] for s in final_candidates_for_discovery] # Extract symbols

            if not top_symbols:
                logger.warning("⚠️ No symbols passed the screening criteria for discovery.")
                return

            logger.info(f"✅ [SymbolScreener] Discovered symbols: {top_symbols}")

            for s in final_candidates_for_discovery:
                symbol = s[0]
                volume = s[1]
                percent_change = s[2]
                accepted_flag = await self._propose(
                    symbol,
                    source="SymbolScreener",
                    metadata={"24h_volume": volume, "24h_percent_change": percent_change}
                )
                if accepted_flag:
                    logger.info(f"[SymbolScreener] ✅ Discovery symbol accepted: {symbol}")
                else:
                    logger.warning(f"[SymbolScreener] ❌ Discovery symbol rejected/buffered: {symbol}")

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
