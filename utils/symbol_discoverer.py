import logging
import asyncio
from math import log # Added import for log

logger = logging.getLogger("SymbolDiscovererAgent")

class SymbolDiscovererAgent:
    async def _maybe_await(self, v):
        return await v if asyncio.iscoroutine(v) else v
    agent_type = "discovery"
    name = "SymbolDiscovererAgent"
    def __init__(
        self,
        shared_state,
        config,
        exchange_client=None,
        symbol_manager=None,
        interval=3600,
        **kwargs
    ):
        self.shared_state = shared_state
        self.config = config
        self.exchange_client = exchange_client
        self.symbol_manager = symbol_manager
        self.interval = interval
        self.logger = logger # Changed to use the module-level logger directly
        self.name = "SymbolDiscovererAgent"
        self.is_discovery_agent = True
        self.stop_flag = False
        self.kwargs = kwargs

        # Allow late binding if created before Phase 6
        if self.exchange_client is None and hasattr(self.shared_state, "exchange_client"):
            self.exchange_client = getattr(self.shared_state, "exchange_client")
        if self.symbol_manager is None and hasattr(self.shared_state, "symbol_manager"):
            self.symbol_manager = getattr(self.shared_state, "symbol_manager")


    async def run_once(self):
        self.logger.info("🔍 Running symbol discovery...")

        # discovery complete?
        is_done = getattr(self.shared_state, "is_discovery_complete", None)
        if callable(is_done) and await self._maybe_await(is_done()):
            self.logger.info("⏸ Discovery skipped: discovery already marked complete.")
            return

        # accepted symbols finalized?
        get_acc = getattr(self.shared_state, "get_accepted_symbols", None)
        accepted = await self._maybe_await(get_acc()) if callable(get_acc) else getattr(self.shared_state, "accepted_symbols", {}) or {}
        if isinstance(accepted, (list, set, tuple)):
            accepted = {s: {"enabled": True} for s in accepted}
        if accepted:
            self.logger.info("🚫 Skipping discovery: accepted symbols already finalized.")
            return

        # discovery enabled flag?
        enabled = getattr(self.shared_state, "symbol_discovery_enabled", True)
        if callable(enabled):
            enabled = await self._maybe_await(enabled())
        if not enabled:
            self.logger.warning("🛑 Symbol discovery disabled. Skipping run.")
            return

        # cap check
        get_cnt = getattr(self.shared_state, "get_symbol_count", None)
        if callable(get_cnt):
            current_count = await self._maybe_await(get_cnt())
        else:
            syms = getattr(self.shared_state, "symbols", {})
            current_count = len(syms) if isinstance(syms, dict) else 0
        if hasattr(self.config, "MAX_DISCOVERED_SYMBOLS") and current_count >= self.config.MAX_DISCOVERED_SYMBOLS:
            self.logger.warning(f"🛑 Discovery cap hit ({current_count} >= {self.config.MAX_DISCOVERED_SYMBOLS}). Skipping.")
            return

        new_symbols = await self.discover_symbols()
        if not new_symbols:
            self.logger.warning("No new symbols discovered.")
            return

        if not self.symbol_manager:
            self.logger.error("SymbolManager is not provided. Cannot propose or add symbols.")
            return


        # Deduplicate against existing accepted symbols using safe getter
        existing_keys = set(accepted.keys())
        unique = [s for s in new_symbols if isinstance(s, dict) and s.get("symbol") not in existing_keys]

        if not unique:
            self.logger.info("🔁 No new symbols after deduplication.")
            return

        # Select top 10 based on score (if available)
        top_symbols = sorted(
            unique,
            key=lambda x: x.get("score", 0),
            reverse=True
        )[:10]


        batch_size = 5
        added_count = 0
        propose_many = getattr(self.symbol_manager, "propose_symbols", None)

        for i in range(0, len(top_symbols), batch_size):
            batch = top_symbols[i:i+batch_size]
            symbols = [s["symbol"] for s in batch]
            try:
                if callable(propose_many):
                    result = await asyncio.wait_for(propose_many(symbols, source=self.name), timeout=10)
                    added_count += len(result) if isinstance(result, list) else 0
                else:
                    # fallback to per-symbol
                    for s in symbols:
                        r = await asyncio.wait_for(self.symbol_manager.propose_symbol(s, source=self.name, metadata={"source": self.name}), timeout=5)
                        added_count += 1 if r else 0
                self.logger.info(f"✅ Proposed: {symbols} → Added so far: {added_count}")
            except asyncio.TimeoutError:
                self.logger.warning(f"⏰ Timeout while proposing: {symbols}")
            except Exception as e:
                self.logger.error(f"❌ Error while proposing {symbols}: {e}", exc_info=True)

        self.logger.info(f"✅ Discovery complete. Added {added_count} new symbols.")

        # Graceful toggle-off after reaching cap
        if hasattr(self.config, "MAX_DISCOVERED_SYMBOLS"):
            get_cnt = getattr(self.shared_state, "get_symbol_count", None)
            cur = await self._maybe_await(get_cnt()) if callable(get_cnt) else current_count
            if cur >= self.config.MAX_DISCOVERED_SYMBOLS:
                setter = getattr(self.shared_state, "set_symbol_discovery_enabled", None)
                if callable(setter):
                    await self._maybe_await(setter(False))
                else:
                    setattr(self.shared_state, "symbol_discovery_enabled", False)
                self.logger.info("Symbol discovery disabled as MAX_DISCOVERED_SYMBOLS cap has been reached.")


    async def run_loop(self):
        self.logger.info(f"🌀 [{self.name}] Background loop started. Interval: {self.interval} seconds.")
        await asyncio.sleep(10)
        while not self.stop_flag:
            # Early return check in loop
            if await self._maybe_await(self.shared_state.is_discovery_complete()):
                self.logger.info("🛑 Discovery marked complete. Exiting run_loop.")
                break
            try:
                await self.run_once()
            except Exception as e:
                self.logger.exception(f"❌ [{self.name}] Error in run_loop: {e}")
            await asyncio.sleep(self.interval)


    async def discover_symbols(self):
        if not self.exchange_client:
            self.logger.error("Exchange client not available. Cannot discover symbols.")
            return []

        if getattr(self.config, "SYMBOL_UNIVERSE", None):
            self.logger.info("🔧 Using SYMBOL_UNIVERSE from config.")
            return [{"symbol": s, "source": "config"} for s in self.config.SYMBOL_UNIVERSE]

        if getattr(self.config, "TRENDING_DISCOVERY_ENABLED", False):
            self.logger.info("📈 Discovering trending symbols from Exchange.")
            try:
                tickers = await self.exchange_client.get_24hr_tickers()

                filtered = []
                min_volume = getattr(self.config, "TRENDING_MIN_VOLUME", 0)
                min_price = getattr(self.config, "TRENDING_MIN_PRICE", 0)
                min_price_change = getattr(self.config, "TRENDING_MIN_PRICE_CHANGE_PERCENT", 0)

                for t in tickers:
                    symbol = t.get("symbol", "")
                    status = t.get("status", "TRADING")
                    if not (symbol.endswith("USDT") and not symbol.endswith("BUSD") and status == "TRADING"):
                        self.logger.debug(f"Skipping {symbol}: Not a valid USDT trading pair or status is not TRADING.")
                        continue

                    try:
                        volume = float(t.get("volume", 0))
                        last_price = float(t.get("lastPrice", 0))
                        price_change = float(t.get("priceChangePercent", 0))
                    except (ValueError, TypeError):
                        self.logger.warning(f"Could not parse numeric ticker data for {symbol}. Skipping.")
                        continue

                    if min_volume > 0 and volume < min_volume:
                        self.logger.debug(f"Skipping {symbol}: Volume {volume} < {min_volume}")
                        continue

                    if min_price > 0 and last_price < min_price:
                        self.logger.debug(f"Skipping {symbol}: Price {last_price} < {min_price}")
                        continue

                    if min_price_change > 0 and abs(price_change) < min_price_change:
                        self.logger.debug(f"Skipping {symbol}: Price Change % {price_change} < {min_price_change}")
                        continue

                    score = log(volume + 1) * abs(price_change)
                    filtered.append({"symbol": symbol, "score": score, "source": self.name, "24h_volume": volume})

                filtered.sort(key=lambda x: x["score"], reverse=True)
                limit = getattr(self.config, "TRENDING_LIMIT", 50)
                top_symbols = filtered[:limit]

                self.logger.info(f"📊 Trending Discovery found {len(top_symbols)} symbols.")
                return top_symbols

            except AttributeError:
                self.logger.warning(
                    "Exchange client does not have 'get_24hr_tickers()' method for trending discovery. "
                    "Falling back to all symbols."
                )
            except Exception as e:
                self.logger.error(f"❌ Trending discovery failed: {e}", exc_info=True)
                return []

        self.logger.info("🔄 Falling back to discovering all active USDT symbols from Exchange.")
        try:
            info = await self.exchange_client.get_exchange_info()
            all_symbols = [s["symbol"] for s in info.get("symbols", [])
                           if s["status"] == "TRADING" and s["symbol"].endswith("USDT") and not s["symbol"].endswith("BUSD")]
            self.logger.info(f"Discovered {len(all_symbols)} active USDT symbols from Exchange.")
            return [{"symbol": s, "source": "Exchange_Fallback"} for s in all_symbols]
        except Exception as e:
            self.logger.error(f"❌ Fallback discovery failed: {e}", exc_info=True)
            return []

try:
    from core.agent_registry import AGENT_CLASS_MAP
    AGENT_CLASS_MAP["SymbolDiscovererAgent"] = SymbolDiscovererAgent
    AGENT_CLASS_MAP["SymbolDiscoverer"] = SymbolDiscovererAgent  # alias for registry compatibility
except ImportError:
    pass
