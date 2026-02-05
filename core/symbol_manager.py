from __future__ import annotations

import asyncio
import datetime as dt
import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple
import inspect

# keep these if the rest of your code imports from here; otherwise you can remove
try:
    from utils.symbol_filter_pipeline import SymbolFilterPipeline_symbols  # noqa: F401
except Exception:
    SymbolFilterPipeline_symbols = None  # runtime noop if unused
try:
    from core.config import Config  # noqa: F401
except Exception:
    Config = object  # type: ignore[misc,assignment]


# ---------- small helpers ----------

def _pair_pattern(base: str) -> re.Pattern:
    base = (base or "USDT").upper()
    return re.compile(rf"^[A-Z0-9]+{re.escape(base)}$")


def _extract_quote_volume(kwargs: Dict[str, Any]) -> Optional[float]:
    if not kwargs:
        return None
    for k in ("quote_volume_usd", "quote_volume", "24h_volume", "volume"):
        v = kwargs.get(k)
        if v is None:
            continue
        try:
            return float(v)
        except (ValueError, TypeError):
            pass
    return None


def _meta_from_kwargs(symbol: str, source: str, **kwargs) -> Dict[str, Any]:
    now = dt.datetime.now(dt.timezone.utc)
    entry = {
        "symbol": symbol.upper(),
        "source": source,
        "added_at": kwargs.get("added_at", now),
        "score": float(kwargs.get("score", 0.0)),
        "24h_volume": float(kwargs.get("24h_volume", kwargs.get("volume", 0.0) or 0.0)),
        "price": float(kwargs.get("price", 0.0)),
    }
    # avoid overwriting normalized fields above
    for k, v in kwargs.items():
        entry.setdefault(k, v)
    return entry


def _clean_meta(meta: Dict[str, Any]) -> Dict[str, Any]:
    # Avoid double-passing fields that are already explicit params
    return {k: v for k, v in (meta or {}).items() if k not in ("source", "symbol")}


# ---------- main manager ----------

class SymbolManager:
    """
    Discovers, validates, and maintains the accepted trading symbol set.

    Highlights:
      • TTL-cached exchange info (monotonic clock, thread/loop safe)
      • Bounded-concurrency validation with a single SharedState write
      • Config snapshot for hot-path lookups; dynamic BASE_CURRENCY pattern
      • Early cheap filters; config-driven blacklist; optional stable-base avoidance
      • Safe fallbacks to avoid rate spikes and missing fields
    """

    STABLES = {"USDT", "USDC", "TUSD", "FDUSD", "BUSD", "DAI"}  # heuristic only
    _now_mono = staticmethod(time.monotonic)

    def __init__(
        self,
        shared_state: Optional[Any] = None,
        config: Optional[Any] = None,
        exchange_client: Optional[Any] = None,
        market_data_feed: Optional[Any] = None,
        database_manager: Optional[Any] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.shared_state = shared_state
        self.config = config
        self.exchange_client = exchange_client
        self.market_data_feed = market_data_feed
        self.database_manager = database_manager
        self.logger = logger or logging.getLogger("SymbolManager")

        # ---- config snapshot (avoid getattr in hot paths) ----
        base = str(getattr(self.config, "BASE_CURRENCY", "USDT")).upper() if self.config else "USDT"
        self._base = base
        self._pair_pat = _pair_pattern(base)

        self._min_trade_volume = int(getattr(self.config, "discovery_min_24h_vol", getattr(self.config, "MIN_TRADE_VOLUME", 1000))) if self.config else 1000
        self._max_conc = int(getattr(self.config, "SYMBOL_VALIDATE_MAX_CONCURRENCY", 24)) if self.config else 24
        self._info_cache_ttl = float(getattr(self.config, "SYMBOL_INFO_CACHE_TTL", 900.0)) if self.config else 900.0
        self._snapshot_ttl = float(getattr(self.config, "SYMBOL_SNAPSHOT_TTL", 10.0)) if self.config else 10.0
        self._exclude_stable_base = bool(getattr(self.config, "EXCLUDE_STABLE_BASE", False)) if self.config else False
        self._stable_thr_pct = float(getattr(self.config, "STABLE_MAX_ABS_PCT_CHANGE_24H", 0.6)) if self.config else 0.6
        self._stable_band = float(getattr(self.config, "STABLE_TARGET_BAND", 0.03)) if self.config else 0.03
        self._stable_target = float(getattr(self.config, "STABLE_TARGET_PRICE", 1.0)) if self.config else 1.0
        self._cap = 0
        if self.config:
            # P9: Try namespace first, then legacy flat keys
            disc = getattr(self.config, "DISCOVERY", None)
            if disc and hasattr(disc, "TOP_N_SYMBOLS"):
                self._cap = int(disc.TOP_N_SYMBOLS or 0)
            else:
                self._cap = int(getattr(self.config, "discovery_top_n_symbols", getattr(self.config, "MAX_ACTIVE_SYMBOLS", 0)) or 0)
        self._accept_new = bool(getattr(self.config, "discovery_accept_new_symbols", True)) if self.config else True

        # blacklist (union of config knobs)
        b1 = set(getattr(self.config, "SYMBOL_BLACKLIST", []) or []) if self.config else set()
        b2 = set(getattr(self.config, "SYMBOL_EXCLUDE_LIST", []) or []) if self.config else set()
        self._blacklist: set[str] = {str(s).upper() for s in (b1 | b2)}

        # ---- concurrency & caches ----
        self._sem = asyncio.Semaphore(self._max_conc)

        self.symbol_info_cache: Dict[str, Dict[str, Any]] = {}
        self._info_cache_ts: float = 0.0

        self._snapshot_symbols_cache: Optional[Dict[str, Any]] = None
        self._snapshot_ts: float = 0.0

        self.proposed_symbols: Dict[str, Dict[str, Any]] = {}
        self.buffered_symbols: List[Dict[str, Any]] = []

        self.logger.info(
            "✅ SymbolManager init(base=%s, min_vol=%s, max_conc=%s, ttl=%ss)",
            self._base, self._min_trade_volume, self._max_conc, self._info_cache_ttl,
        )

        # Runtime sanity check after wiring
        mgr = self
        self.logger.info(
            "SymbolManager has propose_symbol? %s from %s",
            hasattr(mgr, "propose_symbol"),
            type(mgr).__module__,
        )

    # --- add inside SymbolManager (public, spec-aligned API) ---

    async def bootstrap(self) -> None:
        await self.initialize_symbols()

    async def discover_symbols(self) -> Dict[str, Dict[str, Any]]:
        """
        Spec: Discovery → filter/validate → commit to SharedState → emit AcceptedSymbolsReady.
        This delegates to initialize_symbols() and then returns the fresh snapshot.
        """
        await self.initialize_symbols()
        return await self._get_symbols_snapshot(force=True)

    def filter_symbols(self, candidates: List[Any]) -> Dict[str, Dict[str, Any]]:
        """
        Spec wants a filter_symbols() entrypoint. We map it to our filter_pipeline().
        """
        return self.filter_pipeline(candidates)

    async def set_accepted_symbols(
        self,
        symbols_map: Dict[str, Dict[str, Any]],
        *,
        allow_shrink: bool = False,
        source: str = "SymbolManager",
    ):
        """
        Spec: set_accepted_symbols(). We wrap the safe internal setter.
        'source' is available for logging; we emit AcceptedSymbolsReady via _safe_set_accepted_symbols.
        """
        return await self._safe_set_accepted_symbols(symbols_map, allow_shrink=allow_shrink)

    # ---------------- public lifecycle ----------------

    async def initialize_symbols(self) -> None:
        """
        Discovery → prefilter → bounded async validation → single shared write.
        """
        if not self.exchange_client:
            self.logger.error("❌ Exchange client is not set. Cannot initialize symbols.")
            return
        if not self.shared_state:
            self.logger.error("❌ SharedState is not set. Cannot initialize symbols.")
            return

        await self._ensure_exchange_info()

        self.logger.info("🚀 Starting symbol discovery and validation…")
        discovered = await self.run_discovery_agents()
        prelim_map = self.filter_pipeline(discovered)
        self.logger.info("🧹 Pre-filtered to %d candidate(s); validating…", len(prelim_map))

        validated: Dict[str, Dict[str, Any]] = {}
        lock = asyncio.Lock()

        async def _check_one(sym: str, meta: Dict[str, Any]):
            async with self._sem:
                src, meta2 = self._split_source(meta, "discovery")
                ok, reason, validated_price = await self._is_symbol_valid(sym, source=src, **meta2)
                if ok:
                    async with lock:
                        # Ensure we use the freshly validated price
                        validated[sym] = {**meta, "source": src, "price": validated_price}
                else:
                    self.logger.debug("Filtered out %s: %s", sym, reason)

        await asyncio.gather(*(_check_one(s, m) for s, m in prelim_map.items()))

        if self._cap and len(validated) > self._cap:
            validated = self._apply_cap(validated)

        if not validated:
            self.logger.error("❌ No symbols validated after discovery! Universe is empty.")
            return

        # P9 Guard: Catastrophic Collapse Prevention
        # If we previously had a healthy universe (e.g. 30 from WalletScanner) 
        # and our fresh discovery only found 1 (due to filters or network flakiness), 
        # we do NOT want to overwrite the healthy set.
        current_map = await self._get_symbols_snapshot(force=True)
        if len(current_map) > 10 and len(validated) <= 1:
            self.logger.error(
                "🛡️ PANIC GUARD: Refusing to commit discovery (found %d) because it would collapse healthy universe (%d).",
                len(validated), len(current_map)
            )
            return

        # Pass allow_shrink=True when setting initial symbols
        await self._safe_set_accepted_symbols(validated, allow_shrink=True)
        self.logger.info("📦 SharedState updated with %d accepted symbol(s).", len(validated))

    # ---------------- validation chain ----------------

    def validate_symbol_format(self, symbol: str) -> Tuple[bool, Optional[str]]:
        if not isinstance(symbol, str):
            return False, "symbol is not a string"
        if not self._pair_pat.match(symbol.upper()):
            return False, f"invalid format; expected {self._pair_pat.pattern}"
        # block self-pairs like USDTUSDT quickly
        if symbol.upper() == f"{self._base}{self._base}":
            return False, "self-quote pair"
        return True, None

    async def is_valid_symbol(self, symbol: str) -> Tuple[bool, Optional[str]]:
        """Checks tradability using cached exchange info; lazy fetch on miss."""
        if not isinstance(symbol, str):
            return False, "not a valid symbol string"
        if not self.exchange_client:
            return False, "exchange client not available"

        await self._ensure_exchange_info()
        info = self.symbol_info_cache.get(symbol)
        if info is None:
            try:
                info = await self.exchange_client.get_symbol_info(symbol)
                if info:
                    self.symbol_info_cache[symbol] = info
            except Exception as e:
                self.logger.debug("get_symbol_info error for %s: %s", symbol, e, exc_info=True)
                return False, "no symbol info"

        if not info:
            return False, "no symbol info"
        if info.get("status") != "TRADING":
            return False, "not trading"

        base = (info.get("baseAsset") or "").upper()
        quote = (info.get("quoteAsset") or "").upper()
        if base == quote:
            return False, "base equals quote"
        # exclude leveraged tokens quickly
        if any(tag in base for tag in ("UP", "DOWN", "BULL", "BEAR")):
            return False, "leveraged token"
        if info.get("isSpotTradingAllowed") is False:
            return False, "spot trading not allowed"
        if quote != self._base:
            return False, f"quote asset mismatch (expected {self._base})"
        return True, None

    async def _passes_risk_filters(self, symbol: str, source: str = "unknown", **kwargs) -> Tuple[bool, Optional[str]]:
        if symbol in self._blacklist:
            return False, "symbol blacklisted (config)"
        if not self.exchange_client:
            return False, "exchange client unavailable"

        # existence via cache (cheap), otherwise awaited call
        if hasattr(self.exchange_client, "symbol_exists_cached"):
            if not self.exchange_client.symbol_exists_cached(symbol):
                return False, "symbol not trading (cached)"
        elif hasattr(self.exchange_client, "symbol_exists"):
            if not await self.exchange_client.symbol_exists(symbol):
                return False, "symbol not trading"

        # quote volume: try kwargs → client quick calls → cached 24h stats
        qv = _extract_quote_volume(kwargs)
        if qv is None and hasattr(self.exchange_client, "get_24hr_volume"):
            try:
                qv = await self.exchange_client.get_24hr_volume(symbol)
            except Exception as e:
                self.logger.debug("get_24hr_volume fail %s: %s", symbol, e, exc_info=True)

        stats: Dict[str, Any] = {}
        if hasattr(self.exchange_client, "get_cached_24h_stats"):
            stats = self.exchange_client.get_cached_24h_stats(symbol) or {}
        
        # Fallback: if stats missing/empty, try explicit fetch (Bootstrap Safety)
        if not stats and hasattr(self.exchange_client, "get_24h_stats"):
            try:
                stats = await self.exchange_client.get_24h_stats(symbol) or {}
            except Exception:
                pass

        # repair/derive quote vol if possible
        quote_vol = float(stats.get("quoteVolume") or stats.get("volume") or 0.0)
        base_vol = float(stats.get("baseVolume") or 0.0)
        wap = float(stats.get("weightedAvgPrice") or 0.0)
        if quote_vol == 0.0 and base_vol > 0.0 and wap > 0.0:
            quote_vol = base_vol * wap
        if qv is None and quote_vol > 0.0:
            qv = quote_vol

        if qv is None:
            # P9 Guard: If the source is authoritative (WalletScanner), we skip volume check
            # if we can't find volume, as it's better to keep the symbol than to lose it.
            if source == "WalletScannerAgent":
                self.logger.debug(f"[{source}] No volume info for {symbol}; allowing as authoritative")
                return True, None
            return False, "missing 24h quote volume"
            
        if float(qv) < float(self._min_trade_volume):
            if source == "WalletScannerAgent":
                 self.logger.info(f"[WalletScannerAgent] Symbol {symbol} volume {qv} < {self._min_trade_volume}, but allowing as authoritative")
                 return True, None
            return False, f"below min 24h quote volume ({qv} < {self._min_trade_volume})"

        # optional: exclude stable base assets (if not using BASE_CURRENCY as base asset)
        if self._exclude_stable_base:
            info = self.symbol_info_cache.get(symbol) or {}
            base = (info.get("baseAsset") or "").upper()
            if base and base != self._base:
                is_stable = False
                if hasattr(self.exchange_client, "is_stable_asset"):
                    try:
                        is_stable = await self.exchange_client.is_stable_asset(base)
                    except Exception:
                        is_stable = False
                else:
                    # heuristic from 24h stats
                    pct = abs(float(stats.get("priceChangePercent") or 0.0))
                    wap2 = float(stats.get("weightedAvgPrice") or 0.0)
                    band_lo = self._stable_target * (1 - self._stable_band)
                    band_hi = self._stable_target * (1 + self._stable_band)
                    is_stable = (pct <= self._stable_thr_pct) and (band_lo <= wap2 <= band_hi)
                if is_stable:
                    return False, "base asset classified as stable"

        return True, None

    async def _is_symbol_valid(self, symbol: str, source: str = "unknown", **kwargs) -> Tuple[bool, Optional[str], float]:
        s = symbol.upper()
        if s in self._blacklist:
            return False, "blacklisted", 0.0
        ok, reason = self.validate_symbol_format(s)
        if not ok:
            return False, reason, 0.0
        # call is_valid_symbol first for generic checks (ExchangeInfo)
        ok, reason = await self.is_valid_symbol(s)
        if not ok:
            return False, reason, 0.0
        ok, reason = await self._passes_risk_filters(s, source, **kwargs)
        if not ok:
            self.logger.debug("risk filter failed for %s: %s", s, reason)
            return False, reason, 0.0
        
        # P0: Ensure we have a real price before accepting
        price = float(kwargs.get("price", 0.0))
        if price <= 0:
            try:
                price = await self.exchange_client.get_ticker_price(s)
            except Exception:
                price = 0.0
        
        if price <= 0:
             return False, "market price unavailable", 0.0
             
        return True, None, price

    async def validate_symbols(self, symbols: List[str]) -> List[str]:
        """Validate a batch concurrently (bounded)."""
        out: List[str] = []

        async def _one(sym: str):
            async with self._sem:
                ok, _, _ = await self._is_symbol_valid(sym)
                if ok:
                    out.append(sym.upper())

        await asyncio.gather(*(_one(s) for s in symbols))
        return out

    async def _safe_set_accepted_symbols(self, symbols_map: dict, *, allow_shrink: bool = False, source: Optional[str] = None):
        """
        Calls SharedState.set_accepted_symbols, passing allow_shrink only if supported.
        Keeps behavior aligned with the current design while remaining backward compatible.
        """
        if not self.shared_state or not hasattr(self.shared_state, "set_accepted_symbols"):
            self.logger.error("❌ SharedState missing set_accepted_symbols.")
            return

        # Sanitize the metadata for each symbol before passing to SharedState
        sanitized_map = {
            s: {k: v for k, v in m.items() if k != "symbol"}
            for s, m in symbols_map.items()
        }

        fn = self.shared_state.set_accepted_symbols
        try:
            sig = inspect.signature(fn)
            kwargs_call = {}
            if "allow_shrink" in sig.parameters:
                kwargs_call["allow_shrink"] = allow_shrink
            if "source" in sig.parameters and source:
                kwargs_call["source"] = source

            result = fn(sanitized_map, **kwargs_call)
            if asyncio.iscoroutine(result):
                result = await result

            # Inside _safe_set_accepted_symbols, after successful call:
            if hasattr(self.shared_state, "emit_event"):
                try:
                    payload = {
                        "count": len(symbols_map),
                        "reason": "init" if allow_shrink else "finalize",
                        "cap_applied": bool(self._cap and len(symbols_map) > self._cap),
                        "cap_value": int(self._cap or 0),
                        "symbols": list(symbols_map.keys())[:20],
                        "source": source or "unknown"
                    }
                    await self.shared_state.emit_event("AcceptedSymbolsReady", payload)
                except Exception as e:
                    self.logger.warning("Failed to emit AcceptedSymbolsReady: %s", e)

            return result

        except TypeError as e:
            # Extra belt-and-suspenders fallback without allow_shrink/source
            self.logger.warning("set_accepted_symbols signature mismatch (%s) — retrying positional-only.", e)
            result = fn(sanitized_map)
            if asyncio.iscoroutine(result):
                result = await result
            return result

    # ---------------- mutations ----------------

    async def add_symbol(self, symbol: str, source: str = "unknown", **kwargs) -> Tuple[bool, Optional[str]]:
        if not symbol:
            self.logger.warning("❌ Empty symbol.")
            return False, "empty symbol"
        symbol = symbol.upper()
        if not getattr(self, "_accept_new", True):
            self.logger.info("Discovery.accept_new_symbols is False; rejecting %s", symbol)
            return False, "discovery disabled"

        snap = await self._get_symbols_snapshot()
        if symbol in snap:
            self.logger.debug("⚠️ %s already exists; skipping.", symbol)
            return False, "already exists"

        # Avoid duplicate explicit params being passed through **kwargs
        kwargs.pop("source", None)
        kwargs.pop("symbol", None)
        ok, reason, validated_price = await self._is_symbol_valid(symbol, source=source, **kwargs)
        if not ok:
            self.logger.warning("❌ %s rejected from %s: %s", symbol, source, reason)
            return False, reason

        # Update price for metadata
        kwargs["price"] = validated_price
        meta = _meta_from_kwargs(symbol, source, **kwargs)
        meta_clean = {k: v for k, v in meta.items() if k != "symbol"}
        try:
            final_map = dict(await self._get_symbols_snapshot(force=True))
            # store meta WITHOUT "symbol" key (SharedState holds map by symbol)
            final_map[symbol] = {k: v for k, v in meta.items() if k != "symbol"}

            if self._cap and len(final_map) > self._cap:
                final_map = self._apply_cap(final_map)

            # P9 Fix: Use allow_shrink=False for additive additions. Pass source for sticky policies.
            await self._safe_set_accepted_symbols(final_map, allow_shrink=False, source=source)

            if self.database_manager and hasattr(self.database_manager, "add_symbol"):
                self.database_manager.add_symbol(symbol)

            self.logger.info("✅ Accepted %s from %s.", symbol, source)
            return True, None
        except Exception as e:
            self.logger.error("❌ add_symbol exception for %s: %s", symbol, e, exc_info=True)
            return False, f"exception during addition: {e}"

    async def propose_symbol(self, symbol: str, source: str = "unknown", **kwargs) -> Tuple[bool, Optional[str]]:
        if not self.shared_state:
            self.logger.warning("⚠️ SharedState not set; cannot propose.")
            return False, "SharedState not initialized"

        symbol = (symbol or "").upper()
        if not getattr(self, "_accept_new", True):
            self.logger.info("Discovery.accept_new_symbols is False; rejecting proposal %s", symbol)
            return False, "discovery disabled"
        snap = await self._get_symbols_snapshot()
        if symbol in snap:
            self.logger.info("⚠️ %s already exists; skipping proposal.", symbol)
            return False, "already exists"

        if self._cap and len(snap) >= self._cap:
            # try immediate prioritized add; if it fails, buffer for later flush
            self.logger.info("📦 Cap reached; trying prioritized add for %s…", symbol)
            ok, reason = await self.add_symbol(symbol, source=source, **kwargs)
            if ok:
                return True, None
            self.buffered_symbols.append(_meta_from_kwargs(symbol, source, **kwargs))
            self.logger.info("⏳ Queued %s (cap reached).", symbol)
            return False, reason or "cap reached"

        return await self.add_symbol(symbol, source=source, **kwargs)

    async def propose_symbols(self, symbols: List[str], source: str = "unknown", **kwargs) -> List[str]:
        """BATACH PROPOSAL: Validates and adds multiple symbols in one SharedState write."""
        self.logger.info("🧪 Validating batch of %d proposed symbol(s) from %s…", len(symbols), source)
        
        # 1. Fetch current pool once
        final_map = dict(await self._get_symbols_snapshot(force=True))
        added_count = 0
        accepted_names = []

        # 2. Iterate and validate locally
        for s in symbols:
            s_up = s.upper()
            if s_up in final_map:
                 continue
            
            ok, reason, px = await self._is_symbol_valid(s_up, source=source, **kwargs)
            if ok:
                meta = _meta_from_kwargs(s_up, source, price=px, **kwargs)
                final_map[s_up] = {k: v for k, v in meta.items() if k != "symbol"}
                added_count += 1
                accepted_names.append(s_up)
            else:
                self.logger.debug(f"[{source}] Rejected {s_up}: {reason}")

        if added_count > 0:
            # 3. Apply cap if needed
            if self._cap and len(final_map) > self._cap:
                final_map = self._apply_cap(final_map)

            # 4. Commit once with allow_shrink=False to preserve screener work
            await self._safe_set_accepted_symbols(final_map, allow_shrink=False, source=source)
            self.logger.info("✅ Batch update complete: +%d symbol(s) from %s.", added_count, source)
        else:
            self.logger.info("ℹ️ No new symbols added from batch.")

        return accepted_names

    async def flush_buffered_proposals_to_shared_state(self) -> None:
        if not self.shared_state:
            self.logger.warning("⚠️ SharedState not set; cannot flush.")
            return
        if not self.proposed_symbols and not self.buffered_symbols:
            self.logger.info("✅ Flushed 0 buffered symbols (no-op).")
            return
        if not getattr(self, "_accept_new", True):
            self.logger.info("Discovery.accept_new_symbols is False; skipping buffered proposals flush.")
            return

        effective_blacklist = set(self._blacklist)
        current = dict(await self._get_symbols_snapshot(force=True))
        final_map: Dict[str, Dict[str, Any]] = dict(current)
        newly_added = 0

        # sanitize proposed map
        sanitized = {
            s: m for s, m in (self.proposed_symbols or {}).items()
            if isinstance(s, str)
            and self._pair_pat.match(s)
            and s != f"{self._base}{self._base}"
            and s not in effective_blacklist
        }
        self.proposed_symbols.clear()

        async def _try_add(sym: str, meta: Dict[str, Any], src_hint: str):
            nonlocal newly_added
            if sym in final_map:
                return
            src, meta2 = self._split_source(meta, src_hint)
            ok, reason = await self._is_symbol_valid(sym, source=src, **meta2)
            if ok:
                meta_clean = dict(meta2); meta_clean["source"] = src # Reconstruct meta with source for final_map
                final_map[sym] = meta_clean
                newly_added += 1
            else:
                self.logger.warning("❌ Buffered %s failed revalidation: %s", sym, reason)

        await asyncio.gather(*(_try_add(s, m, m.get("source", "buffer")) for s, m in sanitized.items()))

        tmp = list(self.buffered_symbols)
        self.buffered_symbols.clear()
        await asyncio.gather(*[
            _try_add(e.get("symbol"), e, e.get("source", "cap_buffer"))
            for e in tmp
            if isinstance(e, dict)
            and isinstance(e.get("symbol"), str)
            and self._pair_pat.match(e["symbol"])
            and e["symbol"] != f"{self._base}{self._base}"
            and e["symbol"] not in effective_blacklist
        ])

        # apply cap
        if self._cap and len(final_map) > self._cap:
            final_map = self._apply_cap(final_map)

        # final existence sanitation
        if self.exchange_client and hasattr(self.exchange_client, "symbol_exists_cached"):
            final_map = {s: m for s, m in final_map.items() if self.exchange_client.symbol_exists_cached(s)}
        elif self.exchange_client and hasattr(self.exchange_client, "symbol_exists"):
            self.logger.warning("⚠️ Falling back to awaitable symbol_exists. May hit rate limits.")
            exists: Dict[str, Dict[str, Any]] = {}
            sem = asyncio.Semaphore(self._max_conc)

            async def _exists(sym: str, meta: Dict[str, Any]):
                async with sem:
                    if await self.exchange_client.symbol_exists(sym):
                        exists[sym] = meta

            await asyncio.gather(*(_exists(s, m) for s, m in final_map.items()))
            final_map = exists

        # Pass allow_shrink=True when setting symbols after flush
        await self._safe_set_accepted_symbols(final_map, allow_shrink=True)
        self.logger.info("✅ Accepted %d symbol(s) after flush. SharedState updated. (+%d new)", len(final_map), newly_added)

    async def finalize_universe(
        self,
        cap: Optional[int] = None,
        allow_shrink: bool = False,
        source: str = "SymbolManager", # 'source' argument is kept for potential logging/internal use
    ) -> Dict[str, Dict[str, Any]]:
        """
        Trim current accepted symbols to `cap` (or self._cap) and commit back to SharedState.
        Uses allow_shrink to legally reduce the universe when needed.
        """
        if not self.shared_state:
            self.logger.error("❌ SharedState is not set; cannot finalize universe.")
            return {}

        # Pull current accepted set from SharedState (fresh)
        current = dict(await self._get_symbols_snapshot(force=True))

        # Decide cap
        eff_cap = cap if (cap is not None and cap > 0) else self._cap

        final_map = current
        if eff_cap and len(final_map) > eff_cap:
            final_map = self._apply_cap(final_map)

        # Commit with shrink permission if requested
        await self._safe_set_accepted_symbols(final_map, allow_shrink=allow_shrink)
        self.logger.info(
            "✅ Finalized %d symbols (cap=%s, allow_shrink=%s).",
            len(final_map), eff_cap, allow_shrink
        )
        return final_map

    # ---------------- queries & utilities ----------------

    async def get_final_universe(self) -> Dict[str, Dict[str, Any]]:
        """
        Canonical Phase-5 API: return the accepted symbols map that should be used
        to seed MarketDataFeed. Applies cap if configured.
        """
        current = dict(await self._get_symbols_snapshot(force=True))
        if self._cap and len(current) > self._cap:
            current = self._apply_cap(current)
        # Ensure meta has no "symbol" key duplication (SharedState holds map by symbol)
        return {s: {k: v for k, v in (m or {}).items() if k != "symbol"} for s, m in current.items()}

    async def get_valid_symbol_names_async(self) -> List[str]:
        snap = await self._get_symbols_snapshot(force=False)
        names = list((snap or {}).keys())
        self.logger.info("✅ get_valid_symbol_names_async() → %d symbols", len(names))
        return names

    def get_valid_symbol_names(self) -> List[str]:
        if self.shared_state and hasattr(self.shared_state, "get_symbols_snapshot"):
            names = list((self.shared_state.get_symbols_snapshot() or {}).keys())
            self.logger.info("✅ get_valid_symbol_names() → %d symbols", len(names))
            return names
        self.logger.warning("⚠️ SharedState not initialized or empty.")
        return []

    def get_invalid_symbols(self) -> List[str]:
        if self.shared_state and hasattr(self.shared_state, "get_symbols_snapshot"):
            current = self.shared_state.get_symbols_snapshot() or {}
            return [item["symbol"] for item in self.buffered_symbols if item.get("symbol") not in current]
        self.logger.warning("SharedState not available for get_invalid_symbols.")
        return []

    def set_shared_state(self, shared_state):
        self.shared_state = shared_state
        return self

    def set_config(self, config):
        """Hot-swap config; refresh snapshots & pattern."""
        self.config = config
        base = str(getattr(self.config, "BASE_CURRENCY", "USDT")).upper()
        self._base = base
        self._pair_pat = _pair_pattern(base)

        self._min_trade_volume = int(getattr(self.config, "discovery_min_24h_vol", getattr(self.config, "MIN_TRADE_VOLUME", 1000)))
        self._max_conc = int(getattr(self.config, "SYMBOL_VALIDATE_MAX_CONCURRENCY", 24))
        self._info_cache_ttl = float(getattr(self.config, "SYMBOL_INFO_CACHE_TTL", 900.0))
        self._snapshot_ttl = float(getattr(self.config, "SYMBOL_SNAPSHOT_TTL", 10.0))
        self._exclude_stable_base = bool(getattr(self.config, "EXCLUDE_STABLE_BASE", False))
        self._stable_thr_pct = float(getattr(self.config, "STABLE_MAX_ABS_PCT_CHANGE_24H", 0.6))
        self._stable_band = float(getattr(self.config, "STABLE_TARGET_BAND", 0.03))
        self._stable_target = float(getattr(self.config, "STABLE_TARGET_PRICE", 1.0))
        self._cap = int(getattr(self.config, "discovery_top_n_symbols", getattr(self.config, "MAX_ACTIVE_SYMBOLS", 0)) or 0)
        self._accept_new = bool(getattr(self.config, "discovery_accept_new_symbols", True))

        b1 = set(getattr(self.config, "SYMBOL_BLACKLIST", []) or [])
        b2 = set(getattr(self.config, "SYMBOL_EXCLUDE_LIST", []) or [])
        self._blacklist = {str(s).upper() for s in (b1 | b2)}

        self._sem = asyncio.Semaphore(self._max_conc)
        self.logger.info("🔧 Config hot-swapped (base=%s, min_vol=%s, max_conc=%s).", self._base, self._min_trade_volume, self._max_conc)
        return self

    def set_exchange_client(self, exchange_client):
        self.exchange_client = exchange_client
        return self

    def set_market_data_feed(self, market_data_feed):
        self.market_data_feed = market_data_feed
        return self

    async def get_recent_symbols(self, max_symbols: int = 10) -> List[str]:
        try:
            if not self.exchange_client:
                self.logger.warning("[SymbolManager] Exchange client unavailable.")
                return []
            if hasattr(self.exchange_client, "get_new_listings_cached"):
                new_listings = await self.exchange_client.get_new_listings_cached()
            elif hasattr(self.exchange_client, "get_new_listings"):
                new_listings = await self.exchange_client.get_new_listings()
            else:
                self.logger.warning("[SymbolManager] No method for new listings.")
                return []
            
            recent = [s for s in (new_listings or []) if isinstance(s, str) and s.endswith(self._base)]
            self.logger.info("[SymbolManager] Found %d recent %s listings.", len(recent), self._base)
            return recent[:max_symbols]
        except Exception as e:
            self.logger.error("[SymbolManager] Failed to get recent symbols: %s", e, exc_info=True)
            return []

    async def run_loop(self) -> None:
        """Periodic maintenance (refresh cached exchange info)."""
        self.logger.info("🟡 [SymbolManager] Starting continuous symbol management loop.")
        interval = int(getattr(self.config, "SYMBOL_INFO_UPDATE_INTERVAL_SECONDS", 3600)) if self.config else 3600
        while True:
            try:
                await self._ensure_exchange_info(force=True)
                self.logger.debug("Symbol info cache refreshed.")
            except Exception as e:
                self.logger.warning("Symbol info refresh failed: %s", e, exc_info=True)
            await asyncio.sleep(interval)

    # ---------------- discovery & cache ----------------

    async def run_discovery_agents(self) -> List[Dict[str, Any]]:
        """Data-driven discovery from cached exchange_info with robust filtering."""
        self.logger.info("Running discovery from exchange_info…")
        await self._ensure_exchange_info()
        out: List[Dict[str, Any]] = []
        now = dt.datetime.now(dt.timezone.utc)
        
        # P9: Robust filtering to avoid single-symbol locks
        stats_map = {}
        if hasattr(self.exchange_client, "get_all_24h_stats"):
            try:
                stats_map = await self.exchange_client.get_all_24h_stats()
            except Exception:
                pass
        
        min_v = float(self._min_trade_volume)

        for s, info in (self.symbol_info_cache or {}).items():
            if not isinstance(info, dict) or not info.get("symbol"):
                continue
            
            # P9: Integration of user-suggested filtering logic
            if info.get("isSpotTradingAllowed") is False:
                continue
            
            if info.get("status") != "TRADING":
                continue

            if (info.get("quoteAsset") or "").upper() != self._base:
                continue

            stats = stats_map.get(s) or {}
            qv = float(stats.get("quoteVolume") or stats.get("volume") or 0.0)
            
            # Lenient floor (from _min_trade_volume)
            if qv < min_v and min_v > 0:
                continue

            out.append({
                "symbol": s,
                "source": "exchange_info_discovery",
                "added_at": now,
                "score": 0.0,
                "24h_volume": qv,
                "price": float(stats.get("lastPrice", 0.0)),
            })
        self.logger.info("Discovered %d potential symbol(s) from exchange info (Volume floor: %.1f).", len(out), min_v)
        return out

    async def _ensure_exchange_info(self, force: bool = False) -> None:
        """Fetch full exchange info once, with TTL (monotonic)."""
        if not self.exchange_client:
            self.logger.warning("⚠️ Exchange client not available. Cannot fetch symbol info.")
            return
        now = self._now_mono()
        if not force and self.symbol_info_cache and (now - self._info_cache_ts) < self._info_cache_ttl:
            return
        self.logger.info("🔄 Fetching & caching all symbol information from the exchange…")
        try:
            exchange_info = await self.exchange_client.get_exchange_info()
            symbols = (exchange_info or {}).get("symbols") or []
            self.symbol_info_cache = {s.get("symbol"): s for s in symbols if s.get("symbol")}
            self._info_cache_ts = now
            self.logger.info("✅ Cached info for %d symbols.", len(self.symbol_info_cache))
        except Exception as e:
            self.logger.error("❌ Failed to fetch/cache symbol information: %s", e, exc_info=True)

    # ---------------- filter pipeline & accessors ----------------

    def filter_pipeline(self, symbol_list: List[Any]) -> Dict[str, Dict[str, Any]]:
        """Normalize, blacklist-filter, and enrich basic fields (cheap, pre-validation)."""
        self.logger.info("Running filtering pipeline…")
        bl = set(self._blacklist)
        out: Dict[str, Dict[str, Any]] = {}
        now = dt.datetime.now(dt.timezone.utc)

        for entry in symbol_list:
            sym: Optional[str] = None
            meta: Dict[str, Any] = {}

            if isinstance(entry, dict) and "symbol" in entry and isinstance(entry["symbol"], str):
                sym = entry["symbol"].upper()
                meta = dict(entry)
            elif isinstance(entry, str):
                sym = entry.upper()
                meta = {"symbol": sym, "source": "filter_pipeline_conversion"}
            else:
                continue

            if not self._pair_pat.match(sym):
                continue
            if sym in bl or sym == f"{self._base}{self._base}":
                continue

            meta.setdefault("symbol", sym)
            meta.setdefault("source", meta.get("source", "unknown"))
            meta.setdefault("score", float(meta.get("score", 0.0)))
            meta.setdefault("24h_volume", float(meta.get("24h_volume", meta.get("volume", 0.0) or 0.0)))
            meta.setdefault("price", float(meta.get("price", 0.0)))
            meta.setdefault("added_at", meta.get("added_at", now))

            out[sym] = meta

        return out

    def get_filters(self, symbol: str) -> List[Dict[str, Any]]:
        # Prefer local cached exchangeInfo; keep this method sync-safe (no awaits here)
        info = self.symbol_info_cache.get(symbol) or {}
        fx = info.get("filters") or []
        if not fx:
            # Best-effort: if the exchange_client maintains a local filters cache map, try to read it synchronously
            try:
                raw = None
                # Some clients expose a dict-like `symbol_filters` map
                raw = getattr(self.exchange_client, "symbol_filters", None) if self.exchange_client else None
                if isinstance(raw, dict):
                    rf = raw.get(symbol) or raw.get(str(symbol).upper())
                    if isinstance(rf, dict) and rf:
                        fx_list = []
                        for k, v in rf.items():
                            if k.startswith("_"):
                                continue
                            if isinstance(v, dict):
                                d = dict(v); d.setdefault("filterType", k)
                                fx_list.append(d)
                        if fx_list:
                            return fx_list
            except Exception:
                self.logger.debug("get_filters sync read of client cache failed; falling back to local cache", exc_info=True)
        if not fx:
            self.logger.debug("No filters in cache for %s. Cache size: %d", symbol, len(self.symbol_info_cache))
        return fx

    def get_step_size(self, symbol: str) -> str:
        for f in self.get_filters(symbol):
            if f.get("filterType") == "LOT_SIZE":
                return f.get("stepSize", "0.000001")
        return "0.000001"

    def get_min_notional(self, symbol: str) -> float:
        # Keep sync-safe: use local cache or a synchronous peek of the client's `symbol_filters` map
        try:
            if self.exchange_client and isinstance(getattr(self.exchange_client, "symbol_filters", None), dict):
                rf = self.exchange_client.symbol_filters.get(symbol) or self.exchange_client.symbol_filters.get(str(symbol).upper())
                if isinstance(rf, dict):
                    notional = rf.get("NOTIONAL") or rf.get("MIN_NOTIONAL") or {}
                    return float(notional.get("minNotional", 0.0) or 0.0)
        except Exception:
            pass
        for f in self.get_filters(symbol):
            if f.get("filterType") in ("MIN_NOTIONAL", "NOTIONAL"):
                try:
                    return float(f.get("minNotional", 0.0))
                except Exception:
                    return 0.0
        return 0.0

    # ---------------- persistence (optional) ----------------

    async def persist_proposals_to_db(self) -> None:
        if self.database_manager:
            for sym, meta in self.proposed_symbols.items():
                await self.database_manager.save_symbol_proposal(sym, meta)
        else:
            self.logger.debug("No DB manager. Skipping proposal persistence.")

    async def load_proposals_from_db(self) -> None:
        if self.database_manager:
            self.proposed_symbols = await self.database_manager.load_symbol_proposals()
            self.logger.info("✅ Loaded %d proposed symbol(s) from DB.", len(self.proposed_symbols))
        else:
            self.logger.debug("No DB manager. Skipping proposal load.")

    async def finalize_symbol_proposals(self) -> None:
        if not self.shared_state:
            self.logger.error("❌ Cannot finalize proposals: SharedState is not set.")
            return
        
        accepted = [* (await self.get_valid_symbol_names_async())]

        if not accepted:
            self.logger.info("ℹ️ No accepted symbols to finalize.")
            return

        if self.database_manager and hasattr(self.database_manager, "write_symbol_snapshot"):
            try:
                await self.database_manager.write_symbol_snapshot(accepted, phase="discovery")
                self.logger.info("📸 Snapshot of %d symbols saved to DB.", len(accepted))
            except Exception as e:
                self.logger.error("❌ Failed to save symbol snapshot to DB: %s", e)
        else:
            self.logger.debug("Skipping snapshot: DB manager not attached or missing method.")
        self.logger.info("✅ Finalized %d symbols in SharedState.", len(accepted))

    # ---------------- internals ----------------

    def _apply_cap(self, sym_map: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Trim to MAX_ACTIVE_SYMBOLS using score → 24h_volume → price → symbol. 
        Active positions and pending intents are protected."""
        
        # 1. Identify protected symbols (those with balance or active allocation)
        protected = set()
        if self.shared_state:
            # Check positions
            if hasattr(self.shared_state, "get_positions_snapshot"):
                try:
                    snap = self.shared_state.get_positions_snapshot()
                    if asyncio.iscoroutine(snap):
                        # skip async check here as _apply_cap is sync; assume cache exists
                        pass
                    else:
                        protected.update((snap or {}).keys())
                except Exception:
                    pass
            
            # Check pending buy intents (P9-Safe Accumulation)
            if hasattr(self.shared_state, "_pending_position_intents"):
                intents = getattr(self.shared_state, "_pending_position_intents", {})
                for (sym, side), intent in list(intents.items()): # Use list() to avoid mutation during iteration
                    if side == "BUY" and intent.accumulated_quote > 0:
                        protected.update([sym])
            
            # P9: Protect Wallet-Force symbols (Legitimate universe preservation)
            if hasattr(self.shared_state, "accepted_symbols"):
                for sym, meta in self.shared_state.accepted_symbols.items():
                    if meta.get("accept_policy") == "wallet_force":
                        protected.update([sym])

        items = list(sym_map.items())

        def score_key(it: Tuple[str, Dict[str, Any]]):
            sym, m = it
            is_protected = 1 if sym in protected else 0
            return (-is_protected,
                    -float(m.get("score", 0.0)),
                    -float(m.get("24h_volume", 0.0)),
                    -float(m.get("price", 0.0)),
                    sym)

        # Sort so protected items are at the top, then by score
        sorted_items = sorted(items, key=score_key)
        
        # We take the top N. If we have MORE protected items than the cap, 
        # the cap is effectively expanded to include all protected items.
        cap_val = max(self._cap, len(protected))
        trimmed = dict(sorted_items[: cap_val])
        
        self.logger.info("✂️ Trimmed to %d symbols (cap=%d, protected=%d).", len(trimmed), self._cap, len(protected))
        return trimmed

    async def _get_symbols_snapshot(self, *, force: bool = False) -> Dict[str, Any]:
        if not self.shared_state or not hasattr(self.shared_state, "get_symbols_snapshot"):
            return {}
        now = self._now_mono()
        if not force and self._snapshot_symbols_cache and (now - self._snapshot_ts) < self._snapshot_ttl:
            return self._snapshot_symbols_cache
        snap = self.shared_state.get_symbols_snapshot()
        # support both sync/async
        snap = await snap if asyncio.iscoroutine(snap) else snap
        self._snapshot_symbols_cache = dict(snap or {})
        self._snapshot_ts = now
        return self._snapshot_symbols_cache

    # New helper to split source from meta
    def _split_source(self, meta: Dict[str, Any] | None, default_src: str) -> tuple[str, Dict[str, Any]]:
        """
        Return (source, cleaned_meta). Strip keys that map to explicit parameters
        so we can safely do: _is_symbol_valid(symbol, source=..., **meta).
        """
        m = dict(meta or {})
        src = str(m.pop("source", default_src))
        m.pop("symbol", None)  # ← critical: avoid duplicate 'symbol' arg
        return src, m
