import time
import logging
try:
    import _thread  # for low-level lock type detection
except Exception:
    _thread = None
from typing import Any, Dict, Optional, List

logger = logging.getLogger("SharedStateTools")
logger.setLevel(logging.INFO)

# Identify concrete threading lock classes at import time (portable across CPython versions)
try:
    _LOCK_CLASS = threading.Lock().__class__
    _RLOCK_CLASS = threading.RLock().__class__
    _THREAD_LOCK_TYPES = (_LOCK_CLASS, _RLOCK_CLASS)
except Exception:
    _THREAD_LOCK_TYPES = tuple()

# ---------- internal helpers ----------
class _NoopAsyncLock:
    """A no-operation asynchronous context manager for scenarios where a lock isn't needed."""
    async def __aenter__(self):
        return self
    async def __aexit__(self, exc_type, exc, tb):
        return False

# ---------- shared state convenience helpers ----------
def safe_price(shared_state, symbol: str, default: float = 0.0) -> float:
    """Return price for symbol or default if not available."""
    try:
        sym = (symbol or "").upper().replace("/", "")
        return float(shared_state.latest_prices.get(sym, default))
    except Exception:
        return float(default)

def spread_bps(shared_state, symbol: str) -> Optional[float]:
    """Return spread in basis points (bps) for symbol, if bid/ask available."""
    try:
        sym = (symbol or "").upper().replace("/", "")
        f = shared_state.symbol_filters.get(sym, {})
        bid = float(f.get("bidPrice", 0.0))
        ask = float(f.get("askPrice", 0.0))
        if bid > 0 and ask > 0:
            return ((ask - bid) / bid) * 10000.0
    except Exception:
        pass
    return None

def fee_bps(shared_state, kind: str = "taker") -> Optional[float]:
    """Return fee in basis points (bps) for maker/taker from config or filters."""
    try:
        k = str(kind).lower()
        # Prefer config, fallback to symbol_filters (first symbol)
        if hasattr(shared_state, "config"):
            if k == "maker" and hasattr(shared_state.config, "maker_fee_bps"):
                return float(shared_state.config.maker_fee_bps)
            if k == "taker" and hasattr(shared_state.config, "taker_fee_bps"):
                return float(shared_state.config.taker_fee_bps)
        # Fallback: try first symbol's filters
        for sym in shared_state.symbol_filters:
            f = shared_state.symbol_filters[sym]
            if k == "maker" and "makerFee" in f:
                return float(f["makerFee"]) * 10000.0 if f["makerFee"] < 1 else float(f["makerFee"])
            if k == "taker" and "takerFee" in f:
                return float(f["takerFee"]) * 10000.0 if f["takerFee"] < 1 else float(f["takerFee"])
        return None
    except Exception:
        return None

def min_notional(shared_state, symbol: str) -> Optional[float]:
    """Return minNotional (minimum quote value) for symbol, if available."""
    try:
        sym = (symbol or "").upper().replace("/", "")
        f = shared_state.symbol_filters.get(sym, {})
        mn = f.get("minNotional")
        if mn is not None:
            return float(mn)
        # Fallback: nested dicts
        if "MIN_NOTIONAL" in f and "minNotional" in f["MIN_NOTIONAL"]:
            return float(f["MIN_NOTIONAL"]["minNotional"])
        if "notional" in f and "min" in f["notional"]:
            return float(f["notional"]["min"])
    except Exception:
        pass
    return None

def quote_value(shared_state, symbol: str, qty: float) -> Optional[float]:
    """Return estimated quote value for symbol and qty using latest price."""
    try:
        px = safe_price(shared_state, symbol, 0.0)
        return float(qty) * float(px)
    except Exception:
        return None


import asyncio
import threading

def _get_lock(shared_state: Any):
    """
    Obtain an async lock from SharedState, accommodating both .get_lock() method
    and .lock property. Falls back to a no-op lock if no valid lock is found.
    If a non-async lock (e.g., threading.Lock) is found, wrap it in an async-compatible wrapper.
    """
    getter = getattr(shared_state, "get_lock", None)
    if callable(getter):
        try:
            lock = getter()
            # If get_lock() returned an awaitable that yields a lock, wrap it so we can await on enter
            if asyncio.iscoroutine(lock):
                class _AwaitableLockWrapper:
                    def __init__(self, awaitable):
                        self._awaitable = awaitable
                        self._lock = None
                    async def __aenter__(self):
                        self._lock = await self._awaitable
                        # Delegate to the underlying lock if it supports async context
                        if hasattr(self._lock, "__aenter__") and asyncio.iscoroutinefunction(self._lock.__aenter__):
                            return await self._lock.__aenter__()
                        # Otherwise, acquire synchronously and return self
                        if hasattr(self._lock, "acquire"):
                            loop = asyncio.get_event_loop()
                            await loop.run_in_executor(None, self._lock.acquire)
                            return self
                        return self
                    async def __aexit__(self, exc_type, exc, tb):
                        if self._lock is None:
                            return False
                        # Delegate if async exit exists
                        if hasattr(self._lock, "__aexit__") and asyncio.iscoroutinefunction(self._lock.__aexit__):
                            return await self._lock.__aexit__(exc_type, exc, tb)
                        # Otherwise, release synchronously if possible
                        if hasattr(self._lock, "release"):
                            self._lock.release()
                        return False
                return _AwaitableLockWrapper(lock)
            if lock:
                return _wrap_lock_if_needed(lock)
        except Exception:
            logger.debug("get_lock() failed; falling back to .lock", exc_info=True)
    lock = getattr(shared_state, "lock", None)
    return _wrap_lock_if_needed(lock) if lock else _NoopAsyncLock()

def _wrap_lock_if_needed(lock):
    """
    Normalize lock into an async-compatible context manager:
    - If it's already async-context compatible, return as is.
    - If it's a synchronous threading lock (Lock/RLock), wrap it.
    - Otherwise, return a no-op async lock.
    """
    # If it already supports async context management (e.g., asyncio.Lock)
    if hasattr(lock, "__aenter__") and hasattr(lock, "__aexit__"):
        return lock

    # Detect low-level thread locks (Lock/RLock) via their concrete classes
    try:
        if _THREAD_LOCK_TYPES and isinstance(lock, _THREAD_LOCK_TYPES):
            class AsyncThreadLock:
                def __init__(self, lock):
                    self._lock = lock
                async def __aenter__(self):
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(None, self._lock.acquire)
                    return self
                async def __aexit__(self, exc_type, exc, tb):
                    try:
                        self._lock.release()
                    except Exception:
                        pass
                    return False
            return AsyncThreadLock(lock)
    except Exception:
        # Fall through to best-effort detection below
        pass

    # Best-effort detection: has blocking acquire/release but not async methods
    if hasattr(lock, "acquire") and hasattr(lock, "release") and not hasattr(lock, "__aenter__"):
        class AsyncCompatLock:
            def __init__(self, lock):
                self._lock = lock
            async def __aenter__(self):
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, self._lock.acquire)
                return self
            async def __aexit__(self, exc_type, exc, tb):
                try:
                    self._lock.release()
                except Exception:
                    pass
                return False
        return AsyncCompatLock(lock)

    # Otherwise, fallback to a no-op async lock
    return _NoopAsyncLock()

def _schedule(coro_or_value: Any) -> None:
    """
    Schedules a coroutine to run as a task in the event loop without blocking the caller.
    If the input is not a coroutine, it does nothing.
    """
    if asyncio.iscoroutine(coro_or_value):
        try:
            # Get the running event loop and create a task for the coroutine
            asyncio.get_running_loop().create_task(coro_or_value)
        except RuntimeError:
            # If no event loop is running (e.g., in a script without explicit loop setup),
            # run the coroutine directly. This is a best-effort approach.
            asyncio.run(coro_or_value)

# ---------- public API ----------
async def inject_agent_signal(
    shared_state: Any,
    agent_name: str,
    symbol: str,
    signal: Dict[str, Any],
) -> None:
    """
    Injects an agent's signal into the shared state and the unified strategy bus.
    It expects the `signal` dictionary to contain 'action', 'confidence', 'reason',
    and an optional 'timestamp'.
    """
    # Ensure top-level dictionaries for agent signals and last signal timestamps exist
    if not hasattr(shared_state, "agent_signals"):
        shared_state.agent_signals = {}
    if not hasattr(shared_state, "last_signal_timestamp"):
        shared_state.last_signal_timestamp = {} # Changed type hint for per-symbol

    # Normalize symbol once
    sym = str(symbol or "").upper().replace("/", "")

    # Get the appropriate lock for thread-safe access to shared_state
    lock = _get_lock(shared_state)
    now_ts = time.time()

    async with lock:
        # Retrieve or initialize the dictionary for signals specific to this symbol
        per_symbol_signals = shared_state.agent_signals.get(sym, {})
        # Assign the new signal to the specific agent for this symbol
        per_symbol_signals[agent_name] = signal
        # Update the shared_state with the modified per-symbol signals
        shared_state.agent_signals[sym] = per_symbol_signals

        # Fix: Update last_signal_timestamp to per-symbol granularity
        per_symbol_ts = shared_state.last_signal_timestamp.get(sym, {})
        per_symbol_ts[agent_name] = float(signal.get("timestamp", now_ts))
        shared_state.last_signal_timestamp[sym] = per_symbol_ts

    # Normalize & validate action; clamp confidence
    raw_action = str(signal.get("action", "")).strip().upper()
    allowed_actions = {"BUY", "SELL", "HOLD"}
    action = raw_action if raw_action in allowed_actions else "HOLD"
    if action != raw_action:
        logger.warning(f"Signal action normalized to HOLD (was '{raw_action}') for {agent_name}:{sym}")

    try:
        conf = float(signal.get("confidence", 0.0))
    except Exception:
        conf = 0.0
    conf = max(0.0, min(1.0, conf))

    # Normalize timestamp once; reuse in both bus + event
    ts = signal.get("timestamp")
    if not isinstance(ts, (int, float)):
        ts = now_ts

    # Persist the signal to the unified strategy bus in a non-blocking manner.
    # This allows other components (e.g., MetaController) to react to signals.
    try:
        bus_signal = {
            "agent_name": agent_name,
            "action": action, # Use clamped action
            "confidence": conf, # Use clamped confidence
            "timestamp": ts,
            "reason": signal.get("reason", ""),
        }
        # Schedule the `add_strategy_signal` coroutine without blocking
        _schedule(shared_state.add_strategy_signal(sym, bus_signal))
    except Exception as e:
        logger.error(f"[{agent_name}] Failed to append to strategy bus for {sym}: {e}")

    # Add event emission: Emit an event after the signal is injected and processed (non-blocking)
    try:
        _schedule(shared_state.emit_event("AgentSignalInjected", {
            "agent": agent_name,
            "symbol": sym,
            "action": action,
            "confidence": conf,
            "reason": str(signal.get("reason", "")),
            "timestamp": ts,
        }))
    except Exception:
        pass

    # Log the successful injection for debugging
    logger.debug(
        f"[{agent_name}] Injected signal for {sym}: {action} "
        f"(confidence={conf:.2f})"
    )

async def get_latest_signal(shared_state: Any, agent_name: str, symbol: str) -> Optional[Dict[str, Any]]:
    """
    Retrieves the latest signal from a specific agent for a specific symbol.
    Returns None if no signal is found.
    """
    lock = _get_lock(shared_state)
    async with lock:
        symbol_map: Dict[str, Dict[str, Any]] = getattr(shared_state, "agent_signals", {})
        sym = str(symbol or "").upper().replace("/", "")
        return (symbol_map.get(sym) or {}).get(agent_name)

async def get_all_signals_for_symbol(shared_state: Any, symbol: str) -> Dict[str, Dict[str, Any]]:
    """
    Retrieves all latest signals for a given symbol from all agents.
    Returns a shallow copy of the dictionary mapping agent names to their signals.
    """
    lock = _get_lock(shared_state)
    async with lock:
        symbol_map: Dict[str, Dict[str, Any]] = getattr(shared_state, "agent_signals", {})
        sym = str(symbol or "").upper().replace("/", "")
        return dict(symbol_map.get(sym, {}))

async def preload_ohlcv(
    shared_state: Any,
    config: Any,
    exchange_client: Any,
    symbols: List[str],
    interval: str = "5m",
    limit: int = 150,
) -> None:
    """
    Concurrently preloads OHLCV (Open-High-Low-Close-Volume) data for a list of symbols
    into the SharedState. It uses a semaphore to bound concurrency, marks symbols as loaded,
    and updates the latest price based on the last candle's closing price.
    """
    if not symbols:
        logger.info("📦 Preloading OHLCV skipped (no symbols).")
        return

    # Determine maximum concurrent prefetching based on config or default to 20
    max_conc = int(
        getattr(getattr(config, "market_data", {}), "max_prefetch_concurrency", 0)
        or getattr(getattr(config, "execution", {}), "max_concurrency", 0)
        or getattr(config, "MAX_CONCURRENT_PREFETCH", 0)
        or 20
    )
    sem = asyncio.Semaphore(max_conc) # Semaphore to limit concurrent API calls

    async def _fetch_one(sym: str) -> None:
        """Helper async function to fetch OHLCV for a single symbol."""
        async with sem: # Acquire semaphore before fetching
            try:
                # Attempt to get OHLCV data, preferring 'timeframe' argument if supported,
                # otherwise falling back to 'interval'.
                try:
                    ohlcv = await exchange_client.get_ohlcv(sym, timeframe=interval, limit=limit)
                except TypeError:
                    ohlcv = await exchange_client.get_ohlcv(symbol=sym, interval=interval, limit=limit)

                # Ensure ohlcv is a list, defaulting to empty if not
                if not isinstance(ohlcv, list):
                    ohlcv = []

                # Update shared state with OHLCV data and mark the symbol as loaded
                await shared_state.set_ohlcv_data(sym, interval, ohlcv)
                await shared_state.mark_symbol_loaded(sym, interval)

                # If OHLCV data exists, update the latest price in shared state
                if ohlcv:
                    last_candle = ohlcv[-1] # Get the most recent candle
                    close_val: Optional[float] = None
                    if isinstance(last_candle, dict):
                        close_val = last_candle.get("close") # For dict-like candles
                    elif isinstance(last_candle, (list, tuple)) and len(last_candle) > 4:
                        close_val = last_candle[4] # For list/tuple-like candles (index 4 is close)

                    if close_val is not None:
                        try:
                            # Schedule non-blocking update of the latest price
                            _schedule(shared_state.update_latest_price(sym, float(close_val)))
                        except Exception:
                            # Log any errors during price update
                            logger.debug("Failed to update latest price.", exc_info=True)

                logger.info(f"✅ Loaded {len(ohlcv)} OHLCV for {sym}@{interval}")

                # Emit MarketDataUpdate event after successful data load and price update
                # Don’t block preload concurrency on event I/O
                _schedule(shared_state.emit_event("MarketDataUpdate", {
                    "symbol": sym,
                    "interval": interval,
                    "ohlcv_len": len(ohlcv),
                    "timestamp": time.time(),
                }))

            except Exception as e:
                logger.error(f"❌ Failed to fetch OHLCV for {sym}@{interval}: {e}")
                try:
                    # In case of error, set empty OHLCV and mark as loaded to prevent retries
                    await shared_state.set_ohlcv_data(sym, interval, [])
                    await shared_state.mark_symbol_loaded(sym, interval)
                except Exception:
                    logger.debug("Failed to record empty OHLCV after error.", exc_info=True)

    logger.info(f"📦 Preloading OHLCV data for {len(symbols)} symbols (max_concurrency={max_conc})...")
    # Concurrently run _fetch_one for all symbols
    await asyncio.gather(*(_fetch_one(s) for s in symbols))
    logger.info("🎯 OHLCV preload complete.")

async def inject_signals_batch(shared_state: Any, signals: List[Dict[str, Any]]) -> None:
    """
    Injects a batch of agent signals into the shared state.
    Each signal in the list is processed by the inject_agent_signal function.
    Includes a lightweight type guard to skip malformed entries.
    """
    for sig in signals:
        agent_name = sig.get("agent_name")
        symbol = sig.get("symbol")

        # Lightweight type guard: skip if agent_name or symbol is missing
        if not agent_name or not symbol:
            logger.warning(f"Skipping malformed signal in batch due to missing agent_name or symbol: {sig}")
            continue

        await inject_agent_signal(
            shared_state,
            agent_name,
            symbol,
            sig,
        )
