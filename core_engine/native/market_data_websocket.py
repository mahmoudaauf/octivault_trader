"""
Native L2: WebSocket Market Data (Zero API Rate Limits)

Replaces REST polling with Binance WebSocket streams for:
  • Real-time price updates (@ticker)
  • Real-time kline updates (@kline_1m)
  • Background polling (no rate limits, no blocking)

Architecture:
  • Uses python-binance BinanceSocketManager
  • Multiplex socket for 50+ symbols (1024 stream limit per connection)
  • Non-blocking message handlers
  • Exponential backoff reconnection
  • Fallback to REST only for bootstrap
  • @bookTicker (optional, opt-in) runs on its own isolated connection so an
    overflow/disconnect there can never take down @ticker/@kline delivery —
    see _run_connection().

Design choices:
  * WebSocket primary data source (live prices/klines)
  * REST fallback only for missing data or bootstrap
  * No blocking operations in message handlers
  * Buffer updates, don't compute immediately
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)


class NativeMarketDataWebSocket:
    """
    WebSocket market data feed (price + klines via Binance streams).

    Replaces NativeMarketData REST polling with live WebSocket updates.
    """

    def __init__(
        self,
        exchange_client: Any,
        shared_state: Any,
        symbols: Optional[list[str]] = None,
        timeframes: Optional[list[str]] = None,
        *,
        max_reconnect_attempts: int = 10,
        initial_backoff_sec: float = 1.0,
        max_backoff_sec: float = 30.0,
        message_timeout_sec: float = 60.0,
        queue_max_size: Optional[int] = None,
    ):
        """
        Initialize WebSocket market data.

        Args:
            exchange_client: NativeExchangeClient with binance_client
            shared_state: NativeSharedState for updates
            symbols: Symbols to subscribe (default: from wallet scan)
            timeframes: Kline intervals (default: ["1m"])
            max_reconnect_attempts: Max reconnects before giving up
            initial_backoff_sec: Initial backoff for exponential retry
            max_backoff_sec: Max backoff interval
            message_timeout_sec: Timeout for receiving messages
            queue_max_size: python-binance BinanceSocketManager's internal
                message queue size (default: library default of 100 is too
                small for the startup contention window — see WS_QUEUE_MAX_SIZE).
        """
        self._exchange_client = exchange_client
        self._shared_state = shared_state
        self._symbols = list(symbols or [])
        self._timeframes = list(timeframes or ["1m"])

        self._max_reconnect_attempts = max_reconnect_attempts
        self._initial_backoff_sec = initial_backoff_sec
        self._max_backoff_sec = max_backoff_sec
        self._message_timeout_sec = message_timeout_sec
        self._queue_max_size = int(
            queue_max_size
            if queue_max_size is not None
            else (os.getenv("WS_QUEUE_MAX_SIZE", "2000") or 2000)
        )

        # State
        self._running = False
        self._stopped = asyncio.Event()
        self._ws_task: Optional[asyncio.Task] = None  # @ticker + @kline (critical)
        self._bookticker_task: Optional[asyncio.Task] = None  # @bookTicker (isolated, non-critical)
        self._last_msg_ts = time.time()

    async def start(self) -> None:
        """Start WebSocket market data feed."""
        if self._running:
            logger.warning("WebSocket market data already running")
            return

        logger.info(
            f"📡 Starting WebSocket market data ({len(self._symbols)} symbols, {len(self._timeframes)} timeframes)"
        )
        self._running = True
        self._stopped.clear()
        self._ws_task = asyncio.create_task(
            self._run_connection(self._build_primary_streams, "primary", critical=True)
        )
        self._bookticker_task = asyncio.create_task(
            self._run_connection(self._build_bookticker_streams, "bookTicker", critical=False)
        )

    async def stop(self) -> None:
        """Stop WebSocket market data feed."""
        logger.info("⏹️ Stopping WebSocket market data")
        self._running = False
        self._stopped.set()

        for task in (self._ws_task, self._bookticker_task):
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

    async def subscribe(self, symbols: list[str]) -> None:
        """Add symbols to subscription (additive). Prefer set_symbols() for the trading
        universe — additive growth raises the stream/message count unbounded and risks
        the python-binance queue overflow."""
        new_symbols = {s.upper() for s in symbols} - set(self._symbols)
        if new_symbols:
            self._symbols.extend(new_symbols)
            logger.info(f"📡 Added {len(new_symbols)} symbols (total: {len(self._symbols)})")
            # Force reconnect to pick up new symbols
            for task in (self._ws_task, self._bookticker_task):
                if task:
                    task.cancel()

    async def set_symbols(self, symbols: list[str], *, max_symbols: int = 12) -> None:
        """REPLACE the subscription set (order-preserving, capped) and reconnect if changed.

        Unlike subscribe() this replaces rather than appends, so the stream count stays
        bounded (each symbol = 2 streams: @ticker + @kline) and the message queue can't
        overflow from unbounded growth. Caller passes symbols in priority order — held
        positions first (real-time prices for fast SL/TP), then regime anchors (BTC/ETH),
        then the rotator's trading universe.
        """
        seen: list[str] = []
        for s in symbols:
            su = str(s or "").upper()
            if su and su.endswith("USDT") and su not in seen:
                seen.append(su)
            if len(seen) >= max(1, int(max_symbols)):
                break
        if not seen or sorted(seen) == sorted(self._symbols):
            return  # no change → don't churn the connection
        self._symbols = seen
        logger.info("📡 WS universe updated (%d): %s", len(seen), seen)
        # Cleanly RESTART both connections with the new symbol set. Cancelling alone just
        # terminates _run_connection — and on its way out the loop sets self._running=False
        # (primary only) — so we must cancel+await the old tasks, then re-enable _running
        # before spawning fresh tasks (which reset each connection's own reconnect counter
        # and re-subscribe to _symbols). REST polling covers the brief gap.
        for task in (self._ws_task, self._bookticker_task):
            if task is not None and not task.done():
                self._stopped.set()  # so the old loop logs a clean "stopped", not an ERROR
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
        self._running = True
        self._stopped.clear()
        self._ws_task = asyncio.create_task(
            self._run_connection(self._build_primary_streams, "primary", critical=True)
        )
        self._bookticker_task = asyncio.create_task(
            self._run_connection(self._build_bookticker_streams, "bookTicker", critical=False)
        )

    async def _run_connection(self, streams_fn, conn_label: str, *, critical: bool) -> None:
        """Run one independently-managed WebSocket connection (its own AsyncClient,
        BinanceSocketManager, and reconnect/backoff state) for whatever streams_fn()
        returns.

        Isolated from any other connection this instance runs — an overflow or
        disconnect here cannot take down another connection's stream delivery,
        since each has its own multiplexed socket and message queue. This is why
        @bookTicker (a high-frequency, opt-in, observe-only stream) runs on its
        own connection separate from @ticker/@kline (which real trading decisions
        depend on): a bookTicker-side overflow should never cost the trading loop
        its price/candle feed.

        critical=True means this connection permanently failing (reconnect
        attempts exhausted) marks the whole feed as not-running; critical=False
        (bookTicker) just leaves that one connection stopped.
        """
        reconnect_count = 0
        current_backoff = self._initial_backoff_sec

        while self._running and reconnect_count < self._max_reconnect_attempts:
            try:
                streams = streams_fn()
                if not streams:
                    logger.debug("[%s] No streams to subscribe, waiting...", conn_label)
                    await asyncio.sleep(5)
                    continue

                logger.info(
                    "🔌 [%s] Connecting WebSocket (attempt %d/%d)",
                    conn_label,
                    reconnect_count + 1,
                    self._max_reconnect_attempts,
                )

                # Create Binance AsyncClient for WebSocket
                try:
                    from binance import AsyncClient, BinanceSocketManager

                    # Get credentials from exchange_client
                    api_key = getattr(self._exchange_client, "api_key", None)
                    api_secret = getattr(self._exchange_client, "api_secret", None)

                    if not api_key or not api_secret:
                        logger.error("[%s] No API credentials available for WebSocket", conn_label)
                        reconnect_count += 1
                        await asyncio.sleep(current_backoff)
                        continue

                    # Create AsyncClient (public key, secret for authentication)
                    binance_client = AsyncClient(api_key, api_secret)

                except ImportError as e:
                    logger.error("[%s] Failed to import Binance client: %s", conn_label, e)
                    reconnect_count += 1
                    await asyncio.sleep(current_backoff)
                    continue

                logger.info(
                    "📡 [%s] Subscribing to %d streams: %s...", conn_label, len(streams), streams[:5]
                )

                # Connect WebSocket
                try:
                    sm = BinanceSocketManager(binance_client, max_queue_size=self._queue_max_size)
                    got_message = False

                    try:
                        async with sm.multiplex_socket(streams) as stream:
                            self._last_msg_ts = time.time()

                            logger.info("✅ [%s] WebSocket connected, receiving messages...", conn_label)

                            # Message loop
                            while self._running:
                                try:
                                    msg = await asyncio.wait_for(
                                        stream.recv(),
                                        timeout=self._message_timeout_sec,
                                    )
                                    self._last_msg_ts = time.time()

                                    if not got_message:
                                        # Only declare this connection healthy -- and reset
                                        # the reconnect/backoff counters -- once a message
                                        # actually arrives. Resetting immediately on connect
                                        # let a connect -> instant-overflow -> disconnect
                                        # loop report "attempt 1/N" forever, never honoring
                                        # max_reconnect_attempts (a real storm observed live
                                        # 2026-07-14).
                                        got_message = True
                                        reconnect_count = 0
                                        current_backoff = self._initial_backoff_sec

                                    if msg:
                                        await self._handle_message(msg)

                                except asyncio.TimeoutError:
                                    logger.warning(
                                        "⚠️ [%s] No message for %ss, reconnecting...",
                                        conn_label,
                                        self._message_timeout_sec,
                                    )
                                    break
                                except asyncio.CancelledError:
                                    raise
                                except Exception as e:
                                    logger.warning("⚠️ [%s] WebSocket error: %s", conn_label, e)
                                    break
                    finally:
                        # Always close the Binance client
                        await binance_client.close_connection()

                    if not got_message:
                        # This connection attempt never became healthy -- the exact
                        # 2026-07-14 storm mechanism: connect -> instant overflow
                        # (BinanceWebsocketQueueOverflow -> ReadLoopClosed) -> the
                        # message loop's own generic "except Exception: break" catches
                        # this without ever reaching the outer except block below, so
                        # it previously never counted toward reconnect_count and never
                        # applied backoff -- an unbounded, zero-backoff reconnect storm.
                        # Count and back off here explicitly instead.
                        reconnect_count += 1
                        if reconnect_count < self._max_reconnect_attempts:
                            wait_time = min(current_backoff, self._max_backoff_sec)
                            logger.info("⏳ [%s] Reconnecting in %ss...", conn_label, wait_time)
                            await asyncio.sleep(wait_time)
                            current_backoff *= 2

                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    logger.warning("❌ [%s] WebSocket connection failed: %s", conn_label, e)
                    reconnect_count += 1
                    if reconnect_count < self._max_reconnect_attempts:
                        wait_time = min(current_backoff, self._max_backoff_sec)
                        logger.info("⏳ [%s] Reconnecting in %ss...", conn_label, wait_time)
                        await asyncio.sleep(wait_time)
                        current_backoff *= 2

            except asyncio.CancelledError:
                logger.info("[%s] WebSocket loop cancelled", conn_label)
                break
            except Exception as e:
                logger.error("[%s] WebSocket loop error: %s", conn_label, e)
                reconnect_count += 1

        if self._stopped.is_set() or not self._running:
            logger.info("[%s] WebSocket market data stopped", conn_label)
        else:
            logger.error("❌ [%s] WebSocket disconnected (max reconnects reached)", conn_label)
            if critical:
                self._running = False

    def _bookticker_enabled(self) -> bool:
        return str(os.getenv("WS_ENABLE_BOOKTICKER", "false")).lower() in ("1", "true", "yes", "on")

    def _build_primary_streams(self) -> list[str]:
        """@ticker + @kline for every symbol -- the data live trading decisions
        depend on. Always subscribed; never shares a connection with @bookTicker."""
        streams = []
        for symbol in self._symbols:
            symbol_lower = symbol.lower()
            streams.append(f"{symbol_lower}@ticker")
            for tf in self._timeframes:
                streams.append(f"{symbol_lower}@kline_{tf}")
        return streams

    def _build_bookticker_streams(self) -> list[str]:
        """@bookTicker for every symbol -- the highest-frequency stream (fires on
        every best bid/ask change). Opt-in via WS_ENABLE_BOOKTICKER (checked fresh
        on every reconnect attempt, so toggling it takes effect without a restart).
        Runs on its own isolated connection (see _run_connection) so a queue
        overflow here can't take down @ticker/@kline delivery."""
        if not self._bookticker_enabled():
            return []
        return [f"{symbol.lower()}@bookTicker" for symbol in self._symbols]

    async def _handle_message(self, msg: dict[str, Any]) -> None:
        """Handle WebSocket message (non-blocking)."""
        try:
            payload = msg.get("data") if isinstance(msg.get("data"), dict) else msg
            event_type = payload.get("e")

            # @bookTicker carries no "e" field — detect by best bid/ask keys.
            if event_type is None and "b" in payload and "a" in payload:
                symbol = payload.get("s", "").upper()
                try:
                    bid = float(payload.get("b", 0) or 0)
                    bid_qty = float(payload.get("B", 0) or 0)
                    ask = float(payload.get("a", 0) or 0)
                    ask_qty = float(payload.get("A", 0) or 0)
                except (TypeError, ValueError):
                    return
                if symbol and bid > 0 and ask > 0:
                    self._shared_state.update_book(symbol, bid, bid_qty, ask, ask_qty)
                return

            if event_type == "24hrTicker":
                # Price update
                symbol = payload.get("s", "").upper()
                price = float(payload.get("c", 0))
                if symbol and price > 0:
                    self._shared_state.price_cache[symbol] = price
                    self._shared_state.prices[symbol] = price
                    if hasattr(self._shared_state, "_last_tick_timestamps"):
                        self._shared_state._last_tick_timestamps[symbol] = time.time()
                    if hasattr(self._shared_state, "market_data_ready"):
                        self._shared_state.market_data_ready = True

            elif event_type == "kline":
                # Kline update
                data = payload.get("k", {})
                symbol = data.get("s", "").upper()
                interval = data.get("i", "1m")
                is_closed = data.get("x", False)

                if is_closed and symbol and interval:
                    # Store completed kline
                    ohlcv = {
                        "time": int(data.get("t", 0)) / 1000,
                        "open": float(data.get("o", 0)),
                        "high": float(data.get("h", 0)),
                        "low": float(data.get("l", 0)),
                        "close": float(data.get("c", 0)),
                        "volume": float(data.get("v", 0)),
                        "taker_buy_volume": float(data.get("V", 0)),  # taker buy base asset volume
                        "num_trades": float(data.get("n", 0)),  # number of trades
                    }
                    _key = (symbol, interval)
                    _buf = self._shared_state.market_data.get(_key) or []
                    _buf = list(_buf)
                    _buf.append(ohlcv)
                    if len(_buf) > 3500:
                        _buf = _buf[-3500:]
                    self._shared_state.market_data[_key] = _buf
                    if hasattr(self._shared_state, "market_data_ready"):
                        self._shared_state.market_data_ready = True

        except Exception as e:
            logger.debug(f"Error handling message: {e}")

    async def prefetch_klines_history(self, limit: int = 3000, timeframe: str = "1m") -> None:
        """Pre-populate kline buffers via REST on startup so ML models can train immediately.
        Without this, WS-only accumulation takes ~50 minutes to reach 3,000 rows."""
        import asyncio as _asyncio
        symbols = list(self._symbols)
        logger.info("📥 Pre-fetching %d klines for %d symbols via REST...", limit, len(symbols))
        fetched, failed = 0, 0

        def _parse_row(row):
            if isinstance(row, (list, tuple)) and len(row) >= 6:
                return {
                    "time": float(row[0]) / 1000,
                    "open": float(row[1]),
                    "high": float(row[2]),
                    "low": float(row[3]),
                    "close": float(row[4]),
                    "volume": float(row[5]),
                    "taker_buy_volume": float(row[9]) if len(row) > 9 else 0.0,
                    "num_trades": float(row[8]) if len(row) > 8 else 0.0,
                }
            return row if isinstance(row, dict) else None

        async def _fetch_one(sym: str) -> None:
            nonlocal fetched, failed
            try:
                ec = self._exchange_client
                all_rows = []
                end_time = None
                per_call = 1000  # Binance hard limit per request
                calls_needed = (limit + per_call - 1) // per_call

                for _ in range(calls_needed):
                    params = {"symbol": sym, "interval": timeframe, "limit": per_call}
                    if end_time is not None:
                        params["endTime"] = end_time
                    raw = await ec._request("GET", ec.EP_KLINES, params=params)
                    if not raw:
                        break
                    parsed = [r for row in raw if (r := _parse_row(row)) is not None]
                    all_rows = parsed + all_rows  # prepend older pages
                    if len(raw) < per_call:
                        break
                    end_time = int(raw[0][0]) - 1  # step back before earliest row
                    await _asyncio.sleep(0.1)

                if all_rows:
                    self._shared_state.market_data[(sym, timeframe)] = all_rows[-limit:]
                    fetched += 1
            except Exception as _e:
                failed += 1
                logger.debug("prefetch_klines failed for %s: %s", sym, _e)

        # Fetch in small batches to stay within rate limits
        for i in range(0, len(symbols), 3):
            batch = symbols[i:i + 3]
            await _asyncio.gather(*[_fetch_one(s) for s in batch])
            await _asyncio.sleep(0.3)

        if hasattr(self._shared_state, "market_data_ready"):
            self._shared_state.market_data_ready = True
        logger.info("✅ Kline pre-fetch complete: %d fetched, %d failed", fetched, failed)
