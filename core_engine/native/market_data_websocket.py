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

Design choices:
  * WebSocket primary data source (live prices/klines)
  * REST fallback only for missing data or bootstrap
  * No blocking operations in message handlers
  * Buffer updates, don't compute immediately
"""

from __future__ import annotations

import asyncio
import logging
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
        """
        self._exchange_client = exchange_client
        self._shared_state = shared_state
        self._symbols = list(symbols or [])
        self._timeframes = list(timeframes or ["1m"])

        self._max_reconnect_attempts = max_reconnect_attempts
        self._initial_backoff_sec = initial_backoff_sec
        self._max_backoff_sec = max_backoff_sec
        self._message_timeout_sec = message_timeout_sec

        # State
        self._running = False
        self._stopped = asyncio.Event()
        self._ws_task: Optional[asyncio.Task] = None
        self._reconnect_count = 0
        self._current_backoff = initial_backoff_sec
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
        self._ws_task = asyncio.create_task(self._ws_loop())

    async def stop(self) -> None:
        """Stop WebSocket market data feed."""
        logger.info("⏹️ Stopping WebSocket market data")
        self._running = False
        self._stopped.set()

        if self._ws_task:
            self._ws_task.cancel()
            try:
                await self._ws_task
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
            if self._ws_task:
                self._ws_task.cancel()

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
        # Cleanly RESTART the feed task with the new symbol set. Cancelling alone just
        # terminates _ws_loop — and on its way out the loop sets self._running=False — so
        # we must cancel+await the old task, then re-enable _running before spawning a
        # fresh task (which resets the reconnect counter and re-subscribes to _symbols).
        # REST polling covers the brief gap.
        old = self._ws_task
        if old is not None and not old.done():
            self._stopped.set()  # so the old loop logs a clean "stopped", not an ERROR
            old.cancel()
            try:
                await old
            except (asyncio.CancelledError, Exception):
                pass
        self._running = True
        self._stopped.clear()
        self._current_backoff = self._initial_backoff_sec
        self._ws_task = asyncio.create_task(self._ws_loop())

    async def _ws_loop(self) -> None:
        """Main WebSocket connection loop with reconnection."""
        reconnect_count = 0

        while self._running and reconnect_count < self._max_reconnect_attempts:
            try:
                if not self._symbols:
                    logger.debug("No symbols to subscribe, waiting...")
                    await asyncio.sleep(5)
                    continue

                logger.info(
                    f"🔌 Connecting WebSocket (attempt {reconnect_count + 1}/{self._max_reconnect_attempts})"
                )

                # Create Binance AsyncClient for WebSocket
                try:
                    from binance import AsyncClient, BinanceSocketManager

                    # Get credentials from exchange_client
                    api_key = getattr(self._exchange_client, "api_key", None)
                    api_secret = getattr(self._exchange_client, "api_secret", None)

                    if not api_key or not api_secret:
                        logger.error("No API credentials available for WebSocket")
                        reconnect_count += 1
                        await asyncio.sleep(self._current_backoff)
                        continue

                    # Create AsyncClient (public key, secret for authentication)
                    binance_client = AsyncClient(api_key, api_secret)

                except ImportError as e:
                    logger.error(f"Failed to import Binance client: {e}")
                    reconnect_count += 1
                    await asyncio.sleep(self._current_backoff)
                    continue

                # Build streams list
                streams = self._build_streams()
                logger.info(f"📡 Subscribing to {len(streams)} streams: {streams[:5]}...")

                # Connect WebSocket
                try:
                    sm = BinanceSocketManager(binance_client)

                    try:
                        async with sm.multiplex_socket(streams) as stream:
                            self._last_msg_ts = time.time()
                            reconnect_count = 0
                            self._current_backoff = self._initial_backoff_sec

                            logger.info("✅ WebSocket connected, receiving messages...")

                            # Message loop
                            while self._running:
                                try:
                                    msg = await asyncio.wait_for(
                                        stream.recv(),
                                        timeout=self._message_timeout_sec,
                                    )
                                    self._last_msg_ts = time.time()

                                    if msg:
                                        await self._handle_message(msg)

                                except asyncio.TimeoutError:
                                    logger.warning(
                                        f"⚠️ No message for {self._message_timeout_sec}s, reconnecting..."
                                    )
                                    break
                                except asyncio.CancelledError:
                                    raise
                                except Exception as e:
                                    logger.warning(f"⚠️ WebSocket error: {e}")
                                    break
                    finally:
                        # Always close the Binance client
                        await binance_client.close_connection()

                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    logger.warning(f"❌ WebSocket connection failed: {e}")
                    reconnect_count += 1
                    if reconnect_count < self._max_reconnect_attempts:
                        wait_time = min(self._current_backoff, self._max_backoff_sec)
                        logger.info(f"⏳ Reconnecting in {wait_time}s...")
                        await asyncio.sleep(wait_time)
                        self._current_backoff *= 2

            except asyncio.CancelledError:
                logger.info("WebSocket loop cancelled")
                break
            except Exception as e:
                logger.error(f"WebSocket loop error: {e}")
                reconnect_count += 1

        if self._stopped.is_set() or not self._running:
            logger.info("WebSocket market data stopped")
        else:
            logger.error("❌ WebSocket disconnected (max reconnects reached)")
        self._running = False

    def _build_streams(self) -> list[str]:
        """Build Binance stream names.

        @bookTicker is the highest-frequency stream (fires on every best bid/ask
        change). With many symbols it overruns python-binance's internal 100-message
        queue (BinanceWebsocketQueueOverflow), which storms reconnects and can wedge
        the trading loop. It only feeds an OBSERVE-ONLY orderbook-imbalance check in
        regime_gate, so it's off by default to keep the message queue bounded
        (~2 streams/symbol). Re-enable with WS_ENABLE_BOOKTICKER=true if needed.
        """
        import os
        enable_bookticker = str(
            os.getenv("WS_ENABLE_BOOKTICKER", "false")
        ).lower() in ("1", "true", "yes", "on")
        streams = []
        for symbol in self._symbols:
            symbol_lower = symbol.lower()
            # Price stream
            streams.append(f"{symbol_lower}@ticker")
            # Top-of-book stream — high-frequency firehose; opt-in only.
            if enable_bookticker:
                streams.append(f"{symbol_lower}@bookTicker")
            # Kline streams
            for tf in self._timeframes:
                streams.append(f"{symbol_lower}@kline_{tf}")
        return streams

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
