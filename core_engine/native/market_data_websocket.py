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
        """Add symbols to subscription."""
        new_symbols = {s.upper() for s in symbols} - set(self._symbols)
        if new_symbols:
            self._symbols.extend(new_symbols)
            logger.info(f"📡 Added {len(new_symbols)} symbols (total: {len(self._symbols)})")
            # Force reconnect to pick up new symbols
            if self._ws_task:
                self._ws_task.cancel()

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
        """Build Binance stream names."""
        streams = []
        for symbol in self._symbols:
            symbol_lower = symbol.lower()
            # Price stream
            streams.append(f"{symbol_lower}@ticker")
            # Kline streams
            for tf in self._timeframes:
                streams.append(f"{symbol_lower}@kline_{tf}")
        return streams

    async def _handle_message(self, msg: dict[str, Any]) -> None:
        """Handle WebSocket message (non-blocking)."""
        try:
            event_type = msg.get("e")

            if event_type == "24hrTicker":
                # Price update
                symbol = msg.get("s", "").upper()
                price = float(msg.get("c", 0))
                if symbol and price > 0:
                    self._shared_state.price_cache[symbol] = price
                    self._shared_state.prices[symbol] = price
                    if hasattr(self._shared_state, "market_data_ready"):
                        self._shared_state.market_data_ready = True

            elif event_type == "kline":
                # Kline update
                data = msg.get("k", {})
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
                    }
                    self._shared_state.market_data[(symbol, interval)] = [ohlcv]
                    if hasattr(self._shared_state, "market_data_ready"):
                        self._shared_state.market_data_ready = True

        except Exception as e:
            logger.debug(f"Error handling message: {e}")
