"""
core/market_data_websocket.py

WebSocket-based market data feed for real-time price updates via Binance Streams API.

Component: MarketDataWebSocket
Phase: P9 (runtime optimization)

Architecture:
- WebSocket connection to wss://stream.binance.com:9443/ws/!miniTicker@arr
- Receives all symbol prices in single stream
- Updates SharedState.latest_prices in real-time
- Falls back to REST if WebSocket unavailable
- Lightweight, maintains connection with ping/pong
"""

import asyncio
import json
import time
import math
from typing import Any, Dict, List, Optional, Set, Callable, Union
import logging

try:
    import aiohttp
except ImportError:
    aiohttp = None  # type: ignore


# For type hints compatibility
if hasattr(aiohttp, "ClientWebSocketResponse"):
    _ClientWebSocketResponse = aiohttp.ClientWebSocketResponse
    _ClientSession = aiohttp.ClientSession
else:
    _ClientWebSocketResponse = Any  # type: ignore
    _ClientSession = Any  # type: ignore


class MarketDataWebSocket:
    """
    Real-time price feed via WebSocket (Binance Streams API).
    
    Uses !miniTicker@arr to stream all symbol prices in a single connection.
    This is 100x more efficient than individual REST calls.
    
    Spec: wss://stream.binance.com:9443/ws/!miniTicker@arr
    
    Each tick:
    {
        "e": "24hrMiniTicker",
        "E": 1645123456789,  # event time
        "s": "BTCUSDT",      # symbol
        "c": "50000.50",     # close price (string)
        "o": "49500.00",     # open
        "h": "51000.00",     # high
        "l": "48000.00",     # low
        "v": "1234.50",      # volume
        "q": "61234567.89"   # quote asset volume
    }
    """
    
    STREAM_URL = "wss://stream.binance.com:9443/ws/!miniTicker@arr"
    TESTNET_STREAM_URL = "wss://stream.testnet.binance.vision/ws/!miniTicker@arr"
    
    def __init__(
        self,
        shared_state: Any,
        *,
        config: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None,
        is_testnet: bool = False,
        ping_interval: float = 30.0,
        max_reconnect_attempts: int = 10,
        reconnect_backoff_sec: float = 2.0,
        max_reconnect_backoff: float = 60.0,
        health_cadence_sec: float = 10.0,
    ) -> None:
        """
        Initialize WebSocket market data feed.
        
        Args:
            shared_state: SharedState instance for price updates
            config: Config dict/object
            logger: Logger instance
            is_testnet: Use testnet endpoint
            ping_interval: WebSocket ping interval (seconds)
            max_reconnect_attempts: Max reconnection attempts before stopping
            reconnect_backoff_sec: Initial backoff for reconnection
            max_reconnect_backoff: Maximum backoff time
            health_cadence_sec: Health report interval
        """
        self.shared_state = shared_state
        self.config = config or {}
        self._logger = logger or logging.getLogger("MarketDataWebSocket")
        if self._logger.level == logging.NOTSET:
            self._logger.setLevel(logging.INFO)
        
        # Config helpers
        def _cfg(key, default=None):
            if isinstance(self.config, dict):
                return self.config.get(key, default)
            return getattr(self.config, key, default)
        
        self.is_testnet = is_testnet or bool(_cfg("IS_TESTNET", False))
        self.stream_url = (
            self.TESTNET_STREAM_URL if self.is_testnet 
            else _cfg("WS_STREAM_URL", self.STREAM_URL)
        )
        self.ping_interval = float(_cfg("WS_PING_INTERVAL", ping_interval))
        self.max_reconnect_attempts = int(_cfg("WS_MAX_RECONNECT", max_reconnect_attempts))
        self.reconnect_backoff_sec = float(_cfg("WS_RECONNECT_BACKOFF", reconnect_backoff_sec))
        self.max_reconnect_backoff = float(_cfg("WS_MAX_BACKOFF", max_reconnect_backoff))
        self.health_cadence_sec = float(_cfg("WS_HEALTH_CADENCE", health_cadence_sec))
        
        # State
        self._stop = asyncio.Event()
        self._ws: Optional[Any] = None
        self._session: Optional[Any] = None
        self._reconnect_count = 0
        self._backoff = self.reconnect_backoff_sec
        self._price_updates_count = 0
        self._last_tick_ts = 0.0
        self._last_health_ts = 0.0
        self._symbols_seen: Set[str] = set()
        self._connection_established_ts: Optional[float] = None
        
        # Component key for health reporting
        try:
            from core.shared_state import Component
            self._component_key = Component.MARKET_DATA_WEBSOCKET
        except Exception:
            self._component_key = "MarketDataWebSocket"
    
    async def start(self) -> None:
        """
        Start WebSocket connection and price update loop.
        Runs until stop() is called.
        """
        if aiohttp is None:
            self._logger.error("[MDW] aiohttp not available; WebSocket disabled")
            return
        
        self._logger.info(f"[MDW] starting WebSocket feed (url={self.stream_url})")
        
        try:
            async with aiohttp.ClientSession() as session:
                self._session = session
                await self._connect_and_run()
        except Exception as e:
            self._logger.error(f"[MDW] fatal error: {e}", exc_info=True)
        finally:
            self._session = None
            self._ws = None
            self._logger.info("[MDW] WebSocket feed stopped")
    
    async def stop(self) -> None:
        """Signal stop to connection loop."""
        self._stop.set()
        if self._ws and not self._ws.closed:
            try:
                await self._ws.close()
            except Exception:
                pass
    
    async def _connect_and_run(self) -> None:
        """Main connection loop with reconnection logic."""
        while not self._stop.is_set():
            try:
                await self._connect()
                self._reconnect_count = 0
                self._backoff = self.reconnect_backoff_sec
                await self._receive_loop()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                self._logger.warning(
                    f"[MDW] connection lost: {e}; reconnect_count={self._reconnect_count}"
                )
                
                if self._reconnect_count >= self.max_reconnect_attempts:
                    self._logger.error(
                        f"[MDW] max reconnection attempts ({self._max_reconnect_attempts}) exceeded; stopping"
                    )
                    await self._set_health(
                        "ERROR",
                        "ws:max_reconnect_exceeded",
                        metrics={"reconnect_count": self._reconnect_count}
                    )
                    break
                
                # Exponential backoff with jitter
                self._reconnect_count += 1
                wait_time = min(self._backoff, self.max_reconnect_backoff)
                jitter = wait_time * 0.1  # 10% jitter
                actual_wait = wait_time + (jitter * (2 * (time.time() % 1) - 1))
                
                self._logger.info(
                    f"[MDW] reconnecting in {actual_wait:.1f}s (attempt {self._reconnect_count})"
                )
                self._backoff *= 1.5  # Exponential backoff
                
                try:
                    await asyncio.sleep(actual_wait)
                except asyncio.CancelledError:
                    raise
    
    async def _connect(self) -> None:
        """Establish WebSocket connection."""
        if not self._session:
            raise RuntimeError("Session not initialized")
        
        self._logger.info(f"[MDW] connecting to {self.stream_url}")
        self._ws = await self._session.ws_connect(
            self.stream_url,
            heartbeat=self.ping_interval,
            autoclose=False,
        )
        self._connection_established_ts = time.time()
        self._logger.info("[MDW] WebSocket connected")
        await self._set_health("OK", "ws:connected")
    
    async def _receive_loop(self) -> None:
        """Receive and process price ticks from WebSocket."""
        if not self._ws:
            raise RuntimeError("WebSocket not connected")
        
        health_task = asyncio.create_task(self._health_loop())
        
        try:
            async for msg in self._ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    try:
                        await self._handle_message(msg.data)
                    except Exception as e:
                        self._logger.warning(f"[MDW] message handling error: {e}")
                
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    raise RuntimeError(f"WebSocket error: {self._ws.exception()}")
                
                elif msg.type == aiohttp.WSMsgType.CLOSED:
                    self._logger.info("[MDW] WebSocket closed by server")
                    break
        finally:
            health_task.cancel()
            try:
                await health_task
            except asyncio.CancelledError:
                pass
    
    async def _handle_message(self, data: str) -> None:
        """
        Parse miniTicker array and update prices.
        
        Data is array of {e, E, s, c, o, h, l, v, q, ...}
        """
        try:
            ticks = json.loads(data)
        except json.JSONDecodeError as e:
            self._logger.warning(f"[MDW] JSON decode error: {e}")
            return
        
        if not isinstance(ticks, list):
            return
        
        now = time.time()
        self._last_tick_ts = now
        
        for tick in ticks:
            if not isinstance(tick, dict):
                continue
            
            try:
                symbol = str(tick.get("s", "")).upper()
                price_str = str(tick.get("c", ""))  # close price
                
                if not symbol or not price_str:
                    continue
                
                price = float(price_str)
                
                if not math.isfinite(price) or price <= 0:
                    continue
                
                # Track symbols we've seen
                if symbol not in self._symbols_seen:
                    self._symbols_seen.add(symbol)
                    self._logger.debug(f"[MDW] discovered symbol: {symbol}")
                
                # Inject price into SharedState
                await self._inject_price(symbol, price)
                self._price_updates_count += 1
            
            except Exception as e:
                self._logger.debug(f"[MDW] tick parsing error: {e}")
    
    async def _inject_price(self, symbol: str, price: float) -> None:
        """Update SharedState with price."""
        try:
            # Try primary method
            if hasattr(self.shared_state, "update_latest_price"):
                await self._maybe_await(
                    self.shared_state.update_latest_price(symbol, price)
                )
            elif hasattr(self.shared_state, "update_last_price"):
                await self._maybe_await(
                    self.shared_state.update_last_price(symbol, price)
                )
            else:
                # Fallback to direct attribute access
                if hasattr(self.shared_state, "latest_prices"):
                    self.shared_state.latest_prices[symbol] = price
        except Exception as e:
            self._logger.debug(f"[MDW] failed to inject price for {symbol}: {e}")
    
    @staticmethod
    async def _maybe_await(val):
        """Await if coroutine, else return."""
        if asyncio.iscoroutine(val):
            return await val
        return val
    
    async def _health_loop(self) -> None:
        """Periodic health reporting."""
        while True:
            try:
                await asyncio.sleep(self.health_cadence_sec)
                
                now = time.time()
                if now - self._last_health_ts < self.health_cadence_sec:
                    continue
                
                self._last_health_ts = now
                
                # Check connection health
                if not self._ws or self._ws.closed:
                    await self._set_health(
                        "WARN",
                        "ws:disconnected",
                        metrics={"reconnect_count": self._reconnect_count}
                    )
                else:
                    uptime = (
                        (now - self._connection_established_ts)
                        if self._connection_established_ts
                        else 0
                    )
                    await self._set_health(
                        "OK",
                        "ws:streaming",
                        metrics={
                            "price_updates": self._price_updates_count,
                            "symbols_seen": len(self._symbols_seen),
                            "uptime_sec": uptime,
                            "last_tick_ago_sec": now - self._last_tick_ts,
                        }
                    )
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.debug(f"[MDW] health loop error: {e}")
    
    async def _set_health(
        self,
        status: str,
        msg: str,
        metrics: Optional[Dict[str, Any]] = None
    ) -> None:
        """Report health status."""
        try:
            if hasattr(self.shared_state, "set_component_health"):
                await self._maybe_await(
                    self.shared_state.set_component_health(
                        self._component_key,
                        status,
                        msg,
                        metrics=metrics or {}
                    )
                )
        except Exception:
            pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Return statistics."""
        now = time.time()
        return {
            "url": self.stream_url,
            "connected": bool(self._ws and not self._ws.closed),
            "price_updates": self._price_updates_count,
            "symbols_seen": len(self._symbols_seen),
            "reconnect_count": self._reconnect_count,
            "uptime_sec": (
                (now - self._connection_established_ts)
                if self._connection_established_ts
                else 0
            ),
            "last_tick_ago_sec": now - self._last_tick_ts,
        }
