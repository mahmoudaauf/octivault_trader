# WebSocket Market Data - Implementation Cookbook

## Quick Start (5 Minutes)

### 1. Start WebSocket Feed

```python
from core.ws_market_data import WebSocketMarketData
from core.shared_state import SharedState
from core.exchange_client import ExchangeClient

# Initialize
shared_state = SharedState(config=your_config)
exchange_client = ExchangeClient(config=your_config)

ws = WebSocketMarketData(
    shared_state=shared_state,
    exchange_client=exchange_client,
    config=your_config,
)

# Start
await ws.start()

# Subscribe to symbols
await ws.subscribe(["BTCUSDT", "ETHUSDT", "BNBUSDT"])

# Now SharedState gets live updates!
# shared_state.prices["BTCUSDT"] → live price
# shared_state.market_data[("BTCUSDT", "5m")] → live candles
```

### 2. Check Readiness

```python
# Wait for enough candles to be collected
await asyncio.sleep(30)  # Let it warm up

# Check if ready
if shared_state.market_data_ready_event.is_set():
    print("✅ MarketDataReady! Can start trading")
else:
    print("⏳ Still warming up...")
```

### 3. Stop When Done

```python
await ws.stop()
```

---

## Detailed Integration Examples

### Example 1: Add to AppContext

**File: `core/app_context.py`**

```python
class AppContext:
    
    def __init__(self, ...):
        # ... existing init ...
        self.ws_market_data: Optional[Any] = None
    
    async def _setup_market_data(self):
        """Setup WebSocket + MarketDataFeed (fallback)."""
        from core.ws_market_data import WebSocketMarketData
        
        try:
            self.ws_market_data = WebSocketMarketData(
                shared_state=self.shared_state,
                exchange_client=self.exchange_client,
                config=self.config,
                ohlcv_timeframes=getattr(self.config, "SUPPORTED_TIMEFRAMES", ["1m", "5m"]),
                readiness_min_bars=getattr(self.config, "WS_READINESS_MIN_BARS", 50),
                logger=self.logger,
            )
            self.logger.info("[AppCtx] WebSocketMarketData initialized")
        except Exception as e:
            self.logger.error(f"[AppCtx] WebSocketMarketData init failed: {e}")
            self.ws_market_data = None
    
    async def run_phase_4(self):
        """Phase 4: Accept symbols and subscribe WebSocket."""
        
        # Get accepted symbols
        symbols = self.shared_state.accepted_symbols or []
        
        # Subscribe WebSocket
        if self.ws_market_data and symbols:
            await self.ws_market_data.subscribe(symbols)
            self.logger.info(f"[Phase4] WebSocket subscribed to {len(symbols)} symbols")
    
    async def run_with_phases(self):
        """Main execution loop with phases."""
        
        # Setup phase
        await self._setup_market_data()
        
        # Start WebSocket (background task)
        if self.ws_market_data:
            ws_task = asyncio.create_task(self.ws_market_data.start())
        
        # Phase 4
        await self.run_phase_4()
        
        # Wait for MarketDataReady
        md_ready_event = getattr(self.shared_state, "market_data_ready_event", None)
        if md_ready_event:
            try:
                await asyncio.wait_for(md_ready_event.wait(), timeout=60.0)
                self.logger.info("[Main] MarketDataReady event set")
            except asyncio.TimeoutError:
                self.logger.warning("[Main] MarketDataReady timeout, proceeding anyway")
        
        # Continue with main loop...
        await self._main_loop()
    
    async def shutdown(self):
        """Graceful shutdown."""
        if self.ws_market_data:
            await self.ws_market_data.stop()
            self.logger.info("[Shutdown] WebSocket stopped")
```

---

### Example 2: Custom Health Reporter

```python
class CustomHealthReporter:
    """Monitor WebSocket health and report metrics."""
    
    def __init__(self, shared_state, ws_market_data, logger):
        self.shared_state = shared_state
        self.ws = ws_market_data
        self.logger = logger
    
    async def monitor_ws_health(self, interval: float = 10.0):
        """Periodic health check."""
        while True:
            try:
                await asyncio.sleep(interval)
                
                # Gather metrics
                price_count = len(self.ws._price_buffer)
                market_data_count = len(self.shared_state.market_data)
                symbols = len(self.ws._symbols_subscribed)
                is_ready = self.shared_state.market_data_ready_event.is_set()
                
                # Report
                self.logger.info(
                    f"[WS:Health] prices={price_count} "
                    f"market_data_pairs={market_data_count} "
                    f"symbols={symbols} ready={is_ready} "
                    f"reconnects={self.ws._reconnect_count}"
                )
                
                # Check for stale data
                if price_count < symbols * 0.8:  # <80% coverage
                    self.logger.warning(f"[WS:Health] Low price coverage: {price_count}/{symbols}")
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"[WS:Health] Monitor error: {e}")
```

---

### Example 3: Fallback Strategy

```python
class MarketDataFallback:
    """Use WebSocket primary, REST fallback."""
    
    def __init__(self, ws_market_data, market_data_feed, logger):
        self.ws = ws_market_data
        self.mdf = market_data_feed  # REST-based MarketDataFeed
        self.logger = logger
    
    async def start_with_fallback(self):
        """Start both feeds for resilience."""
        
        # Start WebSocket
        try:
            await self.ws.start()
            self.logger.info("[Fallback] WebSocket started")
        except Exception as e:
            self.logger.error(f"[Fallback] WebSocket failed: {e}")
        
        # Start MarketDataFeed as fallback
        try:
            await asyncio.create_task(self.mdf.run())
            self.logger.info("[Fallback] MarketDataFeed (REST) started")
        except Exception as e:
            self.logger.error(f"[Fallback] MarketDataFeed failed: {e}")
    
    async def check_data_freshness(self) -> Dict[str, Any]:
        """Check which feed is fresher."""
        
        ws_fresh = (time.time() - self.ws._last_ws_msg_ts) < 5.0
        mdf_fresh = (time.time() - self.mdf._last_poll_ts) < 10.0
        
        return {
            "ws_fresh": ws_fresh,
            "mdf_fresh": mdf_fresh,
            "primary": "WS" if ws_fresh else "MDF",
        }
```

---

### Example 4: Custom Message Processor

```python
class CustomMessageProcessor:
    """Hook into WebSocket messages for custom logic."""
    
    def __init__(self, ws_market_data, shared_state):
        self.ws = ws_market_data
        self.shared_state = shared_state
        self._original_handle_message = self.ws._handle_message
        # Patch method
        self.ws._handle_message = self._patched_handle_message
    
    async def _patched_handle_message(self, msg: Dict[str, Any]):
        """Process message with custom logic."""
        
        # Call original handler
        await self._original_handle_message(msg)
        
        # Custom processing
        if msg.get("e") == "24hrTicker":
            await self._on_price_update(msg)
        elif msg.get("e") == "kline":
            await self._on_kline_update(msg)
    
    async def _on_price_update(self, msg: Dict[str, Any]):
        """Custom price logic."""
        symbol = msg.get("s")
        price = float(msg.get("c", 0))
        
        # E.g., emit custom event
        if hasattr(self.shared_state, "price_updated_event"):
            self.shared_state.price_updated_event.emit(symbol, price)
    
    async def _on_kline_update(self, msg: Dict[str, Any]):
        """Custom kline logic."""
        symbol = msg.get("s")
        kline = msg.get("k", {})
        tf = kline.get("i")
        
        # E.g., compute indicators
        if hasattr(self.shared_state, "compute_sma"):
            await self.shared_state.compute_sma(symbol, tf, 50)
```

---

### Example 5: Symbol Discovery Integration

```python
class DynamicSymbolSubscription:
    """Subscribe to new symbols as they're discovered."""
    
    def __init__(self, ws_market_data, shared_state):
        self.ws = ws_market_data
        self.shared_state = shared_state
        self._subscribed = set()
    
    async def monitor_for_new_symbols(self, check_interval: float = 30.0):
        """Watch for new accepted symbols."""
        
        while True:
            try:
                await asyncio.sleep(check_interval)
                
                # Get current symbols
                current = set(self.shared_state.accepted_symbols or [])
                new_symbols = current - self._subscribed
                
                if new_symbols:
                    # Subscribe to new ones
                    await self.ws.subscribe(list(new_symbols))
                    self._subscribed.update(new_symbols)
                    # logger.info(f"Subscribed to new symbols: {new_symbols}")
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                # logger.error(f"Symbol monitoring error: {e}")
                pass
```

---

### Example 6: Price Cache Wrapper

```python
class CachedPriceGetter:
    """Wrap shared_state.get_price with local cache."""
    
    def __init__(self, shared_state, cache_ttl_sec: float = 1.0):
        self.shared_state = shared_state
        self.cache_ttl_sec = cache_ttl_sec
        self._price_cache: Dict[str, Tuple[float, float]] = {}  # symbol -> (price, ts)
    
    async def get_price(self, symbol: str) -> Optional[float]:
        """Get price with caching."""
        
        # Check cache
        if symbol in self._price_cache:
            cached_price, cached_ts = self._price_cache[symbol]
            if (time.time() - cached_ts) < self.cache_ttl_sec:
                return cached_price
        
        # Get from shared_state (updated by WebSocket)
        try:
            price = self.shared_state.get_price(symbol)
            if price:
                self._price_cache[symbol] = (price, time.time())
                return price
        except Exception:
            pass
        
        return None
    
    def clear_cache(self):
        """Manual cache clear."""
        self._price_cache.clear()
```

---

### Example 7: Monitoring Dashboard Data

```python
class WebSocketMetricsCollector:
    """Collect metrics for dashboard."""
    
    def __init__(self, ws_market_data, shared_state):
        self.ws = ws_market_data
        self.shared_state = shared_state
        self.metrics = {
            "prices_received": 0,
            "klines_received": 0,
            "last_price_ts": None,
            "last_kline_ts": None,
            "uptime_sec": 0.0,
            "reconnects": 0,
        }
        self._start_ts = time.time()
    
    async def collect_metrics(self) -> Dict[str, Any]:
        """Gather current metrics."""
        
        now = time.time()
        
        return {
            "uptime_sec": now - self._start_ts,
            "subscribed_symbols": len(self.ws._symbols_subscribed),
            "price_buffer_size": len(self.ws._price_buffer),
            "market_data_pairs": len(self.shared_state.market_data),
            "is_ready": self.shared_state.market_data_ready_event.is_set(),
            "last_message_age_sec": now - self.ws._last_ws_msg_ts,
            "reconnect_count": self.ws._reconnect_count,
            "current_backoff_sec": self.ws._current_backoff,
        }
    
    async def export_prometheus_metrics(self) -> str:
        """Export as Prometheus-style metrics."""
        
        m = await self.collect_metrics()
        
        return f"""
# HELP octivault_ws_uptime_seconds WebSocket uptime
# TYPE octivault_ws_uptime_seconds gauge
octivault_ws_uptime_seconds {m['uptime_sec']}

# HELP octivault_ws_subscribed_symbols Number of subscribed symbols
# TYPE octivault_ws_subscribed_symbols gauge
octivault_ws_subscribed_symbols {m['subscribed_symbols']}

# HELP octivault_ws_market_data_ready Market data ready flag
# TYPE octivault_ws_market_data_ready gauge
octivault_ws_market_data_ready {1 if m['is_ready'] else 0}

# HELP octivault_ws_last_message_age_seconds Age of last message
# TYPE octivault_ws_last_message_age_seconds gauge
octivault_ws_last_message_age_seconds {m['last_message_age_sec']}

# HELP octivault_ws_reconnect_count Total reconnects
# TYPE octivault_ws_reconnect_count counter
octivault_ws_reconnect_count {m['reconnect_count']}
"""
```

---

## Error Handling Patterns

### Pattern 1: Graceful Degradation

```python
async def get_market_data_with_fallback(symbol: str, tf: str):
    """Get data, fallback to REST if WebSocket fails."""
    
    # Try WebSocket data first
    ws_data = shared_state.market_data.get((symbol, tf))
    if ws_data and len(ws_data) >= 50:
        return ws_data
    
    # Fallback to REST
    try:
        rest_data = await exchange_client.get_ohlcv(symbol, tf, limit=300)
        return rest_data
    except Exception as e:
        logger.error(f"Both WS and REST failed: {e}")
        return []
```

### Pattern 2: Health Check & Recovery

```python
async def health_check_ws_recovery(ws_market_data, interval: float = 30.0):
    """Check health and trigger recovery if needed."""
    
    while True:
        try:
            await asyncio.sleep(interval)
            
            # Check if stale
            age = time.time() - ws_market_data._last_ws_msg_ts
            
            if age > 120:  # >2 min stale
                logger.warning(f"WebSocket stale ({age}s), triggering recovery")
                
                # Stop and restart
                await ws_market_data.stop()
                await asyncio.sleep(5)
                await ws_market_data.start()
        
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Health check failed: {e}")
```

---

## Performance Tuning

### Reduce Memory Usage

```python
# Limit candle history
max_candles = 500  # Keep only last 500

# In ws_market_data._update_shared_state_kline()
if len(lst) > max_candles:
    lst.pop(0)  # Remove oldest
```

### Reduce CPU Usage

```python
# Throttle price updates
last_price_update = {}

async def throttled_price_update(symbol: str, price: float, min_interval: float = 0.1):
    """Update price only if enough time has passed."""
    
    now = time.time()
    last_ts = last_price_update.get(symbol, 0)
    
    if (now - last_ts) >= min_interval:
        await _update_shared_state_price(symbol, price)
        last_price_update[symbol] = now
```

### Batch Kline Updates

```python
# Batch multiple klines instead of updating immediately
kline_batch = []
batch_size = 10

async def batch_kline_updates(symbol: str, tf: str, candle: List[float]):
    """Batch updates for efficiency."""
    
    kline_batch.append((symbol, tf, candle))
    
    if len(kline_batch) >= batch_size:
        # Flush batch
        for sym, timeframe, candle_data in kline_batch:
            await _update_shared_state_kline(sym, timeframe, *candle_data)
        kline_batch.clear()
```

---

## Testing Utilities

### Mock WebSocket for Testing

```python
class MockWebSocketMarketData:
    """Mock WS for unit testing."""
    
    def __init__(self):
        self._price_updates = {}
        self._kline_updates = {}
    
    async def start(self):
        pass
    
    async def stop(self):
        pass
    
    async def subscribe(self, symbols):
        pass
    
    async def inject_price(self, symbol: str, price: float):
        """Inject synthetic price for testing."""
        self._price_updates[symbol] = price
    
    async def inject_kline(self, symbol: str, tf: str, candle: List[float]):
        """Inject synthetic kline for testing."""
        key = (symbol, tf)
        if key not in self._kline_updates:
            self._kline_updates[key] = []
        self._kline_updates[key].append(candle)
```

---

## Summary

✅ WebSocket implementation complete  
✅ Non-blocking architecture  
✅ Automatic reconnection  
✅ Health monitoring  
✅ Ready for production integration  

Next steps:
1. Add to `core/app_context.py`
2. Subscribe symbols in Phase 4
3. Monitor `MarketDataReady` event
4. Deploy and monitor logs
