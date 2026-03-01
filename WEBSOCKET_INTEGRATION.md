# WebSocket Market Data Integration Guide

## Overview

This guide walks through integrating the new `WebSocketMarketData` (Phase 1) with your existing system while keeping `MarketDataFeed` as fallback.

---

## Architecture: Dual-Mode Operation

```
┌─────────────────────────────────────────────────────────────────┐
│                      AppContext/Phase 3                         │
└────────────────┬────────────────────────────┬────────────────────┘
                 │                            │
          ┌──────▼──────────┐        ┌────────▼──────────┐
          │ WebSocketData   │        │ MarketDataFeed    │
          │ (Primary: Live) │        │ (Fallback: REST)  │
          └──────┬──────────┘        └────────┬──────────┘
                 │                            │
                 │ Async updates              │ Polling
                 │ (latency: 50-150ms)        │ (latency: 1-3s)
                 │                            │
                 └────────┬───────────────────┘
                          │
                    ┌─────▼──────────┐
                    │  SharedState   │
                    │ .prices        │
                    │ .market_data   │
                    │ .markets_ready │
                    └────────────────┘
```

---

## Integration Steps

### Step 1: Add WebSocketMarketData to AppContext

**File: `core/app_context.py`**

In the `_setup_market_data()` method, initialize both:

```python
async def _setup_market_data(self):
    """Bootstrap market data: WebSocket primary, MarketDataFeed fallback."""
    
    # Import new module
    from core.ws_market_data import WebSocketMarketData
    
    # 1. Initialize WebSocketMarketData (primary)
    self.ws_market_data = WebSocketMarketData(
        shared_state=self.shared_state,
        exchange_client=self.exchange_client,
        config=self.config,
        ohlcv_timeframes=getattr(self.config, "SUPPORTED_TIMEFRAMES", ["1m", "5m"]),
        logger=self.logger,
    )
    
    # 2. Keep MarketDataFeed for fallback
    from core.market_data_feed import MarketDataFeed
    self.market_data_feed = MarketDataFeed(
        shared_state=self.shared_state,
        exchange_client=self.exchange_client,
        config=self.config,
    )
```

### Step 2: Subscribe WebSocketMarketData to Symbols

**File: `core/app_context.py`**

In `_phase_4_accept_symbols()` or equivalent:

```python
async def _subscribe_to_market_data(self, symbols: List[str]):
    """Subscribe WebSocket to accepted symbols."""
    
    if hasattr(self, "ws_market_data") and self.ws_market_data:
        await self.ws_market_data.subscribe(symbols)
        self.logger.info(f"[WS:Subscribe] Subscribed to {len(symbols)} symbols")
```

### Step 3: Start WebSocket Feed in Startup

**File: `core/app_context.py`**

In your phase startup (e.g., `async def run_with_phases()`):

```python
# Start WebSocket (non-blocking)
if hasattr(self, "ws_market_data") and self.ws_market_data:
    try:
        await self.ws_market_data.start()
        self.logger.info("[WS] WebSocket market data started")
    except Exception as e:
        self.logger.error(f"[WS] Failed to start: {e}")

# Optionally keep MarketDataFeed as fallback
if hasattr(self, "market_data_feed") and self.market_data_feed:
    try:
        await self.market_data_feed.run()
        self.logger.info("[MDF] MarketDataFeed (fallback) started")
    except Exception as e:
        self.logger.error(f"[MDF] Failed to start: {e}")
```

### Step 4: Graceful Shutdown

**File: `core/app_context.py`**

In your shutdown handler:

```python
async def shutdown(self):
    """Clean shutdown of market data feeds."""
    
    # Stop WebSocket first
    if hasattr(self, "ws_market_data") and self.ws_market_data:
        await self.ws_market_data.stop()
    
    # Stop MarketDataFeed fallback
    if hasattr(self, "market_data_feed") and self.market_data_feed:
        await self.market_data_feed.stop()
```

---

## Configuration

Add to your config file or environment:

```python
# config.py or config.json

SUPPORTED_TIMEFRAMES = ["1m", "5m"]           # WebSocket timeframes to stream
OHLCV_LIMIT = 300                              # Max candles per timeframe
REQUIRE_MARKET_DATA_READY = True               # Gate trading on MarketDataReady

# WebSocket-specific
WS_MAX_RECONNECT_ATTEMPTS = 10
WS_INITIAL_BACKOFF_SEC = 1.0
WS_MAX_BACKOFF_SEC = 30.0
WS_HEARTBEAT_INTERVAL_SEC = 30.0
WS_READINESS_MIN_BARS = 50
WS_HEALTH_CADENCE_SEC = 10.0
```

---

## Monitoring & Observability

### Log Messages to Watch

```
[WS:Start]        WebSocket market data started
[WS:Connect]      Connecting to Binance WebSocket
[WS:Connected]    WebSocket connected, processing messages
[WS:Subscribe]    Added symbols, total=X
[WS:Ready]        MarketDataReady SET
[WS:Health]       symbols=X prices=Y market_data_pairs=Z ready=True
[WS:Timeout]      No message for 60s, reconnecting
[WS:ConnErr]      Connection failed, retrying...
[WS:Failed]       Max reconnect attempts exceeded
```

### Health Status

Check `shared_state.update_component_health()` reports:

- **OK**: Streaming actively, message rate healthy
- **WARN**: Message lag detected, but recovering
- **ERROR**: Max reconnects exceeded, fallback to MarketDataFeed

### Readiness Event

Monitor `shared_state.market_data_ready_event`:

```python
# In your strategy/agent code
is_ready = self.shared_state.market_data_ready_event.is_set()
# True = WebSocket has populated all symbols with min candles
# False = Still warming up or disconnected
```

---

## Metrics & Performance

### Expected Improvements

| Metric | Before (Polling) | After (WebSocket) |
|--------|-----------------|------------------|
| Price latency | 1-3 sec | 50-150 ms |
| Rate limits | Constant hits | ~0 |
| Scalability | 2-3 symbols | 50+ symbols |
| MarketDataReady | Flaky | Stable |
| CPU usage | ~5% (polling) | ~2% (event-driven) |

### Throughput

- **1m candles**: ~50-100 updates/min per symbol
- **5m candles**: ~10-20 updates/min per symbol
- **Prices**: 100-1000 updates/sec per symbol (depends on volume)

---

## Fallback & Resilience

### When WebSocket Fails

1. **Disconnection**: Exponential backoff retry (1s → 30s max)
2. **Max retries exceeded**: Falls back to MarketDataFeed (REST polling)
3. **Message timeout (60s)**: Automatic reconnect

### Dual-Mode Safety

Both feeds can run simultaneously:

```python
# WebSocket for low-latency price updates
await ws_market_data.start()

# MarketDataFeed for backup OHLCV polling
await market_data_feed.run()

# MetaController/Agents use whichever is fresher
price = shared_state.get_price(symbol)  # Auto uses latest
ohlcv = shared_state.market_data[(symbol, "5m")]  # Auto uses latest
```

---

## Testing

### Unit Test Example

```python
import pytest
from core.ws_market_data import WebSocketMarketData

@pytest.mark.asyncio
async def test_ws_market_data_init():
    """Test WebSocketMarketData initialization."""
    ws = WebSocketMarketData(
        shared_state=mock_shared_state,
        exchange_client=mock_exchange_client,
    )
    
    assert ws._running == False
    assert ws._symbols_subscribed == set()

@pytest.mark.asyncio
async def test_ws_subscribe():
    """Test symbol subscription."""
    ws = WebSocketMarketData(...)
    
    await ws.subscribe(["BTCUSDT", "ETHUSDT"])
    
    assert "BTCUSDT" in ws._symbols_subscribed
    assert "ETHUSDT" in ws._symbols_subscribed
```

### Integration Test

```python
@pytest.mark.asyncio
async def test_ws_market_data_integration():
    """Test WebSocket with real SharedState."""
    shared_state = SharedState(config=test_config)
    exchange_client = ExchangeClient(config=test_config)
    
    ws = WebSocketMarketData(
        shared_state=shared_state,
        exchange_client=exchange_client,
        readiness_min_bars=10,  # Lower for testing
    )
    
    await ws.start()
    await ws.subscribe(["BTCUSDT"])
    
    # Wait for MarketDataReady
    await asyncio.wait_for(
        asyncio.create_task(
            wait_for_ready(shared_state)
        ),
        timeout=30.0
    )
    
    assert shared_state.market_data_ready_event.is_set()
    assert ("BTCUSDT", "5m") in shared_state.market_data
    
    await ws.stop()
```

---

## Troubleshooting

### WebSocket Won't Connect

1. **Check Binance connectivity**:
   ```bash
   python3 -c "import asyncio; from binance import AsyncClient; print('OK')"
   ```

2. **Check API credentials** (if using authenticated endpoints):
   ```python
   api_key = config.BINANCE_API_KEY
   api_secret = config.BINANCE_API_SECRET
   assert api_key and api_secret, "Missing API credentials"
   ```

3. **Check symbols format**:
   ```python
   # Ensure uppercase with suffix
   symbols = ["BTCUSDT", "ETHUSDT"]  # ✅ Correct
   # NOT: ["BTC-USD", "ETH_USDT"]     # ❌ Wrong
   ```

### MarketDataReady Never Sets

1. **Check subscription**:
   ```python
   assert len(ws._symbols_subscribed) > 0
   ```

2. **Check min_bars threshold**:
   ```python
   # Lower it for testing
   ws.readiness_min_bars = 10
   ```

3. **Check stream building**:
   ```python
   streams = ws._build_stream_list()
   assert len(streams) > 0  # Should have @ticker + @kline streams
   ```

### High Latency Despite WebSocket

1. **Check message rate** in logs:
   ```
   [WS:Health] symbols=10 prices=50 market_data_pairs=20 ready=True
   ```

2. **Check event loop blocking**:
   - Ensure no heavy computation in `_handle_message()`
   - Use `await` for all I/O operations

3. **Check network latency**:
   ```bash
   ping stream.binance.com
   ```

---

## Next Steps (Phase 2 & 3)

Once WebSocket market data is stable:

1. **Phase 2**: Add User Data Stream for balance updates
   - `@account`: Balance, order updates in real-time
   - Replace `sync_balance()` with stream updates

2. **Phase 3**: REST-only for Order Placement
   - Keep `create_order()`, `cancel_order()` REST
   - Light fallback queries for reconciliation

---

## Files Modified/Created

| File | Change | Status |
|------|--------|--------|
| `core/ws_market_data.py` | **NEW** | ✅ Created |
| `core/app_context.py` | Integration points | ⏳ Pending |
| `core/config.py` | Config additions | ⏳ Pending |
| `core/market_data_feed.py` | Kept as fallback | ✅ No change |

---

## Summary

✅ **WebSocket market data** (Phase 1) is ready for integration  
✅ **Non-invasive**: Doesn't touch MetaController, ExecutionManager, etc.  
✅ **Resilient**: Automatic reconnection with exponential backoff  
✅ **Observable**: Health monitoring and readiness signaling  
✅ **Tested**: Import validation passed  

Next: Add to AppContext and subscribe to accepted symbols!
