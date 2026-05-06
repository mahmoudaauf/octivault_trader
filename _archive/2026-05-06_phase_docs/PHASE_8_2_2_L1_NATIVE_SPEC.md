# Phase 8.2.2: L1 Native Implementation Spec

**Layer:** L1 (Exchange Integration)  
**Scope:** ExchangeClient, BalanceSync, OrderExecution  
**Timeline:** 2-3 weeks  
**Target Completion:** 2026-06-10  
**Expected Cycle Time Improvement:** -20ms (280ms → 260ms)

---

## Current Legacy L1 Analysis

### Legacy Components

| Component | File | Lines | Role | Issues |
|-----------|------|-------|------|--------|
| ExchangeClient | `src/l1_exchange/exchange_client.py` | ~800 | Binance API wrapper, WebSocket management, authentication | High complexity: 3 fallback layers (WS API v3, listenKey, polling) |
| BalanceSync | `src/l2_marketdata/balance_cache_updater.py` | ~300 | Real-time balance polling | Redundant with ExchangeClient |
| OrderExecution | `src/l3_portfolio/execution_manager.py` | ~500 | Order placement/cancellation | Tightly coupled with legacy state |

**Total Legacy L1 Lines:** ~1,600

---

## Native L1 Target Architecture

```
NativeExchangeClient (simpler interface)
├─ REST API wrapper (ccxt-like)
│  ├─ get_balance() → {symbol: qty, ...}
│  ├─ get_ticker(symbol) → {price, bid, ask}
│  ├─ place_order(symbol, side, qty, price) → order_id
│  ├─ cancel_order(symbol, order_id) → bool
│  └─ get_orders(symbol) → [orders]
├─ Polling loop (30s interval, no WS)
│  └─ Background task updates balance cache
└─ Rate limiting (handles Binance 1200 req/min limit)

NativeBalanceSync (lightweight)
├─ Real-time cache (dict[symbol] = qty)
├─ Update callback when balance changes
└─ Graceful fallback to polling

NativeOrderExecution (stateless)
├─ Place orders (no validation, delegated to Binance)
├─ Track orders in memory
└─ Cancel orders
```

**Target Lines:**
- NativeExchangeClient: ~400 lines
- NativeBalanceSync: ~150 lines
- NativeOrderExecution: ~200 lines
- Total: ~750 lines (vs 1,600 legacy = 53% reduction)

---

## Implementation Strategy

### Phase 8.2.2a: NativeExchangeClient (Week 1)

**Goal:** REST API wrapper with no WebSocket complexity

```python
class NativeExchangeClient:
    """Simple Binance REST API wrapper"""
    
    def __init__(self, api_key: str, api_secret: str, testnet: bool = False):
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        self.base_url = "https://testnet.binance.vision" if testnet else "https://api.binance.com"
    
    async def get_balance(self) -> dict[str, float]:
        """Get account balance"""
        # GET /api/v3/account
        # Returns: {symbol: qty, ...}
        pass
    
    async def get_ticker(self, symbol: str) -> dict:
        """Get current price"""
        # GET /api/v3/ticker/24hr?symbol=ETHUSDT
        # Returns: {price, bid, ask, volume, ...}
        pass
    
    async def place_order(
        self,
        symbol: str,
        side: str,  # "BUY" or "SELL"
        qty: float,
        price: float,
    ) -> dict:
        """Place limit order"""
        # POST /api/v3/order
        # Returns: {orderId, symbol, side, qty, price, status, ...}
        pass
    
    async def cancel_order(self, symbol: str, order_id: int) -> bool:
        """Cancel order"""
        # DELETE /api/v3/order
        # Returns: {orderId, symbol, status, ...}
        pass
    
    async def get_orders(self, symbol: str, limit: int = 10) -> list[dict]:
        """Get recent orders"""
        # GET /api/v3/allOrders?symbol=ETHUSDT
        pass
```

**Key Design Decisions:**
- ✅ Simple, synchronous-looking REST API (no WebSocket complexity)
- ✅ Rate limiting handled internally (track request count)
- ✅ No reconnection logic (stateless, let bridge handle retries)
- ✅ All methods async for consistency

**Dependencies:**
- `aiohttp` (already in requirements)
- `hmac`, `hashlib` (for signing)

**Test Coverage:**
- Mock Binance API responses
- Rate limiting validation
- Error handling (401, 429, etc)
- Signature generation

---

### Phase 8.2.2b: NativeBalanceSync (Week 1-2)

**Goal:** Lightweight balance cache with async updates

```python
class NativeBalanceSync:
    """Real-time balance cache updater"""
    
    def __init__(self, exchange_client: NativeExchangeClient, poll_interval_sec: float = 5.0):
        self.exchange_client = exchange_client
        self.poll_interval_sec = poll_interval_sec
        self.balance_cache: dict[str, float] = {}
        self.last_update_ms: int = 0
        self._update_callback = None
    
    def on_balance_update(self, callback):
        """Register callback for balance updates"""
        self._update_callback = callback
    
    async def start_polling(self):
        """Start background polling task"""
        while True:
            try:
                await self.sync_balance()
                await asyncio.sleep(self.poll_interval_sec)
            except Exception as e:
                logger.warning(f"Balance sync failed: {e}")
                await asyncio.sleep(self.poll_interval_sec * 2)  # Backoff
    
    async def sync_balance(self):
        """Sync balance from Binance"""
        balance = await self.exchange_client.get_balance()
        
        if balance != self.balance_cache:
            self.balance_cache = balance
            self.last_update_ms = NativeTimeUtils.unix_now_ms()
            
            # Notify listeners
            if self._update_callback:
                await self._update_callback(balance)
    
    def get_balance(self, symbol: str = None) -> float | dict:
        """Get cached balance"""
        if symbol:
            return self.balance_cache.get(symbol, 0.0)
        return self.balance_cache.copy()
    
    @property
    def total_usdt(self) -> float:
        """Total USDT balance (assuming 'USDT' key)"""
        return self.balance_cache.get("USDT", 0.0)
```

**Key Features:**
- ✅ Callback-based updates (no polling in main loop)
- ✅ Graceful error handling (continues on failures)
- ✅ Configurable polling interval
- ✅ Simple cache (just a dict)

---

### Phase 8.2.2c: NativeOrderExecution (Week 2)

**Goal:** Stateless order placement/cancellation

```python
class NativeOrderExecution:
    """Order execution manager (no state validation)"""
    
    def __init__(self, exchange_client: NativeExchangeClient):
        self.exchange_client = exchange_client
        self.pending_orders: dict[str, dict] = {}  # order_id -> order_info
    
    async def place_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        price: float,
    ) -> tuple[bool, str]:
        """
        Place order
        
        Returns:
            (success: bool, order_id: str)
        """
        try:
            result = await self.exchange_client.place_order(symbol, side, qty, price)
            order_id = str(result["orderId"])
            
            # Track locally
            self.pending_orders[order_id] = {
                "symbol": symbol,
                "side": side,
                "qty": qty,
                "price": price,
                "status": result.get("status", "NEW"),
                "timestamp_ms": NativeTimeUtils.unix_now_ms(),
            }
            
            return True, order_id
        
        except Exception as e:
            logger.error(f"Order placement failed: {e}")
            return False, ""
    
    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        """Cancel order"""
        try:
            await self.exchange_client.cancel_order(symbol, int(order_id))
            self.pending_orders.pop(order_id, None)
            return True
        except Exception as e:
            logger.error(f"Order cancellation failed: {e}")
            return False
    
    async def sync_orders(self, symbol: str):
        """Sync pending orders with exchange"""
        try:
            orders = await self.exchange_client.get_orders(symbol)
            
            for order in orders:
                order_id = str(order["orderId"])
                if order_id in self.pending_orders:
                    self.pending_orders[order_id]["status"] = order["status"]
        
        except Exception as e:
            logger.warning(f"Order sync failed: {e}")
    
    def get_pending_orders(self) -> dict:
        """Get all pending orders"""
        return self.pending_orders.copy()
```

---

## Integration Points

### Update `core_engine/integration.py`

```python
async def create_app_context(production: bool = False, native_l0: bool = False, native_l1: bool = False):
    """Build app context with L0/L1 choice"""
    
    if not production:
        return {}
    
    if native_l0 and native_l1:
        # NEW: Use native L0 + L1
        from core_engine.native import NativeSharedState, NativeTimeUtils, ConfigLoader
        from core_engine.native_l1 import NativeExchangeClient, NativeBalanceSync, NativeOrderExecution
        
        config = ConfigLoader()
        exchange = NativeExchangeClient(
            api_key=os.getenv("BINANCE_API_KEY"),
            api_secret=os.getenv("BINANCE_API_SECRET"),
            testnet=config.get("API", "binance_testnet"),
        )
        
        app_ctx = {
            # L0 native
            "shared_state": NativeSharedState(),
            "time_utils": NativeTimeUtils,
            "config": config,
            
            # L1 native
            "exchange_client": exchange,
            "balance_sync": NativeBalanceSync(exchange),
            "order_execution": NativeOrderExecution(exchange),
        }
        return app_ctx
    
    elif native_l0:
        # Use native L0 + legacy L1-L8
        return await build_production_app_ctx(use_native_l0=True)
    
    else:
        # Full legacy bridge
        return await build_production_app_ctx()
```

### Update `main.py` CLI

```python
parser.add_argument("--native-l0", action="store_true", help="Use native L0")
parser.add_argument("--native-l1", action="store_true", help="Use native L1 (requires --native-l0)")
parser.add_argument("--native-l0-l1", action="store_true", help="Use native L0+L1")
```

---

## Testing Strategy

### Unit Tests (`tests/test_native_l1.py`)

```python
class TestNativeExchangeClient:
    @pytest.mark.asyncio
    async def test_get_balance_mock(self, mocker):
        """Test balance retrieval"""
        # Mock aiohttp response
        mocker.patch("aiohttp.ClientSession.get")
        
        client = NativeExchangeClient("key", "secret")
        balance = await client.get_balance()
        
        assert "USDT" in balance
        assert balance["USDT"] > 0
    
    @pytest.mark.asyncio
    async def test_place_order_mock(self, mocker):
        """Test order placement"""
        pass

class TestNativeBalanceSync:
    @pytest.mark.asyncio
    async def test_sync_balance(self, mocker):
        """Test balance sync"""
        pass

class TestNativeOrderExecution:
    @pytest.mark.asyncio
    async def test_place_order(self, mocker):
        """Test order execution"""
        pass
```

### Integration Tests

```bash
# Test 1: Legacy baseline (30s)
python3 main.py --mode=paper-trade --duration=30s --production
# Expected: nav=$86.99, cycle ~312ms

# Test 2: Native L0 + L1 (30s)
python3 main.py --mode=paper-trade --duration=30s --production --native-l0-l1
# Expected: nav=$86.99, cycle ~272ms (40ms faster)
```

---

## Success Criteria

| Criterion | Target | Notes |
|-----------|--------|-------|
| Code reduction | 50% | 1,600 → 750 lines |
| Cycle time | -20ms (280 → 260ms) | Cumulative with L0 (-40ms total) |
| Balance accuracy | ±0.01 USDT | Matches Binance exactly |
| Order execution | Success rate >99% | Only fails on Binance errors |
| Test coverage | 15+ tests | All L1 components |
| Production ready | ✅ | Passing equivalence test |

---

## Known Limitations (L1 Native)

1. **No WebSocket** — Uses polling only (simpler, more reliable)
2. **No user data stream** — Balance checked via REST API
3. **No advanced features** — No order update notifications (check on demand)
4. **Simple rate limiting** — Counts requests, backs off at 1200/min

These are acceptable because:
- Polling is sufficient for 30-second intervals (current target)
- WebSocket adds complexity without proportional benefit
- Can be added later in Phase 8.3 (optimization)

---

## Rollback Plan

If equivalence test fails:
```bash
# Rollback to legacy
git revert COMMIT_SHA_OF_L1_NATIVE
python3 main.py --production  # Uses L0 native + L1 legacy
```

Or disable L1 native:
```bash
python3 main.py --production --native-l0  # Skips L1 native
```

---

## Timeline

| Week | Tasks | Deliverables |
|------|-------|--------------|
| 1 | NativeExchangeClient + BalanceSync | 400 + 150 lines |
| 2 | NativeOrderExecution + tests | 200 lines + 15 tests |
| 2.5 | Integration + equivalence test | CLI flags, docs |
| 3 | Validation + optimization | Performance confirmed |

---

## Next Phase (8.2.3)

Once L1 complete:
- **L2 (Market Data)** — Native OHLCV cache, price feed
- **Expected gain:** -40ms more (260 → 220ms)
- **Timeline:** 2-3 weeks

---

**Status:** Ready to implement  
**Owner:** @mauf  
**Start Date:** 2026-05-07  
**Target Completion:** 2026-06-10  
**Est. Effort:** 80-120 hours
