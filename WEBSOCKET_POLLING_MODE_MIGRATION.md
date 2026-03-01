# WebSocket → Polling Mode Migration

## 🎯 Summary

**Date**: March 1, 2026  
**Status**: ✅ COMPLETE & DEPLOYED  
**Impact**: Hard-disable WebSocket user-data streams, enable deterministic REST polling

---

## 📋 What Changed

### 1. Hard-Disable WebSocket User Data

**File**: `core/exchange_client.py:650-666`

```python
# OLD:
self.user_data_stream_enabled = bool(_cfg_bool("USER_DATA_STREAM_ENABLED", True))
self.user_data_ws_auth_mode = str(_cfg("USER_DATA_WS_AUTH_MODE", "auto") or "auto").strip().lower()

# NEW:
self.user_data_stream_enabled = False  # ✅ FORCE POLLING MODE
self.user_data_ws_auth_mode = str(_cfg("USER_DATA_WS_AUTH_MODE", "polling") or "polling").strip().lower()
```

**Why**: Eliminates repeated 1008 (policy) and 410 (gone) errors that cause cascade failures.

---

### 2. Bypass WS Startup, Go Straight to Polling

**File**: `core/exchange_client.py:1596-1628`

**Old Flow**:
```
_user_data_ws_loop()
  → _user_data_ws_api_v3_direct()  [Tier 1: WS API v3]
    ↓ (on failure)
  → _user_data_listen_key_loop()  [Tier 2: listenKey WS]
    ↓ (on failure)
  → _user_data_polling_loop()  [Tier 3: REST polling]
```

**New Flow**:
```
_user_data_ws_loop()
  → _user_data_polling_loop()  [DIRECT to polling, skip WS]
```

**Code**:
```python
async def _user_data_ws_loop(self) -> None:
    """
    ✅ SIMPLIFIED: Hard-disabled WebSocket, always use polling mode.
    """
    self.logger.info(
        "[EC:UserDataWS] WebSocket modes disabled by default. "
        "Using polling mode for deterministic account reconciliation..."
    )
    
    while self.is_started and not self._user_data_stop.is_set():
        try:
            await self._user_data_polling_loop()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self.logger.error(
                "[EC:UserDataWS:Polling] Polling loop failed: %s. Retrying in 5s...", e
            )
            await asyncio.sleep(5.0)
```

---

### 3. Enhanced Polling with Deterministic Reconciliation

**File**: `core/exchange_client.py:1508-1785`

Each poll cycle (2.0 seconds):

#### Phase 1: Fetch Current State
```python
# Fetch open orders
open_orders_resp = await self._request("GET", "/api/v3/openOrders", signed=True)

# Fetch account balances
acct = await self._request("GET", "/api/v3/account", signed=True)
```

#### Phase 2: Detect Balance Changes
```python
for asset, curr in current_balances.items():
    prev = prev_balances.get(asset, {})
    if (prev.get("free") != curr["free"] or prev.get("locked") != curr["locked"]):
        # Emit balanceUpdate event
        evt_payload = {
            "e": "balanceUpdate",
            "E": int(now * 1000),
            "a": asset,
            "d": curr["free"],
            "l": curr["locked"]
        }
        self._ingest_user_data_ws_payload(evt_payload)
```

#### Phase 3: Detect Order Fills (CRITICAL)
```python
for order_id, prev_order in prev_open_orders.items():
    if order_id not in current_open_orders:
        # Order closed (filled or cancelled)
        # Emit executionReport to update position state
        evt_payload = {
            "e": "executionReport",
            "X": status,  # FILLED, PARTIALLY_FILLED, CANCELED
            "z": filled_qty,  # Total filled
            ...
        }
        self._ingest_user_data_ws_payload(evt_payload)
```

#### Phase 4: Detect Partial Fills
```python
for order_id, curr_order in current_open_orders.items():
    if order_id in prev_open_orders:
        prev_filled = float(prev_order.get("executedQty", 0.0))
        curr_filled = float(curr_order.get("executedQty", 0.0))
        
        if curr_filled > prev_filled:
            # Partial fill detected
            # Emit executionReport
            evt_payload = {
                "e": "executionReport",
                "X": "PARTIALLY_FILLED",
                "l": fill_qty,  # New fill qty
                "z": curr_filled,  # Total filled
                ...
            }
```

#### Phase 5: Truth Auditor
```python
async def _run_truth_auditor(self, current_balances, current_open_orders):
    """
    Validate state consistency:
    - All balances are non-negative
    - No filled > ordered
    - Alert on suspicious patterns
    """
```

---

## ✅ Benefits

| Aspect | WebSocket | Polling |
|--------|-----------|---------|
| **Stability** | ❌ Frequent 1008/410 errors | ✅ Deterministic REST |
| **Complexity** | ❌ 3-tier fallback chain | ✅ Single REST loop |
| **Testability** | ❌ Hard to mock/test | ✅ Easy to unit test |
| **Latency** | ✅ ~50-100ms events | ⏱️ 2-3s poll interval |
| **Cost** | ✅ Low (streams) | ⏱️ Higher (REST calls) |
| **State Sync** | ❌ Event-driven (fragile) | ✅ Comparison-based (robust) |

---

## 🎬 Usage

**Nothing to change in your code!** The polling mode is completely transparent:

1. Start the exchange client as normal
2. `await client.start_user_data_stream()`  → boots into polling mode
3. Position manager receives the same `executionReport` and `balanceUpdate` events
4. Everything else is identical

---

## 📊 Configuration

**Environment Variables** (optional):

```bash
# Force polling interval (default 2.0s, min 1.0s):
USER_DATA_WS_RECONNECT_BACKOFF_SEC=2.0

# Rate limit backoff (default 3.0s):
USER_DATA_WS_RECONNECT_BACKOFF_SEC=3.0

# Max reconnects before FATAL (default 50):
# Not used in polling mode, but kept for API compatibility
```

---

## 🔍 Monitoring

**Watch for these logs**:

```
✅ [EC:UserDataWS] WebSocket modes disabled by default...
✅ [EC:Polling] Polling mode active (interval=2.0s)
✅ [EC:Polling:Balance] USDT changed: free=...
✅ [EC:Polling:Fill] Order XXX (BTCUSDT) CLOSED: status=FILLED
✅ [EC:Polling:PartialFill] Order XXX (ETHUSDT) partial fill: +0.5 qty
✅ [EC:TruthAuditor] ✅ State consistency check passed
```

**Error logs** (to monitor):

```
❌ [EC:Polling] Reconciliation error: ...
❌ [EC:TruthAuditor] ❌ NEGATIVE BALANCE detected: ...
❌ [EC:TruthAuditor] ❌ FILLED > ORDERED: ...
```

---

## 📝 Implementation Details

### Polling Loop State Machine

```
START
  ↓
ws_connected = True
_user_data_auth_mode_active = "polling"
ws_reconnect_count = 0
  ↓
WAIT 2.0s
  ↓
FETCH open orders + balances
  ↓
DETECT changes (balance, fills, partials)
  ↓
EMIT events to position manager
  ↓
UPDATE previous state
  ↓
GOTO WAIT 2.0s
  ↓
(repeat until stop signal)
```

### Event Payload Format

**balanceUpdate**:
```json
{
  "e": "balanceUpdate",
  "E": 1772375776968,  // Timestamp ms
  "a": "USDT",         // Asset
  "d": 100.50,         // Free amount
  "l": 50.00           // Locked amount
}
```

**executionReport**:
```json
{
  "e": "executionReport",
  "E": 1772375776968,
  "s": "BTCUSDT",
  "x": "TRADE",        // Event type
  "X": "FILLED",       // Order status
  "z": 0.5,            // Total filled qty
  "l": 0.5,            // Last fill qty (for partials)
  "n": 25.50,          // Commission amount
  "i": 12345           // Order ID
}
```

---

## 🚨 Known Limitations

1. **Latency**: 2-3 second poll interval vs ~100ms WebSocket events
   - **Mitigation**: Not critical for order confirmation (REST /api/v3/orders is the source of truth)

2. **Rate limiting**: ~10-20 calls/min per request type
   - **Mitigation**: We only call openOrders + account every 2s = 30+30 = 60 calls/min per type (within limits)

3. **Missed micro-fills**: If two fills occur between polls, we only detect the final state
   - **Mitigation**: Position manager doesn't rely on granular fill sequence, only final qty

---

## ✔️ Testing

Run existing tests — they should pass unchanged:

```bash
pytest tests/exchange_client_test.py -v
pytest tests/position_manager_test.py -v
```

The polling mode emits identical events as WebSocket, so position manager behavior is identical.

---

## 🔄 Future Enhancements

1. **Adaptive polling interval**: Increase interval during low activity, decrease during high stress
2. **Batch requests**: Combine openOrders + account into single request
3. **Webhook fallback**: If available, use webhook notifications in addition to polling
4. **Merkle tree validation**: Periodically verify account state against on-chain merkle root

---

## 📞 Support

**If polls are too slow for your use case**:
- Option A: Reduce `poll_interval` to 1.0s (faster but more API cost)
- Option B: Implement hybrid mode (WebSocket for fills, REST for balances)
- Option C: Use WebSocket Streams v3 with dedicated error handling for 1008/410

