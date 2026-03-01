# WebSocket Polling Mode — Quick Reference

## 🎯 What's Changed (3 Edits)

### Edit 1: Hard-Disable WebSocket (Line ~652)
```python
# BEFORE:
self.user_data_stream_enabled = bool(_cfg_bool("USER_DATA_STREAM_ENABLED", True))
self.user_data_ws_auth_mode = "auto"

# AFTER:
self.user_data_stream_enabled = False  # ✅ FORCE POLLING
self.user_data_ws_auth_mode = "polling"
```

### Edit 2: Bypass WS, Go Straight to Polling (Line ~1596)
```python
# BEFORE: 3-tier fallback (WS API v3 → listenKey → polling)
async def _user_data_ws_loop(self) -> None:
    await self._user_data_ws_api_v3_direct()  # Tier 1
    # ... fallback logic ...
    await self._user_data_polling_loop()  # Tier 3

# AFTER: Direct polling only
async def _user_data_ws_loop(self) -> None:
    await self._user_data_polling_loop()  # Direct to polling
```

### Edit 3: Enhanced Polling with Reconciliation (Line ~1508)
```python
async def _user_data_polling_loop(self) -> None:
    """
    Each 2.0s cycle:
    1. Fetch open orders
    2. Fetch account balances
    3. Detect balance changes → emit balanceUpdate
    4. Detect filled orders → emit executionReport
    5. Detect partial fills → emit executionReport
    6. Run truth auditor (validation)
    """
```

---

## 🚀 Usage (No Changes Required!)

```python
# In your trading bot:
client = ExchangeClient(...)
await client.start_user_data_stream()  # ← Automatically uses polling now

# Position manager receives same events:
# - balanceUpdate: when balances change
# - executionReport: when orders fill/cancel
```

---

## 📊 Poll Cycle Timeline

```
TIME  |  ACTION
------|-------------------------------------------
0.0s  |  Fetch /api/v3/openOrders
0.1s  |  Fetch /api/v3/account
0.2s  |  Compare with previous state
0.3s  |  Emit events (balanceUpdate, executionReport)
0.4s  |  Update internal state
...
2.0s  |  (repeat)
```

**Total API calls per cycle**: 2 (openOrders + account)  
**Rate**: 30 calls/min per endpoint = OK (limits are 1200 calls/min)

---

## ⚠️ Common Issues & Fixes

| Issue | Cause | Fix |
|-------|-------|-----|
| Orders missing from polling | Order filled between polls | Normal; position manager sees final state |
| Delayed fill notification | 2s poll interval | Reduce to 1.0s if needed |
| Rate limit errors | Too many polls | Increase interval or add batch requests |
| Negative balance alert | True auditor error | Check exchange_client logs, may indicate sync issue |

---

## 📈 Monitoring Checklist

- [ ] Check logs start with: `[EC:UserDataWS] WebSocket modes disabled by default...`
- [ ] Polling loop starts: `[EC:Polling] Polling mode active (interval=2.0s)`
- [ ] Balance updates appear: `[EC:Polling:Balance] USDT changed: ...`
- [ ] Order fills appear: `[EC:Polling:Fill] Order XXX (...) CLOSED: status=FILLED`
- [ ] No negative balance alerts: `❌ NEGATIVE BALANCE detected`
- [ ] No "filled > ordered" alerts: `❌ FILLED > ORDERED`

---

## 🔧 Tuning (Optional)

**Faster polling** (1.0s instead of 2.0s):
```python
# In _user_data_polling_loop():
poll_interval = 1.0  # ← change from 2.0
```

**Slower polling** (3.0s for lower API cost):
```python
poll_interval = 3.0  # ← change from 2.0
```

**Disable polling entirely** (for testing):
```python
self.user_data_stream_enabled = False  # ← already set, but if you want to disable:
# Just don't call start_user_data_stream()
```

---

## ✅ Deployment Checklist

- [x] Line ~652: `user_data_stream_enabled = False`
- [x] Line ~658: `user_data_ws_auth_mode = "polling"`
- [x] Line ~1596: `_user_data_ws_loop()` simplified to call polling directly
- [x] Line ~1508: `_user_data_polling_loop()` enhanced with:
  - [x] Fetch openOrders
  - [x] Fetch account
  - [x] Detect balance changes
  - [x] Detect order fills
  - [x] Detect partial fills
  - [x] Truth auditor
- [x] No syntax errors
- [x] Imports available (contextlib, Dict, Any already imported)

---

## 📝 Event Format Reference

**Position manager receives these events exactly as before**:

```python
# Balance changed:
{
  "e": "balanceUpdate",
  "E": 1772375776968,  # ms timestamp
  "a": "USDT",
  "d": 100.50,  # free
  "l": 50.00    # locked
}

# Order filled:
{
  "e": "executionReport",
  "E": 1772375776968,
  "s": "BTCUSDT",
  "x": "TRADE",
  "X": "FILLED",      # Status
  "z": 0.5,           # Total filled qty
  "l": 0.5,           # Last fill qty
  "i": 12345          # Order ID
}

# Order partially filled:
{
  "e": "executionReport",
  "E": 1772375776968,
  "s": "BTCUSDT",
  "x": "TRADE",
  "X": "PARTIALLY_FILLED",
  "z": 0.3,           # Total filled (0.3 + 0.2 = 0.5 total)
  "l": 0.2,           # Last fill qty
  "i": 12345
}
```

---

## 🎓 Why Polling is Better

| Metric | WebSocket | Polling |
|--------|-----------|---------|
| **Error Rate** | ❌ 1008 policies, 410 gone | ✅ Stable REST |
| **Debugging** | ❌ Event-driven (hard to trace) | ✅ State comparison (easy) |
| **Testing** | ❌ Requires mock WebSocket | ✅ Simple mocked REST calls |
| **Scalability** | ✅ Stream-based | ✅ Poll-based (10+ bots fine) |
| **Compliance** | ✅ Per Binance spec | ✅ Per REST spec |
| **Latency** | ✅ ~100ms | ⏱️ ~2000ms (acceptable) |

**Conclusion**: Polling trades latency for stability. For most bots, this is the right choice.

