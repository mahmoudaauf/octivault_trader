# ⚡ ACTIVE TRADES LIFECYCLE - QUICK REFERENCE

## 🎯 One-Liner
**Implement full trade lifecycle in `_handle_post_fill()`: BUY creates `active_trades[symbol]`, SELL reduces/closes + emits PnL**

---

## 📍 Location
- **File**: `core/execution_manager.py`
- **Method**: `_handle_post_fill()`
- **Lines**: 357–473 (right after position update, before record_trade)

---

## ✅ What Was Added

### BUY Side (Lines 359–391)
```
if side_u == "BUY" and exec_qty > 0 and price > 0:
    ├─ Create/init active_trades{}
    ├─ Set active_trades[sym] = {
    │  "symbol": sym,
    │  "entry_price": price,
    │  "qty": exec_qty,
    │  "side": "BUY",
    │  "opened_at": timestamp,
    │  "order_id": order_id,
    │  "fee_quote": fee_quote
    │}
    └─ Log: [LIFECYCLE_BUY_OPEN]
```

### SELL Side (Lines 393–435)
```
elif side_u == "SELL" and exec_qty > 0 and price > 0:
    ├─ Get active_trades[sym]
    ├─ remaining_qty = current_qty - exec_qty
    ├─ If remaining_qty <= 0:
    │  ├─ Delete active_trades[sym]
    │  ├─ Compute PnL = (price - entry) * qty - fees
    │  ├─ Update realized_pnl (atomic)
    │  ├─ Emit RealizedPnlUpdated event
    │  └─ Log: [LIFECYCLE_SELL_CLOSE]
    └─ Else:
       ├─ Update active_trades[sym].qty = remaining_qty
       └─ Log: [LIFECYCLE_SELL_REDUCE]
```

---

## 🔄 State Flow

```
┌─ BUY FILL ──────────────────────────────────────┐
│ active_trades[BTCUSDT] = {                      │
│   entry_price: 67000,                          │
│   qty: 1.0,                                    │
│   ...                                          │
│ }                                              │
└────────────────────────────────────────────────┘
                      │
                      ▼
        TP/SL Engine: len(active_trades) > 0 ✅
                      │
                      ▼
┌─ PARTIAL SELL (0.5) ────────────────────────────┐
│ active_trades[BTCUSDT] = {                      │
│   entry_price: 67000,                          │
│   qty: 0.5,  ← reduced                         │
│   ...                                          │
│ }                                              │
└────────────────────────────────────────────────┘
                      │
                      ▼
┌─ FULL SELL (0.5) ───────────────────────────────┐
│ Delete active_trades[BTCUSDT]                  │
│ Emit: RealizedPnlUpdated {pnl_delta: 1480}     │
│ Update realized_pnl atomically                 │
└────────────────────────────────────────────────┘
```

---

## 🎬 Key Scenarios

### Scenario 1: Full Trade Close
```
1. BUY 1.0 BTCUSDT @ 67000
   → active_trades[BTCUSDT].qty = 1.0
   
2. SELL 1.0 BTCUSDT @ 68000
   → remaining_qty = 0
   → pnl = (68000-67000)*1.0 - fees = 1000 - fees
   → Emit RealizedPnlUpdated
   → Delete active_trades[BTCUSDT]
```

### Scenario 2: Partial Exit
```
1. BUY 2.0 BTCUSDT @ 67000
   → active_trades[BTCUSDT].qty = 2.0
   
2. SELL 0.5 BTCUSDT @ 68000
   → remaining_qty = 1.5
   → active_trades[BTCUSDT].qty = 1.5
   → Log: [LIFECYCLE_SELL_REDUCE]
   
3. SELL 1.5 BTCUSDT @ 68500
   → remaining_qty = 0
   → Emit RealizedPnlUpdated
   → Delete active_trades[BTCUSDT]
```

### Scenario 3: Multiple Symbols
```
active_trades = {
  "BTCUSDT": {qty: 1.0, entry: 67000, ...},
  "ETHUSDT": {qty: 10.0, entry: 3800, ...},
}
```
Each symbol managed independently ✅

---

## 🎯 Why This Matters

| Before | After |
|--------|-------|
| ❌ active_trades missing | ✅ Created on BUY |
| ❌ TPSL can't see trades | ✅ Checks active_trades count |
| ❌ PnL from external sources | ✅ Computed from entry_price |
| ❌ No SELL close events | ✅ Emits RealizedPnlUpdated |
| ❌ TruthAuditor patching | ✅ Built-in lifecycle logic |

---

## 🔧 Integration Points

### TP/SL Engine
```python
# Old (broken)
if len(ss.open_trades) > 0:  # Always 0 ❌

# New (fixed)
if len(ss.active_trades) > 0:  # Counts real trades ✅
```

### Accounting
```python
# Realized PnL now emitted twice:
1. From record_trade() [existing]
2. From SELL close [NEW] - more precise
```

### Position Manager
```python
# Positions updated as before (unchanged)
# But now aligned with active_trades lifecycle
```

---

## 📊 Logging Markers

| Marker | Meaning |
|--------|---------|
| `[LIFECYCLE_BUY_OPEN]` | Created active_trades entry |
| `[LIFECYCLE_SELL_CLOSE]` | Closed trade + emitted PnL |
| `[LIFECYCLE_SELL_REDUCE]` | Partial close (qty reduced) |
| `[LIFECYCLE_*_FAILED]` | Error in lifecycle (non-fatal) |

---

## ✅ Testing Checklist

- [ ] BUY creates active_trades[symbol]
- [ ] SELL with remaining qty reduces correctly
- [ ] SELL with zero remaining deletes entry
- [ ] RealizedPnlUpdated emitted on close
- [ ] PnL calculated correctly: (exit_price - entry_price) * qty - fees
- [ ] Multiple symbols independent
- [ ] Error handling non-blocking

---

## 🚀 Status
- [x] Implemented
- [x] Documented
- [ ] Integration tested
- [ ] Live validation

