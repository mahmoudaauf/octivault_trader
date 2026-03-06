# Order-ID Idempotency Fix - Visual Flow

## 🔄 Call Flow (Before vs After)

### BEFORE: Object-Based Guard Only
```
Call 1: _ensure_post_fill_handled(order_dict_v1)
        ↓
        Check: order_dict_v1["_post_fill_done"]? → NO
        ↓
        Execute _handle_post_fill() ✓
        ↓
        Mark: order_dict_v1["_post_fill_done"] = True

Call 2: _ensure_post_fill_handled(order_dict_v2) ← NEW RECONSTRUCTED DICT
        ↓
        Check: order_dict_v2["_post_fill_done"]? → NO (different object!)
        ↓
        Execute _handle_post_fill() ✗ DUPLICATE!
        ↓
        Mark: order_dict_v2["_post_fill_done"] = True
```

### AFTER: Order-ID Based Guard (NEW)

```
Call 1: _ensure_post_fill_handled(order_dict_v1)
        ↓
        Extract: order_id = "123456" (orderId or clientOrderId)
        ↓
        Check: order_id in _post_fill_processed_ids? → NO
        ↓
        Execute _handle_post_fill() ✓
        ↓
        Mark: _post_fill_processed_ids.add("123456")
        Mark: order_dict_v1["_post_fill_done"] = True

Call 2: _ensure_post_fill_handled(order_dict_v2) ← NEW RECONSTRUCTED DICT
        ↓
        Extract: order_id = "123456" (same order!)
        ↓
        Check: order_id in _post_fill_processed_ids? → YES ✓
        ↓
        Return early with dict(default) ← NO DUPLICATE!
```

---

## 🎯 Key Differences

| Aspect | Before | After |
|--------|--------|-------|
| **Guard Type** | Object identity | Order ID value |
| **Survives** | Single object reference | Order ID across all reconstructions |
| **Handles** | ❌ Reconciliation (new dict) | ✅ Reconciliation (new dict) |
| **Handles** | ❌ Recovery (new dict) | ✅ Recovery (new dict) |
| **Handles** | ❌ Shadow mode mutation | ✅ Shadow mode mutation |
| **Exchange Semantics** | ❌ Ignores order immutability | ✅ Respects order ID uniqueness |

---

## 📊 State Tracking

### ExecutionManager Instance State

```python
class ExecutionManager:
    def __init__(self):
        # NEW: Order-ID based idempotency
        self._post_fill_processed_ids: Set[str] = set()
        #     ↑ Lifetime: entire ExecutionManager instance
        #     ↑ Scope: all symbols, all sides
        #     ↑ Thread-safe: single-threaded event loop
```

### Example Execution Trace

```
Session Start:
  _post_fill_processed_ids = set()

Order: BTCUSDT BUY (orderId=12345)
  Call 1: Extract "12345" → not in set → process → add "12345"
  _post_fill_processed_ids = {"12345"}

Order: ETHUSDT BUY (orderId=67890)
  Call 2: Extract "67890" → not in set → process → add "67890"
  _post_fill_processed_ids = {"12345", "67890"}

Reconciliation (BTC reconciles order again):
  Call 3: Extract "12345" → IN SET → return early
  _post_fill_processed_ids = {"12345", "67890"}  (unchanged)

Recovery (BTC recovery):
  Call 4: Extract "12345" → IN SET → return early
  _post_fill_processed_ids = {"12345", "67890"}  (unchanged)
```

---

## 🛡️ Defense-in-Depth

```
_ensure_post_fill_handled() guards (in order):

1. ORDER-ID CHECK (NEW)
   └─ order_id in _post_fill_processed_ids?
      └─ YES → return early ← STRONGEST (persists across reconstructions)
      
2. OBJECT-LEVEL CHECK (EXISTING)
   └─ order["_post_fill_done"]?
      └─ YES → return early ← covers same-object re-calls

3. EXECUTION QTY CHECK (EXISTING)
   └─ exec_qty > 0?
      └─ NO → return early ← filters non-fills
```

All three guards remain in place. Order-ID check is fastest and most robust.

---

## 🔑 Code Implementation Details

### Extraction
```python
order_id = str(order.get("orderId") or order.get("clientOrderId") or "")
#         └─ tries "orderId" first (Binance native)
#         └─ falls back to "clientOrderId" (if order_id is missing)
#         └─ converts to string (safe for set storage)
#         └─ empty string if both missing
```

### Check
```python
if order_id:  # only if non-empty
    if order_id in self._post_fill_processed_ids:
        return dict(default)
#   └─ constant time lookup (O(1) in hash set)
```

### Mark
```python
if order_id:  # only if non-empty
    self._post_fill_processed_ids.add(order_id)
#   └─ idempotent (adding duplicate does nothing)
#   └─ happens AFTER _handle_post_fill() succeeds
```

---

## ✅ Guarantees

1. **Idempotency:** Same order ID → processed exactly once
2. **Non-breaking:** Preserves all existing guards
3. **Efficient:** O(1) set lookup, minimal overhead
4. **Clear:** Comments explain the fix
5. **Testable:** Can mock order reconstruction scenarios

---

## 🚀 Deployment Info

- **Files:** 1 (`core/execution_manager.py`)
- **Lines added:** ~20 (including comments)
- **Lines modified:** 0 (added new guards, kept existing)
- **Breaking changes:** None
- **New dependencies:** None
- **Date:** March 3, 2026

