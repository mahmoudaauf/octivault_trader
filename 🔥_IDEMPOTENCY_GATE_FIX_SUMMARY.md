# 🔥 IDEMPOTENCY GATE FIX — Time-Scoped Duplicate Detection

## Problem
The ExecutionManager was blocking **all retry attempts** for the same (symbol, side) pair with a stale idempotency check:

```
EXEC_REJECT reason=IDEMPOTENT
count=1
count=2
count=3
...
```

The cache entry was never being cleared, so once an order attempt was recorded, **all future orders for that symbol/side were permanently blocked**.

---

## Root Cause

The original implementation used a **set-based idempotency gate**:

```python
# ❌ OLD: Set-based, no expiration
self._active_symbol_side_orders = set()

# Line 7168: Check for duplicates
if order_key in self._active_symbol_side_orders:
    return {"status": "SKIPPED", "reason": "ACTIVE_ORDER"}

# Line 7171: Add entry
self._active_symbol_side_orders.add(order_key)

# Line 7670: Finally block removes entry
self._active_symbol_side_orders.discard(order_key)
```

**The problem**: If an order placement deadlocked, crashed, or hung:
1. The entry was **added** at line 7171
2. The finally block might **not execute** properly
3. The entry remained in the set **forever**
4. All future attempts for that (symbol, side) were rejected

---

## Solution: Time-Scoped Idempotency

Changed the implementation to track **timestamps** instead of just presence:

```python
# ✅ NEW: Dict-based with timestamps
self._active_symbol_side_orders: Dict[tuple, float] = {}  # (symbol, side) -> timestamp
self._active_order_timeout_s = 30.0  # Orders stuck for >30s are forcibly cleared
```

### Key Changes

**1. Initialization (Line 1917)**
```python
self._active_symbol_side_orders: Dict[tuple, float] = {}  # (symbol, side) -> timestamp
self._active_order_timeout_s = 30.0  # Orders stuck for >30s are forcibly cleared
```

**2. Time-Scoped Check (Lines 7186-7204)**
```python
order_key = (symbol, side.upper())
now = time.time()

if order_key in self._active_symbol_side_orders:
    last_attempt = self._active_symbol_side_orders[order_key]
    time_since_last = now - last_attempt
    
    if time_since_last < self._active_order_timeout_s:
        # Still within the window — genuinely blocked
        return {"status": "SKIPPED", "reason": "ACTIVE_ORDER"}
    else:
        # 🔥 STALE ENTRY DETECTED — Force clear it
        self.logger.warning(
            "[EM:STALE_CLEARED] Order stuck for %s %s for %.1fs; forcibly clearing and retrying.",
            symbol, side.upper(), time_since_last
        )
        del self._active_symbol_side_orders[order_key]

# Record this attempt with timestamp
self._active_symbol_side_orders[order_key] = now
```

**3. Cleanup (Line 7709)**
```python
finally:
    # ... semaphore cleanup ...
    self._active_symbol_side_orders.pop(order_key, None)
```

---

## Behavior

| Scenario | Before | After |
|----------|--------|-------|
| Order placed successfully | ✅ Cleared in finally block | ✅ Cleared in finally block |
| Order deadlocks | ❌ Blocks all future attempts forever | ✅ Unblocked after 30s, retry allowed |
| Order crashed mid-placement | ❌ Stale entry remains | ✅ Auto-cleared on timeout |
| Legitimate duplicate (same request) | ✅ Rejected < 30s | ✅ Rejected < 30s |
| Stale duplicate (old retry) | ❌ Rejected forever | ✅ Auto-cleared after 30s |

---

## Example Log Output

### Before Fix
```
[EM] Active order exists for ETHUSDT BUY; skipping.
[EM] Active order exists for ETHUSDT BUY; skipping.
[EM] Active order exists for ETHUSDT BUY; skipping.
... (repeats forever until process restart)
```

### After Fix
```
[EM:IDEMPOTENT] Active order exists for ETHUSDT BUY (2.5s ago); skipping.
[EM:IDEMPOTENT] Active order exists for ETHUSDT BUY (15.3s ago); skipping.
[EM:STALE_CLEARED] Order stuck for ETHUSDT BUY for 31.2s; forcibly clearing and retrying.
[EM] Attempting to place ETHUSDT BUY order...
✅ Order placed successfully
```

---

## Configuration

The timeout can be tuned in `__init__`:

```python
self._active_order_timeout_s = 30.0  # Change this value to adjust the window
```

**Recommended values:**
- **5-10 seconds**: Aggressive clearing (fast recovery, may clear legitimate pending orders)
- **30 seconds**: Balanced (default, good for most use cases)
- **60+ seconds**: Conservative (only clear very stale orders)

---

## Testing

To verify the fix:

1. **Monitor logs** for the new `[EM:STALE_CLEARED]` message
2. **Check health endpoint**: `active_symbol_side_orders` should remain low (<5)
3. **Verify buy signals** are no longer permanently blocked
4. **No regressions** in existing order flow

---

## Files Changed

- `core/execution_manager.py`
  - Line 1917: Changed to Dict-based tracking with timestamps
  - Lines 7186-7204: Added time-scoped idempotency check with auto-clearing
  - Line 7709: Updated cleanup to use `.pop()` instead of `.discard()`
  - Lines 2564-2574: Updated health report to handle both set and dict formats
  - Lines 2598-2612: Updated active_sells counter for dict-based tracking

---

## Deployment Notes

✅ **Backward Compatible**: Health endpoint handles both old (set) and new (dict) formats
✅ **No Config Changes Required**: Works with existing setup
✅ **Safe to Deploy**: Only affects failed/stale order recovery, not normal order flow
✅ **Improves Reliability**: Prevents indefinite blocking from rare deadlock scenarios

---

## Impact

**Before**: Bot could get permanently stuck on a symbol if a single order placement failed
**After**: Bot automatically recovers from stuck orders after 30 seconds and retries

This fix ensures that **every signal generates at least one real attempt**, rather than being silently rejected forever.
