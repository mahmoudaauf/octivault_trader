# Quick Guide: Switch to Real NAV Mode

## The Problem You're Solving

You want to use **real exchange balances** for NAV calculation instead of the virtual ledger.

## The Solution

### 1. Locate Your Shadow Mode Configuration

**File:** `core/recovery_engine.py` or your app initialization

**Find:**
```python
self._shadow_mode = True  # or similar
```

**Change to:**
```python
self._shadow_mode = False
```

### 2. Where to Make the Change

Three likely locations:

#### Option A: In `recovery_engine.py` __init__
```python
def __init__(self, ...):
    self._shadow_mode = False  # ← Change this
```

#### Option B: In app startup/main.py
```python
recovery_engine = RecoveryEngine(...)
recovery_engine._shadow_mode = False  # ← Add this line
```

#### Option C: In `app_context.py` initialization
```python
async def initialize_recovery_engine(self):
    self.recovery_engine._shadow_mode = False  # ← Add this
```

### 3. Optional: Disable Snapshot Loading

If you also want to force fresh data from exchange:

```python
recovery_engine._use_snapshot = False
```

This ensures `_load_live()` is always called.

## Expected Changes After Switch

| Aspect | Before (Shadow Mode) | After (Real Mode) |
|--------|----------------------|-------------------|
| **NAV** | 0.0 (expected) | Actual balance from exchange |
| **Balances** | Not synced | Synced from exchange |
| **Positions** | From virtual ledger | Reconciled with real orders |
| **API Calls** | Made but not applied | Made and applied |
| **Use Case** | Simulation/Testing | Real trading |

## Verification

After making the change, check the logs:

### In Shadow Mode (Current):
```
[SHADOW MODE - balances not updated, virtual ledger is authoritative]
NAV is 0.0 (SHADOW MODE - this is expected)
```

### In Real Mode (After Fix):
```
[SS:BalanceUpdate] USDT: free=1000.50, locked=0.0
[NAV] Total: 1000.50 | Quotes: {'USDT': {...}} | Positions: 3
NAV is 1000.50 (real exchange balance)
```

## Advanced: Check Current Mode Programmatically

```python
# Check if shadow mode is enabled
is_shadow_mode = getattr(recovery_engine, "_shadow_mode", False)
print(f"Shadow Mode: {is_shadow_mode}")

# Check if virtual ledger is authoritative
is_virtual_auth = getattr(shared_state, "_virtual_ledger_authoritative", False)
print(f"Virtual Ledger Authoritative: {is_virtual_auth}")
```

## Troubleshooting

### If NAV is still 0 after disabling shadow mode:

1. **Check balances are syncing:**
   ```
   Look for: "[SS:BalanceUpdate]" logs
   If missing: Exchange API call failed or balances empty
   ```

2. **Check exchange connection:**
   ```
   recovery_engine.ex  # Should not be None
   ```

3. **Check balances were received:**
   ```
   shared_state.balances  # Should not be empty
   shared_state.balances_ready_event.is_set()  # Should be True
   ```

## Rollback to Shadow Mode

If you need to return to shadow mode:

```python
recovery_engine._shadow_mode = True
```

## Summary

1. **Identify** where `_shadow_mode` is set
2. **Change** from `True` to `False`
3. **Restart** the bot
4. **Verify** NAV is now non-zero from exchange balances
