# The Double-Count Bug: Final Diagnosis & Solution

## What You Observed

```
After BUY trade:
  NAV before: 115.04 USDT
  Position qty: 0.00290846 BTC
  Open trade qty: 0.00145 BTC  ← DIFFERENT!
  Position value: 191.62 USDT
  
User confusion: "I have 191.62 in the position AND 115.04 in cash = 306.62"
               "But I only started with 306 USDT total!"
               "Where did the extra 191.62 come from?!"
```

---

## The Real Issue

### It's NOT a double-count of the position value

The math is actually **correct**:
- Started with: ~306 USDT
- Bought: 0.00290846 BTC for ~191.62 USDT
- Now have: 115.04 USDT cash + 0.00290846 BTC (worth 191.62) = 306.66 total

### It IS a tracking inconsistency

The bug was that:
- `positions["BTCUSDT"]["quantity"]` = 0.00290846
- `open_trades["BTCUSDT"]["quantity"]` = 0.00145

**These should ALWAYS be the same!** They refer to the same position.

The mismatch caused confusion:
- "Which quantity is correct?"
- "Is my position 0.00290846 or 0.00145?"
- "Why is open_trade less than position?"

---

## Why `open_trades` Was Out of Sync

### Possible Causes

1. **Partial fill not fully processed**: Buy order filled in 2 increments
   - Fill 1: 0.00145 BTC → recorded in open_trades
   - Fill 2: 0.00145846 BTC → added to positions but not merged in open_trades
   
2. **Manual position adjustment**: Someone modified positions without updating open_trades

3. **Server restart**: open_trades loaded from disk, positions refreshed from Binance

4. **Race condition**: Concurrent trades processed out of order

### The Core Problem

`open_trades` is meant to be a **mirror of positions** for trade tracking. But there was **no reconciliation logic** to keep them in sync.

When they diverged, it looked like data corruption or double-counting.

---

## The Solution

### Added Reconciliation in `get_portfolio_snapshot()`

**Before** calling `record_fill()` or calculating NAV, we now:

```python
1. Read actual position quantities from Binance balances
2. Check if open_trades records match these quantities
3. If they don't match:
   - Fix open_trades to match reality
   - Log a warning so you can investigate why they diverged
   - Proceed with NAV calculation using correct data
```

### Code

```python
# NEW: Reconciliation step in get_portfolio_snapshot()
for sym in list(self.open_trades.keys()):
    # Get actual qty from Binance balance
    actual_qty = fetch_balance(sym)
    
    # Get recorded qty from open_trades
    recorded_qty = self.open_trades[sym]["quantity"]
    
    if actual_qty != recorded_qty:
        LOG WARNING: f"RECONCILE {sym}: {recorded_qty:.8f} → {actual_qty:.8f}"
        self.open_trades[sym]["quantity"] = actual_qty
```

---

## Why This Fixes It

### Before Fix

```
After buy 0.00290846 BTC:

positions["BTCUSDT"] = {
    "quantity": 0.00290846,
    "value_usdt": 191.62  ← Calculated from 0.00290846 * price
}

open_trades["BTCUSDT"] = {
    "quantity": 0.00145  ← Stale! Should be 0.00290846
}

Snapshot returned:
{
    "nav": 306.66,  ← Correct!
    "positions": {"BTCUSDT": {"quantity": 0.00290846}},
    "open_trades": {"BTCUSDT": {"quantity": 0.00145}}  ← Inconsistent!
}

User reads both values and thinks there's double-counting
```

### After Fix

```
After buy 0.00290846 BTC:

positions["BTCUSDT"] = {
    "quantity": 0.00290846,
    "value_usdt": 191.62  ← Correct
}

open_trades["BTCUSDT"] = {
    "quantity": 0.00290846  ← RECONCILED! Now matches positions
}

Snapshot returned:
{
    "nav": 306.66,  ← Correct
    "positions": {"BTCUSDT": {"quantity": 0.00290846}},
    "open_trades": {"BTCUSDT": {"quantity": 0.00290846}}  ← Consistent!
}

User reads consistent values: Position = 0.00290846 BTC ✓
```

---

## What This DOESN'T Fix

This fix ensures consistency, but it **relies on accurate data from Binance**.

If the issue is **price-based** (not quantity-based):
- `value_usdt = qty * stale_price` ← This would need separate fix

If the issue is **balance refresh failure**:
- Exchange client can't reach Binance ← This would need separate fix

If the issue is **record_fill never called**:
- Position got created but trade not recorded ← This would need separate fix

---

## How to Know It's Working

### Check 1: Position Quantities Match
```python
snapshot = await shared_state.get_portfolio_snapshot()
pos_qty = snapshot["positions"]["BTCUSDT"]["quantity"]
ot_qty = shared_state.open_trades["BTCUSDT"]["quantity"]
assert pos_qty == ot_qty  # Should PASS
```

### Check 2: NAV Math Checks Out
```python
nav = snapshot["nav"]
usdt = snapshot["balances"]["USDT"]["free"]
btc_qty = snapshot["positions"]["BTCUSDT"]["quantity"]
btc_price = snapshot["prices"]["BTCUSDT"]

calculated_nav = usdt + (btc_qty * btc_price)
assert abs(nav - calculated_nav) < 0.01  # Should PASS
```

### Check 3: Reconciliation Logs
```
If qty mismatch detected, you'll see:
[WARNING] [RECONCILE] BTCUSDT: open_trade qty=0.00145 → balance qty=0.00290846
```

---

## Deployment

**File Changed**: `core/shared_state.py` lines 3415-3550

**Change Type**: Non-breaking (adds reconciliation, doesn't remove features)

**Risk**: Low (only reconciles data, doesn't execute trades)

**Testing**: 
1. Run a test BUY
2. Check logs for reconciliation messages (or none if already consistent)
3. Verify NAV = USDT + (position value)

---

## Summary

**Problem**: `open_trades` was out of sync with `positions`, causing apparent double-count
**Cause**: No reconciliation logic between two tracking systems
**Solution**: Added sync step that fixes divergence before calculating NAV
**Result**: Consistent quantity tracking, accurate NAV calculation, no more confusion

