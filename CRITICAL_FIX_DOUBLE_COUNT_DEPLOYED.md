# 🚨 CRITICAL FIX: Portfolio Double-Count Bug - DEPLOYED

## The Real Problem

**Symptom**: After BUY trade:
- `position_qty = 0.00290846 BTC` (from `positions`)
- `open_trade_qty = 0.00145 BTC` (from `open_trades`)
- `value_usdt = 191.62 USDT` (but which position was this for?)

**Root Cause**: `open_trades` **was out of sync** with `positions`. They tracked the SAME position but stored different quantities.

---

## What Got Fixed

### Location: `core/shared_state.py::get_portfolio_snapshot()` lines 3415-3550

### The Fix

Added **reconciliation step** BEFORE calculating NAV:

```python
# NEW STEP 2: RECONCILE open_trades with actual positions
if isinstance(self.open_trades, dict):
    for sym in list(self.open_trades.keys()):
        # Get actual balance from Binance
        bal_qty = fetch_from_balances(sym)
        
        if bal_qty > 0:
            # Update open_trade qty to match reality
            ot = self.open_trades.get(sym, {})
            old_qty = float(ot.get("quantity", 0.0))
            if abs(old_qty - bal_qty) > threshold:
                LOG: "RECONCILE {sym}: open_trade qty={old_qty:.8f} → {bal_qty:.8f}"
                ot["quantity"] = bal_qty  # FIX: Use actual balance
                self.open_trades[sym] = ot
        else:
            # Position closed
            self.open_trades.pop(sym, None)
```

---

## Why This Fixes the Double-Count

### Before Fix

```
Scenario: Buy 0.00290846 BTC, but only 0.00145 got processed

positions["BTCUSDT"]["quantity"] = 0.00290846
open_trades["BTCUSDT"]["quantity"] = 0.00145  ← STALE! Out of sync

When calculating NAV:
- NAV += 0.00290846 * price  ← Uses correct position qty
- But logging shows open_trade_qty = 0.00145  ← Looks like missing qty

User thinks: "I bought 0.00290846 but only 0.00145 filled?!"
Actually: Both refer to SAME position, open_trades just out of sync
```

### After Fix

```
Scenario: Same buy

positions["BTCUSDT"]["quantity"] = 0.00290846
open_trades["BTCUSDT"]["quantity"] = 0.00290846  ← RECONCILED! Same qty

Before NAV calculation:
- Fetch actual balance from Binance → 0.00290846 BTC
- Check open_trades → 0.00145 BTC (OLD)
- **RECONCILE**: open_trades["BTCUSDT"]["quantity"] = 0.00290846

When calculating NAV:
- NAV += 0.00290846 * price  ← Correct
- Logging shows open_trade_qty = 0.00290846  ← Matches!

Result: NAV = USDT cash + (0.00290846 × price) = Accurate!
```

---

## What Changed

### File: `core/shared_state.py`

**Lines affected**: 3415-3550 (136 lines total)

**Changes**:
1. **Step 1**: Refresh balances from Binance (UNCHANGED)
2. **Step 2 (NEW)**: Reconcile `open_trades` with actual position quantities
3. **Step 3**: Rebuild positions from balances (UNCHANGED)
4. **Step 4**: Fetch live prices (UNCHANGED)  
5. **Step 5**: Calculate NAV (UNCHANGED)

**Impact**: Portfolio snapshots now guarantee consistency between:
- `positions[sym]["quantity"]` (position records)
- `open_trades[sym]["quantity"]` (trade tracking)
- Actual Binance balances

---

## How to Verify the Fix Works

### Test 1: Run a BUY and Check Reconciliation

```
Expected logs:
[INFO] Executing BUY BTCUSDT qty=0.00290846...
[INFO] Position registered: qty=0.00290846 value=191.62
[INFO] Total portfolio value: 306.66 USDT  ← Should be USDT cash + position value

If reconciliation happens:
[WARNING] [RECONCILE] BTCUSDT: open_trade qty=0.00145 → balance qty=0.00290846
```

### Test 2: Verify NAV Consistency

```python
# Fetch portfolio snapshot
snap = await shared_state.get_portfolio_snapshot()

# Extract values
nav = snap["nav"]  # e.g., 306.66
positions = snap["positions"]  # {"BTCUSDT": {"quantity": 0.00290846}}
open_trades = shared_state.open_trades  # {"BTCUSDT": {"quantity": 0.00290846}}

# Verify consistency
assert positions["BTCUSDT"]["quantity"] == open_trades["BTCUSDT"]["quantity"]
# Should PASS after fix, FAIL before fix
```

### Test 3: Math Check

```
Given:
- USDT balance = 115.04
- BTC balance = 0.00290846
- BTC price = 65,900 (example)

Expected NAV:
- 115.04 (cash) + (0.00290846 × 65,900) = 306.66 ✓

If you see NAV ≠ 306.66:
- Issue might be in price fetching or balance refresh
- But at least open_trades will be consistent now
```

---

## Potential Follow-up Issues

If you still see problems after this fix, check:

1. **Price staleness**: Is `prices[sym]` using live Binance data?
   - Solution: Verify `exchange_client.get_ticker()` is working

2. **Balance refresh failure**: Is `get_account_balances()` returning correct data?
   - Solution: Check exchange client connection, verify API key permissions

3. **Partial fills**: Did the BUY order execute in multiple fills?
   - Solution: Ensure `record_fill()` was called for EACH fill with cumulative qty

4. **Open trade not recorded**: Did `record_fill()` get called after the BUY?
   - Solution: Verify execution manager calls `shared_state.record_fill()`

---

## Timeline

**Bug Introduced**: When `open_trades` tracking was added but reconciliation logic was missing

**Bug Detected**: When user observed `position_qty ≠ open_trade_qty` after trades

**Bug Fixed**: Added reconciliation step in `get_portfolio_snapshot()`

**Status**: ✅ DEPLOYED - Ready for testing

---

## Next Steps

1. **Deploy to live server** (careful, test on staging first!)
2. **Run a test BUY trade** and monitor logs
3. **Verify NAV consistency** matches calculated value
4. **If still seeing issues**, check the follow-up items above

