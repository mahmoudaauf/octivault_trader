# 🔧 SURGICAL FIX: Shadow Mode Isolation

## 🚨 The Core Architectural Flaw (Explained)

### The Problem

In the previous architecture, **shadow mode had TWO sources of truth**:

1. **Virtual ledger** (`virtual_balances`, `virtual_positions`, `virtual_nav`)
   - Used for trading simulation
   - Updated by ExecutionManager
   - Never touches real exchange

2. **Real balance-based positions** (from `hydrate_positions_from_balances()`)
   - Derived from `self.balances`
   - Updated by `sync_authoritative_balance()`
   - Overwrites positions unconditionally

### What Happened

**Shadow Trade Lifecycle:**

```
T=0s:   BUY 1 BTC
        → ExecutionManager creates virtual_position(BTC/USDT, qty=1)
        → virtual_positions[BTCUSDT] = {qty: 1, ...}
        ✅ SHADOW MODE HAPPY

T=2s:   sync_authoritative_balance() runs
        → Fetches real exchange balances (REAL: 0 BTC because we're in shadow)
        → self.balances[BTC] = {free: 0, locked: 0}
        → Calls hydrate_positions_from_balances()
        → Finds 0 BTC in wallet
        → Clears position: BTCUSDT = {qty: 0, status: CLOSED}
        ❌ SHADOW TRADE ERASED!

T=3s:   System checks position
        → position.quantity == 0
        → System thinks FLAT
        → All simulation breaks
```

### Why This Happens

```python
# OLD CODE in sync_authoritative_balance():
for asset, data in new_bals.items():
    a = asset.upper()
    self.balances[a] = data  # ← ALWAYS updates, even in shadow mode!

# Then later in update_balances() or elsewhere:
if getattr(self.config, "auto_positions_from_balances", True):
    await self.hydrate_positions_from_balances()  # ← Runs ALWAYS, even in shadow!
    # This function:
    # 1. Checks wallet balance (0 BTC in shadow)
    # 2. Overwrites position.quantity to 0
    # 3. Marks position as CLOSED
```

---

## ✅ The Surgical Fixes

### Fix #1: Prevent Position Hydration in Shadow Mode

**Location:** `core/shared_state.py` - `update_balances()` method

**Before:**
```python
try:
    if getattr(self.config, "auto_positions_from_balances", True):
        await self.hydrate_positions_from_balances()
except Exception as e:
    self.logger.warning(f"hydrate_positions_from_balances failed: {e}")
```

**After:**
```python
# CRITICAL: Never hydrate positions from balances in shadow mode
# In shadow mode, positions are managed entirely by virtual_positions
# and must not be overwritten by exchange balances
try:
    if (
        getattr(self.config, "auto_positions_from_balances", True)
        and self.trading_mode != "shadow"
    ):
        await self.hydrate_positions_from_balances()
except Exception as e:
    self.logger.warning(f"hydrate_positions_from_balances failed: {e}")
```

**Locations Fixed:**
1. `update_balances()` @ line ~2714
2. `portfolio_reset()` @ line ~1373

**Effect:**
- When `trading_mode == "shadow"`, `hydrate_positions_from_balances()` NEVER runs
- Virtual positions are isolated and protected
- No more erasure of shadow trades

---

### Fix #2: Prevent Balance Overwrite in Shadow Mode

**Location:** `core/shared_state.py` - `sync_authoritative_balance()` method

**Before:**
```python
async def sync_authoritative_balance(self, force: bool = False) -> None:
    if self._exchange_client and hasattr(self._exchange_client, "get_spot_balances"):
        try:
            new_bals = await self._exchange_client.get_spot_balances()
            if new_bals:
                async with self._lock_context("balances"):
                    for asset, data in new_bals.items():
                        if isinstance(data, dict):
                            a = asset.upper()
                            self.balances[a] = data  # ← ALWAYS overwrites!
```

**After:**
```python
async def sync_authoritative_balance(self, force: bool = False) -> None:
    """
    ...
    SURGICAL FIX #2: In shadow mode, treat real balances as read-only reference snapshot.
    Never overwrite self.balances in shadow mode - all trading must use virtual ledgers.
    This prevents exchange corrections from wiping out shadow positions.
    """
    if self._exchange_client and hasattr(self._exchange_client, "get_spot_balances"):
        try:
            new_bals = await self._exchange_client.get_spot_balances()
            if new_bals:
                async with self._lock_context("balances"):
                    # SURGICAL FIX #2: Only update real balances if NOT in shadow mode
                    # In shadow mode, self.balances is read-only reference snapshot
                    if self.trading_mode != "shadow":
                        for asset, data in new_bals.items():
                            if isinstance(data, dict):
                                a = asset.upper()
                                self.balances[a] = data
                    # Always update last sync timestamp (even in shadow mode)
                    self.last_balance_sync = time.time()
                    # ... rest of code
```

**Effect:**
- When `trading_mode == "shadow"`, `self.balances` becomes **read-only**
- Exchange balance corrections don't affect shadow trading
- Virtual ledger (`virtual_balances`, `virtual_positions`) is the true authority

---

## 🧠 Why This Architecture Is Correct

### The Principle: Two Separate Ledgers

**In LIVE mode:**
- `self.balances` is the authority
- Derived from exchange via `sync_authoritative_balance()`
- Positions hydrated from actual wallet holdings
- Real trading with real money

**In SHADOW mode:**
- `self.virtual_balances` is the authority
- `self.virtual_positions` is the authority
- `self.balances` is a read-only snapshot of exchange reality
- Completely isolated from virtual ledger
- Shadow trading cannot affect real positions

### The Lifecycle In Shadow Mode (Now Fixed)

```
T=0s:   ExecutionManager.execute_order(BUY 1 BTC)
        → Checks: trading_mode == "shadow" ✓
        → Updates: virtual_positions[BTCUSDT] = {qty: 1, ...}
        → Updates: virtual_balances[USDT] -= cost
        → Updates: virtual_nav
        ✅ VIRTUAL LEDGER UPDATED

T=2s:   sync_authoritative_balance() runs
        → Fetches real balances from exchange (REAL: 0 BTC)
        → Checks: trading_mode == "shadow" ✓
        → SKIPS balance update: self.balances[BTC] = ... (NOT executed!)
        → SKIPS position hydration (Fix #1)
        ✅ REAL LEDGER UNTOUCHED

T=3s:   System checks position
        → Reads: virtual_positions[BTCUSDT] (qty: 1)
        → System correctly sees: 1 BTC position
        → All simulation correct!
        ✅ SHADOW MODE WORKING
```

---

## 🔐 Safety Guarantees

### These Fixes Guarantee:

1. **Complete Isolation**: Shadow trades cannot affect real positions
2. **Virtual Authority**: `virtual_positions` is the sole source of truth in shadow mode
3. **Read-Only Balances**: Real exchange balances never overwritten in shadow mode
4. **No Erasure**: Shadow positions persist until explicitly closed
5. **Correct Simulation**: Virtual ledger behaves like real exchange without affecting it

### What Still Works:

- **Live Mode**: Completely unaffected (checks `trading_mode != "shadow"`)
- **Balance Readiness**: `balances_ready_event` still set correctly
- **Sync Timestamps**: `last_balance_sync` always updated
- **Health Monitoring**: All metrics still collected

---

## 📊 Verification Checklist

After deployment:

- [ ] Shadow mode boots without clearing positions
- [ ] BUY order in shadow mode creates virtual_position
- [ ] No automatic erasure after 2-5 seconds
- [ ] SELL order in shadow mode uses virtual_position
- [ ] Portfolio NAV uses virtual_nav in shadow mode
- [ ] Live mode still works (positions hydrated from balances)
- [ ] No test failures in test suite
- [ ] Logs show "[SHADOW MODE - balances not updated, virtual ledger is authoritative]"

---

## 📋 Code Impact Summary

| File | Method | Lines | Change |
|------|--------|-------|--------|
| `core/shared_state.py` | `update_balances()` | ~2714 | Add `and self.trading_mode != "shadow"` |
| `core/shared_state.py` | `portfolio_reset()` | ~1373 | Add `and self.trading_mode != "shadow"` |
| `core/shared_state.py` | `sync_authoritative_balance()` | ~2737 | Wrap balance update with `if self.trading_mode != "shadow":` |

**Total Lines Modified:** ~15 lines of critical logic
**Risk Level:** Very Low (only affects shadow mode, live mode untouched)
**Testing:** Unit tests exist for both shadow and live modes

---

## 🚀 Deployment Notes

1. **No Config Changes Required**: Uses existing `trading_mode` setting
2. **No Data Migration**: No state needs to be migrated
3. **Backward Compatible**: Live mode behavior unchanged
4. **Immediate Effect**: Fixes take effect on next sync cycle
5. **Observable**: Logs show shadow mode detection

---

## 🔄 Next Steps

1. Deploy fixes to staging environment
2. Run shadow mode integration test (1 hour)
3. Verify virtual_positions persist across sync cycles
4. Run live mode sanity check
5. Deploy to production
6. Monitor logs for "[SHADOW MODE - balances not updated...]" messages
7. Verify no position erasure incidents in dashboard

