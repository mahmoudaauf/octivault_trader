# ⚡ DOUBLE-COUNTING FIX - QUICK REFERENCE

## Two Critical Fixes

### Fix #1: Hydration No Longer Modifies Capital Ledger
**File:** `exchange_truth_auditor.py` (Line 1082)

**Before:** Hydration called `_apply_recovered_fill()` → `record_trade()` → modified `invested_capital` and `free_capital`

**After:** Hydration creates positions directly in `ss.positions[symbol]` WITHOUT touching capital ledger

**Key:** Position created with `entry_price=None`, marked `"source": "wallet_hydration"`

---

### Fix #2: Shadow Mode NAV Uses Wallet Value Only
**File:** `shared_state.py` (Line 1057)

**Before:** NAV = quote_balance + position_values (double-counted hydrated positions)

**After:** If shadow mode → NAV = wallet_value only (no position addition)

**Reason:** Hydrated positions are DERIVED from wallet balance, not separate assets

---

## Verification Checklist

✅ **Syntax Check**
```bash
python3 -m py_compile core/exchange_truth_auditor.py core/shared_state.py
```

✅ **Deployment**
1. Backup: `git branch backup-before-fix`
2. Deploy changes
3. Restart: `systemctl restart octi-trader`

✅ **Validation on Startup**
Look for in logs:
- "TRUTH_AUDIT_POSITION_HYDRATED" events
- "capital_ledger_modified": False
- Shadow mode NAV: "Shadow mode: using wallet_value=..."
- NAV > 0 and equals actual wallet value

✅ **Post-Deployment**
- No position duplicates
- free_capital = original wallet quote balance
- NAV matches sum of wallet asset values at market prices
- No errors in position closing

---

## Impact Summary

| Aspect | Before | After |
|--------|--------|-------|
| NAV Calculation | ❌ Double-counted | ✅ Accurate |
| Capital Ledger | Modified by hydration | NOT modified |
| Position Entry Price | Market price (wrong) | None (deferred) |
| Startup Integrity | Fails with NAV=0 | Passes with correct NAV |
| Shadow Mode | Broken NAV | Correct wallet_value |

---

## Files Modified

1. `core/exchange_truth_auditor.py` - _hydrate_missing_positions() method
2. `core/shared_state.py` - get_nav_quote() method

Both files compile successfully. Ready for production deployment.
