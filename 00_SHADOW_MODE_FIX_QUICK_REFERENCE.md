# Shadow Mode Position Source Fix - Quick Reference

## 🎯 One-Line Summary
ExecutionManager now uses `virtual_positions` in shadow mode instead of `positions`, achieving single source of truth for capital visibility.

## 🔧 What Changed

### Before
```python
positions = getattr(self.shared_state, "positions", {}) or {}        # ❌ WRONG in shadow
open_trades = getattr(self.shared_state, "open_trades", {}) or {}    # ❌ WRONG in shadow
```

### After
```python
if getattr(self.shared_state, "trading_mode", "") == "shadow":
    positions = getattr(self.shared_state, "virtual_positions", {}) or {}        # ✅ RIGHT in shadow
    open_trades = getattr(self.shared_state, "virtual_open_trades", {}) or {}    # ✅ RIGHT in shadow
else:
    positions = getattr(self.shared_state, "positions", {}) or {}                # ✅ RIGHT in live
    open_trades = getattr(self.shared_state, "open_trades", {}) or {}            # ✅ RIGHT in live
```

## 📍 Locations Fixed (9 total)

### execution_manager.py
- **Line 4151**: `_audit_post_fill_accounting()` - Accounting audit now reads correct container

### meta_controller.py
- **Line 3724**: `_confirm_position_registered()` - BUY registration check
- **Line 8152**: `_passes_min_hold()` - Min-hold timestamp lookup
- **Line 8689**: Capital Recovery Mode - Position nomination
- **Line 13649**: Min-hold in trade execution
- **Line 13742**: Liquidation min-hold check
- **Line 13842**: Net-PnL exit gate entry price

### tp_sl_engine.py
- **Line 137**: `_auto_arm_existing_trades()` - TP/SL auto-arm
- **Line 1462**: `_check_tpsl_triggers()` - TP/SL trigger scan

## 🧪 How to Verify

### Test 1: Accounting vs Readiness Alignment
```python
# Shadow mode: Both should see same positions
accounting_pos = execution_manager._audit_post_fill_accounting()  # reads virtual_positions
readiness_pos = meta_controller.get_positions_snapshot()         # reads virtual_positions
assert accounting_pos == readiness_pos  # ✅ Should be equal
```

### Test 2: OpsPlaneReady Should Fire
```python
# If position exists in virtual_positions (shadow)
# Then OpsPlaneReady should fire (not deadlock)
[Check logs for]
[OpsPlaneReady] Status: Ready
[Not the deadlock pattern]
```

### Test 3: Min-Hold Gates Work
```python
# SELL should be blocked by min-hold
# even though position is in virtual_positions
[Check logs for]
[Meta:MinHold:PreCheck] SELL blocked for XXXX: age=30s < min_hold=90s
```

## 🔄 Architecture Pattern

**Core Pattern Applied Everywhere**:
```python
# 1. Check trading mode
if getattr(self.shared_state, "trading_mode", "") == "shadow":
    # 2a. Use VIRTUAL containers in shadow
    positions = getattr(self.shared_state, "virtual_positions", {}) or {}
    open_trades = getattr(self.shared_state, "virtual_open_trades", {}) or {}
else:
    # 2b. Use LIVE containers in live
    positions = getattr(self.shared_state, "positions", {}) or {}
    open_trades = getattr(self.shared_state, "open_trades", {}) or {}

# 3. Use selected container (rest of logic unchanged)
```

## ✨ Why This Matters

| Impact | Before | After |
|--------|--------|-------|
| **Accounting reads** | live positions | virtual positions |
| **Readiness reads** | virtual positions | virtual positions |
| **Capital visibility** | Inconsistent | Consistent |
| **OpsPlaneReady** | Blocked (deadlock) | Fires correctly |
| **System state** | Split-brain | Single source |

## 🚀 Key Benefits

1. **No False Deadlocks**: Capital doesn't disappear between audit and readiness
2. **Correct Capital Gating**: Shadow mode truly "100% identical to live"
3. **All Gates Work**: Min-hold, capital recovery, TP/SL all see same positions
4. **Clean Implementation**: Single pattern applied uniformly
5. **Backward Compatible**: Zero impact on live mode

## 🔍 Implementation Details

### Safe Pattern Used
- Uses `getattr()` with defaults → won't crash if attribute missing
- Checks `trading_mode` before branching → handles both modes
- Defaults to empty dict → graceful fallback

### Type Safety
- All containers are dict-like
- Safe dict operations throughout
- Type checking with `isinstance()` before access

## 📚 Related Files
- See: `00_SHADOW_MODE_POSITION_SOURCE_FIX.md` (detailed documentation)
- See: `00_SHADOW_MODE_FIX_TECHNICAL_VERIFICATION.md` (technical verification)
- See: `00_SHADOW_MODE_FIX_COMPLETION_SUMMARY.md` (completion summary)

## ⚠️ Important Notes

- **DO NOT** manually edit position containers - use provided methods
- **DO NOT** assume `positions` exists in shadow mode - use the pattern
- **DO** use this pattern for any NEW code that reads positions/trades
- **DO** report issues if you see capital visibility gaps in shadow mode

## 🎓 For Future Development

When adding new code that reads positions/open_trades:

```python
# ❌ WRONG - doesn't account for shadow mode
positions = self.shared_state.positions

# ✅ RIGHT - handles both modes
if getattr(self.shared_state, "trading_mode", "") == "shadow":
    positions = getattr(self.shared_state, "virtual_positions", {}) or {}
else:
    positions = getattr(self.shared_state, "positions", {}) or {}
```

---

**Status**: ✅ COMPLETE - All 9 locations fixed, ready for production.
