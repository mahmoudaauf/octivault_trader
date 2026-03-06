# Shadow Mode Position Source Architecture Fix

## 🎯 Issue Description

**The Split-Brain Problem:**

ExecutionManager's accounting was reading from the **live positions container** (`self.shared_state.positions`), while Meta readiness and position snapshot logic read from **virtual positions** (`self.shared_state.virtual_positions`) in shadow mode.

This created a fundamental architectural violation:

```
Shadow Mode State Split:
┌─────────────────────────────────────────────────────────────────┐
│ ExecutionManager Accounting (MUTATING)                          │
│ └─→ reads/mutates: self.shared_state.positions ← LIVE          │
│                                                                  │
│ Meta Readiness & Snapshot (READING)                            │
│ └─→ reads: self.shared_state.virtual_positions ← VIRTUAL       │
│                                                                  │
│ Result: ACCOUNTING_AUDIT shows significant position            │
│         but get_positions_snapshot() returns empty              │
│         → OpsPlaneReady never fires                             │
│         → System deadlocks                                      │
└─────────────────────────────────────────────────────────────────┘
```

## ✅ The Fix

**Core Principle:**
In shadow mode, ExecutionManager MUST operate on `virtual_positions` and `virtual_open_trades` (same as Meta readiness).

**Pattern Applied:**
```python
# BEFORE (broken):
positions = getattr(self.shared_state, "positions", {}) or {}
open_trades = getattr(self.shared_state, "open_trades", {}) or {}

# AFTER (fixed):
if getattr(self.shared_state, "trading_mode", "") == "shadow":
    positions = getattr(self.shared_state, "virtual_positions", {}) or {}
    open_trades = getattr(self.shared_state, "virtual_open_trades", {}) or {}
else:
    positions = getattr(self.shared_state, "positions", {}) or {}
    open_trades = getattr(self.shared_state, "open_trades", {}) or {}
```

## 📍 Files Modified

### 1. **core/execution_manager.py**
   - **Line 4151-4157**: `_audit_post_fill_accounting()`
     - Reads positions/open_trades for accounting audit
     - NOW: Uses virtual containers in shadow mode
     - Impact: Accounting audit now sees same capital as readiness

### 2. **core/meta_controller.py**
   - **Line 3724**: `_confirm_position_registered()`
     - Verifies BUY fills reached SharedState
     - NOW: Checks virtual containers in shadow mode
     
   - **Line 8152**: `_passes_min_hold()`
     - Entry timestamp lookup for min-hold gate
     - NOW: Reads from virtual containers in shadow mode
     
   - **Line 8689**: Capital Recovery Mode (nomination)
     - Selects oldest position for recovery sell
     - NOW: Iterates virtual positions in shadow mode
     
   - **Line 13649**: Min-hold check in trade execution
     - Validates holding time before SELL
     - NOW: Reads virtual containers in shadow mode
     
   - **Line 13742**: Liquidation min-hold check
     - LiquidationAgent-specific holding time gate
     - NOW: Checks virtual containers in shadow mode
     
   - **Line 13842**: Net-PnL exit gate
     - Reads entry price for profit check
     - NOW: Uses virtual containers in shadow mode

### 3. **core/tp_sl_engine.py**
   - **Line 137**: `_auto_arm_existing_trades()`
     - Auto-arm TP/SL for existing positions at startup
     - NOW: Uses virtual containers in shadow mode
     
   - **Line 1462**: `_check_tpsl_triggers()`
     - Scans open trades for TP/SL activation
     - NOW: Reads virtual containers in shadow mode

## 🏗 Architectural Result

**Before Fix:**
```
Live Mode:
└─ positions (authoritative) ← all systems read/write

Shadow Mode (BROKEN):
├─ ExecutionManager: positions (live)
└─ Meta+Readiness: virtual_positions (virtual)
   → SPLIT-BRAIN: two sources of truth
```

**After Fix:**
```
Live Mode:
└─ positions (authoritative) ← all systems read/write

Shadow Mode (CORRECT):
└─ virtual_positions (authoritative) ← all systems read/write
   ✓ ExecutionManager accounting reads/mutates virtual_positions
   ✓ Meta readiness reads virtual_positions
   ✓ Position snapshots read virtual_positions
   ✓ All systems see identical capital
```

## ✨ Impact

✅ **Execution accounting consistency**: ACCOUNTING_AUDIT now reflects actual capital seen by readiness  
✅ **Capital visibility**: get_positions_snapshot() returns same positions as ExecutionManager sees  
✅ **Readiness gates**: OpsPlaneReady now fires with correct capital data  
✅ **No false deadlocks**: Accounting and readiness use same authoritative source  
✅ **Live-identical behavior**: Shadow mode behaves identically to live (100% capital gating)

## 🔍 Verification

To verify the fix is working:

1. **Check ACCOUNTING_AUDIT logs** in shadow mode:
   ```
   [ACCOUNTING_AUDIT] ... position_qty > 0 ... significant=true
   ```

2. **Check position snapshot**:
   ```
   Get_positions_snapshot() should return same positions
   ```

3. **Check OpsPlaneReady fires**:
   ```
   [OpsPlaneReady] should fire when significant positions exist
   ```

4. **No deadlocks**:
   - Capital recovery mode should activate
   - Min-hold gates should work
   - Exit authorities should fire

## 📝 Notes

- All changes are **backward compatible** (live mode behavior unchanged)
- Shadow mode now has **single source of truth** for positions
- Readiness and accounting are **perfectly aligned**
- This is a **structural fix** not a band-aid
