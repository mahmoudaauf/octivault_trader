# Phase B: MetaController Integration - Capital Governor Position Limits

**Status**: Implementation Guide  
**Component**: `core/meta_controller.py`  
**Dependencies**: `core/capital_governor.py` (already created)  
**Time Estimate**: 30 minutes  
**Scope**: Add position limit validation before BUY execution  

---

## Overview

**Objective**: Integrate Capital Governor into MetaController to enforce position limits before executing BUY signals.

**Decision Point**: In `_execute_decision()` method, before calling `ExecutionManager.create_order()`, check:
1. Current open position count
2. Capital Governor bracket limits
3. Block BUY if position limit exceeded

**Flow**:
```
BUY Signal Arrives
    ↓
_execute_decision() called
    ↓
[NEW] Check Capital Governor Position Limits
    ├─ Get NAV → Determine bracket
    ├─ Get open positions → Count them
    ├─ Compare: open_count vs max_concurrent_positions
    └─ If exceeded → REJECT with "position_limit_exceeded"
    ↓
Continue with normal execution gates (P9, fees, etc.)
    ↓
ExecutionManager.create_order()
```

---

## Implementation Steps

### Step 1: Add Capital Governor Initialization (Lines ~505-750)

In `MetaController.__init__()`, after other component initializations (around line 700 where we see `self.execution_logic = ...`), add:

```python
# Initialize Capital Governor for position limiting
from core.capital_governor import CapitalGovernor

self.capital_governor = CapitalGovernor(config)
self.logger.info("[Meta:Init] Capital Governor initialized")
```

**Location Reference**: After line ~700 (ExecutionLogic initialization)

**Why Here**: Capital Governor is a lightweight permission system that doesn't need async lifecycle. Initialized once and reused for every BUY signal.

---

### Step 2: Add Position Count Helper Method (New method, ~20 lines)

Add a helper method to MetaController to count current open positions:

```python
def _count_open_positions(self) -> int:
    """Count currently open positions across all symbols.
    
    Returns: Integer count of positions with quantity > 0
    """
    try:
        # Try SharedState positions snapshot first
        if hasattr(self.shared_state, "get_positions_snapshot"):
            snap = self.shared_state.get_positions_snapshot()
            count = 0
            for symbol, pos_data in snap.items():
                qty = float(pos_data.get("quantity", 0.0) or pos_data.get("qty", 0.0) or 0.0)
                if qty > 0:
                    count += 1
            return count
        
        # Fallback: Try per-symbol lookup
        if hasattr(self.shared_state, "get_position_qty"):
            # Would need list of symbols - use cache if available
            symbols = self.FOCUS_SYMBOLS or set()
            count = sum(1 for sym in symbols if float(self.shared_state.get_position_qty(sym) or 0.0) > 0)
            return count
            
    except Exception as e:
        self.logger.warning("[Meta:PositionCount] Failed to count positions: %s", e)
    
    return 0
```

**Location**: Add after `_reset_bootstrap_override_if_deadlocked()` method (around line ~440)

---

### Step 3: Add Capital Governor Check in _execute_decision() (Lines ~10882+)

In the `_execute_decision()` method, add this check right after the P9 readiness gate (around line ~10930):

**Find This Section** (existing code):
```python
    async def _execute_decision(self, symbol: str, side: str, signal: Dict[str, Any], accepted_symbols_set: set):
        """Standardized execution path with P9-aligned readiness and trade frequency gating."""
        
        # ...existing code...
        
        # P9 HARD READINESS GATE (around line 10920)
        if side == "BUY":
            md_ready = False
            as_ready = False
            # ...existing P9 gate logic...
            if not (md_ready and as_ready):
                self.logger.warning(...)
                return {"ok": False, "status": "skipped", "reason": "p9_readiness_gate"}
```

**Add AFTER the P9 gate check** (~10935):

```python
            # ─────────────────────────────────────────────────────────────
            # CAPITAL GOVERNOR: Position Limit Check (NEW - Phase B)
            # ─────────────────────────────────────────────────────────────
            try:
                nav = float(getattr(self.shared_state, "nav", 0.0) or 
                           getattr(self.shared_state, "total_value", 0.0) or 0.0)
                
                limits = self.capital_governor.get_position_limits(nav)
                max_positions = limits.get("max_concurrent_positions", 1)
                open_positions = self._count_open_positions()
                
                if open_positions >= max_positions:
                    self.logger.warning(
                        "[Meta:CaptialGovernor] Blocking BUY %s: Position limit reached (%d/%d open)",
                        symbol, open_positions, max_positions
                    )
                    return {"ok": False, "status": "skipped", "reason": "position_limit_exceeded"}
                
                # Optional: Log remaining capacity
                remaining = max_positions - open_positions
                if remaining <= 1:
                    self.logger.warning(
                        "[Meta:CapitalGovernor] ⚠️ Position capacity low: %d/%d (only %d slot(s) remaining)",
                        open_positions, max_positions, remaining
                    )
                
            except Exception as e:
                self.logger.error("[Meta:CapitalGovernor] Position limit check failed: %s", e)
                # CRITICAL: Do NOT block on exception - let execution proceed with warning
                self.logger.warning("[Meta:CapitalGovernor] Proceeding with BUY (limit check failed)")
```

**Exact Location**: After the `if not (md_ready and as_ready):` block that ends with the readiness gate rejection (around line ~10932)

---

## Implementation Verification

### Test 1: Position Limit Blocking ($350 MICRO account)

**Setup**:
- Account NAV: $350 (MICRO bracket)
- Open positions: 1 (BTCUSDT)
- Governor max: 1 position

**Action**: Send BUY signal for ETHUSDT

**Expected**: 
```
[Meta:CapitalGovernor] Blocking BUY ETHUSDT: Position limit reached (1/1 open)
```
Signal rejected, no order placed.

---

### Test 2: Position Limit Allowing ($350 with 0 open)

**Setup**:
- Account NAV: $350 (MICRO bracket)
- Open positions: 0
- Governor max: 1 position

**Action**: Send BUY signal for BTCUSDT

**Expected**:
```
[Meta:CapitalGovernor] ✓ Position limit OK: 0/1 open, proceeding with BUY
```
Signal proceeds to execution gates.

---

### Test 3: Bracket Scaling ($2,500 SMALL account)

**Setup**:
- Account NAV: $2,500 (SMALL bracket)
- Open positions: 1
- Governor max: 2 positions

**Action**: Send BUY signal for ETHUSDT

**Expected**:
```
[Meta:CapitalGovernor] ✓ Position limit OK: 1/2 open
```
Signal proceeds (SMALL bracket allows 2 positions vs MICRO's 1).

---

## Code Changes Summary

| File | Lines | Change | Type |
|------|-------|--------|------|
| `meta_controller.py` | ~700 | Add `self.capital_governor = CapitalGovernor(config)` | Init |
| `meta_controller.py` | ~440 | Add `_count_open_positions()` helper | New method |
| `meta_controller.py` | ~10935 | Add position limit check in `_execute_decision()` | Validation gate |

**Total Changes**: 3 small, focused modifications  
**Lines Added**: ~50  
**Backward Compatible**: Yes (check only affects BUY, non-blocking on error)  

---

## Integration Points

### Where Capital Governor Now Checks

```python
# MetaController.evaluate_and_act()
#   → Loop through decisions
#     → _execute_decision() for each BUY/SELL
#       → [NEW] Capital Governor check here
#         ├─ get_position_limits(nav)
#         ├─ _count_open_positions()
#         └─ Compare & block if exceeded

# Data Flow:
NAV (from SharedState)
  ↓
CapitalGovernor.get_position_limits()
  ↓
Returns: {"max_concurrent_positions": 1, ...}
  ↓
Compared against: _count_open_positions() result
  ↓
Decision: Allow BUY or Reject with reason
```

---

## Logging Output

When Capital Governor blocks a BUY:

```
[Meta:CapitalGovernor] Blocking BUY ETHUSDT: Position limit reached (1/1 open)
_loop_summary_state["rejection_reason"] = "position_limit_exceeded"
```

When Capital Governor allows a BUY:

```
[Meta:CapitalGovernor] ✓ Position limit OK: 0/1 open, proceeding with BUY
# Continues to next execution gate
```

---

## Phase B Completion Checklist

- [ ] Add `self.capital_governor = CapitalGovernor(config)` in `__init__()`
- [ ] Add `_count_open_positions()` helper method
- [ ] Add position limit check in `_execute_decision()` after P9 gate
- [ ] Test: BUY blocked when limit reached
- [ ] Test: BUY allowed when positions available
- [ ] Test: Bracket-specific limits work ($350 vs $2,500 vs $10,000)
- [ ] Commit: `git commit -m "feat: Capital Governor position limits in MetaController (Phase B)"`
- [ ] Log output shows correct position counts
- [ ] No regressions in existing BUY/SELL execution

---

## Next Phases (After Phase B)

**Phase C**: Symbol Rotation Manager Integration
- Prevent rotation in MICRO bracket
- Restrict to core symbols only

**Phase D**: Position Manager Integration
- Use bracket-specific position sizing ($12 for MICRO)
- Apply EV multiplier per bracket

**Phase E**: End-to-End Testing
- Test complete flow: Governor permission → Allocator budget → Execution

---

## Troubleshooting

### Issue: Position count always returns 0

**Cause**: `shared_state.get_positions_snapshot()` not available

**Solution**: Add fallback in `_count_open_positions()`:
```python
# Try agent_manager's position tracking
if hasattr(self.agent_manager, "get_all_positions"):
    positions = self.agent_manager.get_all_positions()
    return len([p for p in positions if float(p.get("quantity", 0.0)) > 0])
```

### Issue: Capital Governor not initialized

**Cause**: Missing import or config not passed

**Solution**:
1. Verify: `from core.capital_governor import CapitalGovernor` at top
2. Check: `config` parameter exists in `__init__()`
3. Log: Add `self.logger.info("[Meta:Init] Capital Governor initialized")` to verify

### Issue: Position limit not enforced

**Cause**: Check for exception silencing

**Solution**: Look for logs:
```
[Meta:CapitalGovernor] Position limit check failed
```
If present, add print statements in try/except to debug the actual error.

---

## Files Modified

✅ `core/meta_controller.py` - 3 modification points

## Files Created

None (reuses existing Capital Governor from Phase A)

## Files Committed

After implementation:
```bash
git add core/meta_controller.py
git commit -m "feat: Capital Governor position limits in MetaController (Phase B)"
git push origin main
```

---

## Expected Behavior After Phase B

Your $350 MICRO account will now:

1. ✅ Allow first BUY (0 < 1 position)
2. ✅ Block second BUY (1 = 1 position limit)
3. ✅ Allow next BUY after SELL (0 < 1)
4. ✅ Log warnings when approaching limits

This is the **structure enforcement** layer - ensuring your account trades 1 symbol at a time, focused learning.

The **budget enforcement** (Capital Allocator) comes in Phase F.
