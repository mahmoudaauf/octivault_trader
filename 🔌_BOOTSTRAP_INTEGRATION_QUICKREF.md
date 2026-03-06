# 🔌 Bootstrap Signal Validation - MetaController Integration Guide

**Quick Reference for Integration**

---

## Where to Add the Call

**File**: `core/meta_controller.py`  
**Method**: `propose_exposure_directive()` or your signal validation method  
**Location**: Right AFTER validation gates pass, BEFORE execution  

---

## Code Pattern

### Current Pattern (Broken in Shadow Mode)
```python
async def propose_exposure_directive(self, directive):
    # ... validation logic ...
    
    meta_approved = await self.should_place_buy(symbol=symbol, ...)
    
    if meta_approved:
        # Execute immediately
        result = await self.execute_via_execution_manager(directive)
        return result
    else:
        return {"ok": False}
```

**Problem**: Bootstrap only completes when `result` has a successful trade. In shadow mode, no trade executes, so bootstrap never completes.

---

### Fixed Pattern (Works in Shadow Mode)
```python
async def propose_exposure_directive(self, directive):
    # ... validation logic ...
    
    meta_approved = await self.should_place_buy(symbol=symbol, ...)
    
    if meta_approved:
        # 🔧 NEW: Mark bootstrap complete (signal validated)
        # This must happen BEFORE execution (or skip)
        # to prevent shadow mode deadlock
        self.shared_state.mark_bootstrap_signal_validated()
        
        # Now execute (or skip if shadow mode)
        result = await self.execute_via_execution_manager(directive)
        return result
    else:
        return {"ok": False}
```

**Fix**: Mark bootstrap on signal validation, not trade execution. Works in all modes.

---

## Integration Points

### 1. In `propose_exposure_directive()` Method

```python
if meta_approved:
    # ✅ Mark bootstrap complete on signal validation
    self.shared_state.mark_bootstrap_signal_validated()
    
    # Execute
    result = await self.execute_via_execution_manager(directive)
```

### 2. In Other Signal Generation Methods

If you have other places where signals are validated (e.g., `add_agent_signal()`), also call:

```python
# After validation passes
await self.shared_state.mark_bootstrap_signal_validated()
```

### 3. Search Pattern

```bash
# Find all locations where signals are validated:
grep -r "should_place_buy\|should_place_sell\|meta_approved\|validation_passed" \
  core/meta_controller.py | head -20
```

Add the call after each validation that passes ✅

---

## Testing the Integration

### Step 1: Verify the Method Exists

```python
# In MetaController.__init__() or test:
assert hasattr(self.shared_state, 'mark_bootstrap_signal_validated'), \
    "mark_bootstrap_signal_validated not found in SharedState"
```

### Step 2: Test Shadow Mode

```bash
# Start in shadow mode
TRADING_MODE=shadow python main.py

# In logs, look for:
# [BOOTSTRAP] ✅ Bootstrap completed by first signal validation
```

### Step 3: Verify Bootstrap Only Fires Once

```bash
# Count bootstrap-related log lines:
grep -c "Bootstrap\|is_cold_bootstrap" logs/*.log

# Should be small (e.g., 1-2 times), not hundreds of times
```

### Step 4: Test Live Mode

```bash
# Start in live mode
python main.py

# Should complete bootstrap on first signal validation ✅
# Then proceed normally with trade execution
```

---

## Error Handling

### If Method Not Found

```
AttributeError: 'SharedState' object has no attribute 'mark_bootstrap_signal_validated'
```

**Fix**: Verify the code change was applied to `core/shared_state.py`:
```bash
grep -n "mark_bootstrap_signal_validated" core/shared_state.py
```

### If Bootstrap Still Active

```
# Logs show bootstrap firing repeatedly
[BOOTSTRAP] Confidence override: increasing to 0.95
[BOOTSTRAP] Forced seed trade: placing initial capital...
```

**Fix**: Verify you're calling the method in the right place:
```bash
# Search for all validation entry points
grep -n "meta_approved\|validation_passed" core/meta_controller.py

# Add the call after each one
```

---

## Timeline

| Step | Action | Status |
|------|--------|--------|
| 1 | Add `mark_bootstrap_signal_validated()` to SharedState | ✅ Done |
| 2 | Modify `is_cold_bootstrap()` to check signal validation | ✅ Done |
| 3 | Call method in MetaController.propose_exposure_directive() | ⏳ TODO |
| 4 | Test in shadow mode | ⏳ TODO |
| 5 | Test in live mode | ⏳ TODO |
| 6 | Deploy | ⏳ TODO |

---

## Quick Copy-Paste

```python
# Add this import if not already present
from core.shared_state import SharedState

# In propose_exposure_directive() or your signal validation method:
if meta_approved:  # After validation passes
    # 🔧 Mark bootstrap complete (signal validated)
    # Prevents shadow mode deadlock
    self.shared_state.mark_bootstrap_signal_validated()
    
    # Now proceed with execution (or skip in shadow mode)
    result = await self.execute_via_execution_manager(directive)
```

---

## Validation Checklist

- [ ] Located `propose_exposure_directive()` in MetaController
- [ ] Found the validation gate (`meta_approved = await ...`)
- [ ] Added `mark_bootstrap_signal_validated()` call right after gate passes
- [ ] Confirmed method is called BEFORE execution attempt
- [ ] Compiled and no syntax errors
- [ ] Tested in shadow mode - bootstrap completes ✅
- [ ] Tested in live mode - bootstrap completes on first valid signal ✅
- [ ] Verified bootstrap logic stops firing after first signal ✅

---

## Common Issues

### Issue: "Is this the only place to add the call?"

**Answer**: Add it in the main signal validation path. If there are multiple:
- `propose_exposure_directive()` - YES, add here
- `add_agent_signal()` - Maybe, if it validates
- `should_place_buy()` / `should_place_sell()` - No, these are validators, not decision points

**Best practice**: Add it once, right after the main validation gate in the decision method.

---

### Issue: "What if validation fails?"

**Answer**: DON'T call the method if validation fails. Only call if `meta_approved = True`:

```python
if meta_approved:  # ✅ Validation PASSED
    self.shared_state.mark_bootstrap_signal_validated()  # ✅ Mark complete
else:  # ✗ Validation FAILED
    # Don't mark - bootstrap stays active
    return {"ok": False, "reason": "validation_failed"}
```

---

### Issue: "Multiple validation gates?"

**Answer**: Use the LAST gate (closest to execution):

```python
if gate1_passed:
    if gate2_passed:
        if gate3_passed:
            # Call here (all gates passed)
            self.shared_state.mark_bootstrap_signal_validated()
            result = await self.execute(...)
```

---

## Questions?

Refer to the full guide: `🔧_BOOTSTRAP_SIGNAL_VALIDATION_FIX.md`

Key concept: **Bootstrap completes on signal validation, not trade execution.**

---

**Status**: Ready for integration 🟢
