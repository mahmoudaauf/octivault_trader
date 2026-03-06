# 🚀 Bootstrap Signal Validation Fix - Deployment Guide

**Target**: Complete shadow mode bootstrap deadlock fix  
**Status**: Code changes ✅ applied, Integration ⏳ pending  
**ETA**: ~15 minutes to complete and test  

---

## Pre-Deployment Checklist

- [ ] Read all three documentation files:
  - `✅_BOOTSTRAP_FIX_IMPLEMENTATION_SUMMARY.md`
  - `🔧_BOOTSTRAP_SIGNAL_VALIDATION_FIX.md`
  - `🔌_BOOTSTRAP_INTEGRATION_QUICKREF.md`

- [ ] Verify code changes applied to `core/shared_state.py`:
  ```bash
  grep -n "mark_bootstrap_signal_validated" core/shared_state.py
  ```
  Should show: method definition around line 5818

- [ ] Verify is_cold_bootstrap() check added:
  ```bash
  grep -A2 "first_signal_validated_at" core/shared_state.py | head -5
  ```
  Should show: check in is_cold_bootstrap() around line 5879

---

## Deployment Steps

### Step 1: Integration (5 min)

**File**: `core/meta_controller.py`  
**Method**: `propose_exposure_directive()`

1. Open the file:
   ```bash
   nano core/meta_controller.py
   ```

2. Find the `propose_exposure_directive()` method (use Ctrl+F for "propose_exposure_directive")

3. Locate the signal validation gate:
   ```python
   meta_approved = await self.should_place_buy(
       symbol=symbol,
       ...
   )
   ```

4. Right after this validation, add:
   ```python
   if meta_approved:
       # 🔧 Mark bootstrap complete on signal validation
       # This prevents shadow mode deadlock by completing bootstrap
       # on signal validation, not trade execution
       self.shared_state.mark_bootstrap_signal_validated()
       
       # Now execute the directive
       result = await self.execute_via_execution_manager(directive)
   ```

5. Save and exit (Ctrl+X, then Y)

6. **Verify syntax**:
   ```bash
   python -m py_compile core/meta_controller.py
   # Should exit with code 0 (no errors)
   ```

---

### Step 2: Pre-Flight Testing (5 min)

**Before deploying to production**

#### Test 1: Syntax Check
```bash
python -c "from core.shared_state import SharedState; print('OK')"
```
Expected: `OK` (no errors)

#### Test 2: Method Exists
```bash
python -c "
from core.shared_state import SharedState
import inspect
assert hasattr(SharedState, 'mark_bootstrap_signal_validated')
print('✓ Method found')
"
```
Expected: `✓ Method found`

#### Test 3: Import Check
```bash
python -c "from core import meta_controller; print('✓ Imports OK')"
```
Expected: `✓ Imports OK`

---

### Step 3: Local Testing (10 min)

#### Test 3a: Shadow Mode
```bash
# Start in shadow mode
TRADING_MODE=shadow python main.py > /tmp/shadow.log 2>&1 &
SHADOW_PID=$!

# Wait 30 seconds for startup and first signal
sleep 30

# Check for bootstrap completion
grep "Bootstrap completed by first signal" /tmp/shadow.log
# Expected: Should find the message

# Check bootstrap logic didn't re-fire
BOOTSTRAP_COUNT=$(grep -c "Bootstrap\|is_cold_bootstrap" /tmp/shadow.log)
# Expected: Small number (< 5)

# Kill shadow mode
kill $SHADOW_PID 2>/dev/null
```

#### Test 3b: Live Mode
```bash
# Start in live mode
python main.py > /tmp/live.log 2>&1 &
LIVE_PID=$!

# Wait 30 seconds
sleep 30

# Check bootstrap worked
grep "Bootstrap" /tmp/live.log
# Expected: Should see bootstrap messages or completion

# Verify no obvious errors
grep "ERROR\|CRITICAL" /tmp/live.log | head -3
# Expected: Should be minimal or none

# Kill live mode
kill $LIVE_PID 2>/dev/null
```

---

### Step 4: Commit & Deploy

#### If Tests Pass
```bash
# Commit changes
git add core/shared_state.py core/meta_controller.py

git commit -m "🔧 Fix: Bootstrap completion on signal validation (prevents shadow mode deadlock)

- Add mark_bootstrap_signal_validated() to SharedState
- Modify is_cold_bootstrap() to check signal validation
- Call method in MetaController.propose_exposure_directive()
- Shadow mode no longer deadlocks in bootstrap phase
- Persistence survives restart

Fixes: Shadow mode infinite bootstrap loop
Related: TRADING_MODE=shadow environment"

# Push to main
git push origin main
```

#### If Tests Fail
```bash
# Revert changes
git checkout core/meta_controller.py
git checkout core/shared_state.py

# Debug
# See troubleshooting section below
```

---

## Post-Deployment Monitoring

### Hour 1: Immediate Monitoring
```bash
# Watch logs for bootstrap behavior
tail -f logs/octivault_trader.log | grep -i "bootstrap\|signal_validated"

# Look for:
# ✅ "[BOOTSTRAP] ✅ Bootstrap completed by first signal validation"
# ❌ Repeated bootstrap messages
# ❌ Syntax errors

# Kill with Ctrl+C
```

### Hour 2-4: System Health
- Monitor error rates
- Check for infinite loops
- Verify shadow mode doesn't deadlock
- Ensure live mode still works

### Daily: Long-Term Monitoring
- Check for bootstrap re-entry (should only happen once per lifetime)
- Verify metrics persistence across restarts
- Monitor for any bootstrap-related errors

---

## Rollback Plan

If issues occur, rollback is safe:

```bash
# Immediate rollback
git revert HEAD

# Or manual revert
git checkout HEAD~1 core/meta_controller.py core/shared_state.py

# Restart system
pkill -f octivault_trader
# Wait 5 seconds
python main.py
```

**Note**: System will revert to old behavior (bootstrap triggers on trade execution instead of signal validation). In live mode, this still works. Shadow mode may deadlock again, but system is stable.

---

## Troubleshooting

### Issue: "mark_bootstrap_signal_validated not found"

**Cause**: Code changes to SharedState not applied

**Fix**:
```bash
# Verify file has the changes
grep -n "mark_bootstrap_signal_validated" core/shared_state.py

# If not found, manually apply changes from documentation
# Or reapply file from version control
```

### Issue: "Shadow mode still deadlocks"

**Cause**: Method not called in MetaController

**Fix**:
```bash
# Verify MetaController call
grep -n "mark_bootstrap_signal_validated" core/meta_controller.py

# If not found, add it in propose_exposure_directive()
# Follow integration guide in 🔌_BOOTSTRAP_INTEGRATION_QUICKREF.md
```

### Issue: "Syntax errors in MetaController"

**Cause**: Improper indentation or placement

**Fix**:
```bash
# Check syntax
python -m py_compile core/meta_controller.py

# If error, review the changes:
git diff core/meta_controller.py

# Manually fix indentation:
# - Use 4 spaces per level
# - Match indentation of surrounding code
# - Verify bracket/parenthesis matching
```

### Issue: "Bootstrap fires multiple times"

**Cause**: Method not idempotent or called in wrong place

**Fix**:
```bash
# Verify method checks for existing first_signal_validated_at:
grep -A5 "def mark_bootstrap_signal_validated" core/shared_state.py

# Should show:
#   if self.metrics.get("first_signal_validated_at") is not None:
#       return  # Already marked

# If missing, check that changes were fully applied
```

---

## Verification Checklist (Post-Deployment)

- [ ] Syntax compiles without errors
- [ ] Shadow mode starts without deadlock
- [ ] First signal validation is logged
- [ ] Bootstrap logic stops firing after signal
- [ ] Live mode trades normally
- [ ] System restarts preserve bootstrap completion
- [ ] No new errors in logs
- [ ] Performance is normal
- [ ] All metrics recorded correctly

---

## Key Metrics to Monitor

### In Logs
```
✅ SHOULD SEE:
[BOOTSTRAP] ✅ Bootstrap completed by first signal validation at 1234567890.5

❌ SHOULD NOT SEE:
[BOOTSTRAP] Confidence override: increasing to 0.95
[BOOTSTRAP] Forced seed trade...
(repeated bootstrap messages)
```

### In Metrics
```python
# Should have at least one of these set:
metrics["first_signal_validated_at"]  # New in this fix
metrics["first_trade_at"]             # Existing

# Should be True after bootstrap
metrics["bootstrap_completed"]  # True
```

### In Events
```
Look for: "BootstrapSignalValidated" event
Expected: Once per startup (not repeated)
```

---

## Timeline

| Time | Activity | Status |
|------|----------|--------|
| -5 min | Read documentation | ⏳ Now |
| 0 min | Begin integration | ⏳ Next |
| 5 min | Add call to MetaController | ⏳ 5 min |
| 10 min | Test syntax | ⏳ 10 min |
| 15 min | Shadow mode test | ⏳ 15 min |
| 20 min | Live mode test | ⏳ 20 min |
| 25 min | Deploy if tests pass | ⏳ 25 min |
| 30 min | Monitor hour 1 | ⏳ 30 min |

---

## Success Criteria

✅ Fix is successful if:

1. **Shadow mode**: Starts without deadlock, completes bootstrap on first signal ✅
2. **Live mode**: Trades normally, bootstrap still works ✅
3. **Restart**: Preserves bootstrap completion in metrics file ✅
4. **Logs**: Show bootstrap completed, logic stops firing ✅
5. **Performance**: No degradation, no new errors ✅

---

## Support

### Quick Reference
- Fast integration: See `🔌_BOOTSTRAP_INTEGRATION_QUICKREF.md`
- Full details: See `🔧_BOOTSTRAP_SIGNAL_VALIDATION_FIX.md`
- Implementation: See `✅_BOOTSTRAP_FIX_IMPLEMENTATION_SUMMARY.md`

### Key Insight
**Bootstrap now completes on signal validation, not trade execution.**

This fixes shadow mode deadlock while maintaining compatibility with live mode.

---

## Next Steps

1. **NOW**: Read the three documentation files
2. **NEXT**: Add one line to MetaController
3. **THEN**: Test shadow mode
4. **THEN**: Test live mode
5. **FINALLY**: Deploy and monitor

---

**Estimated total time: 25-30 minutes**

**Status**: Ready to deploy 🟢

---

*Questions? See the full guides or contact the system architect.*
