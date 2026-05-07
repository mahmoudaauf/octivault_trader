# Quick Reference Guide — May 7, 2026

## Current Status (15:58 UTC)

### Automated Test Running
```bash
# Process: test_after_throttle_expires.py
# Status: Monitoring throttle countdown
# Log file: throttle_expiry_test.log
# Expected completion: ~15:20 UTC (when throttle expires)
```

### Monitor in Real-Time
```bash
# Watch countdown every 5 seconds
tail -f throttle_expiry_test.log

# Or check progress every 10 seconds
watch -n 10 'tail -20 throttle_expiry_test.log'

# Or check if process is running
ps aux | grep "test_after_throttle_expires" | grep -v grep
```

---

## What's Happening Now

### Timeline
```
Current time: 15:58 UTC
Throttle expires: 15:19:53 UTC (May 7)
Time remaining: ~21 minutes
Test status: Counting down every 5 seconds
```

### What the Test Will Do
```
1. Monitor throttle state (exchange_throttle_until_ts)
2. When throttle_ts <= time.time(), automatically transition
3. Run 100 trading cycles (should take ~7 seconds)
4. Verify:
   ├─ NAV > 0 (balance synced)
   ├─ Zero throttle errors
   ├─ Signals generated
   ├─ Positions created
   └─ System stable
5. Save results to THROTTLE_EXPIRY_TEST_RESULTS.md
```

---

## Expected Results (Best Case)

```
✅ Throttle Expired!
✅ Starting 100-cycle test...

Cycle | NAV ($) | Positions | Status
────────────────────────────────────
    1 | $50.23  | 0         | OK
   10 | $51.00  | 1         | BUY
   20 | $51.80  | 2         | OK
   30 | $52.10  | 1         | SELL
   ...
  100 | $55.00  | 1         | OK

Duration: 7.2s
NAV Start: $50.23
NAV End: $55.00
NAV Peak: $55.50
Errors: 0
Throttle Errors: 0

✅ VERDICT:
✅ All four throttle fixes are working correctly!
✅ Capital compounding working! NAV growing.
```

---

## What Each Part Means

### Throttle Expired ✅
- System detected that `exchange_throttle_until_ts <= current_time`
- Old ban timestamp has passed
- Fix 1 (bootstrap expiry check) would have cleared it

### NAV > $0 ✅
- Balance fetch succeeded after throttle expired
- Fix 4 (initial balance defer) allowed balance sync to resume
- System can now trade with real capital

### Signals Generated ✅
- Multi-timeframe indicator analysis running
- System found trading opportunities
- Confidence scores computed

### Positions Created ✅
- BUY decisions executed
- Orders placed on Binance
- Capital deployed

### Zero Throttle Errors ✅
- No fresh 418 bans detected
- All four throttle protection layers working
- Safe API weight maintained (< 1200/min)

---

## What to Watch For

### Good Signs ✅
- Log shows "Throttle expires in: 0m 0s" then "✅ Throttle expired!"
- Cycles complete 100/100
- NAV starts > $0
- NAV changes (growth)
- No "418" or "throttle" error messages

### Bad Signs ❌
- Test hangs (doesn't transition from countdown)
- NAV stays $0 (balance didn't sync)
- "418 error" or "throttled" in output
- Cycles < 100/100
- Process crashes (exit code != 0)

---

## If Something Goes Wrong

### Test Hangs (Countdown Doesn't End)
```bash
# Check throttle state manually
python3 -c "
import json, time
from pathlib import Path
state = json.loads(Path('runtime_state_snapshot.json').read_text())
ts = state['exchange_throttle_until_ts']
print(f'Throttle expires at: {ts}')
print(f'Current time:      {time.time()}')
print(f'Remaining:         {ts - time.time():.0f}s')
"

# Kill test and restart if needed
pkill -f "test_after_throttle_expires"
python3 test_after_throttle_expires.py 2>&1 | tee throttle_expiry_test.log &
```

### NAV Stays $0
```bash
# Check if polling_coordinator started
grep -i "polling\|balance" throttle_expiry_test.log | head -20

# Check if Fix 4 deferred balance fetch
grep -i "throttled at startup\|deferring" throttle_expiry_test.log

# This is EXPECTED if throttle is still active
# NAV should sync once throttle expires
```

### 418 Errors Appear
```bash
# Extract all error lines
grep -i "418\|throttle.*error\|exception" throttle_expiry_test.log

# This would indicate a problem with one of the fixes
# But we expect ZERO 418 errors
```

---

## Files to Check

### Main Results
```
throttle_expiry_test.log
├─ Real-time output from test
├─ Updated every 5 seconds during countdown
└─ Contains full cycle output when test runs

THROTTLE_EXPIRY_TEST_RESULTS.md
├─ Created after test completes
├─ Contains summary metrics
├─ Indicates success/failure
└─ Can be compared with ASSESSMENT_RESULTS_MAY_7.md
```

### System State
```
runtime_state_snapshot.json
├─ Current throttle_until_ts
├─ Current balance
├─ Current positions
└─ Updated every cycle

objective_controller_state.json
├─ Created if OFC is running (not yet, future phase)
└─ Tracks adaptive parameters
```

---

## Commands for Monitoring

### Watch Test Progress
```bash
tail -f throttle_expiry_test.log
```

### Extract Key Metrics
```bash
grep -E "Throttle|NAV|Error|Duration" throttle_expiry_test.log
```

### Count by Type
```bash
# Count throttle countdowns
grep "Throttle expires in" throttle_expiry_test.log | wc -l

# Count cycles (when test runs)
grep "^[0-9].*\|" throttle_expiry_test.log | wc -l

# Count errors
grep -i "error\|fail\|exception" throttle_expiry_test.log | wc -l
```

### Compare with Previous Test
```bash
# Show differences between old and new results
diff ASSESSMENT_RESULTS_MAY_7.md THROTTLE_EXPIRY_TEST_RESULTS.md
```

---

## After Test Completes

### If Successful ✅
```bash
# Review results
cat THROTTLE_EXPIRY_TEST_RESULTS.md

# Commit the verification
git add THROTTLE_EXPIRY_TEST_RESULTS.md
git commit -m "test: Verify all throttle fixes working end-to-end"

# Start Phase 2 planning (ACE + OFC)
cat NEXT_PHASE_PLAN.md
```

### If Failed ❌
```bash
# Check error log
grep -i "error\|fail\|exception" throttle_expiry_test.log | head -20

# Check throttle state
python3 -c "
import json
from pathlib import Path
state = json.loads(Path('runtime_state_snapshot.json').read_text())
print(f'Throttled: {state[\"exchange_throttled\"]}')
print(f'Until TS: {state[\"exchange_throttle_until_ts\"]}')
print(f'Reason: {state[\"exchange_throttle_reason\"]}')
"

# May need to debug specific phase
grep -E "Phase|phase" throttle_expiry_test.log | tail -10
```

---

## Key Commits for Reference

```
96ee86a fix: Check throttle before initial balance fetch
a81b24e fix: Implement three-layer throttle state management
```

## Architecture Documents

**New (May 7)**:
- `SYSTEM_ARCHITECTURE_MAY_7_2026.md` — Full system architecture with throttle fixes
- `THROTTLE_FIXES_FINAL_SUMMARY.md` — Complete throttle solution explanation
- `NEXT_PHASE_PLAN.md` — ACE + OFC integration roadmap

**Previous**:
- `ASSESSMENT_RESULTS_MAY_7.md` — 100-cycle test results (before expiry)
- `FIXES_ARCHITECTURE_DIAGRAM.md` — Visual diagrams of fixes

---

## Next Steps (After Test)

1. **Review results** (5 minutes)
   - Check THROTTLE_EXPIRY_TEST_RESULTS.md
   - Verify NAV > 0, zero 418 errors

2. **Plan Phase 2** (15 minutes)
   - Review NEXT_PHASE_PLAN.md
   - Decide: implement ACE + OFC now?

3. **ACE Integration** (2-4 hours)
   - Copy src/l6_governance/adaptive_capital_engine.py
   - Wire trade_history tracking
   - Test intelligent risk sizing

4. **OFC Integration** (2-4 hours)
   - Copy src/l5_strategy/objective_feedback_controller.py
   - Wire runtime_overrides
   - Test self-tuning

5. **Live 4-8 Hour Test** (4-8 hours)
   - Run full trading session
   - Verify 1-2% NAV growth per hour
   - Monitor for edge cases

---

## TL;DR

- **What's running**: Automatic throttle expiry test
- **Where**: `throttle_expiry_test.log`
- **When it'll finish**: 15:19:53 UTC (when throttle expires)
- **How long**: ~7 seconds for 100 cycles
- **What to expect**: NAV > 0, zero throttle errors
- **Success metric**: All 100 cycles complete, NAV growing
- **Next phase**: ACE + OFC integration if test passes

---

**Last Updated**: May 7, 2026, 15:58 UTC
**Test Status**: Running (21 min countdown)
**Next Update**: When test completes (~15:20-15:30 UTC)
