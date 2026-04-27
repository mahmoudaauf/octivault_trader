# ✅ ACTION CHECKLIST - Fix Logs Issues

**Priority**: URGENT  
**Date**: April 27, 2026  
**Status**: Ready to Execute

---

## 🎯 ROOT CAUSE

Phase 2 deployment claimed to update 8 parameters to 25 USDT, but they're still at 15 USDT in the `.env` file.

This single issue is causing:
- ❌ 424 balance allocation errors
- ❌ 43% quote mismatch in execution
- ❌ 1.8 million debug warning spam

---

## ✅ STEP-BY-STEP CHECKLIST

### STEP 1: STOP BOT ✓

```bash
pkill -f MASTER_SYSTEM_ORCHESTRATOR
sleep 2
# Verify stopped:
pgrep -f MASTER_SYSTEM_ORCHESTRATOR
# Should return nothing (exit code 1)
```

**✓ Checkbox**: [ ] Bot stopped

---

### STEP 2: UPDATE .ENV FILE ✓

**File**: `/Users/mauf/Desktop/Octi AI Trading Bot/octivault_trader/.env`

**Lines to Update**:

**Lines 44-52** (BUY SIZING section):
```
Change from 15 to 25:
- Line 44: DEFAULT_PLANNED_QUOTE=15 → DEFAULT_PLANNED_QUOTE=25
- Line 45: MIN_TRADE_QUOTE=15 → MIN_TRADE_QUOTE=25
- Line 47: MIN_ENTRY_USDT=15 → MIN_ENTRY_USDT=25
- Line 48: TRADE_AMOUNT_USDT=15 → TRADE_AMOUNT_USDT=25
- Line 49: MIN_ENTRY_QUOTE_USDT=15 → MIN_ENTRY_QUOTE_USDT=25
- Line 50: EMIT_BUY_QUOTE=15 → EMIT_BUY_QUOTE=25
- Line 51: META_MICRO_SIZE_USDT=15 → META_MICRO_SIZE_USDT=25
```

**Line 140** (STRATEGY THRESHOLDS section):
```
- Line 140: MIN_SIGNIFICANT_POSITION_USDT=12 → MIN_SIGNIFICANT_POSITION_USDT=25
```

**✓ Checkbox**: [ ] All 8 parameters updated to 25

---

### STEP 3: VERIFY CHANGES ✓

```bash
# Verify the changes were made:
grep "=25\|=15" /Users/mauf/Desktop/Octi\ AI\ Trading\ Bot/octivault_trader/.env | \
  grep -E "DEFAULT_PLANNED_QUOTE|MIN_TRADE_QUOTE|MIN_ENTRY_USDT|TRADE_AMOUNT_USDT|MIN_ENTRY_QUOTE_USDT|EMIT_BUY_QUOTE|META_MICRO_SIZE_USDT|MIN_SIGNIFICANT_POSITION_USDT"
```

**Expected Output** (all should show 25):
```
DEFAULT_PLANNED_QUOTE=25
MIN_TRADE_QUOTE=25
MIN_ENTRY_USDT=25
TRADE_AMOUNT_USDT=25
MIN_ENTRY_QUOTE_USDT=25
EMIT_BUY_QUOTE=25
META_MICRO_SIZE_USDT=25
MIN_SIGNIFICANT_POSITION_USDT=25
```

**✓ Checkbox**: [ ] Changes verified (all 8 = 25)

---

### STEP 4: CLEAR OLD LOG FILE (OPTIONAL) ✓

```bash
# Archive old log (381 MB):
mv /tmp/octivault_master_orchestrator.log \
   /tmp/octivault_master_orchestrator.log.backup.$(date +%Y%m%d_%H%M%S)

# OR just truncate it:
> /tmp/octivault_master_orchestrator.log
```

**✓ Checkbox**: [ ] Old logs archived/cleared

---

### STEP 5: START BOT WITH CORRECTED CONFIG ✓

```bash
# Navigate to workspace
cd /Users/mauf/Desktop/Octi\ AI\ Trading\ Bot/octivault_trader

# Start bot with live trading approval
export APPROVE_LIVE_TRADING=YES
nohup python3 🎯_MASTER_SYSTEM_ORCHESTRATOR.py > /tmp/octivault_master_orchestrator.log 2>&1 &

# Get the PID (should be a new number):
sleep 2
pgrep -f MASTER_SYSTEM_ORCHESTRATOR
```

**Expected**: Process ID should be printed (e.g., 12345)

**✓ Checkbox**: [ ] Bot restarted with corrected config

---

### STEP 6: MONITOR INITIALIZATION (5 minutes) ✓

```bash
# Watch logs for initialization:
tail -f /tmp/octivault_master_orchestrator.log

# In another terminal, after 30 seconds, check for errors:
grep -i "error\|critical" /tmp/octivault_master_orchestrator.log | head -20
```

**✓ Checkbox**: [ ] Monitored initialization (no critical errors)

---

### STEP 7: VERIFY FIXES ✓

After 5 minutes, run these checks:

**Check 1: No allocation errors?**
```bash
grep "Invalid allocation amount" /tmp/octivault_master_orchestrator.log | wc -l
# Should be 0 (or very few, not hundreds)
```
**✓ Checkbox**: [ ] Allocation errors resolved (count < 5)

**Check 2: Entry sizing at 25.00?**
```bash
grep "quote=25.00" /tmp/octivault_master_orchestrator.log | head -10
# Should show multiple entries with quote=25.00
```
**✓ Checkbox**: [ ] Entry sizing showing 25.00 USDT

**Check 3: Quote mismatch resolved?**
```bash
grep "quote mismatch\|planned=.*execute=" /tmp/octivault_master_orchestrator.log | tail -5
# Should show planned ≈ executed (not 11.57 vs 20.18)
```
**✓ Checkbox**: [ ] Quote mismatch resolved

**Check 4: System healthy?**
```bash
grep "\[INFO.*MetaController\]\|System ready\|main loop" /tmp/octivault_master_orchestrator.log | tail -3
# Should show system running normally
```
**✓ Checkbox**: [ ] System running normally

---

### STEP 8: DISABLE DEBUG LOGGING (Follow-up) ✓

**After bot is running well**, disable the DEBUG spam:

**File**: `core/shared_state.py`

**Find**: Lines with `[DEBUG:CLASSIFY]` logging

**Change**: Set logging level to ERROR (or disable completely)

```python
# Before:
self.logger.warning(f"[DEBUG:CLASSIFY] {symbol} qty=...")

# After:
# self.logger.warning(f"[DEBUG:CLASSIFY] {symbol} qty=...")  # DISABLED
# OR
if DEBUG_LOGGING:  # New flag
    self.logger.warning(f"[DEBUG:CLASSIFY] {symbol} qty=...")
```

**✓ Checkbox**: [ ] Debug logging disabled (optional, do later)

---

## 📋 FINAL VERIFICATION

After completing all steps, verify:

- [x] Bot restarted
- [x] .env updated (all 8 parameters = 25)
- [x] No allocation errors
- [x] Entry sizing = 25.00
- [x] Quote mismatch resolved
- [x] System running normally
- [x] Log file size reasonable (~50-100 MB over time, not 381 MB)

---

## ⏱️ ESTIMATED TIME

- Stop bot: 1 minute
- Update .env: 5 minutes
- Verify changes: 1 minute
- Clear logs: 1 minute
- Restart bot: 2 minutes
- Monitor: 5 minutes
- Verify fixes: 5 minutes

**Total: ~20 minutes**

---

## 🆘 IF ISSUES PERSIST

If after this checklist you still see:
- Allocation errors
- Quote mismatches
- Other problems

Then:
1. Check account balance (should be > $100 USDT recommended)
2. Review other .env parameters that might conflict
3. Check for any other manual edits made to .env
4. Review the full LOG_ANALYSIS_REPORT.md for more details

---

## 📞 DOCUMENTATION

- **Full Analysis**: LOG_ANALYSIS_REPORT.md
- **Quick Summary**: LOGS_SUMMARY.md
- **This Checklist**: ACTION_CHECKLIST.md

---

**Status**: Ready to execute  
**Risk**: Low (config update only, no code changes)  
**Expected Outcome**: All 4 issues resolved  
**Time to Resolution**: ~20 minutes

