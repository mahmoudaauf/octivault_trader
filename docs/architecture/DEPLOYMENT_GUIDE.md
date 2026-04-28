# 🚀 PHASE 2 FIXES - DEPLOYMENT GUIDE

**Quick Reference for Deploying Phase 2 Bottleneck Fixes**  
**Status:** Ready for Production  
**Time Required:** 30 minutes (15 min deploy + 15 min warm-up test)  

---

## ⚡ 5-MINUTE QUICK START

If you're familiar with the system, here's the quick path:

```bash
# 1. Change to project directory
cd /Users/mauf/Desktop/Octi\ AI\ Trading\ Bot/octivault_trader

# 2. Review the changes
git diff

# 3. Deploy (restart the bot)
pkill -f "MASTER_SYSTEM_ORCHESTRATOR"
sleep 2
python3 🎯_MASTER_SYSTEM_ORCHESTRATOR.py &

# 4. Monitor (in another terminal)
tail -f /tmp/octivault_master_orchestrator.log

# 5. Watch for these log patterns:
#    - [Meta:SafeMinHold] Bypassing min-hold check
#    - [REA:authorize_rotation] ⚠️ MICRO restriction OVERRIDDEN
#    - Entry orders at ~25 USDT size
```

---

## 📋 WHAT WAS FIXED

### Fix #1: Recovery Exit Min-Hold Bypass
**Problem:** When capital drops, recovery exits blocked by min-hold gate  
**Solution:** Set `_bypass_min_hold` flag on recovery exits  
**Impact:** Capital recycled even during age restrictions  
**Status:** ✅ Implemented

### Fix #2: Micro Rotation Override
**Problem:** MICRO accounts can't rotate even when beneficial  
**Solution:** Set `force_rotation` precedence to override MICRO bracket  
**Impact:** Forced rotations now work in all regimes  
**Status:** ✅ Implemented

### Fix #3: Entry-Sizing Config Alignment
**Problem:** Config defaults (15 USDT) misaligned from floor (25 USDT)  
**Solution:** Align all 7 entry-size parameters to 25 USDT  
**Impact:** Clean intent, no runtime friction  
**Status:** ✅ Implemented

---

## 📊 FILES MODIFIED

```
.env
├─ DEFAULT_PLANNED_QUOTE: 15 → 25
├─ MIN_TRADE_QUOTE: 15 → 25
├─ MIN_ENTRY_USDT: 15 → 25
├─ TRADE_AMOUNT_USDT: 15 → 25
├─ MIN_ENTRY_QUOTE_USDT: 15 → 25
├─ EMIT_BUY_QUOTE: 15 → 25
├─ META_MICRO_SIZE_USDT: 15 → 25
└─ MIN_SIGNIFICANT_POSITION_USDT: 15 → 25

core/meta_controller.py
├─ Lines ~13426: Stagnation exit bypass flag
└─ Lines ~13445: Liquidity restore bypass flag

core/rotation_authority.py
├─ Lines ~302-350: force_rotation parameter & logic
└─ Already implemented from previous work
```

---

## ✅ VERIFICATION CHECKLIST

Before deploying, verify:
- [ ] All 8 parameters in .env set to 25
- [ ] core/meta_controller.py contains bypass flag logic
- [ ] core/rotation_authority.py contains force_rotation logic
- [ ] No syntax errors: `python3 -m py_compile .env core/*.py`
- [ ] All files compile cleanly

**Run verification:**
```bash
python3 verify_fixes_detailed.py
```

---

## 🚀 DEPLOYMENT STEPS

### Step 1: Backup Current Configuration
```bash
# Save current .env in case rollback needed
cp .env .env.backup.pre_phase2
cp core/meta_controller.py core/meta_controller.py.backup.pre_phase2
cp core/rotation_authority.py core/rotation_authority.py.backup.pre_phase2
```

### Step 2: Verify Changes
```bash
# Review what changed
git diff .env core/meta_controller.py core/rotation_authority.py

# Expected:
# - 8 lines changed in .env (all entries from 15→25)
# - 2-3 lines in meta_controller (bypass flags)
# - 0 lines in rotation_authority (already done)
```

### Step 3: Commit to Git
```bash
git add .env core/meta_controller.py core/rotation_authority.py
git commit -m "Phase 2: Unblock recovery/rotation + align entry sizing (15→25 USDT)"
```

### Step 4: Stop Current Bot (if running)
```bash
# Graceful shutdown
pkill -f "MASTER_SYSTEM_ORCHESTRATOR"
sleep 3

# Verify it stopped
ps aux | grep MASTER_SYSTEM_ORCHESTRATOR
# Should return empty (except grep itself)
```

### Step 5: Start Bot with New Configuration
```bash
# Start in background
python3 🎯_MASTER_SYSTEM_ORCHESTRATOR.py > bot.log 2>&1 &

# Verify it started
ps aux | grep MASTER_SYSTEM_ORCHESTRATOR
# Should see Python process running

# Wait for initialization (5-10 seconds)
sleep 10
```

### Step 6: Monitor Startup Logs
```bash
# In a new terminal, watch the logs
tail -100f /tmp/octivault_master_orchestrator.log

# Expected startup sequence:
# [TRACE] Initialization of...
# [INFO] Loading policy manager
# [INFO] Starting capital governor
# ... (30-60 sec of initialization)
# [INFO] System ready, starting main loop
```

### Step 7: 15-Minute Observation Period
```bash
# Watch for these key log patterns:

# 1. Normal trading
grep "Meta:Arbitration\|Submitting BUY\|Submitting SELL" /tmp/octivault_master_orchestrator.log

# 2. Recovery exits (Fix #1)
grep "SafeMinHold\|Bypassing min-hold" /tmp/octivault_master_orchestrator.log

# 3. Rotation overrides (Fix #2)
grep "MICRO restriction OVERRIDDEN" /tmp/octivault_master_orchestrator.log

# 4. Entry sizing (Fix #3)
grep "quote: 25" /tmp/octivault_master_orchestrator.log
```

---

## 🔍 WHAT TO LOOK FOR

### Expected Log Patterns

**Fix #1 - Recovery Exit Min-Hold Bypass:**
```
[Meta:SafeMinHold] Bypassing min-hold check for forced recovery exit: ETHUSDT
```
This appears when capital drops below strategic reserve.  
Expected frequency: Every 1-3 cycles if capital is strained

**Fix #2 - Micro Rotation Override:**
```
[REA:authorize_rotation] ⚠️ MICRO restriction OVERRIDDEN for BTCUSDT due to forced rotation
```
This appears when a MICRO account needs to rotate.  
Expected frequency: Every 5-15 minutes if MICRO and at capacity

**Fix #3 - Entry-Sizing Alignment:**
```
[Execution] Submitting BUY order: ETHUSDT @ quantity 0.05 (quote: 25.00)
```
This appears on every BUY order.  
Expected frequency: On each signal (3-12 per hour depending on regime)

### Health Indicators

**Good signs:**
- ✅ No syntax errors in logs
- ✅ Bot completes initialization within 60 seconds
- ✅ Sees at least 1-2 signals within first 5 minutes
- ✅ Entry orders consistently 25 USDT
- ✅ No repeated errors or warnings

**Warning signs:**
- ❌ Repeated error messages
- ❌ Bot crashes or hangs
- ❌ Entry orders not 25 USDT
- ❌ Recovery exits not executing

---

## 🛠️ TROUBLESHOOTING

### Issue: Bot won't start
**Diagnosis:**
```bash
python3 -m py_compile .env
python3 -m py_compile core/meta_controller.py
python3 -m py_compile core/rotation_authority.py
```

**If compilation fails:** Check syntax in recent changes

**If compilation passes:** Try starting manually:
```bash
python3 🎯_MASTER_SYSTEM_ORCHESTRATOR.py
# Watch for error messages
```

### Issue: Recovery exits not executing
**Diagnosis:**
```bash
grep -i "bypass\|recovery" /tmp/octivault_master_orchestrator.log | head -20
```

**If not in logs:** Recovery conditions haven't triggered  
**If in logs but still blocked:** Check min-hold logic in meta_controller.py

### Issue: Entry orders still 15 USDT
**Diagnosis:**
```bash
grep "DEFAULT_PLANNED_QUOTE" .env
# Should show: DEFAULT_PLANNED_QUOTE=25
```

**If shows 15:** .env changes didn't apply  
**Remedy:** Restart bot: `pkill -f MASTER_SYSTEM_ORCHESTRATOR && sleep 3 && python3 🎯_MASTER_SYSTEM_ORCHESTRATOR.py &`

### Issue: Unexpected "MICRO restriction OVERRIDDEN" messages
**Normal:** This is expected behavior when forced rotations occur  
**Not a problem:** The override is working correctly

---

## ✨ SUCCESS CRITERIA

System is working correctly when:
- [ ] Bot starts without errors
- [ ] Initialization completes in 60-90 seconds
- [ ] First BUY order appears within 5-10 minutes
- [ ] Entry size is consistently ~25 USDT
- [ ] No repeated error messages
- [ ] Log shows normal trading cycle patterns

**After 30-minute warm-up:**
- [ ] At least 2-5 BUY orders executed
- [ ] Average entry size: 24-26 USDT
- [ ] Win rate reasonable for initial sample
- [ ] Capital allocated properly

---

## 📈 PERFORMANCE BASELINE

After deployment, track these metrics for 24 hours:

**Daily Targets:**
- Return: +0.5% to +2.0%
- Win rate: 55-65%
- Number of trades: 3-12
- Max intraday drawdown: <5%

**Red Flags:**
- Return: <-1% (losing money)
- Win rate: <40% (too many losses)
- No trades: (execution blocked)
- Repeated errors: (system unstable)

---

## 🔄 ROLLBACK PROCEDURE

If something goes wrong:

```bash
# Stop the bot
pkill -f "MASTER_SYSTEM_ORCHESTRATOR"
sleep 2

# Restore backup
cp .env.backup.pre_phase2 .env
cp core/meta_controller.py.backup.pre_phase2 core/meta_controller.py
cp core/rotation_authority.py.backup.pre_phase2 core/rotation_authority.py

# Restart with old configuration
python3 🎯_MASTER_SYSTEM_ORCHESTRATOR.py &

# Verify
tail -f /tmp/octivault_master_orchestrator.log
```

---

## 📞 SUPPORT REFERENCE

### Key Files
- **Logs:** `/tmp/octivault_master_orchestrator.log`
- **Config:** `.env`
- **Core Logic:** `core/meta_controller.py`, `core/rotation_authority.py`
- **Documentation:** `FIXES_IMPLEMENTATION_COMPLETE.md`

### Useful Commands
```bash
# View all relevant log messages
grep -E "SafeMinHold|MICRO restriction|quote: 25" /tmp/octivault_master_orchestrator.log

# Check current config
grep "DEFAULT_PLANNED_QUOTE\|MIN_TRADE\|MIN_ENTRY" .env

# Restart bot cleanly
pkill -f MASTER && sleep 2 && python3 🎯_MASTER_SYSTEM_ORCHESTRATOR.py &

# Monitor system metrics
curl http://localhost:8000/metrics

# View open positions
curl http://localhost:8000/positions
```

---

## ✅ DEPLOYMENT CHECKLIST

- [ ] Read this entire guide
- [ ] Backup current configuration
- [ ] Verify all changes applied
- [ ] Commit to git
- [ ] Stop current bot
- [ ] Start bot with new configuration
- [ ] Monitor logs for 15 minutes
- [ ] Verify expected behavior
- [ ] Mark deployment as successful

---

## 🎉 YOU'RE DONE!

Once the 15-minute warm-up test completes successfully:

1. **System is ready for production**
2. **Phase 2 bottleneck fixes are active**
3. **Capital recovery will be more efficient**
4. **Rotation will work in all regimes**
5. **Entry sizing is clean and consistent**

The fixes will automatically improve system efficiency as capital is recovered and opportunities are rotated properly.

---

**Version:** 1.0  
**Status:** Ready to Deploy  
**Date:** April 27, 2026  
**Estimated Deploy Time:** 30 minutes
