# 🚨 EMERGENCY ACTION CHECKLIST - GET SYSTEM TRADING AGAIN

**Status**: ⚠️ **ACCOUNT CRITICAL - $103.71 REMAINING**  
**Problem**: Cannot execute trades (need $25 entry, have $21.85)  
**Solution**: Reduce entry size to $5-10 USDT and test

---

## ✅ IMMEDIATE ACTIONS (Next 5 minutes)

### Step 1: Reduce Entry Size to Enable Trading
```bash
# Edit .env and change all entry sizing parameters to 5 USDT
# Lines to change:
DEFAULT_PLANNED_QUOTE=5          # was 25
MIN_TRADE_QUOTE=5                # was 25
MIN_ENTRY_USDT=5                 # was 25
TRADE_AMOUNT_USDT=5              # was 25
MIN_ENTRY_QUOTE_USDT=5           # was 25
EMIT_BUY_QUOTE=5                 # was 25
META_MICRO_SIZE_USDT=5           # was 25
MIN_SIGNIFICANT_POSITION_USDT=5  # was 25
```

### Step 2: Verify Change Applied
```bash
grep "DEFAULT_PLANNED_QUOTE\|MIN_TRADE_QUOTE\|MIN_ENTRY_USDT" .env
# Should see all = 5
```

### Step 3: Restart Bot
```bash
pkill -f MASTER_SYSTEM_ORCHESTRATOR
sleep 2
export APPROVE_LIVE_TRADING=YES
nohup python3 🎯_MASTER_SYSTEM_ORCHESTRATOR.py > /tmp/octivault_master_orchestrator.log 2>&1 &
```

### Step 4: Monitor First Trades
```bash
tail -f /tmp/octivault_master_orchestrator.log | grep "exec_attempted=True\|exec_result"
# Watch for:
# - exec_attempted=True (trading resuming)
# - exec_result=SUCCESS or REJECTED
```

---

## 📊 WHAT TO WATCH FOR (Next 30 minutes)

### Good Signs ✅
- [ ] `exec_attempted=True` appearing in logs
- [ ] Some trades executing (not all rejected)
- [ ] Win rate improving (> 0%)
- [ ] Balance slowly increasing

### Bad Signs ❌
- [ ] `exec_attempted=False` still dominant
- [ ] Still rejecting ALL trades
- [ ] Still showing quote mismatch errors
- [ ] Balance continues dropping

### If Still Not Trading:
Check for:
1. `MICRO_CAPITAL_ADAPTIVE_ENTRY` warning with $5 entry
2. Position limit gate (max 2 positions)
3. Bootstrap gate blocking trades
4. Win rate gate

---

## 📈 TESTING PLAN (Next 2 hours)

### Test Phase 1: Can We Execute at All?
- Goal: Get 5-10 trades executed
- Watch: Win rate (target > 40%)
- If win rate < 20%: Signal quality issue

### Test Phase 2: Profitability Check
- Goal: After 20 trades, check if cumulative P&L positive
- If negative: Strategy needs debugging
- If positive: Ready to scale up

### Test Phase 3: Stability Check
- Goal: Run for 2 hours without crashes
- Watch: Memory usage, CPU, errors
- If stable: Ready for Phase 2 implementation

---

## 🔧 ALTERNATIVE: If Entry Size Reduction Doesn't Work

### Issue: Still can't trade at $5 entry
**Reason**: Position limit gate or other blocker

**Solution**: 
```bash
# Disable some strategies temporarily
# Edit core/app_context.py and comment out:
# - SwingTradeHunter (not performing well - 0% win rate)
# - MLForecaster (insufficient data - bootstrap phase)

# Keep enabled:
# - TrendHunter (simplest, most reliable)
# - One other proven strategy
```

### Issue: Still getting quote mismatch
**Reason**: Execution logic mismatch

**Solution**:
```bash
# Check ExecutionManager.place_order() implementation
# Verify meta controller quote == execution quote
# May need to enforce exact quote matching
```

---

## 🎯 SUCCESS CRITERIA

| Metric | Current | Target | Timeline |
|--------|---------|--------|----------|
| Trades/hour | 0 | 2-4 | 30 min |
| Win rate | 0% | 40%+ | 2 hours |
| Daily P&L | -$34 | +$0.50+ | 4 hours |
| Capital | $103.71 | Growing | 2 hours |

---

## 📋 FILES THAT NEED EDITING

### 1. `.env` - MOST IMPORTANT
```
Change these 8 parameters from 25 → 5:
- DEFAULT_PLANNED_QUOTE=5
- MIN_TRADE_QUOTE=5
- MIN_ENTRY_USDT=5
- TRADE_AMOUNT_USDT=5
- MIN_ENTRY_QUOTE_USDT=5
- EMIT_BUY_QUOTE=5
- META_MICRO_SIZE_USDT=5
- MIN_SIGNIFICANT_POSITION_USDT=5
```

### 2. `core/app_context.py` (Optional - if still blocking)
```
Comment out problematic strategies:
- SwingTradeHunter initialization
- MLForecaster (keep bootstrap off)
```

### 3. `core/meta_controller.py` (Optional - if quote mismatch persists)
```
Check quote passing logic in:
- _generate_entry_decision()
- _attempt_execution()
```

---

## ⏱️ TIMELINE

| Time | Action | Expected Result |
|------|--------|-----------------|
| Now | Edit .env entries (5 min) | Entry size = $5 |
| +5 min | Restart bot (2 min) | Bot starts fresh |
| +7 min | First signals (3-5 min) | Trades attempting |
| +12 min | First trades execute (30 sec) | `exec_attempted=True` |
| +13 min | Monitor first 5 trades (5 min) | Win rate visible |
| +18 min | First P&L snapshot (2 min) | See if profitable |
| +20 min | Decide: scale or debug (varies) | Adjust strategy |

---

## 🎊 SUCCESS: If Trades Start Executing

Once trades are executing at $5 size:

1. **Run for 30 minutes** - collect 5-10 trades
2. **Calculate win rate** - should be > 40%
3. **Check cumulative P&L** - should be positive
4. **If all good**: Gradually increase to $10, then $15, then $25
5. **If not good**: Debug signal quality (TrendHunter tuning)

---

## 🔴 FAILURE: If Trades Still Not Executing

**Check in this order:**

1. **Capital floor gate** (most likely)
   - [ ] Verify $5 entry size was applied
   - [ ] Check adaptive floor logic
   - [ ] May need to disable temporarily

2. **Position limits** (second most likely)
   - [ ] Check if 2/2 positions filled
   - [ ] May need to liquidate losing positions first

3. **Bootstrap gates** (third likely)
   - [ ] Win rate gate rejecting signals
   - [ ] Need to lower confidence threshold temporarily

4. **Quote mismatch** (if seeing EXEC_QUOTE_MISMATCH)
   - [ ] Check execution manager
   - [ ] May need to enforce exact matching

---

## 📞 NEED HELP?

If stuck:
1. Check `CRITICAL_PROFITABILITY_ISSUES.md` for full root cause analysis
2. Check `/tmp/octivault_master_orchestrator.log` for specific error
3. Look for the rejection_reason in LOOP_SUMMARY lines

---

**READY TO FIX?** → Apply the changes above and monitor for 30 minutes. Report win rate + P&L results.

