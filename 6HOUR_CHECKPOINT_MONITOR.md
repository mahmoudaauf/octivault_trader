# 6-Hour Trading Session: Checkpoint & Active Monitoring Guide

**Phase 2 Validation with Real-Time Checkpoints**

---

## 📊 Overview

This guide sets up **active monitoring with 50-minute checkpoints** to validate Phase 2 implementations:
- ✅ Recovery exit min-hold bypass
- ✅ Forced rotation micro bracket override  
- ✅ Entry sizing alignment (25 USDT)
- ✅ Capital velocity governance
- ✅ Risk management compliance

---

## 🔖 Checkpoint Schedule (50-minute intervals)

| # | Checkpoint | Time | Duration | Purpose |
|---|-----------|------|----------|---------|
| 1 | INITIALIZATION & WARMUP | 0:00-0:15 | 15 min | Load config, connect exchange, validate credentials |
| 2 | PHASE 2 VERIFICATION | 0:15-0:50 | 35 min | Verify recovery bypass & forced rotation wiring |
| 3 | ENTRY SIZING CHECK | 0:50-1:40 | 50 min | Validate 25 USDT alignment across all entries |
| 4 | CAPITAL ALLOCATION | 1:40-2:30 | 50 min | Review allocation efficiency & velocity |
| 5 | MID-SESSION STATUS | 2:30-3:00 | 30 min | P&L check, trade quality, active positions |
| 6 | ROTATION ESCAPE | 3:00-3:50 | 50 min | Validate forced rotation overrides working |
| 7 | LIQUIDITY RESTORATION | 3:50-4:40 | 50 min | Check recovery exit bypass triggers |
| 8 | FINAL PERFORMANCE | 4:40-5:50 | 70 min | Performance review, sharpe ratio, drawdown |
| 9 | SESSION COMPLETION | 5:50-6:00 | 10 min | Final summary, data export |

---

## 📈 What to Monitor at Each Checkpoint

### ✅ Checkpoint 1: INITIALIZATION (15 minutes)

**Expected Behavior:**
```
✅ Configuration loaded
✅ Exchange connection established
✅ Credentials validated
✅ Market data streaming
✅ Signal generation active
✅ Trading engine ready
```

**Red Flags:**
- ❌ Connection errors
- ❌ Missing configuration
- ❌ Signal generation delays

---

### ✅ Checkpoint 2: PHASE 2 VERIFICATION (35 minutes)

**Phase 2 Fix #1: Recovery Exit Bypass**
```
Look for log lines:
[Meta:SafeMinHold] Bypassing min-hold check for forced recovery exit: BTCUSDT

Expected: 1-3 recovery exits triggered
Behavior: Recovery exits should execute without min-hold blocking
```

**Phase 2 Fix #2: Forced Rotation Override**
```
Look for log lines:
[REA:authorize_rotation] ⚠️ MICRO restriction OVERRIDDEN for ETHUSDT

Expected: 1-2 forced rotations with override
Behavior: Forced rotations should override MICRO bracket restrictions
```

**Phase 2 Fix #3: Entry Sizing**
```
Expected: All entry signals use 25 USDT base size
Log lines should show consistent sizing:
Entry: BTCUSDT @ 25.0 USDT
Entry: ETHUSDT @ 25.1 USDT (within ±1 tolerance)
```

**Metrics to Check:**
- [ ] Recovery bypasses triggered: ✅ (should be 1-3)
- [ ] Forced rotations overridden: ✅ (should be 1-2)
- [ ] Entry size consistency: ✅ (within ±1 USDT of 25.0)

---

### ✅ Checkpoint 3: ENTRY SIZING CHECK (50 minutes)

**Validation Points:**
```
Configuration Parameters:
✅ DEFAULT_PLANNED_QUOTE: 25 USDT
✅ MIN_ENTRY_USDT: 25 USDT
✅ MIN_ENTRY_QUOTE_USDT: 25 USDT
✅ EMIT_BUY_QUOTE: 25 USDT
✅ META_MICRO_SIZE_USDT: 25 USDT

Actual Entry Sizes (sample):
✅ Entry 1: 25.0 USDT
✅ Entry 2: 24.9 USDT (within tolerance)
✅ Entry 3: 25.1 USDT (within tolerance)
✅ Entry 4: 25.0 USDT
```

**Red Flags:**
- ❌ Any entry < 23 USDT or > 27 USDT
- ❌ Inconsistent sizing across symbols
- ❌ Mismatch between config and actual

---

### ✅ Checkpoint 4: CAPITAL ALLOCATION (50 minutes)

**Check These Metrics:**
```
Total Capital: $10,000 USDT (example)
├─ Allocated to Positions: $7,500 (75%)
├─ Reserve (MICRO): $2,500 (25%)
└─ Available for New Entries: Active

Allocation Efficiency:
✅ 70-80% actively deployed is healthy
✅ Capital velocity showing rotations
✅ Reserve sufficient for emergency rotations
```

**Red Flags:**
- ❌ Capital stuck in low-performing positions
- ❌ Reserve depleted
- ❌ Poor capital velocity (< 2 rotations/hour)

---

### ✅ Checkpoint 5: MID-SESSION STATUS (30 minutes)

**Trading Metrics:**
```
Active Positions: 6-10 (healthy range)
├─ Winning Trades: 60%+ (target)
├─ Losing Trades: 40%- (managed)
└─ Current P&L: Monitor trend

Performance:
✅ Win Rate: 60-75% (Phase 2 should improve this)
✅ Avg Trade Duration: 10-30 minutes
✅ Capital Velocity: 1-2 rotations/hour
```

**Expected P&L at 50% through session:**
- Conservative: +2% to +5%
- Optimistic: +5% to +10%
- Concerning: < 0%

---

### ✅ Checkpoint 6: ROTATION ESCAPE VALIDATION (50 minutes)

**Phase 2 Fix Validation:**
```
Verify forced rotations are executing:

Expected Pattern:
1. Position accumulates (MICRO bracket fills)
2. Forced rotation triggered
3. [REA:authorize_rotation] ⚠️ MICRO restriction OVERRIDDEN
4. Position exits at market
5. Capital re-deployed

Success Indicators:
✅ Override messages appear in logs
✅ Average exit time: 1-3 minutes
✅ Success rate: 100%
```

**Red Flags:**
- ❌ No override messages
- ❌ Rotations timing out
- ❌ Capital stuck in forced positions

---

### ✅ Checkpoint 7: LIQUIDITY RESTORATION CHECK (50 minutes)

**Phase 2 Fix Validation:**
```
Recovery Exit Bypass Monitoring:

Expected Pattern:
1. Position stagnates (price not moving favorably)
2. Recovery exit triggered
3. [Meta:SafeMinHold] Bypassing min-hold check detected
4. Position exits despite min-hold being active
5. Capital restored to reserve

Metrics:
✅ Recovery exits triggered: 2-5 (depends on market)
✅ Min-hold bypasses: 2-5 (should match recovery exits)
✅ Success rate: 100%
✅ Avg restoration time: 2-5 minutes
```

**Red Flags:**
- ❌ No recovery exit triggers
- ❌ Recovery exits blocked (min-hold not bypassed)
- ❌ Capital not being restored

---

### ✅ Checkpoint 8: FINAL PERFORMANCE (70 minutes)

**Comprehensive Review:**
```
Trading Metrics:
✅ Total trades: 20-30 (depending on market activity)
✅ Win rate: 60-75%
✅ Avg trade profit/loss: +0.5% to +2% per trade
✅ Final P&L: +5% to +15%

Risk Metrics:
✅ Max drawdown: -2% to -5%
✅ Sharpe ratio: > 1.5 (Phase 2 should improve)
✅ Recovery time from drawdown: < 30 minutes

Phase 2 Impact:
✅ Recovery bypasses executed: 2-5
✅ Forced rotation overrides: 1-3
✅ Entry sizing consistency: > 95%
```

---

### ✅ Checkpoint 9: SESSION COMPLETION (10 minutes)

**Final Data Export:**
```
Generate and verify:
✅ 6hour_session_report_monitored.json
✅ 6hour_session_checkpoint_summary.txt
✅ 6hour_session_monitored.log

Export should contain:
✅ All checkpoint data
✅ Phase 2 metrics
✅ Alert history
✅ Trading P&L
```

---

## 🚀 Quick Start

### Option 1: Monitored Session (Recommended for Phase 2 Validation)

```bash
# Run with checkpoints and active monitoring
python3 RUN_6HOUR_SESSION_MONITORED.py

# Watch logs in real-time
tail -f 6hour_session_monitored.log
```

### Option 2: Manual Checkpoint System

```bash
# Start trading
python3 run_trading.sh &

# At each checkpoint time, run:
python3 verify_fixes.py  # Check Phase 2 fixes still active

# After session, generate report:
python3 -c "
import json
checkpoints = {
    'checkpoint_2': {'recovery_bypasses': 3, 'forced_rotations': 2},
    'checkpoint_3': {'entry_sizes': [25.0, 25.1, 24.9]},
    # ... etc
}
print(json.dumps(checkpoints, indent=2))
" > session_checkpoints.json
```

---

## 📊 Expected Phase 2 Outcomes

By end of 6-hour session, you should see:

### Recovery Exit Bypass
- ✅ 2-5 recovery exit triggers
- ✅ All bypassed min-hold restriction
- ✅ 100% success rate (exits executed)
- ✅ ~$250-$500 capital freed up

### Forced Rotation Override
- ✅ 1-3 forced rotation triggers
- ✅ All overrode MICRO bracket
- ✅ 100% success rate (rotations completed)
- ✅ Capital reallocated efficiently

### Entry Sizing Consistency
- ✅ 20-30 entries tracked
- ✅ All within ±1 USDT of 25.0
- ✅ < 1% mismatch rate
- ✅ Improved capital efficiency

### Overall P&L Impact
- ✅ Phase 2 improvements visible: +1% to +3%
- ✅ Better capital utilization: +2-3% ROC
- ✅ Improved exit timing: lower slippage
- ✅ Better rotation execution: higher throughput

---

## 🔴 If Things Go Wrong

### Recovery Bypass Not Triggering
```
Symptom: No [Meta:SafeMinHold] messages in log

Fix:
1. Verify bypass flag in meta_controller.py:
   stagnation_exit_sig["_bypass_min_hold"] = True
2. Check signal format: _bypass_min_hold field exists
3. Run: python3 verify_fixes.py
```

### Forced Rotation Override Not Working
```
Symptom: No [REA:authorize_rotation] override messages

Fix:
1. Verify override logic in rotation_authority.py:
   if owned_positions and not force_rotation:
2. Check force_rotation flag is set
3. Run: python3 verify_fixes.py
```

### Entry Sizes Misaligned
```
Symptom: Entry sizes vary wildly (10-50 USDT)

Fix:
1. Verify .env values all set to 25:
   grep "USDT=25\|QUOTE=25" .env
2. Check config.py floor alignment
3. Run config verification:
   python3 -c "from core.config import *; print(SIGNIFICANT_POSITION_FLOOR)"
```

---

## 📝 Checkpoint Recording Template

For manual monitoring, record at each checkpoint:

```
CHECKPOINT {N}: {NAME}
Time: {HH:MM:SS}
Elapsed: {X} minutes / 360 minutes

Phase 2 Status:
  Recovery Bypasses: {count}
  Forced Rotations: {count}
  Entry Size Range: ${min}-${max}
  
Trading Status:
  Active Positions: {count}
  Current P&L: {$amount}
  Win Rate: {X}%
  
Issues:
  - {issue if any}
  
Notes:
  - {observation if any}
```

---

## ✅ Success Criteria

Session is successful if:

1. ✅ **All 3 Phase 2 fixes demonstrated**
   - Recovery bypass working (logs show triggers)
   - Forced rotation override working (logs show triggers)
   - Entry sizing consistent (all 25 ± 1 USDT)

2. ✅ **No blocking errors**
   - No crashed processes
   - All checkpoints completed
   - Trading continuing throughout

3. ✅ **Positive P&L**
   - Net gain > 0% (even +1% is success)
   - No catastrophic drawdowns
   - Capital preserved

4. ✅ **Reports generated**
   - 6hour_session_report_monitored.json exists
   - 6hour_session_checkpoint_summary.txt exists
   - 6hour_session_monitored.log has no errors

---

## 🎯 Next Steps After Session

1. **Analyze Results**
   - Review checkpoint_summary.txt
   - Check recovery bypass count vs expected
   - Check forced rotation override count
   
2. **Compare Baselines**
   - Compare P&L before/after Phase 2
   - Compare capital velocity improvement
   - Compare exit timing improvement

3. **Document Findings**
   - Create post-session analysis
   - Record Phase 2 impact metrics
   - Plan Phase 3 if needed

4. **Deploy to Production**
   - If metrics positive, deploy with confidence
   - Monitor first 24 hours closely
   - Compare against baseline

---

## 📞 Support

If you need to check Phase 2 status:

```bash
# Quick verification
python3 verify_fixes.py

# Full system diagnostic
python3 component_validator.py

# Check logs for Phase 2 triggers
grep "Bypassing min-hold\|MICRO restriction OVERRIDDEN" 6hour_session_monitored.log
```

---

**Ready to monitor Phase 2? Run:** `python3 RUN_6HOUR_SESSION_MONITORED.py`
