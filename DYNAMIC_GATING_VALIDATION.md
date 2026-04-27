# Dynamic Gating Implementation - Validation & Monitoring Guide

## 🧪 Quick Start: Verifying Dynamic Gating is Active

### Step 1: Check for Gating Logs

Run this command to monitor the dynamic gating system in real-time:

```bash
tail -f logs/trading_run_*.log | grep -E "\[Meta:DynamicGating\]|\[Meta:Gating\]"
```

**Expected Output** (you should see these patterns):

**Minutes 0-5 (BOOTSTRAP)**:
```
[Meta:DynamicGating] phase=BOOTSTRAP, should_relax=False, gated_reasons=[], success_rate=0.0%, attempts=0/2
[Meta:Gating] Recorded execution: attempted=True, successful=False, total_attempts=1, total_fills=0, recent_success_rate=0.0%
```

**Minutes 5-15 (INITIALIZATION)**:
```
[Meta:DynamicGating] phase=INITIALIZATION, should_relax=False, gated_reasons=[], success_rate=33.3%, attempts=1/2
[Meta:Gating] Recorded execution: attempted=True, successful=True, total_attempts=2, total_fills=1, recent_success_rate=50.0%
[Meta:DynamicGating] phase=INITIALIZATION, should_relax=True, gated_reasons=[], success_rate=50.0%, attempts=2/2
```

**Minutes 20+ (STEADY_STATE)**:
```
[Meta:DynamicGating] phase=STEADY_STATE, should_relax=True, gated_reasons=[], success_rate=75.0%, attempts=8/50
```

---

## 📊 Monitoring Commands

### Command 1: Watch Phase Transitions

```bash
# Show only phase transitions
tail -f logs/trading_run_*.log | grep "phase=" | head -20
```

**Expected Pattern**:
```
phase=BOOTSTRAP (0-5 min)
phase=BOOTSTRAP (continues...)
phase=INITIALIZATION (5-20 min)
phase=INITIALIZATION (continues...)
phase=STEADY_STATE (20+ min)
```

### Command 2: Track Success Rate Progression

```bash
# Extract success rate over time
tail -f logs/trading_run_*.log | grep "\[Meta:DynamicGating\]" | grep -oE "success_rate=[0-9.%]+" | tail -50
```

**Expected Pattern**:
```
success_rate=0.0%
success_rate=0.0%
success_rate=50.0%   ← Gates relax at this point!
success_rate=50.0%
success_rate=60.0%
success_rate=70.0%
success_rate=75.0%
```

### Command 3: Check Trading Signal Generation

```bash
# Show decision transitions from NONE to actual signals
tail -f logs/trading_run_*.log | grep "decision=" | head -30
```

**Expected Pattern**:
```
decision=NONE (Bootstrap - signals blocked)
decision=NONE (Bootstrap - signals blocked)
decision=HOLD (Initialization - SELL-only)
decision=HOLD (Initialization - SELL-only)
decision=BUY  ← GATES RELAXED! BUY signals now allowed
decision=BUY  ← Multiple BUY signals appear
decision=HOLD
decision=BUY
```

### Command 4: Monitor Trade Execution

```bash
# Show when trades actually open
tail -f logs/trading_run_*.log | grep "trade_opened=" | head -20
```

**Expected Pattern**:
```
trade_opened=False (first 5+ minutes)
trade_opened=False (continues...)
trade_opened=True  ← First trade opens!
trade_opened=False (not every loop opens trades)
trade_opened=True  ← Second trade opens
```

### Command 5: Track PnL Accumulation

```bash
# Show realized_pnl progression
tail -f logs/trading_run_*.log | grep "\[LOOP_SUMMARY\]" | grep -oE "pnl=[^]]*" | head -30
```

**Expected Pattern**:
```
pnl=0.00 (bootstrap - no trades)
pnl=0.00 (bootstrap - no trades)
pnl=+2.50 ← First trade profit!
pnl=+1.25 ← Second trade profit
pnl=+3.75 ← Accumulating
pnl=+5.50 ← Getting closer to $10 target
pnl=+7.25
pnl=+10.00+ ← TARGET REACHED! ✅
```

---

## 🔍 Deep Dive: Understanding Each Log Entry

### Log Format

```
[Meta:DynamicGating] phase=<PHASE>, should_relax=<BOOL>, gated_reasons=<LIST>, 
                     success_rate=<PERCENT>%, attempts=<CURRENT>/<MINIMUM>
```

### Phase Values

| Value | Duration | Meaning |
|-------|----------|---------|
| `BOOTSTRAP` | 0-5 min | System initializing, gates STRICT (readiness required) |
| `INITIALIZATION` | 5-20 min | System warming up, gates ADAPTIVE (success-rate based) |
| `STEADY_STATE` | 20+ min | System stable, gates RELAXED (health metrics only) |

### `should_relax` Values

| Value | Meaning |
|-------|---------|
| `False` | Gates are STRICT - only SELL/HOLD signals allowed |
| `True` | Gates are RELAXED - BUY/SELL/HOLD all allowed |

### `gated_reasons` Values

When gates are strict, this shows why:
- Empty `[]` = No specific gate reasons (normal during relaxed)
- `['MarketData']` = Market data not ready yet
- `['Balances']` = Balance snapshot incomplete
- `['OpsPlane']` = Operations plane not initialized
- Multiple reasons possible: `['MarketData', 'Balances']`

When relaxed, only critical issues gate:
- `['Balances_CRITICAL']` = Balance issue detected despite relaxed mode

### `success_rate` Values

Percentage of recent execution attempts that succeeded:
- `0.0%` = No successful fills yet (all rejected)
- `25.0%` = 1 out of 4 attempts succeeded
- `50.0%` = 50% threshold reached, GATES RELAX!
- `75.0%` = 3 out of 4 recent attempts succeeded

### `attempts=X/Y` Values

- `X` = Current number of recorded attempts in rolling window
- `Y` = Minimum attempts required to check success rate (default: 2)

Examples:
- `attempts=0/2` = No attempts recorded yet
- `attempts=1/2` = 1 attempt recorded, still need 1 more to check rate
- `attempts=2/2` = 2 attempts recorded, can now check if >= threshold
- `attempts=5/2` = 5 attempts recorded (window keeps max 50)

---

## ✅ Validation Checklist

Run this checklist to verify dynamic gating is working:

### Phase 1: System Start (Minutes 0-5)

- [ ] Orchestrator starts and enters main loop
- [ ] Logs show `phase=BOOTSTRAP` entries
- [ ] `should_relax=False` in all BOOTSTRAP logs
- [ ] Decision shows `NONE` or `HOLD` (BUY signals blocked)
- [ ] `success_rate=0.0%` in early logs

### Phase 2: First Execution (Minutes 5-10)

- [ ] After ~5 min, logs transition to `phase=INITIALIZATION`
- [ ] First execution attempt recorded: `Recorded execution: attempted=True`
- [ ] `success_rate` begins updating (shows 0%, 25%, 33%, etc.)
- [ ] Decision still shows `NONE` or `HOLD` (waiting for threshold)
- [ ] `attempts` counter increments toward minimum

### Phase 3: Gate Relaxation (Minutes 10-15)

- [ ] Success rate reaches threshold (≥50%)
- [ ] Logs show `should_relax=True` ✅
- [ ] Decision transitions from `NONE` to `BUY` ✅
- [ ] First `BUY` decision appears in LOOP_SUMMARY
- [ ] `trade_opened=True` appears shortly after

### Phase 4: Trading Execution (Minutes 15-20)

- [ ] First trade opens successfully
- [ ] `realized_pnl` shows positive value (e.g., `+2.50`)
- [ ] Multiple BUY/SELL decisions appear in logs
- [ ] `decision` varies between BUY, SELL, HOLD
- [ ] PnL accumulates (each loop shows new total)

### Phase 5: Steady State (Minutes 20+)

- [ ] Logs transition to `phase=STEADY_STATE`
- [ ] `should_relax=True` (gates always relaxed now)
- [ ] `success_rate` continues improving (70%+)
- [ ] Trades execute regularly (not every loop, but many)
- [ ] PnL continues accumulating

### Phase 6: Target Achievement (24-hour goal)

- [ ] `realized_pnl` approaches $10 USDT
- [ ] System shows `pnl=10.00+` or `pnl=10.50`, etc.
- [ ] All 24 hours worth of trades executed successfully
- [ ] System demonstrates consistent execution capability

---

## 🐛 Troubleshooting Guide

### Issue 1: "No DynamicGating logs appearing"

**Check**:
```bash
# Verify logs exist and are being written
ls -lah logs/trading_run_*.log

# Verify gating logs in the file
grep "\[Meta:DynamicGating\]" logs/trading_run_*.log | head -5
```

**If no logs**: 
- [ ] Is the orchestrator running? Check PID
- [ ] Is the log file being written? Check modification time
- [ ] Did implementation add the logging? Check meta_controller.py for log statements

### Issue 2: "Stuck on phase=BOOTSTRAP after 10 minutes"

**Check**:
```bash
# Verify phase transitions aren't happening
grep "phase=" logs/trading_run_*.log | tail -20 | uniq -c

# Check elapsed time logic
grep "_gating_start_time\|elapsed_sec" logs/trading_run_*.log
```

**Likely Causes**:
- [ ] `_gating_start_time` not set correctly in __init__
- [ ] `_update_gating_phase()` not being called
- [ ] Time calculation off (wrong duration thresholds)

**Fix**:
```python
# Verify in __init__:
self._gating_start_time = time.time()  # ✅ Must be set

# Verify _should_relax_gates() calls _update_gating_phase():
phase = self._update_gating_phase()  # ✅ Must be called
```

### Issue 3: "Success rate stuck at 0% even with execution attempts"

**Check**:
```bash
# Verify execution results are recorded
grep "\[Meta:Gating\] Recorded execution" logs/trading_run_*.log | head -10
```

**Likely Causes**:
- [ ] `_record_execution_result()` not being called
- [ ] All execution attempts are failing (rejected)
- [ ] `execution_successful` always False

**Diagnostic**:
```bash
# Check if trades are being rejected
grep "exec_result" logs/trading_run_*.log | sort | uniq -c

# Example output:
# 150 exec_result=REJECTED  ← All rejections = 0% success
# 100 exec_result=SUCCESS   ← 40% success = gates relax soon
```

### Issue 4: "Gates never relax to should_relax=True"

**Check**:
```bash
# Verify threshold logic
grep "success_rate" logs/trading_run_*.log | tail -10

# See if any reach 50%
grep "success_rate" logs/trading_run_*.log | grep -E "50\.0|[6-9][0-9]\.[0-9]|100\.0"
```

**Likely Causes**:
- [ ] Success rate never reaches threshold (all attempts fail)
- [ ] `_gating_min_attempts` requirement not met
- [ ] Threshold value too high (default 50%)

**Fix**:
```python
# If too many rejections, check execution issues
# If threshold too high, lower it temporarily:
self._gating_success_threshold = 0.30  # Try 30% instead of 50%
```

### Issue 5: "BUY signals appear but trades don't open"

**Check**:
```bash
# See if BUY decisions convert to trades
grep "decision=BUY\|trade_opened" logs/trading_run_*.log | head -20

# Look for rejection logs
grep "REJECT\|rejection" logs/trading_run_*.log | tail -20
```

**Likely Causes**:
- [ ] BUY signals generated but execution fails
- [ ] Insufficient balance despite gates relaxing
- [ ] Notional or risk checks failing

**Next Steps**:
- Check execution logs for detailed rejection reasons
- Verify capital is available before trades
- Check if risk limits are being enforced

---

## 📈 Performance Metrics

### Key Metrics to Track

```bash
# Count phase transitions
grep "\[Meta:DynamicGating\]" logs/trading_run_*.log | grep -c "phase=BOOTSTRAP"
grep "\[Meta:DynamicGating\]" logs/trading_run_*.log | grep -c "phase=INITIALIZATION"
grep "\[Meta:DynamicGating\]" logs/trading_run_*.log | grep -c "phase=STEADY_STATE"

# Count gate relaxations
grep "gates_relaxed" logs/trading_run_*.log | tail -1

# Track success rate improvement
grep "\[Meta:DynamicGating\]" logs/trading_run_*.log | tail -30 | grep -oE "success_rate=[0-9.]+" | sort -u

# Count successful trades
grep "trade_opened=True" logs/trading_run_*.log | wc -l

# Sum total PnL
grep "\[LOOP_SUMMARY\]" logs/trading_run_*.log | tail -1 | grep -oE "pnl=[+-]?[0-9.]+"
```

---

## 🎯 Success Criteria

Dynamic gating implementation is **SUCCESSFUL** if:

1. **✅ Phase Progression**: System moves through BOOTSTRAP → INITIALIZATION → STEADY_STATE
2. **✅ Signal Generation**: `decision` changes from `NONE` to `BUY`/`SELL`/`HOLD` within 15 minutes
3. **✅ Trade Execution**: `trade_opened=True` appears within 20 minutes
4. **✅ Profit Accumulation**: `pnl` shows positive values and grows
5. **✅ Target Achievement**: PnL reaches $10+ USDT within session

---

## 📞 Quick Reference

### All Gating-Related Log Entries

```bash
# Filter to see just gating logs
tail -f logs/trading_run_*.log | grep -E "\[Meta:(Gating|DynamicGating)\]"
```

### Configuration to Adjust

```python
# In config.py:
GATING_BOOTSTRAP_DURATION_SEC = 300.0   # Default: 5 min (too short? increase)
GATING_INIT_DURATION_SEC = 900.0        # Default: 15 min (adjust init window)
GATING_SUCCESS_THRESHOLD = 0.50         # Default: 50% (gates too strict? lower to 0.30)
GATING_MIN_ATTEMPTS = 2                 # Default: 2 (need more data? increase to 3-5)
```

### Emergency Debugging

If system not working as expected:

```bash
# 1. Verify orchestrator is running with dynamic gating enabled
ps aux | grep -i orchestrator | grep -v grep

# 2. Check for errors in logs
grep -i error logs/trading_run_*.log | tail -20

# 3. See full gating lifecycle
grep "\[Meta" logs/trading_run_*.log | head -100 | tail -50

# 4. Compare before/after gate relaxation
grep "\[Meta:DynamicGating\]" logs/trading_run_*.log | \
  grep -B5 "should_relax=True" | head -10
```

---

## 🚀 Next: Reaching the $10 USDT Target

Once dynamic gating is confirmed working (trades opening, PnL positive):

1. Let the system run continuously (24-hour session)
2. Monitor PnL accumulation every 30 minutes
3. Gates should stay relaxed in STEADY_STATE (20+ min)
4. Multiple trades per hour should execute
5. PnL should accumulate at ~$0.50-$1.00 per hour
6. **Target**: Reach $10.00+ USDT within 24 hours

Estimated timeline:
- 0-5 min: BOOTSTRAP (preparing)
- 5-15 min: INITIALIZATION (proving capability)
- 15-20 min: Gates relax, first trades open
- 20 min - 24 hours: STEADY_STATE (accumulating profits)
- **Goal**: 10 USDT profit ÷ 12 hours trading = ~$0.83/hour

---

