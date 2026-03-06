# Signal Filtering Debug - Implementation Summary

## What Was Added

I've added comprehensive diagnostic logging to help identify why signals are being generated and cached but not converted to trading decisions.

### Code Changes to meta_controller.py

**1. Signal Intake Logging (line ~9475)**
```python
if all_signals:
    self.logger.warning(
        "[Meta:SIGNAL_INTAKE] Retrieved %d signals from cache: %s",
        len(all_signals),
        [(s.get("symbol"), s.get("action"), float(s.get("confidence", 0.0))) for s in all_signals]
    )
```
**Purpose:** Shows all signals retrieved from SignalManager cache

**2. Gate Entry Tracing (line ~9598)**
```python
self.logger.debug(
    "[Meta:GATE_TRACE] Processing %s %s from %s (conf=%.2f)",
    sym, action, sig.get("agent", "?"), float(sig.get("confidence", 0.0))
)
```
**Purpose:** Logs each signal as it enters the filtering pipeline

**3. Capital Recovery Gate Drop (line ~9611)**
```python
self.logger.warning(
    "[Meta:GATE_DROP_RECOVERY] %s BUY dropped at CAPITAL_RECOVERY gate (not bootstrap)",
    sym
)
```
**Purpose:** Shows when signals are rejected due to capital recovery mode

**4. Tradeability Gate Drop (line ~9679)**
```python
self.logger.warning(
    "[Meta:GATE_DROP_TRADEABILITY] %s BUY dropped at TRADEABILITY gate: conf=%.3f floor=%.3f gate=%s",
    sym, conf, final_exec_floor, gate_reason
)
```
**Purpose:** Shows when signals fail confidence floor check (most common cause)

**5. One-Position Gate Drop (line ~9871)**
```python
self.logger.warning(
    "[Meta:GATE_DROP_ONE_POSITION] %s BUY dropped at ONE_POSITION_PER_SYMBOL gate (qty=%.6f)",
    sym, existing_qty
)
```
**Purpose:** Shows when signals blocked by existing positions

**6. Gate Passed (line ~9993)**
```python
self.logger.warning(
    "[Meta:GATE_PASSED] %s %s PASSED ALL GATES and ADDED to valid_signals (conf=%.3f agent=%s)",
    sym, action, float(sig.get("confidence", 0.0)), sig.get("agent", "?")
)
```
**Purpose:** Shows when a signal successfully passes ALL filters

**7. After-Filter Summary (line ~10002)**
```python
self.logger.warning(
    "[Meta:AFTER_FILTER] valid_signals_by_symbol has %d symbols with signals: %s",
    len(valid_signals_by_symbol),
    {sym: [(s.get("action"), float(s.get("confidence", 0.0))) for s in sigs] 
     for sym, sigs in valid_signals_by_symbol.items()}
)
```
**Purpose:** Shows final state of signal dict after all filtering

**8. Deadlock Diagnostic (line ~10005)**
```python
if not valid_signals_by_symbol:
    self.logger.error(
        "[Meta:DEADLOCK_DIAGNOSTIC] 🔴 NO SIGNALS PASSED FILTERS! "
        "all_signals=%d, signals_by_sym=%d (pre-filter), valid_signals_by_symbol=%d (post-filter). "
        "LIKELY CAUSES: TRADEABILITY gate dropped by conf floor, CAPITAL_RECOVERY suppressed BUYs, "
        "ONE_POSITION_GATE blocked existing symbols, or PROBING gate blocked new symbols. "
        "Check logs for [Meta:GATE_DROP_*] messages to identify which gate(s) are filtering.",
        len(all_signals), len(signals_by_sym), len(valid_signals_by_symbol)
    )
```
**Purpose:** Error message explaining why no signals passed filters

## How to Use

### Step 1: Run the Bot
```bash
cd /Users/mauf/Desktop/Octi\ AI\ Trading\ Bot/octivault_trader
python main_phased.py 2>&1 | tee logs/diagnostic_run.log
```

### Step 2: Quick Analysis with Python Script
```bash
python analyze_diagnostics.py logs/diagnostic_run.log
```

Or manually:

### Step 3: Manual Analysis - Check Signal Generation
```bash
grep "\[Meta:SIGNAL_INTAKE\]" logs/diagnostic_run.log | tail -5
```
Expected: `Retrieved X signals` where X > 0

### Step 4: Manual Analysis - Trace Gate Drops
```bash
grep "\[Meta:GATE_DROP_\|Meta:GATE_PASSED\|Meta:GATE_TRACE\]" logs/diagnostic_run.log | head -50
```
Look for patterns:
- Many `GATE_TRACE` but no `GATE_PASSED` = signals are being dropped
- Many `GATE_DROP_TRADEABILITY` = confidence floor issue
- Many `GATE_DROP_RECOVERY` = capital recovery suppressing BUYs

### Step 5: Check Final State
```bash
grep "\[Meta:AFTER_FILTER\]\|\[Meta:DEADLOCK_DIAGNOSTIC\]" logs/diagnostic_run.log | tail -10
```
- If `valid_signals_by_symbol has 0 symbols` → all signals dropped
- If `valid_signals_by_symbol has >0 symbols` → signals made it through

## Expected Outcomes

### Good Flow (Signals Converting to Decisions)
```
[Meta:SIGNAL_INTAKE] Retrieved 3 signals from cache: [('OPNUSDT', 'BUY', 0.77), ...]
[Meta:GATE_TRACE] Processing OPNUSDT BUY from TrendHunter (conf=0.77)
[Meta:GATE_PASSED] OPNUSDT BUY PASSED ALL GATES and ADDED to valid_signals (conf=0.77 agent=TrendHunter)
[Meta:AFTER_FILTER] valid_signals_by_symbol has 1 symbols with signals: {'OPNUSDT': [('BUY', 0.77)]}
```

### Problem: Confidence Floor Drop
```
[Meta:SIGNAL_INTAKE] Retrieved 3 signals from cache: [('OPNUSDT', 'BUY', 0.45), ...]
[Meta:GATE_TRACE] Processing OPNUSDT BUY from TrendHunter (conf=0.45)
[Meta:GATE_DROP_TRADEABILITY] OPNUSDT BUY dropped at TRADEABILITY gate: conf=0.45 floor=0.55 gate=MODE_RECOVERY
[Meta:AFTER_FILTER] valid_signals_by_symbol has 0 symbols with signals: {}
[Meta:DEADLOCK_DIAGNOSTIC] 🔴 NO SIGNALS PASSED FILTERS! ... TRADEABILITY gate dropped by conf floor ...
```
**Solution:** TrendHunter needs to generate higher confidence signals, or bot needs to exit recovery mode

### Problem: Capital Recovery Gate
```
[Meta:GATE_DROP_RECOVERY] OPNUSDT BUY dropped at CAPITAL_RECOVERY gate (not bootstrap)
[Meta:GATE_DROP_RECOVERY] BARDUSDT BUY dropped at CAPITAL_RECOVERY gate (not bootstrap)
[Meta:DEADLOCK_DIAGNOSTIC] ... CAPITAL_RECOVERY suppressed BUYs ...
```
**Solution:** Exit capital recovery mode or wait for bootstrap signal

### Problem: No Signals Generated
```
[Meta:SIGNAL_INTAKE] Retrieved 0 signals from cache: []
[Meta:AFTER_FILTER] valid_signals_by_symbol has 0 symbols with signals: {}
[Meta:DEADLOCK_DIAGNOSTIC] all_signals=0 ...
```
**Solution:** Check TrendHunter agent, verify market data, check agent configuration

## Files Created/Modified

1. **meta_controller.py** - Added 8 diagnostic log points
2. **DIAGNOSTIC_GUIDE.md** - Comprehensive diagnostic guide (NEW)
3. **analyze_diagnostics.py** - Python script to auto-analyze logs (NEW)

## Next Steps After Running

1. Run bot with new diagnostics
2. Look for the dominant `[Meta:GATE_DROP_*]` message
3. Read DIAGNOSTIC_GUIDE.md for the specific scenario matching your logs
4. Adjust settings based on root cause identified
5. Re-run and verify signals are now passing through

## Example Analysis Command
```bash
# Get quick summary
grep "\[Meta:\(SIGNAL_INTAKE\|GATE_PASSED\|GATE_DROP\|AFTER_FILTER\|DEADLOCK\)\]" logs/diagnostic_run.log | tail -50

# Count drops by gate
grep "\[Meta:GATE_DROP" logs/diagnostic_run.log | sed 's/.*GATE_DROP_//' | sed 's/\].*//' | sort | uniq -c | sort -rn

# Find affected symbols
grep "\[Meta:GATE_DROP_TRADEABILITY\]" logs/diagnostic_run.log | awk '{print $7}' | sort | uniq -c
```

## Questions This Diagnostic System Answers

- ✅ Are signals being generated? (Check SIGNAL_INTAKE)
- ✅ Are signals entering the filter? (Check GATE_TRACE)
- ✅ Which gate is dropping signals? (Check GATE_DROP_* messages)
- ✅ How many signals made it through? (Check GATE_PASSED and AFTER_FILTER)
- ✅ What's the likely root cause? (Check DEADLOCK_DIAGNOSTIC)
- ✅ Is this a configuration issue or signal issue? (Compare GATE_TRACE vs GATE_PASSED)

The diagnostic logs create a complete trace of the signal filtering pipeline, making it easy to identify exactly where and why signals are being dropped.
